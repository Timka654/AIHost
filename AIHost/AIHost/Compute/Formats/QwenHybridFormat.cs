using AIHost.Inference;
using Microsoft.Extensions.Logging;

namespace AIHost.Compute.Formats;

/// <summary>
/// Qwen3.6 hybrid format: SSM (Mamba-2 style) + Attention layers.
/// Two block types:
///   Type A (blk.0,1,2,4..): combined attn_qkv.weight (gated Q) + SSM + FFN
///   Type B (blk.3,7,11..): separate attn_q/k/v.weight (gated Q + QK-norm) + FFN
///
/// Key characteristics:
///   - Every 4th layer is Type B (separate Q/K/V), rest are Type A (combined QKV)
///   - Type A: gated Q (qTotalDim = 2× kvDim), attn_gate.weight present
///   - Type B: gated Q (qTotalDim = 2× kvDim), attn_gate.weight present, QK-norm
///   - SSM recurrence in Type A layers
///   - post_attention_norm.weight used for FFN norm
/// </summary>
public class QwenHybridFormat : ITransformerFormat
{
    private readonly ILogger<QwenHybridFormat> _logger = AppLogger.Create<QwenHybridFormat>();

    public Tensor ApplyLayer(TransformerBase transformer, Tensor x, int layerIdx, uint position, KVCache? kvCache, SSMState? ssmState)
    {
        int g = transformer.GlobalLayer(layerIdx);
        var nm = transformer._nameMapper!;

        // Determine per-layer attention type
        bool hasCombinedQKV = transformer.HasWeight($"blk.{g}.attn_qkv.weight");
        bool hasSeparateQKV = transformer.HasWeight($"blk.{g}.attn_q.weight");

        if (!hasCombinedQKV && !hasSeparateQKV)
            return ApplyLayerSSMFallback(transformer, x, g, nm);

        if (hasSeparateQKV)
            return ApplyLayerTypeB(transformer, x, g, layerIdx, position, kvCache);

        // hasCombinedQKV
        return ApplyLayerCombinedQKV(transformer, x, g, layerIdx, position, kvCache, nm, ssmState);
    }

    /// <summary>
    /// Type B attention block (Qwen3.6 blk.3, 7, 11…):
    ///   separate attn_q/k/v weights, gated-Q (Q has 2× normal dim), QK-norm, standard W_o.
    ///
    /// Q [seqLen,12288] is split into q_proj[6144] and q_gate[6144]; gated_q = q_proj * SiLU(q_gate).
    /// Then standard GQA(gated_q[24h×256], K[4h×256], V[4h×256]) → attn_out[6144] → W_o → [5120].
    /// QK normalization weights (q_norm/k_norm [256]) applied as per-head LayerNorm approximation.
    /// Post-attention norm → FFN completes the block.
    /// </summary>
    private static Tensor ApplyLayerTypeB(TransformerBase transformer, Tensor x, int g, int layerIdx, uint position, KVCache? kvCache)
    {
        var ops = transformer.Ops;

        // 1. Pre-attention RMSNorm
        var xNorm = ops.Clone(x, "attn_norm_in");
        { var (wAN, sAN) = transformer.TempF32Named($"blk.{g}.attn_norm.weight"); ops.LayerNorm(xNorm, wAN); if (!sAN) wAN.Dispose(); }

        // 2. Separate Q, K, V projections
        var (wQ, sQ) = transformer.TempF32Named($"blk.{g}.attn_q.weight");
        var (wK, sK) = transformer.TempF32Named($"blk.{g}.attn_k.weight");
        var (wV, sV) = transformer.TempF32Named($"blk.{g}.attn_v.weight");
        var rawQ = ops.MatMulWeights(xNorm, wQ, "rawQ");
        var K = ops.MatMulWeights(xNorm, wK, "K");
        var V = ops.MatMulWeights(xNorm, wV, "V");
        if (!sQ) wQ.Dispose(); if (!sK) wK.Dispose(); if (!sV) wV.Dispose();
        xNorm.Dispose();

        // 3. Gated Q detection via attn_gate.weight presence
        int kvDim = K.Shape[1];
        int qTotalDim = rawQ.Shape[1];
        bool isGatedQ = transformer.HasWeight($"blk.{g}.attn_gate.weight");
        int headDim = 256;
        int nQH, qEffectiveDim;
        Tensor gatedQ;

        if (isGatedQ)
        {
            int qDim = qTotalDim / 2;
            var qProj = ops.SliceCols(rawQ, 0, qDim, "q_proj");
            var qGate = ops.SliceCols(rawQ, qDim, qDim, "q_gate");
            rawQ.Dispose();
            ops.SiLU(qGate);
            gatedQ = ops.Multiply(qProj, qGate, "gated_q");
            qProj.Dispose(); qGate.Dispose();
            nQH = qDim / headDim;
            qEffectiveDim = qDim;
        }
        else
        {
            gatedQ = rawQ;
            nQH = qTotalDim / headDim;
            qEffectiveDim = qTotalDim;
        }

        // 4. QK normalization
        var qNormW = transformer.GetOrBuildTiledNorm($"blk.{g}.attn_q_norm.weight", headDim, qEffectiveDim);
        var kNormW = transformer.GetOrBuildTiledNorm($"blk.{g}.attn_k_norm.weight", headDim, kvDim);
        ops.LayerNorm(gatedQ, qNormW);
        ops.LayerNorm(K, kNormW);

        // 5. RoPE
        int nKVH = kvDim / headDim;
        ops.ApplyRoPEFull(gatedQ, position, nQH, headDim, transformer._ropeFreqBase);
        ops.ApplyRoPEFull(K, position, nKVH, headDim, transformer._ropeFreqBase);

        // 6. GQA
        Tensor attnOut;
        if (kvCache != null)
        {
            kvCache.Add(layerIdx, K, V);
            var (cachedK, cachedV) = kvCache.Get(layerIdx);
            attnOut = ops.MultiHeadAttention(gatedQ, cachedK!, cachedV!, nQH, position, "attn_out_B");
            ops.DeferExternal(gatedQ);
        }
        else
        {
            attnOut = ops.MultiHeadAttention(gatedQ, K, V, nQH, position, "attn_out_B");
            ops.DeferExternal(gatedQ); ops.DeferExternal(K); ops.DeferExternal(V);
        }

        // 7. Output projection
        var (wO, sO) = transformer.TempF32Named($"blk.{g}.attn_output.weight");
        var logger = AppLogger.Create<QwenHybridFormat>();
        logger.LogWarning("[TypeB] Layer {Layer} attnOut=[{A0}×{A1}] wO=[{B0}×{B1}] dModel={DM}",
            g, attnOut.Shape[0], attnOut.Shape[1], wO.Shape[0], wO.Shape[1], transformer._dModel);
        Tensor attnProj;
        if (wO.Shape[0] == attnOut.Shape[1])
        {
            attnProj = ops.MatMulWeights(attnOut, wO, "attn_proj_B");
        }
        else if (wO.Shape[1] == attnOut.Shape[1])
        {
            attnProj = ops.MatMulWeightsT(attnOut, wO, "attn_proj_B");
        }
        else
        {
            throw new InvalidOperationException(
                $"TypeB Layer {g}: cannot project attnOut [{attnOut.Shape[0]}×{attnOut.Shape[1]}] " +
                $"with W_o [{wO.Shape[0]}×{wO.Shape[1]}]");
        }
        if (!sO) wO.Dispose();
        attnOut.Dispose();

        var x1 = ops.Add(x, attnProj, "x_after_attn_B");
        attnProj.Dispose();

        // 8. Post-attention norm → FFN
        var x1Norm = ops.Clone(x1, "post_attn_norm_in");
        { var (wPN, sPN) = transformer.TempF32Named($"blk.{g}.post_attention_norm.weight"); ops.LayerNorm(x1Norm, wPN); if (!sPN) wPN.Dispose(); }

        var (wG, sG) = transformer.TempF32Named($"blk.{g}.ffn_gate.weight");
        var (wU, sU) = transformer.TempF32Named($"blk.{g}.ffn_up.weight");
        var (wD, sD) = transformer.TempF32Named($"blk.{g}.ffn_down.weight");
        var ffnOut = ops.FeedForward(x1Norm, wG, wU, wD, "ffn_B");
        if (!sG) wG.Dispose(); if (!sU) wU.Dispose(); if (!sD) wD.Dispose();
        x1Norm.Dispose();

        var output = ops.Add(x1, ffnOut, "layer_out_B");
        x1.Dispose(); ffnOut.Dispose(); x.Dispose();
        return output;
    }

    /// <summary>
    /// Type A combined QKV block with SSM recurrence.
    /// </summary>
    private static Tensor ApplyLayerCombinedQKV(TransformerBase transformer, Tensor x, int g, int layerIdx, uint position,
                                          KVCache? kvCache, TensorNameMapper nm,
                                          SSMState? ssmState)
    {
        var ops = transformer.Ops;

        // Attention: xnorm → QKV → split
        var xNorm = ops.Clone(x, "attn_norm_in");
        var (wAN, sAN) = transformer.TempF32(nm.AttnNorm(g));
        ops.LayerNorm(xNorm, wAN);
        if (!sAN) wAN.Dispose();

        var (wQKV, sQKV) = transformer.TempF32(nm.AttnQKV(g));
        var qkv = ops.MatMulWeights(xNorm, wQKV, "qkv");
        if (!sQKV) wQKV.Dispose();

        int totalQKV = qkv.Shape[1];

        // Check if the output projection weight tells us Q_dim directly
        var wAOkey = nm.AttnOutput(g);
        bool isGatedAttn = transformer.HasWeight(wAOkey) &&
                           transformer._weightCache[wAOkey].Shape[0] == x.Shape[1] &&
                           transformer._weightCache[wAOkey].Shape[1] != x.Shape[1];

        int qDim, kvDim, headDim, nKvH;
        if (isGatedAttn)
        {
            qDim = transformer._weightCache[wAOkey].Shape[1];
            headDim = qDim / transformer._numHeads;
            kvDim = (totalQKV - qDim) / 2;
            nKvH = kvDim / headDim;
            if (g == 0) Console.WriteLine($"[Attn] headDim={headDim} qDim={qDim} kvDim={kvDim} nKvH={nKvH}");
        }
        else
        {
            nKvH = transformer._numKVHeads;
            headDim = totalQKV / (transformer._numHeads + 2 * nKvH);
            qDim = transformer._numHeads * headDim;
            kvDim = nKvH * headDim;
        }

        var Q = ops.SliceCols(qkv, 0, qDim, "Q");
        var K = ops.SliceCols(qkv, qDim, kvDim, "K");
        var V = ops.SliceCols(qkv, qDim + kvDim, kvDim, "V");
        qkv.Dispose();

        ops.ApplyRoPEFull(Q, position, transformer._numHeads, headDim, transformer._ropeFreqBase);
        ops.ApplyRoPEFull(K, position, nKvH, headDim, transformer._ropeFreqBase);

        Tensor attnOut;
        if (kvCache != null)
        {
            kvCache.Add(layerIdx, K, V);
            var (cachedK, cachedV) = kvCache.Get(layerIdx);
            attnOut = ops.MultiHeadAttention(Q, cachedK!, cachedV!, transformer._numHeads, position, "attn_out");
            ops.DeferExternal(Q);
        }
        else
        {
            attnOut = ops.MultiHeadAttention(Q, K, V, transformer._numHeads, position, "attn_out");
            ops.DeferExternal(Q); ops.DeferExternal(K); ops.DeferExternal(V);
        }

        // Output projection
        var (wAO, sAO) = transformer.TempF32(wAOkey);
        Tensor attnProj;
        if (isGatedAttn)
        {
            attnProj = ops.MatMulWeightsT(attnOut, wAO, "attn_proj");
            attnOut.Dispose();
        }
        else
        {
            attnProj = ops.MatMulWeights(attnOut, wAO, "attn_proj");
            attnOut.Dispose();
        }
        if (!sAO) wAO.Dispose();
        xNorm.Dispose();

        var x1 = ops.Add(x, attnProj, "x_after_attn");
        attnProj.Dispose();

        // Post-attention norm: shared pre-norm for both SSM and FFN branches
        var x1Norm = ops.Clone(x1, "post_attn_norm_in");
        var (wFN, sFN) = transformer.TempF32(nm.FfnNorm(g));
        ops.LayerNorm(x1Norm, wFN);
        if (!sFN) wFN.Dispose();

        // SSM branch (parallel to FFN)
        int seqLen_ = x1.Shape[0];
        var x2 = ssmState != null && HasSSMWeights(transformer, g) && seqLen_ == 1
            ? ApplySSMRecurrence(transformer, x1, x1Norm, g, ssmState)
            : x1;

        // FFN branch
        var (wG, sG) = transformer.TempF32(nm.FfnGate(g));
        var (wU, sU) = transformer.TempF32(nm.FfnUp(g));
        var (wD, sD) = transformer.TempF32(nm.FfnDown(g));
        var ffnOut = ops.FeedForward(x1Norm, wG, wU, wD, "ffn_out");
        if (!sG) wG.Dispose(); if (!sU) wU.Dispose(); if (!sD) wD.Dispose();
        x1Norm.Dispose();

        var output = ops.Add(x2, ffnOut, "layer_out");
        if (!ReferenceEquals(x1, x2)) { x1.Dispose(); }
        x2.Dispose(); ffnOut.Dispose(); x.Dispose();
        return output;
    }

    private static bool HasSSMWeights(TransformerBase transformer, int g)
        => transformer.HasWeight($"blk.{g}.ssm_a");

    /// <summary>
    /// SSM recurrence for Type-A Qwen3.6 blocks. Runs fully on CPU.
    /// </summary>
    private static Tensor ApplySSMRecurrence(TransformerBase transformer, Tensor xResidual, Tensor xNorm,
                                       int g, SSMState ssmState)
    {
        const int N = 48;
        const int D = 128;
        var ops = transformer.Ops;

        var Wa = transformer._weightCache[$"blk.{g}.ssm_alpha.weight"].ReadF32();
        var Wb = transformer._weightCache[$"blk.{g}.ssm_beta.weight"].ReadF32();
        var ssA = transformer._weightCache[$"blk.{g}.ssm_a"].ReadF32();
        var ssB = transformer._weightCache[$"blk.{g}.ssm_dt.bias"].ReadF32();
        var sNw = transformer._weightCache[$"blk.{g}.ssm_norm.weight"].ReadF32();

        var x = xNorm.ReadData();
        int seqLen = xNorm.Shape[0];
        int dModel = xNorm.Shape[1];

        var B = new float[N];
        var C = new float[N];
        var dt = new float[N];
        for (int n = 0; n < N; n++)
        {
            float bv = 0, cv = 0;
            for (int k = 0; k < dModel; k++)
            {
                float xk = x[k];
                bv += xk * Wa[k + n * dModel];
                cv += xk * Wb[k + n * dModel];
            }
            B[n] = bv;
            C[n] = cv;
            dt[n] = MathF.Log(1f + MathF.Exp(bv + ssB[n]));
        }

        var h = ssmState.GetLayer(g);
        for (int n = 0; n < N; n++)
        {
            float decay = MathF.Exp(dt[n] * ssA[n]);
            float input = dt[n] * B[n];
            int base_ = n * D;
            for (int d = 0; d < D; d++)
                h[base_ + d] = decay * h[base_ + d] + input;
        }

        var y = new float[N * D];
        for (int n = 0; n < N; n++)
        {
            float cn = C[n];
            int base_ = n * D;
            for (int d = 0; d < D; d++)
                y[base_ + d] = cn * h[base_ + d];
            float sumSq = 0f;
            for (int d = 0; d < D; d++) sumSq += y[base_ + d] * y[base_ + d];
            float rms = MathF.Sqrt(sumSq / D + 1e-5f);
            for (int d = 0; d < D; d++)
                y[base_ + d] = y[base_ + d] / rms * sNw[d];
        }

        var yMatrix = Tensor.FromData(ops.Device, y, TensorShape.Matrix(1, N * D), "ssm_y");
        var (wOut, sOut) = transformer.TempF32Named($"blk.{g}.ssm_out.weight");
        var ssmProj = ops.MatMulWeights(yMatrix, wOut, "ssm_proj");
        if (!sOut) wOut.Dispose();
        yMatrix.Dispose();

        var result = ops.Add(xResidual, ssmProj, "x_after_ssm");
        ssmProj.Dispose();
        return result;
    }

    /// <summary>
    /// Fallback for SSM-only layers (no attention weights).
    /// </summary>
    private static Tensor ApplyLayerSSMFallback(TransformerBase transformer, Tensor x, int g, TensorNameMapper nm)
    {
        var ops = transformer.Ops;

        string ffnGateName = nm.FfnGate(g);
        if (!transformer.HasWeight(ffnGateName))
        {
            var logger = AppLogger.Create<QwenHybridFormat>();
            logger.LogTrace("[SSM] Layer {Layer} skipped (no FFN weights, pure SSM)", g);
            return x;
        }

        var x1Norm = ops.Clone(x, "ssm_ffn_norm_in");
        var (wFN, sFN) = transformer.TempF32(nm.FfnNorm(g));
        ops.LayerNorm(x1Norm, wFN);
        if (!sFN) wFN.Dispose();

        var (wG, sG) = transformer.TempF32(nm.FfnGate(g));
        var (wU, sU) = transformer.TempF32(nm.FfnUp(g));
        var (wD, sD) = transformer.TempF32(nm.FfnDown(g));
        var ffnOut = ops.FeedForward(x1Norm, wG, wU, wD, "ssm_ffn");
        if (!sG) wG.Dispose(); if (!sU) wU.Dispose(); if (!sD) wD.Dispose();
        x1Norm.Dispose();

        var output = ops.Add(x, ffnOut, "ssm_layer_out");
        ffnOut.Dispose(); x.Dispose();
        return output;
    }
}
