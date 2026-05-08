using AIHost.Inference;
using Microsoft.Extensions.Logging;

namespace AIHost.Compute.Formats;

/// <summary>
/// Format for models with a single fused QKV weight (attn_qkv.weight).
/// Used by: Phi-3/4, Falcon, Qwen2.5, etc.
///
/// Layer structure:
///   x → RMSNorm → QKV projection → split → RoPE → GQA → W_o → +x → RMSNorm → FFN(SiLU) → +x
///
/// Key characteristics:
///   - Single attn_qkv.weight [dModel, (nHeads + 2*nKVHeads) * headDim]
///   - Standard GQA (no gated Q)
///   - attn_output.weight [qTotalDim, dModel] (normal layout)
///   - SiLU-gated FFN: gate/up/down
/// </summary>
public class CombinedQKVFormat : ITransformerFormat
{
    private readonly ILogger<CombinedQKVFormat> _logger = AppLogger.Create<CombinedQKVFormat>();

    public Tensor ApplyLayer(TransformerBase transformer, Tensor x, int layerIdx, uint position, KVCache? kvCache, SSMState? ssmState)
    {
        int g = transformer.GlobalLayer(layerIdx);
        var ops = transformer.Ops;
        var nm = transformer._nameMapper!;

        // 1. Pre-attention RMSNorm
        var xNorm = ops.Clone(x, "attn_norm_in");
        var (wAN, sAN) = transformer.TempF32(nm.AttnNorm(g));
        ops.LayerNorm(xNorm, wAN);
        if (!sAN) wAN.Dispose();

        // 2. Combined QKV projection
        var (wQKV, sQKV) = transformer.TempF32(nm.AttnQKV(g));
        var qkv = ops.MatMulWeights(xNorm, wQKV, "qkv");
        if (!sQKV) wQKV.Dispose();

        // 3. Split QKV
        int totalQKV = qkv.Shape[1];
        int nKvH = transformer._numKVHeads;
        int headDim = totalQKV / (transformer._numHeads + 2 * nKvH);
        int qDim = transformer._numHeads * headDim;
        int kvDim = nKvH * headDim;

        var Q = ops.SliceCols(qkv, 0, qDim, "Q");
        var K = ops.SliceCols(qkv, qDim, kvDim, "K");
        var V = ops.SliceCols(qkv, qDim + kvDim, kvDim, "V");
        qkv.Dispose();

        // 4. RoPE
        ops.ApplyRoPEFull(Q, position, transformer._numHeads, headDim, transformer._ropeFreqBase);
        ops.ApplyRoPEFull(K, position, nKvH, headDim, transformer._ropeFreqBase);

        // 5. GQA
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

        // 6. Output projection
        var (wAO, sAO) = transformer.TempF32(nm.AttnOutput(g));
        var attnProj = ops.MatMulWeights(attnOut, wAO, "attn_proj");
        if (!sAO) wAO.Dispose();
        attnOut.Dispose();
        xNorm.Dispose();

        var x1 = ops.Add(x, attnProj, "x_after_attn");
        attnProj.Dispose();

        // 7. Post-attention norm → FFN
        var x1Norm = ops.Clone(x1, "post_attn_norm_in");
        var (wFN, sFN) = transformer.TempF32(nm.FfnNorm(g));
        ops.LayerNorm(x1Norm, wFN);
        if (!sFN) wFN.Dispose();

        var (wG, sG) = transformer.TempF32(nm.FfnGate(g));
        var (wU, sU) = transformer.TempF32(nm.FfnUp(g));
        var (wD, sD) = transformer.TempF32(nm.FfnDown(g));
        var ffnOut = ops.FeedForward(x1Norm, wG, wU, wD, "ffn_out");
        if (!sG) wG.Dispose(); if (!sU) wU.Dispose(); if (!sD) wD.Dispose();
        x1Norm.Dispose();

        var output = ops.Add(x1, ffnOut, "layer_out");
        x1.Dispose(); ffnOut.Dispose(); x.Dispose();
        return output;
    }
}
