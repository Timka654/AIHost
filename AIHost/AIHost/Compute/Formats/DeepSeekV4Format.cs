using AIHost.Inference;
using Microsoft.Extensions.Logging;

namespace AIHost.Compute.Formats;

/// <summary>
/// DeepSeek V2/V3/V4 transformer format (simplified).
///
/// Architecture overview (from llama.cpp deepseek2.cpp):
///   - Dense lead layers (first n_layer_dense_lead): standard attention + SiLU FFN
///   - MoE layers (remaining): standard attention + MoE FFN with shared expert
///   - MLA (Multi-head Latent Attention): NOT YET IMPLEMENTED
///     MLA uses compressed KV cache (kv_lora_rank) with wk_b/wv_b decompression.
///     This requires custom GPU shaders. For MLA models, only dense lead layers
///     will work correctly; MoE layers fall back to shared expert only.
///
/// Key tensor names (GGUF):
///   Standard (dense lead): blk.{N}.attn_q.weight, attn_k.weight, attn_v.weight, attn_output.weight
///   MLA: blk.{N}.attn_q_a.weight, attn_q_b.weight, attn_kv_a_mqa.weight, attn_k_b.weight, attn_v_b.weight
///   Dense FFN: blk.{N}.ffn_gate.weight, ffn_up.weight, ffn_down.weight
///   MoE FFN (shared): blk.{N}.ffn_gate_shexp.weight, ffn_up_shexp.weight, ffn_down_shexp.weight
///   MoE FFN (routed): blk.{N}.ffn_gate_exps.weight, ffn_up_exps.weight, ffn_down_exps.weight
///
/// Reference: llama.cpp src/models/deepseek2.cpp, src/models/deepseek.cpp
/// </summary>
public class DeepSeekV4Format : ITransformerFormat
{
    private readonly ILogger<DeepSeekV4Format> _logger = AppLogger.Create<DeepSeekV4Format>();

    /// <summary>Number of initial dense (non-MoE) layers. Read from metadata or detected.</summary>
    private int _denseLeadLayers = -1; // -1 = not yet detected

    public Tensor ApplyLayer(TransformerBase t, Tensor x, int li, uint pos, KVCache? kvc, SSMState? ss)
    {
        var _ts = GlobalProfiler.Start();
        int g = t.GlobalLayer(li);
        var nm = t._nameMapper!;
        var o = t.Ops;
        int sl = x.Shape[0];
        bool b = sl > 1;

        // Detect dense lead count once
        if (_denseLeadLayers < 0)
        {
            _denseLeadLayers = DetectDenseLeadLayers(t);
            t._logger.LogInformation("[DeepSeekV4] Dense lead layers: {Count}, total: {Total}",
                _denseLeadLayers, t._numLayers);
        }

        bool isDenseLead = g < _denseLeadLayers;
        bool isMoeLayer = !isDenseLead;

        if (b) o.BeginBatch();

        // ── 1. Pre-attention norm ──────────────────────────────────────────
        var (wAttnNorm, sAN) = t.TempF32Named(nm.AttnNorm(g));
        var xn = o.Clone(x, "an");
        o.LayerNorm(xn, wAttnNorm);
        if (!sAN) o.DeferExternal(wAttnNorm);

        // ── 2. Q/K/V projections ──────────────────────────────────────────
        int nHeads = t._numHeads;
        int nKVHeads = t._numKVHeads;
        int hd = t._headDim;
        int qd = nHeads * hd;
        int kd = nKVHeads * hd;

        var (wQ, sQ) = t.TempF32Named(nm.AttnQ(g));
        var (wK, sK) = t.TempF32Named(nm.AttnK(g));
        var (wV, sV) = t.TempF32Named(nm.AttnV(g));
        var (wO, sO) = t.TempF32Named(nm.AttnOutput(g));

        var Q = o.MatMulWeights(xn, wQ, "Q");
        if (!sQ) o.DeferExternal(wQ);
        var K = o.MatMulWeights(xn, wK, "K");
        if (!sK) o.DeferExternal(wK);
        var V = o.MatMulWeights(xn, wV, "V");
        if (!sV) o.DeferExternal(wV);

        if (b) o.DeferExternal(xn); else xn.Dispose();

        // ── 3. RoPE ────────────────────────────────────────────────────────
        o.ApplyRoPEFull(Q, pos, nHeads, hd, t._ropeFreqBase, t._ropeDimCount);
        o.ApplyRoPEFull(K, pos, nKVHeads, hd, t._ropeFreqBase, t._ropeDimCount);

        // ── 4. Multi-Head Attention ────────────────────────────────────────
        Tensor ao;
        if (kvc != null)
        {
            kvc.Add(li, K, V);
            var (ck, cv) = kvc.Get(li);
            ao = o.MultiHeadAttention(Q, ck!, cv!, nHeads, pos, "ao");
            o.DeferExternal(Q);
        }
        else
        {
            ao = o.MultiHeadAttention(Q, K, V, nHeads, pos, "ao");
            o.DeferExternal(Q);
            o.DeferExternal(K);
            o.DeferExternal(V);
        }

        // ── 5. Output projection + residual ───────────────────────────────
        var ap = wO.Shape[0] == ao.Shape[1]
            ? o.MatMulWeights(ao, wO, "ap")
            : o.MatMulWeightsT(ao, wO, "ap");
        if (!sO) o.DeferExternal(wO);
        if (b) o.DeferExternal(ao); else ao.Dispose();

        var attnOut = o.Add(x, ap, "attn_out");
        if (b) o.DeferExternal(ap); else ap.Dispose();

        // ── 6. FFN ────────────────────────────────────────────────────────
        var ffnNormName = nm.FfnNorm(g);
        var (wFfnNorm, sFN) = t.TempF32Named(ffnNormName);
        var attnOutNorm = o.Clone(attnOut, "fn");
        o.LayerNorm(attnOutNorm, wFfnNorm);
        if (!sFN) o.DeferExternal(wFfnNorm);

        Tensor ffnOut;
        if (isDenseLead)
        {
            // Standard SiLU-gated FFN for dense lead layers
            var (wGate, sG) = t.TempF32Named(nm.FfnGate(g));
            var (wUp, sU) = t.TempF32Named(nm.FfnUp(g));
            var (wDown, sD) = t.TempF32Named(nm.FfnDown(g));

            ffnOut = o.FeedForward(attnOutNorm, wGate, wUp, wDown, "ffn_out");

            if (!sG) o.DeferExternal(wGate);
            if (!sU) o.DeferExternal(wUp);
            if (!sD) o.DeferExternal(wDown);
        }
        else
        {
            // MoE layer: use shared expert FFN (shexp) as fallback
            // Full MoE routing (top-k expert selection) not yet implemented
            string? gateShexp = nm.FfnGateShexp(g);
            string? upShexp = nm.FfnUpShexp(g);
            string? downShexp = nm.FfnDownShexp(g);

            if (gateShexp != null && upShexp != null && downShexp != null)
            {
                var (wGate, sG) = t.TempF32Named(gateShexp);
                var (wUp, sU) = t.TempF32Named(upShexp);
                var (wDown, sD) = t.TempF32Named(downShexp);

                ffnOut = o.FeedForward(attnOutNorm, wGate, wUp, wDown, "ffn_moe_shexp");

                if (!sG) o.DeferExternal(wGate);
                if (!sU) o.DeferExternal(wUp);
                if (!sD) o.DeferExternal(wDown);
            }
            else
            {
                // Fallback: use dense FFN weights if shexp not available
                t._logger.LogWarning("[DeepSeekV4] Layer {Layer}: no shexp weights, using dense FFN", g);
                var (wGate, sG) = t.TempF32Named(nm.FfnGate(g));
                var (wUp, sU) = t.TempF32Named(nm.FfnUp(g));
                var (wDown, sD) = t.TempF32Named(nm.FfnDown(g));

                ffnOut = o.FeedForward(attnOutNorm, wGate, wUp, wDown, "ffn_out");

                if (!sG) o.DeferExternal(wGate);
                if (!sU) o.DeferExternal(wUp);
                if (!sD) o.DeferExternal(wDown);
            }
        }

        if (b) o.DeferExternal(attnOutNorm); else attnOutNorm.Dispose();

        // ── 7. Residual connection ────────────────────────────────────────
        var result = o.Add(attnOut, ffnOut, "layer_out");
        if (b) { o.DeferExternal(attnOut); o.DeferExternal(ffnOut); }
        else { attnOut.Dispose(); ffnOut.Dispose(); }

        // ── 8. Consume input ──────────────────────────────────────────────
        if (b) o.DeferExternal(x); else x.Dispose();
        if (b) o.Flush();

        GlobalProfiler.End(_ts, "DeepSeekV4.ApplyLayer");
        return result;
    }

    /// <summary>
    /// Detect number of dense lead layers (non-MoE).
    /// MoE layers have ffn_gate_inp.weight or ffn_gate_exps.weight.
    /// </summary>
    private static int DetectDenseLeadLayers(TransformerBase t)
    {
        // Check if any layer has MoE router weight (ffn_gate_inp.weight)
        // The first layer with ffn_gate_inp.weight marks the start of MoE layers.
        for (int g = 0; g < t._numLayers; g++)
        {
            if (t.HasWeight($"blk.{g}.ffn_gate_inp.weight") ||
                t.HasWeight($"blk.{g}.ffn_gate_exps.weight"))
            {
                return g; // This layer is the first MoE layer
            }
        }
        // No MoE layers detected — all layers are dense
        return t._numLayers;
    }
}
