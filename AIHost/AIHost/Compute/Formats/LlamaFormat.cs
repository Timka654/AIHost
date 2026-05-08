using AIHost.Inference;
using Microsoft.Extensions.Logging;

namespace AIHost.Compute.Formats;

/// <summary>
/// Standard LLaMA-family format: separate Q/K/V weights, GQA, RoPE, SiLU-gated FFN.
/// Used by: TinyLlama, LLaMA 2/3, Mistral, Qwen2, Gemma, etc.
///
/// Layer structure:
///   x → RMSNorm → Q/K/V projections → RoPE → GQA → W_o → +x → RMSNorm → FFN(SiLU) → +x
///
/// Key characteristics:
///   - Separate attn_q.weight, attn_k.weight, attn_v.weight
///   - Standard GQA (no gated Q — qTotalDim = nHeads × headDim, kvDim = nKVHeads × headDim)
///   - No QK-norm weights (unlike Qwen3.6 Type B)
///   - attn_output.weight [qTotalDim, dModel] (normal layout)
///   - SiLU-gated FFN: gate/up/down
/// </summary>
public class LlamaFormat : ITransformerFormat
{
    private readonly ILogger<LlamaFormat> _logger = AppLogger.Create<LlamaFormat>();

    public Tensor ApplyLayer(TransformerBase transformer, Tensor x, int layerIdx, uint position, KVCache? kvCache, SSMState? ssmState)
    {
        int g = transformer.GlobalLayer(layerIdx);
        var ops = transformer.Ops;
        var nm = transformer._nameMapper!;

        int seqLen = x.Shape[0];
        bool useBatch = seqLen == 1;

        if (useBatch)
        {
            ops.BeginBatch();

            (Tensor t, bool scratch) Wb(string name)
            {
                var (t2, s) = transformer.TempF32(name);
                if (!s) ops.DeferExternal(t2);
                return (t2, s);
            }

            var (wAttnNorm, _) = Wb(nm.AttnNorm(g));
            var (wQ, _) = Wb(nm.AttnQ(g));
            var (wK, _) = Wb(nm.AttnK(g));
            var (wV, _) = Wb(nm.AttnV(g));
            var (wAttnOut, _) = Wb(nm.AttnOutput(g));
            var (wFfnNorm, _) = Wb(nm.FfnNorm(g));
            var (wGate, _) = Wb(nm.FfnGate(g));
            var (wUp, _) = Wb(nm.FfnUp(g));
            var (wDown, _) = Wb(nm.FfnDown(g));
            ops.InsertBarrier();

            var outputB = ops.TransformerLayer(
                x, wAttnNorm, wQ, wK, wV, wAttnOut,
                wFfnNorm, wGate, wUp, wDown,
                transformer._numHeads, position, kvCache, layerIdx);

            ops.DeferExternal(x);
            ops.Flush();
            return outputB;
        }

        // Prefill: per-op flush
        var (wAN, sAN) = transformer.TempF32(nm.AttnNorm(g));
        var (wQ2, sQ) = transformer.TempF32(nm.AttnQ(g));
        var (wK2, sK) = transformer.TempF32(nm.AttnK(g));
        var (wV2, sV) = transformer.TempF32(nm.AttnV(g));
        var (wAO, sAO) = transformer.TempF32(nm.AttnOutput(g));
        var (wFN, sFN) = transformer.TempF32(nm.FfnNorm(g));
        var (wG, sG) = transformer.TempF32(nm.FfnGate(g));
        var (wU, sU) = transformer.TempF32(nm.FfnUp(g));
        var (wD, sD) = transformer.TempF32(nm.FfnDown(g));

        var output = ops.TransformerLayer(
            x, wAN, wQ2, wK2, wV2, wAO,
            wFN, wG, wU, wD,
            transformer._numHeads, position, kvCache, layerIdx);

        void Dispose2(Tensor t, bool isScratch) { if (!isScratch) t.Dispose(); }
        Dispose2(wAN, sAN); Dispose2(wQ2, sQ); Dispose2(wK2, sK); Dispose2(wV2, sV);
        Dispose2(wAO, sAO); Dispose2(wFN, sFN); Dispose2(wG, sG); Dispose2(wU, sU);
        Dispose2(wD, sD);
        x.Dispose();
        return output;
    }
}
