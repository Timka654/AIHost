using AIHost.Inference;

namespace AIHost.Compute;

/// <summary>
/// Interface for model-architecture-specific transformer layer logic.
/// Each format implements the per-layer forward pass for a specific model family
/// (e.g. standard LLaMA, Qwen3.6 hybrid, Phi combined-QKV, etc.).
/// </summary>
public interface ITransformerFormat
{
    /// <summary>
    /// Apply one transformer layer.
    /// </summary>
    /// <param name="transformer">The TransformerBase instance providing weight cache and helpers.</param>
    /// <param name="x">Input tensor [seqLen, dModel]</param>
    /// <param name="layerIdx">Local layer index (0-based for this device)</param>
    /// <param name="position">Current token position for RoPE</param>
    /// <param name="kvCache">KV cache (null if not caching)</param>
    /// <param name="ssmState">SSM state for hybrid models (null if not applicable)</param>
    /// <returns>Output tensor [seqLen, dModel]</returns>
    Tensor ApplyLayer(TransformerBase transformer, Tensor x, int layerIdx, uint position, KVCache? kvCache, SSMState? ssmState);
}


