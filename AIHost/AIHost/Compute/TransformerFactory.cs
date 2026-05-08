using AIHost.Compute.Formats;
using AIHost.GGUF;
using AIHost.ICompute;

namespace AIHost.Compute;

/// <summary>
/// Factory that creates the appropriate TransformerBase with the correct format
/// for a given model architecture.
/// </summary>
public static class TransformerFactory
{
    /// <summary>
    /// Create a TransformerBase instance with the correct format for the given model.
    /// </summary>
    public static TransformerBase Create(IComputeDevice device, IGGUFModel model)
    {
        var arch = model.Metadata.GetValue<string>("general.architecture", "llama").ToLowerInvariant();
        var allTensorNames = model.Tensors.Select(t => t.Name).ToHashSet(StringComparer.Ordinal);

        // Detect format based on tensor names
        bool hasCombinedQKV = allTensorNames.Any(n => n.Contains("attn_qkv.weight") || n.Contains("attention.wqkv.weight"));
        bool hasSeparateQKV = allTensorNames.Any(n => n.Contains("attn_q.weight"));
        bool hasSSM = allTensorNames.Any(n => n.Contains(".ssm_out.weight") || n.Contains(".ssm_a"));
        bool hasQKNorm = allTensorNames.Any(n => n.Contains("attn_q_norm.weight"));
        bool hasGatedQ = allTensorNames.Any(n => n.Contains("attn_gate.weight"));

        Console.WriteLine($"[Factory] arch='{arch}' combinedQKV={hasCombinedQKV} separateQKV={hasSeparateQKV} SSM={hasSSM} QKNorm={hasQKNorm} gatedQ={hasGatedQ}");

        ITransformerFormat format;

        if (hasSSM && hasSeparateQKV && hasQKNorm)
        {
            // Qwen3.6 hybrid: SSM + Type A (combined QKV) + Type B (separate QKV + QK-norm)
            Console.WriteLine("[Factory] Selected: QwenHybridFormat (SSM + Attention hybrid)");
            format = new QwenHybridFormat();
        }
        else if (hasCombinedQKV && !hasSeparateQKV)
        {
            // Phi, Falcon, Qwen2.5: fused QKV
            Console.WriteLine("[Factory] Selected: CombinedQKVFormat (fused QKV)");
            format = new CombinedQKVFormat();
        }
        else
        {
            // Default: standard LLaMA (TinyLlama, LLaMA 2/3, Mistral, Qwen2, Gemma)
            Console.WriteLine("[Factory] Selected: LlamaFormat (standard separate Q/K/V)");
            format = new LlamaFormat();
        }

        // Create the real TransformerBase with the correct format
        return new TransformerBase(device, model, format);
    }
}
