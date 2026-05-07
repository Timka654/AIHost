using AIHost.GGUF;

namespace AIHost.Compute;

/// <summary>
/// Maps logical weight roles to actual GGUF tensor names for different model architectures.
/// GGUF standardises most names, but some architectures diverge
/// (e.g. combined QKV, MoE experts, MLA attention).
/// </summary>
public class TensorNameMapper
{
    private readonly string _arch;
    private readonly HashSet<string> _allTensorNames;

    public string Architecture => _arch;

    public TensorNameMapper(IGGUFModel model)
    {
        _arch = model.Metadata.GetValue<string>("general.architecture", "llama").ToLowerInvariant();
        _allTensorNames = model.Tensors.Select(t => t.Name).ToHashSet(StringComparer.Ordinal);

        Console.WriteLine($"[Arch] model architecture = '{_arch}'");

        // Warn about unsupported features detected in tensor names
        if (HasAny("attn_q_a.weight", "attn_kv_a_mqa.weight"))
            Console.WriteLine("[Arch] MLA attention detected (DeepSeek-style) — not fully supported");
        if (HasAny("ffn_gate_exps.weight", ".ffn_exp.0."))
            Console.WriteLine("[Arch] MoE FFN detected — using expert 0 as representative (single-path inference only)");
        if (HasAny("attn_qkv.weight"))
            Console.WriteLine("[Arch] Combined QKV weight detected (Phi/Falcon-style)");
    }

    // ── Layer weight names ────────────────────────────────────────────────────

    public string AttnNorm(int layer)        => Try($"blk.{layer}.attn_norm.weight",
                                                    $"blk.{layer}.ln1.weight");

    public string AttnQ(int layer)           => Try($"blk.{layer}.attn_q.weight",
                                                    $"blk.{layer}.attn_q_proj.weight");

    public string AttnK(int layer)           => Try($"blk.{layer}.attn_k.weight",
                                                    $"blk.{layer}.attn_k_proj.weight");

    public string AttnV(int layer)           => Try($"blk.{layer}.attn_v.weight",
                                                    $"blk.{layer}.attn_v_proj.weight");

    public string AttnOutput(int layer)      => Try($"blk.{layer}.attn_output.weight",
                                                    $"blk.{layer}.attn_out_proj.weight",
                                                    $"blk.{layer}.attn_o_proj.weight");

    public string FfnNorm(int layer)         => Try($"blk.{layer}.ffn_norm.weight",
                                                    $"blk.{layer}.ln2.weight");

    public string FfnGate(int layer)         => TryFfn(layer, "gate");
    public string FfnUp(int layer)           => TryFfn(layer, "up");
    public string FfnDown(int layer)         => TryFfn(layer, "down");

    // ── Global weight names ───────────────────────────────────────────────────

    public string TokenEmbd         => Try("token_embd.weight", "tok_embeddings.weight");
    public string OutputNorm        => Try("output_norm.weight", "norm.weight");
    public string OutputWeight      => Try("output.weight", "lm_head.weight");

    // ── Optional biases (Qwen2 etc.) ─────────────────────────────────────────

    public string? AttnQBias(int layer)      => TryOpt($"blk.{layer}.attn_q.bias");
    public string? AttnKBias(int layer)      => TryOpt($"blk.{layer}.attn_k.bias");
    public string? AttnVBias(int layer)      => TryOpt($"blk.{layer}.attn_v.bias");

    // ── Existence checks ─────────────────────────────────────────────────────

    public bool HasBias(int layer) =>
        _allTensorNames.Contains($"blk.{layer}.attn_q.bias");

    public bool IsMoE(int layer) =>
        _allTensorNames.Contains($"blk.{layer}.ffn_gate_exps.weight") ||
        _allTensorNames.Contains($"blk.{layer}.ffn_exp.0.ffn_gate.weight");

    // ── Diagnostics ──────────────────────────────────────────────────────────

    /// <summary>Print all tensors whose names contain the given substring.</summary>
    public void DumpMatching(string substring)
    {
        foreach (var n in _allTensorNames.Where(n => n.Contains(substring)).OrderBy(n => n))
            Console.WriteLine($"  [Tensor] {n}");
    }

    // ── Internals ────────────────────────────────────────────────────────────

    private string Try(params string[] candidates)
    {
        foreach (var c in candidates)
            if (_allTensorNames.Contains(c)) return c;

        // Nothing matched — return first candidate so caller gets a clear error
        Console.WriteLine($"[Arch] Warning: none of [{string.Join(", ", candidates)}] found in GGUF");
        return candidates[0];
    }

    private string? TryOpt(string name) =>
        _allTensorNames.Contains(name) ? name : null;

    private bool HasAny(params string[] substrings) =>
        substrings.Any(s => _allTensorNames.Any(n => n.Contains(s)));

    /// <summary>
    /// FFN gate/up/down with support for dense and MoE layouts.
    /// MoE: blk.N.ffn_gate_exps.weight (grouped experts) or blk.N.ffn_exp.0.ffn_gate.weight
    /// Dense: blk.N.ffn_gate.weight
    /// </summary>
    private string TryFfn(int layer, string role)
    {
        // Grouped MoE (newer llama.cpp)
        string moeGrouped = $"blk.{layer}.ffn_{role}_exps.weight";
        if (_allTensorNames.Contains(moeGrouped)) return moeGrouped;

        // Per-expert MoE — use expert 0 as representative
        string moeExpert = $"blk.{layer}.ffn_exp.0.ffn_{role}.weight";
        if (_allTensorNames.Contains(moeExpert)) return moeExpert;

        // Standard dense
        string dense = $"blk.{layer}.ffn_{role}.weight";
        if (_allTensorNames.Contains(dense)) return dense;

        Console.WriteLine($"[Arch] Warning: no FFN {role} weight found for layer {layer}");
        return dense; // let CacheWeight throw a clear error
    }
}
