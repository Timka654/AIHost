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
        PrintDiagnostics(model);
    }

    private void PrintDiagnostics(IGGUFModel model)
    {
        // Print all blk.0.* tensor names so mismatches are immediately visible in logs.
        var layer0 = _allTensorNames
            .Where(n => n.StartsWith("blk.0.") || n.StartsWith("blk.0_"))
            .OrderBy(n => n).ToList();

        if (layer0.Count > 0)
        {
            Console.WriteLine("[Arch] Layer-0 tensors in GGUF:");
            foreach (var n in layer0) Console.WriteLine($"  {n}");
        }
        else
        {
            Console.WriteLine("[Arch] Warning: no 'blk.0.*' tensors found — tensor name format may differ");
            Console.WriteLine("[Arch] First 20 tensor names:");
            foreach (var n in _allTensorNames.OrderBy(x => x).Take(20))
                Console.WriteLine($"  {n}");
        }

        // Print shapes for key tensors in layers 0 and 3 (Type A and Type B representatives)
        foreach (var layer in new[] { 0, 3 })
        {
            var layerTensors = model.Tensors
                .Where(t => t.Name.StartsWith($"blk.{layer}."))
                .OrderBy(t => t.Name)
                .ToList();
            if (layerTensors.Count > 0)
            {
                Console.WriteLine($"[Arch] Layer-{layer} tensor shapes:");
                foreach (var t in layerTensors)
                {
                    var shapeStr = string.Join("×", t.Shape.Select(s => s.ToString()));
                    Console.WriteLine($"  {t.Name} [{shapeStr}] type={t.Type}");
                }
            }
        }

        // Print SSM metadata
        var meta = model.Metadata;
        Console.WriteLine("[Arch] SSM metadata:");
        foreach (var key in new[] { "ssm.inner_size", "ssm.state_size", "ssm.group_count", "ssm.time_step_rank",
                                     "attention.key_length", "attention.value_length",
                                     "attention.head_count", "attention.head_count_kv",
                                     "embedding_length", "block_count" })
        {
            var val = meta.GetArchValue<object>(key);
            Console.WriteLine($"  {key} = {val ?? "(not found)"}");
        }

        // Detect and warn about special architectures
        if (HasAny("attn_q_a.weight", "attn_kv_a_mqa.weight"))
            Console.WriteLine("[Arch] MLA attention detected (DeepSeek-style) — not fully supported");
        if (HasAny("ffn_gate_exps.weight", ".ffn_exp.0."))
            Console.WriteLine("[Arch] MoE FFN detected — expert 0 used for single-path inference");
        if (HasCombinedQKV)
            Console.WriteLine("[Arch] Combined QKV weight — using ApplyLayerCombinedQKV");
        if (HasAny("attn_q.weight") && !HasCombinedQKV)
            Console.WriteLine("[Arch] Separate Q/K/V weights — standard attention path");
        if (IsHybridSSM)
            Console.WriteLine("[Arch] Hybrid SSM+Attention model detected — SSM layers will use attention fallback (output may be incorrect for SSM layers)");
    }

    // ── Attention layout ─────────────────────────────────────────────────────

    /// <summary>True when the model stores Q, K, V in a single fused weight.</summary>
    public bool HasCombinedQKV => HasAny("attn_qkv.weight", "attention.wqkv.weight");

    // ── Layer weight names ────────────────────────────────────────────────────

    public string AttnNorm(int layer)        => Try($"blk.{layer}.attn_norm.weight",
                                                    $"blk.{layer}.ln1.weight");

    /// <summary>
    /// Combined QKV weight for architectures that don't have separate Q/K/V.
    /// Shape (GGUF column-major): [n_embd, (n_q_heads + 2*n_kv_heads) * head_dim]
    /// </summary>
    public string AttnQKV(int layer)         => Try($"blk.{layer}.attn_qkv.weight",
                                                    $"blk.{layer}.attention.wqkv.weight");

    public string AttnQ(int layer)           => HasCombinedQKV
                                                ? AttnQKV(layer)  // caller must handle split
                                                : Try($"blk.{layer}.attn_q.weight",
                                                      $"blk.{layer}.attn_q_proj.weight");

    public string AttnK(int layer)           => HasCombinedQKV
                                                ? AttnQKV(layer)
                                                : Try($"blk.{layer}.attn_k.weight",
                                                      $"blk.{layer}.attn_k_proj.weight");

    public string AttnV(int layer)           => HasCombinedQKV
                                                ? AttnQKV(layer)
                                                : Try($"blk.{layer}.attn_v.weight",
                                                      $"blk.{layer}.attn_v_proj.weight");

    public string AttnOutput(int layer)      => Try($"blk.{layer}.attn_output.weight",
                                                    $"blk.{layer}.attn_gate.weight",       // Qwen3.5/qwen35
                                                    $"blk.{layer}.attn_out_proj.weight",
                                                    $"blk.{layer}.attn_o_proj.weight");

    public string FfnNorm(int layer)         => Try($"blk.{layer}.ffn_norm.weight",
                                                    $"blk.{layer}.post_attention_norm.weight", // Qwen3.5
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

    /// <summary>True if this specific layer has attention weights (vs pure SSM layer).</summary>
    public bool LayerHasAttention(int layer) =>
        _allTensorNames.Contains($"blk.{layer}.attn_qkv.weight") ||
        _allTensorNames.Contains($"blk.{layer}.attn_q.weight")   ||
        _allTensorNames.Contains($"blk.{layer}.attn_q_proj.weight");

    /// <summary>True if this specific layer has SSM (Mamba/state-space) weights.</summary>
    public bool LayerHasSSM(int layer) =>
        _allTensorNames.Contains($"blk.{layer}.ssm_out.weight") ||
        _allTensorNames.Contains($"blk.{layer}.ssm_a");

    /// <summary>True if this is a hybrid SSM+Attention model (e.g. Jamba, Qwen3.6).</summary>
    public bool IsHybridSSM => HasAny(".ssm_out.weight", ".ssm_a");

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

    private readonly HashSet<string> _warnedMissing = [];

    private string Try(params string[] candidates)
    {
        foreach (var c in candidates)
            if (_allTensorNames.Contains(c)) return c;

        // Log once per missing set to avoid spamming on every token generation
        string key = candidates[0];
        if (_warnedMissing.Add(key))
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
