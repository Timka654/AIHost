namespace AIHost.GGUF;

/// <summary>
/// Метаданные GGUF модели
/// </summary>
public class GGUFMetadata
{
    private readonly Dictionary<string, object> _values = new();

    public void Add(string key, object value)
    {
        _values[key] = value;
    }

    public bool TryGetValue<T>(string key, out T? value)
    {
        if (_values.TryGetValue(key, out var obj) && obj is T typedValue)
        {
            value = typedValue;
            return true;
        }
        value = default;
        return false;
    }

    public T? GetValue<T>(string key, T? defaultValue = default)
    {
        return TryGetValue<T>(key, out var value) ? value : defaultValue;
    }

    public IEnumerable<string> Keys => _values.Keys;
    public int Count => _values.Count;

    // ── Architecture-aware key resolution ────────────────────────────────────

    /// <summary>
    /// Returns the model's architecture string (e.g. "llama", "qwen2", "phi3").
    /// </summary>
    public string Architecture => GetValue<string>(KeyArchitecture, "llama")!.ToLowerInvariant();

    /// <summary>
    /// Looks up {arch}.{suffix}, then "llama.{suffix}" as fallback, then returns defaultValue.
    /// Handles all GGUF models regardless of architecture prefix.
    /// </summary>
    public T? GetArchValue<T>(string suffix, T? defaultValue = default)
    {
        string arch = Architecture;
        // Try architecture-specific key first (e.g. "qwen2.block_count")
        if (!arch.Equals("llama", StringComparison.OrdinalIgnoreCase) &&
            TryGetValue<T>($"{arch}.{suffix}", out var v1))
            return v1;
        // Try llama-prefixed (most models, and legacy LLaMA-derived)
        if (TryGetValue<T>($"llama.{suffix}", out var v2))
            return v2;
        // Try the arch key again in case it IS llama
        if (TryGetValue<T>($"{arch}.{suffix}", out var v3))
            return v3;
        return defaultValue;
    }

    // ── Well-known key constants (llama-prefixed kept for back-compat) ────────
    public const string KeyName              = "general.name";
    public const string KeyArchitecture      = "general.architecture";
    public const string KeyFileType          = "general.file_type";
    public const string KeyContextLength     = "llama.context_length";
    public const string KeyEmbeddingLength   = "llama.embedding_length";
    public const string KeyBlockCount        = "llama.block_count";
    public const string KeyFeedForwardLength = "llama.feed_forward_length";
    public const string KeyAttentionHeadCount   = "llama.attention.head_count";
    public const string KeyAttentionHeadCountKV = "llama.attention.head_count_kv";
    public const string KeyRopeFreqBase      = "llama.rope.freq_base";
    public const string KeyTokenizerModel    = "tokenizer.ggml.model";
    public const string KeyVocabSize         = "llama.vocab_size";

    public override string ToString()
    {
        var name = GetValue<string>(KeyName, "Unknown");
        var arch = GetValue<string>(KeyArchitecture, "Unknown");
        return $"{name} ({arch}) - {Count} metadata entries";
    }
}
