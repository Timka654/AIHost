using AIHost.Compute;
using AIHost.ICompute;
using AIHost.Tokenizer;

namespace AIHost.Inference;

/// <summary>
/// Configuration for text generation
/// </summary>
public class GenerationConfig
{
    public int MaxNewTokens { get; set; } = 100;
    public float Temperature { get; set; } = 0.7f;
    public int TopK { get; set; } = 50;
    public float TopP { get; set; } = 0.9f;
    public int Seed { get; set; } = -1;
    public bool UseKVCache { get; set; } = true;
}

/// <summary>
/// High-level inference engine with sampling strategies and KV-cache
/// </summary>
public class InferenceEngine : IDisposable
{
    private readonly Transformer _model;
    private readonly BPETokenizer _tokenizer;
    private readonly ComputeOps _ops;
    private Random _random;
    private KVCache? _kvCache;
    private bool _disposed;

    public InferenceEngine(Transformer model, BPETokenizer tokenizer, ComputeOps ops)
    {
        _model = model;
        _tokenizer = tokenizer;
        _ops = ops;
        _random = new Random();
    }

    /// <summary>
    /// Generate text from a prompt
    /// </summary>
    public string Generate(string prompt, GenerationConfig config)
    {
        if (config.Seed >= 0)
        {
            _random = new Random(config.Seed);
        }

        // Encode prompt
        var tokens = _tokenizer.Encode(prompt, addBos: true, addEos: false).ToList();
        Console.WriteLine($"Prompt tokens: [{string.Join(", ", tokens)}]");

        // Initialize or reset KV cache for this generation
        if (config.UseKVCache)
        {
            if (_kvCache == null)
                _kvCache = new KVCache(_ops);
            else
                _kvCache.Clear();
        }
        else
        {
            _kvCache?.Clear();
        }

        // Generation loop
        int eosToken = _tokenizer.EosToken;
        
        for (int i = 0; i < config.MaxNewTokens; i++)
        {
            // Forward pass
            uint startPos = _kvCache != null ? (uint)_kvCache.SequenceLength : 0;
            
            // With KV-cache: pass only new tokens after first iteration
            int[] inputTokens;
            if (_kvCache != null && _kvCache.SequenceLength > 0)
            {
                // Only the last token (newly generated)
                inputTokens = new[] { tokens[^1] };
            }
            else
            {
                // First iteration: all tokens
                inputTokens = tokens.ToArray();
            }
            
            var logits = _model.Forward(inputTokens, startPos, _kvCache);

            // Get logits for last token
            var lastLogits = ExtractLastToken(logits);
            logits.Dispose();

            // Sample next token
            int nextToken = Sample(lastLogits, config);

            tokens.Add(nextToken);

            if (nextToken == eosToken)
            {
                break;
            }
        }

        // Decode
        return _tokenizer.Decode(tokens.ToArray());
    }

    /// <summary>
    /// Generate text with streaming callback
    /// </summary>
    public void GenerateStreaming(string prompt, GenerationConfig config, Action<string> onTokenGenerated)
    {
        if (config.Seed >= 0)
        {
            _random = new Random(config.Seed);
        }

        var tokens = _tokenizer.Encode(prompt, addBos: true, addEos: false).ToList();

        if (config.UseKVCache)
        {
            if (_kvCache == null)
                _kvCache = new KVCache(_ops);
            else
                _kvCache.Clear();
        }
        else
        {
            _kvCache?.Clear();
        }

        int eosToken = _tokenizer.EosToken;

        for (int i = 0; i < config.MaxNewTokens; i++)
        {
            uint startPos = _kvCache != null ? (uint)_kvCache.SequenceLength : 0;
            
            // With KV-cache: pass only new tokens after first iteration
            int[] inputTokens;
            if (_kvCache != null && _kvCache.SequenceLength > 0)
            {
                inputTokens = new[] { tokens[^1] };
            }
            else
            {
                inputTokens = tokens.ToArray();
            }
            
            var logits = _model.Forward(inputTokens, startPos, _kvCache);

            var lastLogits = ExtractLastToken(logits);
            logits.Dispose();

            int nextToken = Sample(lastLogits, config);
            tokens.Add(nextToken);

            // Stream token
            string tokenText = _tokenizer.GetToken(nextToken);
            onTokenGenerated(tokenText);

            if (nextToken == eosToken)
            {
                break;
            }
        }
    }

    private float[] ExtractLastToken(Tensor logits)
    {
        // logits shape: [seqLen, vocabSize]
        var data = logits.ReadData();
        int seqLen = logits.Shape.Dimensions[0];
        int vocabSize = logits.Shape.Dimensions[1];

        float[] lastLogits = new float[vocabSize];
        int offset = (seqLen - 1) * vocabSize;
        Array.Copy(data, offset, lastLogits, 0, vocabSize);

        return lastLogits;
    }

    private int Sample(float[] logits, GenerationConfig config)
    {
        // Apply temperature
        if (Math.Abs(config.Temperature - 1.0f) > 0.001f)
        {
            for (int i = 0; i < logits.Length; i++)
            {
                logits[i] /= config.Temperature;
            }
        }

        // Convert to probabilities
        var probs = Softmax(logits);

        // Top-K filtering
        if (config.TopK > 0)
        {
            probs = ApplyTopK(probs, config.TopK);
        }

        // Top-P (nucleus) filtering
        if (config.TopP < 1.0f)
        {
            probs = ApplyTopP(probs, config.TopP);
        }

        // Sample from distribution
        return SampleFromDistribution(probs);
    }

    private float[] Softmax(float[] logits)
    {
        float max = logits.Max();
        float[] exps = logits.Select(x => MathF.Exp(x - max)).ToArray();
        float sum = exps.Sum();
        return exps.Select(x => x / sum).ToArray();
    }

    private float[] ApplyTopK(float[] probs, int k)
    {
        // Get top-k indices
        var indexed = probs.Select((p, i) => (prob: p, index: i))
                           .OrderByDescending(x => x.prob)
                           .Take(k)
                           .ToArray();

        // Zero out all others
        float[] filtered = new float[probs.Length];
        float sum = 0f;
        
        foreach (var (prob, index) in indexed)
        {
            filtered[index] = prob;
            sum += prob;
        }

        // Renormalize
        if (sum > 0)
        {
            for (int i = 0; i < filtered.Length; i++)
            {
                filtered[i] /= sum;
            }
        }

        return filtered;
    }

    private float[] ApplyTopP(float[] probs, float p)
    {
        // Sort by probability descending
        var indexed = probs.Select((prob, i) => (prob, index: i))
                           .OrderByDescending(x => x.prob)
                           .ToArray();

        // Find cutoff
        float cumSum = 0f;
        int cutoff = 0;
        
        for (int i = 0; i < indexed.Length; i++)
        {
            cumSum += indexed[i].prob;
            cutoff = i + 1;
            if (cumSum >= p) break;
        }

        // Zero out tail
        float[] filtered = new float[probs.Length];
        float sum = 0f;

        for (int i = 0; i < cutoff; i++)
        {
            var (prob, index) = indexed[i];
            filtered[index] = prob;
            sum += prob;
        }

        // Renormalize
        if (sum > 0)
        {
            for (int i = 0; i < filtered.Length; i++)
            {
                filtered[i] /= sum;
            }
        }

        return filtered;
    }

    private int SampleFromDistribution(float[] probs)
    {
        float rand = (float)_random.NextDouble();
        float cumSum = 0f;

        for (int i = 0; i < probs.Length; i++)
        {
            cumSum += probs[i];
            if (rand < cumSum)
            {
                return i;
            }
        }

        // Fallback: return last non-zero index
        for (int i = probs.Length - 1; i >= 0; i--)
        {
            if (probs[i] > 0) return i;
        }

        return 0;
    }

    public void Dispose()
    {
        if (_disposed) return;
        _kvCache?.Dispose();
        _disposed = true;
    }
}

/// <summary>
/// KV cache for efficient autoregressive generation
/// </summary>
public class KVCache : IDisposable
{
    private readonly ComputeOps _ops;
    private readonly List<(Tensor key, Tensor value)> _cache = new();
    private bool _disposed;

    public int SequenceLength { get; private set; }

    public KVCache(ComputeOps ops)
    {
        _ops = ops;
    }

    public void Add(int layer, Tensor key, Tensor value)
    {
        if (layer >= _cache.Count)
        {
            // First time for this layer - take ownership of tensors
            _cache.Add((key, value));
            SequenceLength = key.Shape.Dimensions[0]; // seq_len is first dimension
        }
        else
        {
            // Concatenate with existing cache along seq_len axis (axis 0)
            var (oldKey, oldValue) = _cache[layer];
            
            var newKey = _ops.Concat(oldKey, key, axis: 0, $"kv_cache_k_layer{layer}");
            var newValue = _ops.Concat(oldValue, value, axis: 0, $"kv_cache_v_layer{layer}");
            
            oldKey.Dispose();
            oldValue.Dispose();
            key.Dispose(); // Dispose incoming tensors after concat
            value.Dispose();
            
            _cache[layer] = (newKey, newValue);
            SequenceLength = newKey.Shape.Dimensions[0]; // seq_len is first dimension
        }
    }

    public (Tensor? key, Tensor? value) Get(int layer)
    {
        if (layer < _cache.Count)
        {
            return _cache[layer];
        }
        return (null, null);
    }

    public void Clear()
    {
        foreach (var (key, value) in _cache)
        {
            key.Dispose();
            value.Dispose();
        }
        _cache.Clear();
        SequenceLength = 0;
    }

    public void Dispose()
    {
        if (_disposed) return;
        Clear();
        _disposed = true;
    }
}
