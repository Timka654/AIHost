using AIHost.Compute;
using AIHost.ICompute;
using AIHost.Tokenizer;
using Microsoft.Extensions.Logging;

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
    public float RepetitionPenalty { get; set; } = 1.0f;
    public int Seed { get; set; } = -1;
    public bool UseKVCache { get; set; } = true;
    public KVCacheQuantization KVCacheQuantization { get; set; } = KVCacheQuantization.None;
    /// <summary>Max prompt tokens before truncation (0 = no limit).</summary>
    public int MaxPromptTokens { get; set; } = 0;
    /// <summary>Stop generation when any of these strings appears at the end of generated text.</summary>
    public List<string> StopSequences { get; set; } = [];
    /// <summary>
    /// Frequency penalty: positive values penalize tokens that have already appeared,
    /// reducing repetition. Applied as: logit -= freq_penalty * count(token).
    /// Typical range: 0.0 to 1.0. 0 = disabled.
    /// </summary>
    public float FrequencyPenalty { get; set; } = 0.0f;
    /// <summary>
    /// Presence penalty: penalizes tokens that have appeared at least once,
    /// encouraging the model to talk about new topics. Applied as: logit -= presence_penalty * (count > 0 ? 1 : 0).
    /// Typical range: 0.0 to 1.0. 0 = disabled.
    /// </summary>
    public float PresencePenalty { get; set; } = 0.0f;
    /// <summary>
    /// N-gram repetition penalty size: if > 0, penalizes repeating n-grams of this length.
    /// E.g. 3 means "don't repeat the same 3-token sequence". Higher = more aggressive.
    /// </summary>
    public int RepeatLastN { get; set; } = 64;
}

/// <summary>
/// High-level inference engine with pseudo-parallel processing.
///
/// GPU is only locked during the forward pass (the only GPU-bound operation).
/// Tokenization, sampling, and stop-sequence checking run on the CPU without
/// holding the GPU lock, allowing another request to use the GPU concurrently.
///
/// Each call to Generate/GenerateStreaming creates its own local KV cache and
/// SSM state, so multiple concurrent requests on the same engine are safe.
///
/// A CPU throttle (SemaphoreSlim) prevents too many concurrent CPU phases
/// from overwhelming system resources.
/// </summary>
public class InferenceEngine : IInferenceEngine
{
    private readonly TransformerBase _model;
    private readonly BPETokenizer _tokenizer;
    private readonly ComputeOps _ops;
    private bool _disposed;
    private readonly int _batchSize;
    private readonly ILogger<InferenceEngine> _logger = AppLogger.Create<InferenceEngine>();

    // Serializes GPU access: Vulkan command queues are NOT thread-safe.
    // Without this, concurrent requests cause vkQueueSubmit ErrorDeviceLost.
    // maxConcurrency controls how many threads may enter the GPU at once.
    // 1 = fully serialized (safe for single-GPU), 2+ = parallel (requires thread-safe command queue).
    private readonly SemaphoreSlim _gpuLock;

    // Throttles CPU-bound phases (tokenization, sampling) to avoid overwhelming
    // system memory / CPU cores when many requests are in-flight concurrently.
    // Set to Environment.ProcessorCount or a user-specified limit.
    private readonly SemaphoreSlim _cpuThrottle;

    // Random number generator for sampling (thread-safe via lock if needed,
    // but each call to Generate/GenerateStreaming is sequential from the caller's perspective)
    private readonly Random _random = new();

    public BPETokenizer Tokenizer => _tokenizer;
    public int BatchSize => _batchSize;
    public int ContextLength => _model.ContextLength;

    public InferenceEngine(TransformerBase model, BPETokenizer tokenizer, ComputeOps ops,
                           int batchSize = 8, int maxConcurrency = 1, int maxCpuConcurrency = 0)
    {
        _model = model;
        _tokenizer = tokenizer;
        _ops = ops;
        _batchSize = Math.Max(1, batchSize); // Ensure at least 1
        _gpuLock = new SemaphoreSlim(Math.Max(1, maxConcurrency), Math.Max(1, maxConcurrency));
        // Default CPU throttle = number of logical processors (avoid oversubscription)
        int cpuLimit = maxCpuConcurrency > 0 ? maxCpuConcurrency : Environment.ProcessorCount;
        _cpuThrottle = new SemaphoreSlim(cpuLimit, cpuLimit);
    }

    /// <summary>
    /// Generate text from a prompt with pseudo-parallel execution.
    ///
    /// The GPU lock is only held during the forward pass. CPU work
    /// (tokenization, sampling) runs outside the GPU lock so other
    /// requests can use the GPU concurrently.
    /// </summary>
    public string Generate(string prompt, GenerationConfig config,
                            CancellationToken cancellationToken = default)
    {
        // Phase 1: Tokenize prompt (CPU only, no GPU lock needed)
        // Use a flag to track whether we need to release the CPU throttle in finally
        bool cpuThrottleAcquired = false;
        _cpuThrottle.Wait(cancellationToken);
        cpuThrottleAcquired = true;
        try
        {
            var tokens = TokenizePrompt(prompt, config);
            _cpuThrottle.Release();
            cpuThrottleAcquired = false;

            // Phase 2: Generation loop — GPU forward pass under _gpuLock,
            //          CPU sampling outside _gpuLock
            var resultTokens = RunGenerationLoop(tokens, config, onToken: null, cancellationToken);
            return _tokenizer.Decode(resultTokens.ToArray());
        }
        finally
        {
            // Ensure throttle is released if TokenizePrompt threw
            if (cpuThrottleAcquired)
                _cpuThrottle.Release();
        }
    }

    public void GenerateStreaming(string prompt, GenerationConfig config,
                                   Action<string> onTokenGenerated,
                                   CancellationToken cancellationToken = default)
    {
        bool cpuThrottleAcquired = false;
        _cpuThrottle.Wait(cancellationToken);
        cpuThrottleAcquired = true;
        try
        {
            var tokens = TokenizePrompt(prompt, config);
            _cpuThrottle.Release();
            cpuThrottleAcquired = false;

            // Use a stateful UTF-8 Decoder so that multi-byte characters split across
            // tokens (e.g. 3-byte CJK chars in BPE) are buffered and emitted whole.
            var utf8Decoder = System.Text.Encoding.UTF8.GetDecoder();
            var charBuf = new char[256];
            RunGenerationLoop(tokens, config,
                onToken: tokenId =>
                {
                    var bytes = _tokenizer.GetTokenBytes(tokenId);
                    if (bytes.Length == 0) return;
                    int charCount = utf8Decoder.GetChars(bytes, 0, bytes.Length, charBuf, 0);
                    if (charCount > 0)
                        onTokenGenerated(new string(charBuf, 0, charCount));
                },
                cancellationToken);
        }
        finally
        {
            if (cpuThrottleAcquired)
                _cpuThrottle.Release();
        }
    }

    /// <summary>
    /// Generate text for multiple prompts sequentially.
    /// True parallel batch inference would require one InferenceEngine per prompt
    /// (Vulkan command queues are not thread-safe).
    /// </summary>
    public string[] BatchGenerate(string[] prompts, GenerationConfig config)
    {
        var results = new string[prompts.Length];
        for (int i = 0; i < prompts.Length; i++)
            results[i] = Generate(prompts[i], config);
        return results;
    }

    // ── Tokenization (CPU only) ──────────────────────────────────────────────

    /// <summary>
    /// Tokenize the prompt.
    /// Runs on CPU only — no GPU lock needed.
    /// KV cache and SSM state are created locally per-call in RunGenerationLoop for thread safety.
    /// </summary>
    private List<int> TokenizePrompt(string prompt, GenerationConfig config)
    {
        // Don't add BOS when the prompt already starts with a chat-template special token
        // (e.g. Qwen's <|im_start|>). Adding BOS=1 before special tokens is wrong.
        bool needsBos = !(prompt.StartsWith("<|im_start|>", StringComparison.Ordinal) ||
                          prompt.StartsWith("<|system|>",   StringComparison.Ordinal) ||
                          prompt.StartsWith("<s>",          StringComparison.Ordinal));
        var tokens = _tokenizer.Encode(prompt, addBos: needsBos, addEos: false).ToList();

        if (config.MaxPromptTokens > 0 && tokens.Count > config.MaxPromptTokens)
        {
            int removed = tokens.Count - config.MaxPromptTokens;
            // Keep BOS + tail of prompt (preserve most-recent context)
            tokens = [.. tokens.Take(1), .. tokens.Skip(removed + 1)];
            _logger.LogWarning("[Inference] Prompt truncated: {Original} → {Trimmed} tokens", tokens.Count + removed, tokens.Count);
        }

        _logger.LogDebug("[Inference] Prompt tokens: {Count} ids=[{Ids}]", tokens.Count, string.Join(",", tokens));

        return tokens;
    }

    // ── Generation loop with split GPU/CPU phases ────────────────────────────

    /// <summary>
    /// Runs the autoregressive generation loop.
    ///
    /// Each iteration:
    ///   1. GPU phase: forward pass under _gpuLock
    ///   2. CPU phase: ExtractLastToken + Sample + stop-check (outside _gpuLock)
    ///
    /// This allows another request to acquire the GPU lock for its forward pass
    /// while this request is doing CPU work (sampling, stop checking).
    ///
    /// KV cache and SSM state are local to this call, so concurrent requests
    /// on the same engine are safe.
    /// </summary>
    private List<int> RunGenerationLoop(List<int> tokens, GenerationConfig config,
                                         Action<int>? onToken,
                                         CancellationToken cancellationToken = default)
    {
        // Create local KV cache and SSM state for this generation
        using var kvCache = config.UseKVCache ? new KVCache(_ops) : null;
        using var ssmState = config.UseKVCache ? new SSMState(_ops.Device) : null;

        int eosToken = _tokenizer.EosToken;

        // Pre-encode stop sequences
        var stopSeqTokens = config.StopSequences
            .Select(s => _tokenizer.EncodeStopSequence(s))
            .Where(seq => seq.Length > 0)
            .ToList();

        var stopSingleTokens = new HashSet<int>(stopSeqTokens
            .Where(s => s.Length == 1).Select(s => s[0]));
        stopSingleTokens.Add(eosToken);

        var sw = System.Diagnostics.Stopwatch.StartNew();
        bool prefillDone = false;
        int generatedCount = 0;

        for (int i = 0; i < config.MaxNewTokens; i++)
        {
            if (cancellationToken.IsCancellationRequested)
            {
                _logger.LogDebug("[Inference] Cancelled at token {N}", generatedCount);
                break;
            }

            // ── GPU phase: forward pass ──────────────────────────────────────
            // Only this section holds the GPU lock. Other requests can queue
            // their forward passes while this one does CPU work.
            float[] lastLogits;
            {
                _logger.LogInformation("[INF] Acquiring GPU lock...");
                _gpuLock.Wait(cancellationToken);
                _logger.LogInformation("[INF] GPU lock acquired");
                try
                {
                    uint startPos = kvCache != null ? (uint)kvCache.SequenceLength : 0;
                    int[] inputTokens = kvCache != null && kvCache.SequenceLength > 0
                        ? new[] { tokens[^1] }
                        : tokens.ToArray();

                    _logger.LogInformation("[INF] Calling _model.Forward tokens={Cnt} pos={Pos}", inputTokens.Length, startPos);
                    var iterSw = System.Diagnostics.Stopwatch.StartNew();
                    var logits = _model.Forward(inputTokens, startPos, kvCache, ssmState);
                    _logger.LogInformation("[INF] _model.Forward done in {Ms}ms", iterSw.ElapsedMilliseconds);
                    lastLogits = ExtractLastToken(logits);
                    logits.Dispose();

                    if (!prefillDone)
                    {
                        _logger.LogDebug("[Inference] Prefill {Count} tokens: {Ms}ms", inputTokens.Length, iterSw.ElapsedMilliseconds);
                        prefillDone = true;
                    }
                    else if (generatedCount % 10 == 0)
                    {
                        double tps = generatedCount / sw.Elapsed.TotalSeconds;
                        _logger.LogDebug("[Inference] Token {N}: {Ms}ms | avg {Tps:F1} tok/s | kvLen={KvLen}", generatedCount, iterSw.ElapsedMilliseconds, tps, startPos);
                    }

                    if (generatedCount == 0)
                    {
                        var sorted = lastLogits.Select((v,i)=>(v,i)).OrderByDescending(x=>x.v).ToArray();
                        int rankQuin = Array.FindIndex(sorted, x => x.i == 24150);
                        int rankComma = Array.FindIndex(sorted, x => x.i == 1919);
                        _logger.LogTrace("[LogitCmp] 'quin'(24150)=rank{RankQuin},logit{LQuin:F3} | ' ,'(1919)=rank{RankComma},logit{LComma:F3}", rankQuin+1, lastLogits[24150], rankComma+1, lastLogits[1919]);
                        _logger.LogTrace("[LogitCmp] top1={Id}('{Tok}')={Val:F3}", sorted[0].i, _tokenizer.GetToken(sorted[0].i), sorted[0].v);
                        // Print top-5 tokens for debugging
                        _logger.LogWarning("[LogitCmp] Top-5 tokens at first step:");
                        for (int t = 0; t < 5 && t < sorted.Length; t++)
                        {
                            var tokStr = _tokenizer.GetToken(sorted[t].i);
                            _logger.LogWarning("  #{Rank}: id={Id} token='{Tok}' logit={Logit:F4}",
                                t + 1, sorted[t].i, tokStr, sorted[t].v);
                        }
                        // Print statistics
                        float maxL = sorted[0].v, minL = sorted[^1].v, meanL = lastLogits.Average();
                        _logger.LogWarning("[LogitCmp] Stats: max={Max:F4} min={Min:F4} mean={Mean:F4} range={Range:F4}",
                            maxL, minL, meanL, maxL - minL);
                    }
                }
                finally
                {
                    _gpuLock.Release();
                }
            }

            // ── CPU phase: sampling + stop check ─────────────────────────────
            // GPU lock is released — another request can now do its forward pass.
            int nextToken = Sample(lastLogits, tokens, config);
            tokens.Add(nextToken);
            generatedCount++;

            // Check stop BEFORE sending to client so stop tokens never appear in output
            if (stopSingleTokens.Contains(nextToken))
            {
                _logger.LogDebug("[Inference] Stop token {T} at {N}, total {Ms}ms", nextToken, generatedCount, sw.ElapsedMilliseconds);
                break;
            }

            // Multi-token stop sequence check (suffix match on recent tokens)
            if (stopSeqTokens.Any(seq =>
                tokens.Count >= seq.Length &&
                tokens.Skip(tokens.Count - seq.Length).SequenceEqual(seq)))
            {
                _logger.LogDebug("[Inference] Stop sequence matched at {N}, total {Ms}ms", generatedCount, sw.ElapsedMilliseconds);
                break;
            }

            onToken?.Invoke(nextToken);  // only called for non-stop tokens
        }

        if (generatedCount == config.MaxNewTokens)
            _logger.LogDebug("[Inference] Hit MaxNewTokens={Limit} limit, total {Ms}ms", config.MaxNewTokens, sw.ElapsedMilliseconds);

        return tokens;
    }

    private float[] ExtractLastToken(Tensor logits)
    {
        // logits shape: [seqLen, vocabSize] — only the last row is needed.
        // ReadRow avoids a full-buffer GPU→CPU transfer (e.g. 284 MB for 2214-token prefill).
        int lastRow = logits.Shape.Dimensions[0] - 1;
        return logits.ReadRow(lastRow);
    }

    private int Sample(float[] logits, List<int> generatedTokens, GenerationConfig config)
    {
        // Apply repetition penalty
        if (config.RepetitionPenalty != 1.0f)
        {
            logits = ApplyRepetitionPenalty(logits, generatedTokens, config.RepetitionPenalty);
        }

        // Apply frequency penalty (logit -= freq_penalty * count(token))
        if (config.FrequencyPenalty > 0.0f)
        {
            logits = ApplyFrequencyPenalty(logits, generatedTokens, config.FrequencyPenalty);
        }

        // Apply presence penalty (logit -= presence_penalty if token has appeared)
        if (config.PresencePenalty > 0.0f)
        {
            logits = ApplyPresencePenalty(logits, generatedTokens, config.PresencePenalty);
        }

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

    private float[] ApplyRepetitionPenalty(float[] logits, List<int> tokens, float penalty)
    {
        if (penalty == 1.0f || tokens.Count == 0)
            return logits;

        float[] penalized = (float[])logits.Clone();
        
        foreach (int tokenId in tokens)
        {
            if (tokenId >= 0 && tokenId < penalized.Length)
            {
                // If logit is positive, divide; if negative, multiply
                if (penalized[tokenId] > 0)
                    penalized[tokenId] /= penalty;
                else
                    penalized[tokenId] *= penalty;
            }
        }

        return penalized;
    }

    /// <summary>
    /// Frequency penalty: reduces logits for tokens that have appeared many times.
    /// logit -= freq_penalty * count(token)
    /// </summary>
    private float[] ApplyFrequencyPenalty(float[] logits, List<int> tokens, float freqPenalty)
    {
        if (freqPenalty <= 0.0f || tokens.Count == 0)
            return logits;

        float[] penalized = (float[])logits.Clone();
        
        // Count token frequencies
        var counts = new Dictionary<int, int>();
        foreach (int tokenId in tokens)
        {
            if (tokenId >= 0 && tokenId < penalized.Length)
            {
                counts.TryGetValue(tokenId, out int c);
                counts[tokenId] = c + 1;
            }
        }

        // Apply penalty proportional to frequency
        foreach (var (tokenId, count) in counts)
        {
            penalized[tokenId] -= freqPenalty * count;
        }

        return penalized;
    }

    /// <summary>
    /// Presence penalty: reduces logits for tokens that have appeared at least once.
    /// logit -= presence_penalty if token has appeared
    /// </summary>
    private float[] ApplyPresencePenalty(float[] logits, List<int> tokens, float presencePenalty)
    {
        if (presencePenalty <= 0.0f || tokens.Count == 0)
            return logits;

        float[] penalized = (float[])logits.Clone();
        var seen = new HashSet<int>(tokens);

        foreach (int tokenId in seen)
        {
            if (tokenId >= 0 && tokenId < penalized.Length)
            {
                penalized[tokenId] -= presencePenalty;
            }
        }

        return penalized;
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
        _gpuLock.Dispose();
        _cpuThrottle.Dispose();
        _disposed = true;
    }
}

/// <summary>
/// KV cache for efficient autoregressive generation
/// </summary>
public class KVCache : IDisposable
{
    private readonly ComputeOps _ops;
    // Dictionary keyed by ABSOLUTE layer index so sparse hybrid models
    // (SSM+Attention) with non-contiguous attention layers work correctly.
    // A List assumed 0,1,2,… which broke when SSM layers were skipped.
    private readonly Dictionary<int, (Tensor key, Tensor value)> _cache = new();
    private bool _disposed;

    public int SequenceLength { get; private set; }

    public KVCache(ComputeOps ops)
    {
        _ops = ops;
    }

    public void Add(int layer, Tensor key, Tensor value)
    {
        if (!_cache.ContainsKey(layer))
        {
            // First time for this layer — take ownership of tensors
            _cache[layer] = (key, value);
            SequenceLength = key.Shape.Dimensions[0];
        }
        else
        {
            var (oldKey, oldValue) = _cache[layer];

            var newKey   = _ops.Concat(oldKey,   key,   axis: 0, $"kv_cache_k_layer{layer}");
            var newValue = _ops.Concat(oldValue, value, axis: 0, $"kv_cache_v_layer{layer}");

            // Defer all four: Concat still references them in the GPU command buffer.
            _ops.DeferExternal(oldKey);
            _ops.DeferExternal(oldValue);
            _ops.DeferExternal(key);
            _ops.DeferExternal(value);

            _cache[layer] = (newKey, newValue);
            SequenceLength = newKey.Shape.Dimensions[0];
        }
    }

    public (Tensor? key, Tensor? value) Get(int layer)
        => _cache.TryGetValue(layer, out var kv) ? kv : (null, null);

    public void Clear()
    {
        foreach (var (_, (key, value)) in _cache)
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
