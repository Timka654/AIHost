using AIHost.Compute;
using AIHost.GGUF;
using AIHost.ICompute;
using AIHost.Tokenizer;
using Microsoft.Extensions.Logging;

namespace AIHost.Inference;

/// <summary>
/// Inference engine that splits a transformer model across multiple GPU devices.
/// Drop-in replacement for single-GPU InferenceEngine when DeviceIndices is set.
///
/// Pseudo-parallel: GPU lock is only held during the forward pass.
/// Tokenization, sampling, and stop-sequence checking run on the CPU without
/// holding the GPU lock, allowing another request to use the GPU concurrently.
/// </summary>
public class MultiGPUInferenceEngine : IInferenceEngine
{
    private readonly MultiGPUTransformer _model;
    private readonly BPETokenizer _tokenizer;
    private readonly int _batchSize;
    private readonly SemaphoreSlim _gpuLock;
    private readonly SemaphoreSlim _cpuThrottle;
    private readonly Random _random = new();
    private bool _disposed;
    private readonly ILogger<MultiGPUInferenceEngine> _logger = AppLogger.Create<MultiGPUInferenceEngine>();

    public BPETokenizer Tokenizer    => _tokenizer;
    public int DeviceCount           => _model.DeviceCount;
    public int ContextLength         => 0; // populated from model metadata if needed

    /// <summary>
    /// Construct from an already-built MultiGPUTransformer and tokenizer.
    /// ModelManager calls this after creating transformer + tokenizer.
    /// </summary>
    public MultiGPUInferenceEngine(
        MultiGPUTransformer model,
        BPETokenizer        tokenizer,
        int                 batchSize = 8,
        int                 maxConcurrency = 1,
        int                 maxCpuConcurrency = 0)
    {
        _model     = model;
        _tokenizer = tokenizer;
        _batchSize = Math.Max(1, batchSize);
        // Multi-GPU всегда использует maxConcurrency=1, так как каждый запрос
        // уже последовательно проходит через все GPU. Параллельные запросы
        // на одном наборе GPU вызывают vkQueueSubmit ErrorDeviceLost.
        _gpuLock   = new SemaphoreSlim(1, 1);
        // Default CPU throttle = number of logical processors (avoid oversubscription)
        int cpuLimit = maxCpuConcurrency > 0 ? maxCpuConcurrency : Environment.ProcessorCount;
        _cpuThrottle = new SemaphoreSlim(cpuLimit, cpuLimit);
    }

    // ── Public generation API (mirrors InferenceEngine) ──────────────────────

    public string Generate(string prompt, GenerationConfig config,
                            CancellationToken cancellationToken = default)
    {
        // Phase 1: Tokenize prompt (CPU only, no GPU lock needed)
        bool cpuThrottleAcquired = false;
        _cpuThrottle.Wait(cancellationToken);
        cpuThrottleAcquired = true;
        try
        {
            var promptTokens = _tokenizer.Encode(prompt, addBos: true, addEos: false).ToList();
            _cpuThrottle.Release();
            cpuThrottleAcquired = false;

            // Phase 2: Generation loop — GPU forward pass under _gpuLock,
            //          CPU sampling outside _gpuLock
            var allTokens = RunGenerationLoop(promptTokens, config, onToken: null, cancellationToken);
            // Return only newly generated tokens (skip prompt tokens)
            var newTokens = allTokens.Skip(promptTokens.Count).ToArray();
            return _tokenizer.Decode(newTokens);
        }
        finally
        {
            // Ensure throttle is released if tokenization threw
            if (cpuThrottleAcquired)
                _cpuThrottle.Release();
        }
    }

    public void GenerateStreaming(string prompt, GenerationConfig config,
                                   Action<string> onToken,
                                   CancellationToken cancellationToken = default)
    {
        bool cpuThrottleAcquired = false;
        _cpuThrottle.Wait(cancellationToken);
        cpuThrottleAcquired = true;
        try
        {
            var promptTokens = _tokenizer.Encode(prompt, addBos: true, addEos: false).ToList();
            _cpuThrottle.Release();
            cpuThrottleAcquired = false;

            RunGenerationLoop(promptTokens, config, id => onToken(_tokenizer.GetToken(id)), cancellationToken);
        }
        finally
        {
            if (cpuThrottleAcquired)
                _cpuThrottle.Release();
        }
    }

    // ── Generation loop with split GPU/CPU phases ────────────────────────────

    /// <summary>
    /// Runs the autoregressive generation loop.
    ///
    /// Each iteration:
    ///   1. GPU phase: forward pass under _gpuLock
    ///   2. CPU phase: ReadRow + Sample + stop-check (outside _gpuLock)
    ///
    /// This allows another request to acquire the GPU lock for its forward pass
    /// while this request is doing CPU work (sampling, stop checking).
    ///
    /// KV cache is created locally per-call for thread safety.
    /// </summary>
    private List<int> RunGenerationLoop(List<int> tokens, GenerationConfig config,
                                         Action<int>? onToken,
                                         CancellationToken cancellationToken = default)
    {
        // Create local KV cache and SSM state for this generation
        using var kvCache = config.UseKVCache ? _model.CreateKVCache() : null;
        using var ssmState = config.UseKVCache ? new SSMState(_model.PrimaryDevice) : null;

        if (config.Seed >= 0)
        {
            // Note: _random is readonly, but we can't replace it.
            // For deterministic generation with seed, we'd need a local Random.
            // For now, seed is best-effort (not perfectly deterministic across calls).
        }

        if (config.MaxPromptTokens > 0 && tokens.Count > config.MaxPromptTokens)
        {
            int removed = tokens.Count - config.MaxPromptTokens;
            tokens = [.. tokens.Take(1), .. tokens.Skip(removed + 1)];
            _logger.LogWarning("[MultiGPU] Prompt truncated to {Count} tokens", tokens.Count);
        }

        _logger.LogDebug("[MultiGPU] Prompt tokens: {Count} ids=[{Ids}]", tokens.Count, string.Join(",", tokens));

        int eosToken = _tokenizer.EosToken;
        var sw = System.Diagnostics.Stopwatch.StartNew();
        bool prefillDone = false;
        int generated = 0;

        for (int i = 0; i < config.MaxNewTokens; i++)
        {
            if (cancellationToken.IsCancellationRequested)
            {
                _logger.LogDebug("[MultiGPU] Cancelled at token {N}", generated);
                break;
            }

            // ── GPU phase: forward pass ──────────────────────────────────────
            // Only this section holds the GPU lock. Other requests can queue
            // their forward passes while this one does CPU work.
            float[] lastLogits;
            {
                _gpuLock.Wait(cancellationToken);
                try
                {
                    uint startPos = (uint)(kvCache?.SequenceLength ?? 0);
                    int[] inputTokens = (kvCache != null && kvCache.SequenceLength > 0)
                        ? new[] { tokens[^1] }
                        : tokens.ToArray();

                    var iterSw = System.Diagnostics.Stopwatch.StartNew();
                    var logitsTensor = _model.Forward(inputTokens, startPos, kvCache, ssmState);
                    int lastRow = logitsTensor.Shape[0] - 1;
                    lastLogits = logitsTensor.ReadRow(lastRow);
                    logitsTensor.Dispose();

                    if (!prefillDone)
                    {
                        _logger.LogDebug("[MultiGPU] Prefill {Count} tok: {Ms}ms", inputTokens.Length, iterSw.ElapsedMilliseconds);
                        prefillDone = true;
                    }
                    else if (generated % 10 == 0)
                    {
                        double tps = generated / sw.Elapsed.TotalSeconds;
                        _logger.LogDebug("[MultiGPU] Token {N}: {Ms}ms | {Tps:F1} tok/s", generated, iterSw.ElapsedMilliseconds, tps);
                    }
                }
                finally
                {
                    _gpuLock.Release();
                }
            }

            // ── CPU phase: sampling + stop check ─────────────────────────────
            // GPU lock is released — another request can now do its forward pass.
            int next = Sample(lastLogits, tokens, config);
            tokens.Add(next);
            generated++;

            // Check stop BEFORE sending to client so stop tokens never appear in output
            if (next == eosToken) { _logger.LogDebug("[MultiGPU] EOS at {N}", generated); break; }

            // Check stop sequences (suffix match on recently generated text only)
            if (config.StopSequences.Count > 0)
            {
                // Only check the last N tokens to avoid matching stop sequences from the prompt
                int checkLen = Math.Min(generated, 20);
                var recentText = _tokenizer.Decode(tokens.Skip(tokens.Count - checkLen).ToArray());
                foreach (var stop in config.StopSequences)
                {
                    if (recentText.EndsWith(stop, StringComparison.Ordinal))
                    {
                        _logger.LogDebug("[MultiGPU] Stop sequence '{Stop}' at {N}", stop, generated);
                        // Remove the stop sequence from tokens
                        int[] stopTokenIds = _tokenizer.Encode(stop, addBos: false, addEos: false);
                        int stopTokenCount = stopTokenIds.Length;
                        if (stopTokenCount > 0 && tokens.Count >= stopTokenCount)
                            tokens.RemoveRange(tokens.Count - stopTokenCount, stopTokenCount);
                        return tokens;
                    }
                }
            }

            onToken?.Invoke(next);  // only called for non-stop tokens
        }

        return tokens;
    }

    // ── Sampling (mirrors InferenceEngine) ───────────────────────────────────

    private int Sample(float[] logits, List<int> prev, GenerationConfig cfg)
    {
        if (cfg.RepetitionPenalty != 1f)
            foreach (int t in prev)
                if (t >= 0 && t < logits.Length)
                    logits[t] = logits[t] > 0 ? logits[t] / cfg.RepetitionPenalty : logits[t] * cfg.RepetitionPenalty;

        if (cfg.FrequencyPenalty > 0f)
        {
            var freq = prev.GroupBy(t => t).ToDictionary(g => g.Key, g => g.Count());
            foreach (var (token, count) in freq)
                if (token >= 0 && token < logits.Length)
                    logits[token] -= cfg.FrequencyPenalty * count;
        }

        if (cfg.PresencePenalty > 0f)
        {
            var seen = new HashSet<int>(prev);
            foreach (int token in seen)
                if (token >= 0 && token < logits.Length)
                    logits[token] -= cfg.PresencePenalty;
        }

        if (Math.Abs(cfg.Temperature - 1f) > 0.001f)
            for (int i = 0; i < logits.Length; i++) logits[i] /= cfg.Temperature;

        float[] p = Softmax(logits);
        if (cfg.TopK > 0)  p = ApplyTopK(p, cfg.TopK);
        if (cfg.TopP < 1f) p = ApplyTopP(p, cfg.TopP);
        return SampleFromDistribution(p);
    }

    private static float[] Softmax(float[] x)
    {
        float max = x.Max();
        float[] e = x.Select(v => MathF.Exp(v - max)).ToArray();
        float s = e.Sum();
        return e.Select(v => v / s).ToArray();
    }

    private static float[] ApplyTopK(float[] p, int k)
    {
        var topk = p.Select((v, i) => (v, i)).OrderByDescending(x => x.v).Take(k).ToArray();
        var f = new float[p.Length];
        float s = 0;
        foreach (var (v, i) in topk) { f[i] = v; s += v; }
        if (s > 0) for (int i = 0; i < f.Length; i++) f[i] /= s;
        return f;
    }

    private static float[] ApplyTopP(float[] p, float nucleus)
    {
        var sorted = p.Select((v, i) => (v, i)).OrderByDescending(x => x.v).ToArray();
        float cum = 0; int cut = 0;
        for (int i = 0; i < sorted.Length; i++) { cum += sorted[i].v; cut = i + 1; if (cum >= nucleus) break; }
        var f = new float[p.Length]; float s = 0;
        for (int i = 0; i < cut; i++) { f[sorted[i].i] = sorted[i].v; s += sorted[i].v; }
        if (s > 0) for (int i = 0; i < f.Length; i++) f[i] /= s;
        return f;
    }

    private int SampleFromDistribution(float[] p)
    {
        float r = (float)_random.NextDouble(), cum = 0;
        for (int i = 0; i < p.Length; i++) { cum += p[i]; if (r < cum) return i; }
        for (int i = p.Length - 1; i >= 0; i--) if (p[i] > 0) return i;
        return 0;
    }

    public void Dispose()
    {
        if (_disposed) return;
        _model.Dispose();
        _gpuLock.Dispose();
        _cpuThrottle.Dispose();
        _disposed = true;
    }
}
