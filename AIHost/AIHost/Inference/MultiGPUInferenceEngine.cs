using AIHost.Compute;
using AIHost.GGUF;
using AIHost.ICompute;
using AIHost.Tokenizer;

namespace AIHost.Inference;

/// <summary>
/// Inference engine that splits a transformer model across multiple GPU devices.
/// Drop-in replacement for single-GPU InferenceEngine when DeviceIndices is set.
/// </summary>
public class MultiGPUInferenceEngine : IInferenceEngine
{
    private readonly MultiGPUTransformer _model;
    private readonly BPETokenizer _tokenizer;
    private readonly int _batchSize;
    private Random _random;
    private MultiDeviceKVCache? _kvCache;
    private bool _disposed;

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
        int                 batchSize = 8)
    {
        _model     = model;
        _tokenizer = tokenizer;
        _batchSize = Math.Max(1, batchSize);
        _random    = new Random();
    }

    // ── Public generation API (mirrors InferenceEngine) ──────────────────────

    public string Generate(string prompt, GenerationConfig config)
    {
        var tokens = Run(prompt, config, onToken: null);
        return _tokenizer.Decode(tokens.ToArray());
    }

    public void GenerateStreaming(string prompt, GenerationConfig config, Action<string> onToken)
        => Run(prompt, config, onToken: id => onToken(_tokenizer.GetToken(id)));

    // ── Core loop ────────────────────────────────────────────────────────────

    private List<int> Run(string prompt, GenerationConfig config, Action<int>? onToken)
    {
        if (config.Seed >= 0) _random = new Random(config.Seed);

        var tokens = _tokenizer.Encode(prompt, addBos: true, addEos: false).ToList();

        if (config.MaxPromptTokens > 0 && tokens.Count > config.MaxPromptTokens)
        {
            int removed = tokens.Count - config.MaxPromptTokens;
            tokens = [.. tokens.Take(1), .. tokens.Skip(removed + 1)];
            Console.WriteLine($"[MultiGPU] Prompt truncated to {tokens.Count} tokens");
        }

        Console.WriteLine($"[MultiGPU] Prompt tokens: {tokens.Count} ids=[{string.Join(",", tokens)}]");

        if (config.UseKVCache)
        {
            if (_kvCache == null) _kvCache = _model.CreateKVCache();
            else _kvCache.Clear();
        }
        else _kvCache?.Clear();

        int eosToken = _tokenizer.EosToken;
        var sw = System.Diagnostics.Stopwatch.StartNew();
        bool prefillDone = false;
        int generated = 0;

        for (int i = 0; i < config.MaxNewTokens; i++)
        {
            uint startPos = (uint)(_kvCache?.SequenceLength ?? 0);
            int[] inputTokens = (_kvCache != null && _kvCache.SequenceLength > 0)
                ? new[] { tokens[^1] }
                : tokens.ToArray();

            var iterSw = System.Diagnostics.Stopwatch.StartNew();
            var logitsTensor = _model.Forward(inputTokens, startPos, _kvCache);
            int lastRow = logitsTensor.Shape[0] - 1;
            float[] lastLogits = logitsTensor.ReadRow(lastRow);
            logitsTensor.Dispose();

            if (!prefillDone)
            {
                Console.WriteLine($"[MultiGPU] Prefill {inputTokens.Length} tok: {iterSw.ElapsedMilliseconds}ms");
                prefillDone = true;
            }
            else if (generated % 10 == 0)
            {
                double tps = generated / sw.Elapsed.TotalSeconds;
                Console.WriteLine($"[MultiGPU] Token {generated}: {iterSw.ElapsedMilliseconds}ms | {tps:F1} tok/s");
            }

            int next = Sample(lastLogits, tokens, config);
            tokens.Add(next);
            generated++;
            onToken?.Invoke(next);

            if (next == eosToken) { Console.WriteLine($"[MultiGPU] EOS at {generated}"); break; }
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
        _kvCache?.Dispose();
        _model.Dispose();
        _disposed = true;
    }
}
