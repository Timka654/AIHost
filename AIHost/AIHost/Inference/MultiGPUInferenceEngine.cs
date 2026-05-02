using AIHost.Compute;
using AIHost.GGUF;
using AIHost.ICompute;
using AIHost.ICompute.Vulkan;
using AIHost.Tokenizer;

namespace AIHost.Inference;

/// <summary>
/// Inference engine that splits a transformer model across multiple GPU devices.
/// Layer distribution is automatic (even split) or configurable via layerSplit.
/// </summary>
public class MultiGPUInferenceEngine : IDisposable
{
    private readonly MultiGPUTransformer _model;
    private readonly BPETokenizer _tokenizer;
    private Random _random;
    private MultiDeviceKVCache? _kvCache;
    private bool _disposed;

    public int DeviceCount => _model.DeviceCount;

    /// <summary>
    /// Load a model split across the given Vulkan device indices.
    /// layerSplit: optional array of N-1 layer boundaries (where device i+1 starts).
    /// If null, layers are distributed evenly.
    /// </summary>
    public MultiGPUInferenceEngine(IGGUFModel model, BPETokenizer tokenizer,
                                    int[] deviceIndices, int[]? layerSplit = null)
    {
        _tokenizer = tokenizer;
        _random    = new Random();

        var devices = deviceIndices.Select(i => (IComputeDevice)new VulkanComputeDevice(i)).ToArray();
        _model = new MultiGPUTransformer(devices, model, layerSplit);
    }

    /// <summary>Generate text from a prompt.</summary>
    public string Generate(string prompt, GenerationConfig config)
    {
        var tokens = Run(prompt, config, onToken: null);
        return _tokenizer.Decode(tokens.ToArray());
    }

    /// <summary>Generate with streaming callback.</summary>
    public void GenerateStreaming(string prompt, GenerationConfig config, Action<string> onToken)
        => Run(prompt, config, onToken: id => onToken(_tokenizer.GetToken(id)));

    private List<int> Run(string prompt, GenerationConfig config, Action<int>? onToken)
    {
        if (config.Seed >= 0) _random = new Random(config.Seed);

        var tokens = _tokenizer.Encode(prompt, addBos: true, addEos: false).ToList();

        if (config.MaxPromptTokens > 0 && tokens.Count > config.MaxPromptTokens)
        {
            int removed = tokens.Count - config.MaxPromptTokens;
            tokens = [.. tokens.Take(1), .. tokens.Skip(removed + 1)];
        }

        Console.WriteLine($"[MultiGPU] Prompt tokens: {tokens.Count}");

        if (config.UseKVCache)
        {
            if (_kvCache == null) _kvCache = _model.CreateKVCache();
            else _kvCache.Clear();
        }

        int eosToken = _tokenizer.EosToken;
        var sw = System.Diagnostics.Stopwatch.StartNew();

        for (int i = 0; i < config.MaxNewTokens; i++)
        {
            uint startPos = (uint)(_kvCache?.SequenceLength ?? 0);
            int[] inputTokens = (_kvCache != null && _kvCache.SequenceLength > 0)
                ? new[] { tokens[^1] }
                : tokens.ToArray();

            var logitsTensor = _model.Forward(inputTokens, startPos, _kvCache);
            int lastRow = logitsTensor.Shape[0] - 1;
            float[] lastLogits = logitsTensor.ReadRow(lastRow);
            logitsTensor.Dispose();

            if (i == 0) Console.WriteLine($"[MultiGPU] Prefill {inputTokens.Length} tok: {sw.ElapsedMilliseconds}ms");

            int next = Sample(lastLogits, tokens, config);
            tokens.Add(next);
            onToken?.Invoke(next);

            if (next == eosToken) break;
        }

        return tokens;
    }

    private int Sample(float[] logits, List<int> prev, GenerationConfig cfg)
    {
        // Repetition penalty
        if (cfg.RepetitionPenalty != 1f)
            foreach (int t in prev)
                if (t >= 0 && t < logits.Length)
                    logits[t] = logits[t] > 0 ? logits[t] / cfg.RepetitionPenalty : logits[t] * cfg.RepetitionPenalty;

        // Temperature
        if (Math.Abs(cfg.Temperature - 1f) > 0.001f)
            for (int i = 0; i < logits.Length; i++) logits[i] /= cfg.Temperature;

        // Softmax
        float max = logits.Max();
        float[] p = logits.Select(v => MathF.Exp(v - max)).ToArray();
        float sum = p.Sum(); for (int i = 0; i < p.Length; i++) p[i] /= sum;

        // Top-K
        if (cfg.TopK > 0)
        {
            var topk = p.Select((v, i) => (v, i)).OrderByDescending(x => x.v).Take(cfg.TopK).ToArray();
            var filtered = new float[p.Length];
            float s2 = 0; foreach (var (v, i) in topk) { filtered[i] = v; s2 += v; }
            if (s2 > 0) for (int i = 0; i < filtered.Length; i++) filtered[i] /= s2;
            p = filtered;
        }

        // Top-P
        if (cfg.TopP < 1f)
        {
            var sorted = p.Select((v, i) => (v, i)).OrderByDescending(x => x.v).ToArray();
            float cum = 0; int cut = 0;
            for (int i = 0; i < sorted.Length; i++) { cum += sorted[i].v; cut = i + 1; if (cum >= cfg.TopP) break; }
            var filtered = new float[p.Length]; float s2 = 0;
            for (int i = 0; i < cut; i++) { filtered[sorted[i].i] = sorted[i].v; s2 += sorted[i].v; }
            if (s2 > 0) for (int i = 0; i < filtered.Length; i++) filtered[i] /= s2;
            p = filtered;
        }

        // Sample
        float r = (float)_random.NextDouble(), cumSum = 0;
        for (int i = 0; i < p.Length; i++) { cumSum += p[i]; if (r < cumSum) return i; }
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
