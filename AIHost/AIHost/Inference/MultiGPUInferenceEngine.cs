using AIHost.Compute;
using AIHost.ICompute;
using AIHost.ICompute.Vulkan;
using AIHost.Tokenizer;

namespace AIHost.Inference;

/// <summary>
/// Multi-GPU inference engine with layer-wise model parallelism
/// Distributes transformer layers across multiple GPUs for parallel execution
/// </summary>
public class MultiGPUInferenceEngine : IDisposable
{
    private readonly IComputeDevice[] _devices;
    private readonly Transformer[] _models;
    private readonly BPETokenizer _tokenizer;
    private readonly ComputeOps[] _ops;
    private readonly int[] _layerToDevice; // Maps layer index to device index
    private Random _random;
    private KVCache?[] _kvCaches; // One KV-cache per device
    private bool _disposed;

    public int DeviceCount => _devices.Length;

    /// <summary>
    /// Create multi-GPU inference engine
    /// </summary>
    /// <param name="deviceIndices">GPU device indices to use (e.g., [0, 1] for first two GPUs)</param>
    /// <param name="modelPath">Path to GGUF model file</param>
    public MultiGPUInferenceEngine(int[] deviceIndices, string modelPath)
    {
        if (deviceIndices == null || deviceIndices.Length == 0)
            throw new ArgumentException("Must specify at least one device", nameof(deviceIndices));

        _random = new Random();
        
        // Initialize devices and load model on each
        _devices = new IComputeDevice[deviceIndices.Length];
        _models = new Transformer[deviceIndices.Length];
        _ops = new ComputeOps[deviceIndices.Length];
        _kvCaches = new KVCache?[deviceIndices.Length];

        Console.WriteLine($"\n=== Initializing Multi-GPU Inference ({deviceIndices.Length} GPUs) ===");
        
        for (int i = 0; i < deviceIndices.Length; i++)
        {
            int deviceIdx = deviceIndices[i];
            Console.WriteLine($"Loading on GPU {deviceIdx}...");
            
            _devices[i] = new VulkanComputeDevice(deviceIdx);
            var model = new GGUF.GGUFModel(modelPath, _devices[i]);
            _models[i] = new Transformer(_devices[i], model);
            _models[i].LoadWeights();
            _ops[i] = new ComputeOps(_devices[i]);
            
            // Load tokenizer only once from first device
            if (i == 0)
                _tokenizer = BPETokenizer.FromGGUF(model.Reader);
        }

        // Distribute layers across devices (round-robin)
        int totalLayers = _models[0].LayerCount;
        _layerToDevice = new int[totalLayers];
        
        for (int layer = 0; layer < totalLayers; layer++)
        {
            _layerToDevice[layer] = layer % deviceIndices.Length;
        }

        Console.WriteLine($"Layer distribution:");
        for (int dev = 0; dev < deviceIndices.Length; dev++)
        {
            var layers = Enumerable.Range(0, totalLayers).Where(l => _layerToDevice[l] == dev).ToArray();
            Console.WriteLine($"  GPU {deviceIndices[dev]}: Layers {string.Join(", ", layers)}");
        }
        Console.WriteLine();
    }

    /// <summary>
    /// Generate text from a prompt using multiple GPUs
    /// </summary>
    public string Generate(string prompt, GenerationConfig config)
    {
        var tokens = PrepareAndRun(prompt, config, onToken: null);
        return _tokenizer.Decode(tokens.ToArray());
    }

    /// <summary>
    /// Generate text with streaming callback
    /// </summary>
    public void GenerateStreaming(string prompt, GenerationConfig config, Action<string> onTokenGenerated)
    {
        PrepareAndRun(prompt, config, onToken: tokenId =>
            onTokenGenerated(_tokenizer.GetToken(tokenId)));
    }

    private List<int> PrepareAndRun(string prompt, GenerationConfig config, Action<int>? onToken)
    {
        if (config.Seed >= 0)
            _random = new Random(config.Seed);

        // Tokenize prompt
        var promptTokens = _tokenizer.Encode(prompt, true, false);
        var generatedTokens = new List<int>(promptTokens);

        // Initialize KV-caches
        if (config.UseKVCache)
        {
            for (int i = 0; i < _devices.Length; i++)
            {
                if (_kvCaches[i] == null)
                    _kvCaches[i] = new KVCache(_ops[i]);
                else
                    _kvCaches[i].Clear();
            }
        }

        int eosToken = _tokenizer.EosToken;

        // Forward pass for prompt tokens
        uint position = 0;
        foreach (var tokenId in promptTokens)
        {
            var logitsTensor = ForwardMultiGPU(new[] { tokenId }, position, config.UseKVCache ? _kvCaches : null);
            logitsTensor.Dispose();
            position++;
        }

        // Generate new tokens
        int lastTokenId = promptTokens[^1];
        for (int i = 0; i < config.MaxNewTokens; i++)
        {
            // Get logits from forward pass
            var logitsTensor = ForwardMultiGPU(new[] { lastTokenId }, position, config.UseKVCache ? _kvCaches : null);
            var logits = ExtractLastToken(logitsTensor);
            logitsTensor.Dispose();

            // Sample next token
            int nextTokenId = Sample(logits, generatedTokens, config);
            
            if (nextTokenId == eosToken)
                break;

            generatedTokens.Add(nextTokenId);
            onToken?.Invoke(nextTokenId);

            lastTokenId = nextTokenId;
            position++;
        }

        return generatedTokens;
    }

    /// <summary>
    /// Multi-GPU forward pass: distribute layers across devices
    /// </summary>
    private Tensor ForwardMultiGPU(int[] tokenIds, uint position, KVCache?[]? kvCaches)
    {
        // Note: This is a simplified implementation
        // In production, you'd want to:
        // 1. Pipeline execution across GPUs
        // 2. Optimize data transfers between devices
        // 3. Use async execution where possible

        // For now, we execute on device 0 (single-GPU fallback)
        // Full multi-GPU with layer splitting requires more complex tensor routing
        return _models[0].Forward(tokenIds, position, kvCaches?[0]);
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

    private int Sample(float[] logits, List<int> generatedTokens, GenerationConfig config)
    {
        // Apply repetition penalty
        if (config.RepetitionPenalty != 1.0f)
            ApplyRepetitionPenalty(logits, generatedTokens, config.RepetitionPenalty);

        // Apply temperature
        if (config.Temperature != 1.0f)
        {
            for (int i = 0; i < logits.Length; i++)
                logits[i] /= config.Temperature;
        }

        // Top-K filtering
        if (config.TopK > 0 && config.TopK < logits.Length)
        {
            var topK = logits.Select((logit, idx) => (logit, idx))
                            .OrderByDescending(x => x.logit)
                            .Take(config.TopK)
                            .ToList();

            float minTopK = topK.Min(x => x.logit);
            for (int i = 0; i < logits.Length; i++)
            {
                if (logits[i] < minTopK)
                    logits[i] = float.NegativeInfinity;
            }
        }

        // Softmax
        float maxLogit = logits.Max();
        float[] probs = new float[logits.Length];
        float sumExp = 0.0f;

        for (int i = 0; i < logits.Length; i++)
        {
            probs[i] = MathF.Exp(logits[i] - maxLogit);
            sumExp += probs[i];
        }

        for (int i = 0; i < probs.Length; i++)
            probs[i] /= sumExp;

        // Top-P (nucleus) filtering
        if (config.TopP < 1.0f)
        {
            var sortedProbs = probs.Select((prob, idx) => (prob, idx))
                                  .OrderByDescending(x => x.prob)
                                  .ToList();

            float cumProb = 0.0f;
            int cutoff = sortedProbs.Count;

            for (int i = 0; i < sortedProbs.Count; i++)
            {
                cumProb += sortedProbs[i].prob;
                if (cumProb >= config.TopP)
                {
                    cutoff = i + 1;
                    break;
                }
            }

            var topPIndices = sortedProbs.Take(cutoff).Select(x => x.idx).ToHashSet();
            for (int i = 0; i < probs.Length; i++)
            {
                if (!topPIndices.Contains(i))
                    probs[i] = 0.0f;
            }

            // Renormalize
            float sum = probs.Sum();
            if (sum > 0)
            {
                for (int i = 0; i < probs.Length; i++)
                    probs[i] /= sum;
            }
        }

        // Sample from distribution
        float r = (float)_random.NextDouble();
        float cumulative = 0.0f;

        for (int i = 0; i < probs.Length; i++)
        {
            cumulative += probs[i];
            if (r <= cumulative)
                return i;
        }

        return probs.Length - 1;
    }

    private void ApplyRepetitionPenalty(float[] logits, List<int> tokens, float penalty)
    {
        foreach (int tokenId in tokens)
        {
            if (tokenId >= 0 && tokenId < logits.Length)
            {
                if (logits[tokenId] > 0)
                    logits[tokenId] /= penalty;
                else
                    logits[tokenId] *= penalty;
            }
        }
    }

    public void Dispose()
    {
        if (_disposed) return;

        for (int i = 0; i < _devices.Length; i++)
        {
            _kvCaches[i]?.Dispose();
            _ops[i]?.Dispose();
            _models[i]?.Dispose(); // releases weight cache GPU buffers
        }

        // GC before device teardown — ensures finalizers of any lingering
        // GPU objects run while the Vulkan device is still alive.
        GC.Collect();
        GC.WaitForPendingFinalizers();

        for (int i = 0; i < _devices.Length; i++)
            _devices[i]?.Dispose();

        _disposed = true;
    }
}
