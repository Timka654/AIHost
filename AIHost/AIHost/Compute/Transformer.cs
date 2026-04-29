using AIHost.GGUF;
using AIHost.ICompute;
using AIHost.Inference;

namespace AIHost.Compute;

/// <summary>
/// LLM Transformer model for inference.
///
/// Memory strategy: weights are stored QUANTIZED in the cache (~600 MB for Q4_K_M).
/// Each forward pass dequantizes only the weights needed for the current operation into
/// temporary F32 tensors that are freed immediately after use.
/// Peak GPU memory per layer ≈ 160 MB F32 (vs 4.4 GB if all pre-dequantized).
/// </summary>
public class Transformer : IDisposable
{
    private readonly ComputeOps _ops;
    private readonly IGGUFModel _model;
    private readonly int _numLayers;
    private readonly int _dModel;
    private readonly int _numHeads;
    private bool _disposed;

    // Stores QUANTIZED tensors (or F32 for weights already stored as F32 in GGUF).
    private readonly Dictionary<string, Tensor> _weightCache = [];

    public int LayerCount => _numLayers;
    public ComputeOps Ops => _ops;

    public Transformer(IComputeDevice device, IGGUFModel model)
    {
        _ops = new ComputeOps(device);
        _model = model;

        var metadata = model.Metadata;
        _numLayers = metadata.GetValue<int>(GGUFMetadata.KeyBlockCount, 22);
        _dModel = metadata.GetValue<int>(GGUFMetadata.KeyEmbeddingLength, 2048);
        _numHeads = metadata.GetValue<int>(GGUFMetadata.KeyAttentionHeadCount, 32);

        Console.WriteLine($"Transformer initialized: layers={_numLayers} d_model={_dModel} heads={_numHeads}");
    }

    /// <summary>
    /// Upload all model weights to GPU (quantized). Fast: no dequantization here.
    /// </summary>
    public void LoadWeights()
    {
        Console.WriteLine("Loading model weights...");

        CacheWeight("token_embd.weight");
        CacheWeight("output_norm.weight");
        CacheWeight("output.weight");

        for (int i = 0; i < _numLayers; i++)
        {
            string p = $"blk.{i}";
            CacheWeight($"{p}.attn_norm.weight");
            CacheWeight($"{p}.attn_q.weight");
            CacheWeight($"{p}.attn_k.weight");
            CacheWeight($"{p}.attn_v.weight");
            CacheWeight($"{p}.attn_output.weight");
            CacheWeight($"{p}.ffn_norm.weight");
            CacheWeight($"{p}.ffn_gate.weight");
            CacheWeight($"{p}.ffn_up.weight");
            CacheWeight($"{p}.ffn_down.weight");

            if ((i + 1) % 5 == 0 || i == _numLayers - 1)
                Console.WriteLine($"  Uploaded layer {i + 1}/{_numLayers}");
        }

        Console.WriteLine($"✓ Weights loaded ({_weightCache.Count} tensors, quantized in VRAM)\n");
    }

    /// <summary>
    /// Forward pass through transformer.
    /// </summary>
    public Tensor Forward(int[] tokenIds, uint startPosition = 0, KVCache? kvCache = null)
    {
        if (!_weightCache.ContainsKey("token_embd.weight"))
            throw new InvalidOperationException("Weights not loaded. Call LoadWeights() first.");
#if DEEP_DEBUG
        Console.WriteLine($"Forward pass: {tokenIds.Length} tokens at position {startPosition}");
#endif

        // 1. Token embedding: dequantize table → lookup → free table
        Tensor x;
        using (var embF32 = TempF32("token_embd.weight"))
            x = _ops.EmbeddingLookup(tokenIds, embF32, "embeddings");

        // 2. All transformer layers (each layer dequantizes its own weights transiently)
        for (int i = 0; i < _numLayers; i++)
        {
#if DEEP_DEBUG
            if (i % 5 == 0 || i == _numLayers - 1)
                Console.WriteLine($"  Layer {i}/{_numLayers}...");
#endif
            x = ApplyLayer(x, i, startPosition, kvCache);
        }

        // 3. Final layer norm
        using (var normF32 = TempF32("output_norm.weight"))
            _ops.LayerNorm(x, normF32);

        // 4. Vocab projection
        Tensor logits;
        using (var outF32 = TempF32("output.weight"))
            logits = _ops.MatMul(x, outF32, "logits");

        x.Dispose();
        return logits;
    }

    private Tensor ApplyLayer(Tensor x, int layerIdx, uint position, KVCache? kvCache = null)
    {
        string p = $"blk.{layerIdx}";

        // All ops within a layer are batched into one GPU submission.
        // Temp F32 tensors and the incoming x are deferred — they stay alive until
        // after the single fence wait, then ComputeOps.Flush() disposes them.
        _ops.BeginBatch();

        Tensor W(string name)
        {
            var t = TempF32(name);
            _ops.DeferExternal(t);  // freed after layer flush
            return t;
        }

        var output = _ops.TransformerLayer(
            x,
            W($"{p}.attn_norm.weight"),
            W($"{p}.attn_q.weight"),
            W($"{p}.attn_k.weight"),
            W($"{p}.attn_v.weight"),
            W($"{p}.attn_output.weight"),
            W($"{p}.ffn_norm.weight"),
            W($"{p}.ffn_gate.weight"),
            W($"{p}.ffn_up.weight"),
            W($"{p}.ffn_down.weight"),
            _numHeads, position, kvCache, layerIdx);

        _ops.DeferExternal(x);  // x no longer needed after this layer
        _ops.Flush();           // 1 fence wait for the entire layer

        return output;
    }

    /// <summary>
    /// Returns a new F32 tensor from the named cached weight.
    /// The caller MUST dispose the returned tensor.
    /// </summary>
    private Tensor TempF32(string name)
    {
        var cached = _weightCache[name];
        // F32 weights (e.g., norm weights): Clone so caller can safely dispose without
        // touching the cached original. Quantized weights: Dequantize creates a new buffer.
        return cached.DataType == DataType.F32
            ? _ops.Clone(cached)
            : _ops.Dequantize(cached);
    }

    /// <summary>
    /// Uploads a single weight to GPU (quantized, no dequantization).
    /// </summary>
    private void CacheWeight(string name)
    {
        if (_weightCache.ContainsKey(name)) return;

        var info = _model.Tensors.FirstOrDefault(t => t.Name == name)
            ?? throw new ArgumentException($"Tensor '{name}' not found in GGUF");

        var buffer = _model.LoadTensor(name);
        var dtype = MapType(info.Type);
        var dims = info.Shape.Select(s => (int)s).ToArray();

        _weightCache[name] = new Tensor(buffer, new TensorShape(dims), dtype, name);
    }

    private static DataType MapType(GGUFTensorType t) => t switch
    {
        GGUFTensorType.F32  => DataType.F32,
        GGUFTensorType.F16  => DataType.F16,
        GGUFTensorType.Q4_0 => DataType.Q4_0,
        GGUFTensorType.Q4_1 => DataType.Q4_1,
        GGUFTensorType.Q5_0 => DataType.Q5_0,
        GGUFTensorType.Q5_1 => DataType.Q5_1,
        GGUFTensorType.Q8_0 => DataType.Q8_0,
        GGUFTensorType.Q8_1 => DataType.Q8_1,
        GGUFTensorType.Q2_K => DataType.Q2_K,
        GGUFTensorType.Q3_K => DataType.Q3_K,
        GGUFTensorType.Q4_K => DataType.Q4_K,
        GGUFTensorType.Q5_K => DataType.Q5_K,
        GGUFTensorType.Q6_K => DataType.Q6_K,
        GGUFTensorType.Q8_K => DataType.Q8_K,
        _ => throw new NotSupportedException($"Unsupported tensor type: {t}")
    };

    public void Dispose()
    {
        if (_disposed) return;

        foreach (var t in _weightCache.Values)
            t.Dispose();
        _weightCache.Clear();

        _ops.Dispose();

        _disposed = true;
        GC.SuppressFinalize(this);
    }
}
