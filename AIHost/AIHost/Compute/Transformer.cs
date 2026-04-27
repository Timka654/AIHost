using AIHost.GGUF;
using AIHost.ICompute;

namespace AIHost.Compute;

/// <summary>
/// LLM Transformer model for inference
/// </summary>
public class Transformer : IDisposable
{
    private readonly IComputeDevice _device;
    private readonly ComputeOps _ops;
    private readonly GGUFModel _model;
    private readonly int _numLayers;
    private readonly int _dModel;
    private readonly int _numHeads;
    private readonly int _vocabSize;
    private bool _disposed;

    // Model weights
    private Tensor? _tokenEmbedding;
    private Tensor? _outputNormWeight;
    private Tensor? _outputWeight;
    
    // Weight cache to avoid repeated dequantization
    private readonly Dictionary<string, Tensor> _weightCache = new();

    public Transformer(IComputeDevice device, GGUFModel model)
    {
        _device = device;
        _ops = new ComputeOps(device);
        _model = model;

        var metadata = model.Metadata;
        _numLayers = metadata.GetValue<int>(GGUFMetadata.KeyBlockCount, 22);
        _dModel = metadata.GetValue<int>(GGUFMetadata.KeyEmbeddingLength, 2048);
        _numHeads = metadata.GetValue<int>(GGUFMetadata.KeyAttentionHeadCount, 32);
        _vocabSize = metadata.GetValue<int>(GGUFMetadata.KeyVocabSize, 32000);

        Console.WriteLine($"Transformer initialized:");
        Console.WriteLine($"  Layers: {_numLayers}");
        Console.WriteLine($"  d_model: {_dModel}");
        Console.WriteLine($"  Num heads: {_numHeads}");
        Console.WriteLine($"  Vocab size: {_vocabSize}\n");
    }

    /// <summary>
    /// Load model weights from GGUF
    /// </summary>
    public void LoadWeights()
    {
        Console.WriteLine("Loading model weights...");

        // Token embedding
        _tokenEmbedding = LoadWeight("token_embd.weight");
        Console.WriteLine($"  token_embd.weight: {_tokenEmbedding.Shape}");

        // Output layers
        _outputNormWeight = LoadWeight("output_norm.weight");
        _outputWeight = LoadWeight("output.weight");
        Console.WriteLine($"  output_norm.weight: {_outputNormWeight.Shape}");
        Console.WriteLine($"  output.weight: {_outputWeight.Shape}");

        Console.WriteLine($"✓ Weights loaded\n");
    }

    /// <summary>
    /// Forward pass through transformer
    /// </summary>
    public Tensor Forward(int[] tokenIds, uint startPosition = 0, Inference.KVCache? kvCache = null)
    {
        if (_tokenEmbedding == null)
            throw new InvalidOperationException("Weights not loaded. Call LoadWeights() first.");

        int seqLen = tokenIds.Length;
        Console.WriteLine($"Forward pass: {seqLen} tokens starting at position {startPosition}");

        // 1. Token embedding lookup
        var x = EmbeddingLookup(_tokenEmbedding, tokenIds);
        Console.WriteLine($"  Embedding: {x.Shape}");

        // 2. Apply all transformer layers
        for (int i = 0; i < _numLayers; i++)
        {
            if (i % 5 == 0 || i == _numLayers - 1)
            {
                Console.WriteLine($"  Applying layer {i}/{_numLayers}...");
            }
            x = ApplyLayer(x, i, startPosition, kvCache);
            
            // Sync GPU after each layer to prevent command buffer overflow
            _device.Synchronize();
        }

        // 3. Final layer norm
        _ops.LayerNorm(x, _outputNormWeight!);
        Console.WriteLine($"  Final norm: {x.Shape}");

        // 4. Project to vocab
        var logits = _ops.MatMul(x, _outputWeight!, "logits");
        x.Dispose();
        Console.WriteLine($"  Logits: {logits.Shape}");

        return logits;
    }

    private Tensor ApplyLayer(Tensor x, int layerIdx, uint position, Inference.KVCache? kvCache = null)
    {
        // Load layer weights
        string prefix = $"blk.{layerIdx}";
        var attnNorm = LoadWeight($"{prefix}.attn_norm.weight");
        var wQ = LoadWeight($"{prefix}.attn_q.weight");
        var wK = LoadWeight($"{prefix}.attn_k.weight");
        var wV = LoadWeight($"{prefix}.attn_v.weight");
        var wAttnOut = LoadWeight($"{prefix}.attn_output.weight");
        var ffnNorm = LoadWeight($"{prefix}.ffn_norm.weight");
        var wGate = LoadWeight($"{prefix}.ffn_gate.weight");
        var wUp = LoadWeight($"{prefix}.ffn_up.weight");
        var wDown = LoadWeight($"{prefix}.ffn_down.weight");

        // Apply layer
        var output = _ops.TransformerLayer(
            x, attnNorm, wQ, wK, wV, wAttnOut,
            ffnNorm, wGate, wUp, wDown,
            _numHeads, position, kvCache, layerIdx);

        // Don't dispose weights - they're cached!
        x.Dispose();

        return output;
    }

    private Tensor LoadWeight(string name)
    {
        // Check cache first
        if (_weightCache.TryGetValue(name, out var cached))
            return cached;
        
        // Find tensor info
        var tensorInfo = _model.Tensors.FirstOrDefault(t => t.Name == name);
        if (tensorInfo == null)
            throw new ArgumentException($"Tensor '{name}' not found");

        // Load buffer
        var buffer = _model.LoadTensor(name);
        
        // Get data type and shape
        var dataType = MapTensorTypeToDataType(tensorInfo.Type);
        
        // Use GGUF dimensions as-is - they are already in correct layout for MatMul
        // GGUF stores weight matrices as [in_features, out_features] (transposed PyTorch format)
        var dims = tensorInfo.Shape.Select(s => (int)s).ToArray();
        var shape = new TensorShape(dims);
        
        // Create tensor wrapper
        var tensor = new Tensor(buffer, shape, dataType, name);
        
        // Dequantize if needed
        if (dataType != DataType.F32)
        {
            var dequantized = _ops.Dequantize(tensor, $"{name}_f32");
            tensor.Dispose();
            tensor = dequantized;
        }

        // Cache the dequantized weight
        _weightCache[name] = tensor;
        
        return tensor;
    }

    private DataType MapTensorTypeToDataType(GGUFTensorType type)
    {
        return type switch
        {
            GGUFTensorType.F32 => DataType.F32,
            GGUFTensorType.F16 => DataType.F16,
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
            _ => throw new NotSupportedException($"Unsupported tensor type: {type}")
        };
    }

    private Tensor EmbeddingLookup(Tensor embeddingTable, int[] tokenIds)
    {
        return _ops.EmbeddingLookup(tokenIds, embeddingTable, "embeddings");
    }

    public void Dispose()
    {
        if (_disposed) return;

        _tokenEmbedding?.Dispose();
        _outputNormWeight?.Dispose();
        _outputWeight?.Dispose();
        _ops.Dispose();

        _disposed = true;
    }
}
