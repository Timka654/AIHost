using AIHost.Compute;
using AIHost.ICompute;

namespace AIHost.Inference;

/// <summary>
/// Memory optimization strategies for inference
/// </summary>
public enum KVCacheQuantization
{
    None,   // No quantization (FP32)
    Int8,   // 8-bit quantization
    Int4    // 4-bit quantization (experimental)
}

/// <summary>
/// Quantized KV cache for memory-efficient inference
/// Stores keys and values in quantized format (8-bit or 4-bit)
/// </summary>
public class QuantizedKVCache : IDisposable
{
    private readonly IComputeDevice _device;
    private readonly ComputeOps _ops;
    private readonly KVCacheQuantization _quantization;
    private readonly List<CacheEntry> _cache = new();
    private bool _disposed;

    public int SequenceLength { get; private set; }
    public KVCacheQuantization Quantization => _quantization;

    private struct CacheEntry
    {
        public Tensor Key;
        public Tensor Value;
        public float KeyScale;   // For dequantization
        public float ValueScale;
    }

    public QuantizedKVCache(ComputeOps ops, KVCacheQuantization quantization = KVCacheQuantization.None)
    {
        _device = ops.Device;
        _ops = ops;
        _quantization = quantization;
    }

    public void Add(int layer, Tensor key, Tensor value)
    {
        Tensor quantizedKey, quantizedValue;
        float keyScale = 1.0f, valueScale = 1.0f;

        if (_quantization == KVCacheQuantization.Int8)
        {
            // Quantize to INT8
            (quantizedKey, keyScale) = QuantizeToInt8(key);
            (quantizedValue, valueScale) = QuantizeToInt8(value);
        }
        else if (_quantization == KVCacheQuantization.Int4)
        {
            // Quantize to INT4 (pack 2 values per byte)
            (quantizedKey, keyScale) = QuantizeToInt4(key);
            (quantizedValue, valueScale) = QuantizeToInt4(value);
        }
        else
        {
            // No quantization
            quantizedKey = key;
            quantizedValue = value;
        }

        if (layer >= _cache.Count)
        {
            // First time for this layer
            _cache.Add(new CacheEntry
            {
                Key = quantizedKey,
                Value = quantizedValue,
                KeyScale = keyScale,
                ValueScale = valueScale
            });
            SequenceLength = key.Shape.Dimensions[0];
        }
        else
        {
            // Concatenate with existing cache
            var entry = _cache[layer];
            
            // Dequantize, concatenate, re-quantize
            var oldKey = DequantizeTensor(entry.Key, entry.KeyScale);
            var oldValue = DequantizeTensor(entry.Value, entry.ValueScale);
            
            var newKey = _ops.Concat(oldKey, key, axis: 0, $"kv_cache_k_layer{layer}");
            var newValue = _ops.Concat(oldValue, value, axis: 0, $"kv_cache_v_layer{layer}");
            
            oldKey.Dispose();
            oldValue.Dispose();
            entry.Key.Dispose();
            entry.Value.Dispose();

            if (_quantization != KVCacheQuantization.None)
            {
                Tensor reqKey, reqValue;
                if (_quantization == KVCacheQuantization.Int8)
                {
                    (reqKey, keyScale) = QuantizeToInt8(newKey);
                    (reqValue, valueScale) = QuantizeToInt8(newValue);
                }
                else
                {
                    (reqKey, keyScale) = QuantizeToInt4(newKey);
                    (reqValue, valueScale) = QuantizeToInt4(newValue);
                }
                
                newKey.Dispose();
                newValue.Dispose();
                quantizedKey = reqKey;
                quantizedValue = reqValue;
            }
            else
            {
                quantizedKey = newKey;
                quantizedValue = newValue;
            }

            _cache[layer] = new CacheEntry
            {
                Key = quantizedKey,
                Value = quantizedValue,
                KeyScale = keyScale,
                ValueScale = valueScale
            };
            SequenceLength = quantizedKey.Shape.Dimensions[0];
        }
    }

    public (Tensor? key, Tensor? value) Get(int layer)
    {
        if (layer >= _cache.Count)
            return (null, null);

        var entry = _cache[layer];
        
        // Return dequantized tensors
        if (_quantization != KVCacheQuantization.None)
        {
            var key = DequantizeTensor(entry.Key, entry.KeyScale);
            var value = DequantizeTensor(entry.Value, entry.ValueScale);
            return (key, value);
        }

        return (entry.Key, entry.Value);
    }

    public void Clear()
    {
        foreach (var entry in _cache)
        {
            entry.Key?.Dispose();
            entry.Value?.Dispose();
        }
        _cache.Clear();
        SequenceLength = 0;
    }

    private (Tensor quantized, float scale) QuantizeToInt8(Tensor tensor)
    {
        var data = tensor.ReadData();
        
        // Calculate scale: max absolute value / 127
        float maxAbs = data.Max(Math.Abs);
        float scale = maxAbs > 0 ? maxAbs / 127.0f : 1.0f;

        // Quantize: round(value / scale)
        byte[] quantized = new byte[data.Length];
        for (int i = 0; i < data.Length; i++)
        {
            int qValue = (int)MathF.Round(data[i] / scale);
            quantized[i] = (byte)(qValue + 128); // Shift to 0-255 range
        }

        // Note: In production, you'd create a specialized INT8 tensor type
        // For now, we'll store as bytes in a FP32 tensor (inefficient but functional)
        float[] quantizedFloat = quantized.Select(b => (float)b).ToArray();
        var quantizedTensor = Tensor.FromData(_device, quantizedFloat, tensor.Shape, tensor.Name + "_q8");
        
        return (quantizedTensor, scale);
    }

    private (Tensor quantized, float scale) QuantizeToInt4(Tensor tensor)
    {
        var data = tensor.ReadData();
        
        // Calculate scale: max absolute value / 7 (4-bit signed: -8 to 7)
        float maxAbs = data.Max(Math.Abs);
        float scale = maxAbs > 0 ? maxAbs / 7.0f : 1.0f;

        // Quantize and pack 2 values per byte
        int packedLength = (data.Length + 1) / 2;
        byte[] packed = new byte[packedLength];
        
        for (int i = 0; i < data.Length; i += 2)
        {
            int q1 = (int)MathF.Round(data[i] / scale) + 8;     // 0-15
            int q2 = i + 1 < data.Length 
                ? (int)MathF.Round(data[i + 1] / scale) + 8 
                : 0;
            
            q1 = Math.Clamp(q1, 0, 15);
            q2 = Math.Clamp(q2, 0, 15);
            
            packed[i / 2] = (byte)((q1 << 4) | q2);
        }

        // Store packed data as FP32 tensor (inefficient but functional)
        float[] packedFloat = packed.Select(b => (float)b).ToArray();
        var quantizedTensor = Tensor.FromData(_device, packedFloat, 
            new TensorShape(new[] { packedLength }), tensor.Name + "_q4");
        
        return (quantizedTensor, scale);
    }

    private Tensor DequantizeTensor(Tensor quantizedTensor, float scale)
    {
        if (_quantization == KVCacheQuantization.None)
            return quantizedTensor;

        var quantizedData = quantizedTensor.ReadData();

        if (_quantization == KVCacheQuantization.Int8)
        {
            // Dequantize INT8
            float[] dequantized = new float[quantizedData.Length];
            for (int i = 0; i < quantizedData.Length; i++)
            {
                int qValue = (int)quantizedData[i] - 128;
                dequantized[i] = qValue * scale;
            }
            
            return Tensor.FromData(_device, dequantized, 
                quantizedTensor.Shape, quantizedTensor.Name + "_dq");
        }
        else if (_quantization == KVCacheQuantization.Int4)
        {
            // Unpack and dequantize INT4
            int originalLength = quantizedTensor.Shape.Dimensions[0] * 2;
            float[] dequantized = new float[originalLength];
            
            for (int i = 0; i < quantizedData.Length; i++)
            {
                byte packed = (byte)quantizedData[i];
                int q1 = (packed >> 4) - 8;
                int q2 = (packed & 0x0F) - 8;
                
                dequantized[i * 2] = q1 * scale;
                if (i * 2 + 1 < originalLength)
                    dequantized[i * 2 + 1] = q2 * scale;
            }
            
            return Tensor.FromData(_device, dequantized,
                new TensorShape(new[] { originalLength }), quantizedTensor.Name + "_dq");
        }

        return quantizedTensor;
    }

    public void Dispose()
    {
        if (_disposed) return;
        Clear();
        _disposed = true;
    }

    /// <summary>
    /// Get memory usage statistics
    /// </summary>
    public long GetMemoryUsageBytes()
    {
        long totalBytes = 0;
        foreach (var entry in _cache)
        {
            totalBytes += entry.Key.Shape.TotalElements * sizeof(float);
            totalBytes += entry.Value.Shape.TotalElements * sizeof(float);
        }
        return totalBytes;
    }
}
