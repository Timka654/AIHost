using AIHost.ICompute;

namespace AIHost.GGUF;

/// <summary>
/// GGUF модель с удобными методами загрузки тензоров
/// </summary>
public class GGUFModel : IDisposable
{
    private readonly GGUFReader _reader;
    private readonly IComputeDevice _device;
    private readonly Dictionary<string, IComputeBuffer> _tensorBuffers = new();
    private bool _disposed;

    public GGUFHeader Header => _reader.Header;
    public GGUFMetadata Metadata => _reader.Metadata;
    public IReadOnlyList<GGUFTensorInfo> Tensors => _reader.Tensors;

    public GGUFModel(string filePath, IComputeDevice device)
    {
        _reader = new GGUFReader(filePath);
        _device = device;
        _reader.Load();
    }

    /// <summary>
    /// Загрузить тензор в GPU память
    /// </summary>
    public unsafe IComputeBuffer LoadTensor(string tensorName)
    {
        // Проверяем кэш
        if (_tensorBuffers.TryGetValue(tensorName, out var cachedBuffer))
            return cachedBuffer;

        // Находим тензор
        var tensor = Tensors.FirstOrDefault(t => t.Name == tensorName);
        if (tensor == null)
            throw new ArgumentException($"Tensor '{tensorName}' not found in model");

        // Читаем данные
        byte[] data = _reader.ReadTensorData(tensor);

        // Создаём буфер
        var dataType = MapTensorTypeToDataType(tensor.Type);
        var buffer = _device.CreateBuffer(tensor.SizeInBytes, BufferType.Storage, dataType);

        // Загружаем данные (пока как byte[], позже добавим typed arrays)
        unsafe
        {
            fixed (byte* src = data)
            {
                var dest = buffer.GetPointer();
                System.Buffer.MemoryCopy(src, dest.ToPointer(), (long)buffer.Size, data.Length);
            }
        }

        // Кэшируем
        _tensorBuffers[tensorName] = buffer;

        Console.WriteLine($"Loaded tensor: {tensor}");
        return buffer;
    }

    /// <summary>
    /// Загрузить несколько тензоров по паттерну (например, "model.layers.0.*")
    /// </summary>
    public Dictionary<string, IComputeBuffer> LoadTensors(Func<GGUFTensorInfo, bool> predicate)
    {
        var result = new Dictionary<string, IComputeBuffer>();
        foreach (var tensor in Tensors.Where(predicate))
        {
            result[tensor.Name] = LoadTensor(tensor.Name);
        }
        return result;
    }

    /// <summary>
    /// Получить информацию о модели
    /// </summary>
    public ModelInfo GetModelInfo()
    {
        return new ModelInfo
        {
            Name = Metadata.GetValue<string>(GGUFMetadata.KeyName) ?? "Unknown",
            Architecture = Metadata.GetValue<string>(GGUFMetadata.KeyArchitecture) ?? "Unknown",
            ContextLength = Metadata.GetValue<uint>(GGUFMetadata.KeyContextLength),
            EmbeddingLength = Metadata.GetValue<uint>(GGUFMetadata.KeyEmbeddingLength),
            BlockCount = Metadata.GetValue<uint>(GGUFMetadata.KeyBlockCount),
            AttentionHeadCount = Metadata.GetValue<uint>(GGUFMetadata.KeyAttentionHeadCount),
            VocabSize = Metadata.GetValue<uint>(GGUFMetadata.KeyVocabSize),
            TensorCount = (uint)Tensors.Count,
            TotalSizeMB = Tensors.Sum(t => (long)t.SizeInBytes) / 1024.0 / 1024.0
        };
    }

    private static DataType MapTensorTypeToDataType(GGUFTensorType tensorType)
    {
        return tensorType switch
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
            GGUFTensorType.I8 => DataType.I8,
            GGUFTensorType.I16 => DataType.I16,
            GGUFTensorType.I32 => DataType.I32,
            _ => DataType.F32
        };
    }

    public void Dispose()
    {
        if (_disposed) return;

        foreach (var buffer in _tensorBuffers.Values)
            buffer.Dispose();

        _tensorBuffers.Clear();
        _reader.Dispose();
        _disposed = true;
    }
}

/// <summary>
/// Информация о модели
/// </summary>
public class ModelInfo
{
    public string Name { get; set; } = string.Empty;
    public string Architecture { get; set; } = string.Empty;
    public uint? ContextLength { get; set; }
    public uint? EmbeddingLength { get; set; }
    public uint? BlockCount { get; set; }
    public uint? AttentionHeadCount { get; set; }
    public uint? VocabSize { get; set; }
    public uint TensorCount { get; set; }
    public double TotalSizeMB { get; set; }

    public override string ToString()
    {
        return $"{Name} ({Architecture})\n" +
               $"  Context: {ContextLength}, Embedding: {EmbeddingLength}\n" +
               $"  Layers: {BlockCount}, Heads: {AttentionHeadCount}\n" +
               $"  Vocab: {VocabSize}, Tensors: {TensorCount}\n" +
               $"  Size: {TotalSizeMB:F2} MB";
    }
}
