namespace AIHost.GGUF;

/// <summary>
/// Информация о тензоре в GGUF файле
/// </summary>
public class GGUFTensorInfo
{
    /// <summary>
    /// Имя тензора (например, "model.layers.0.attention.wq.weight")
    /// </summary>
    public string Name { get; set; } = string.Empty;

    /// <summary>
    /// Количество измерений
    /// </summary>
    public uint Dimensions { get; set; }

    /// <summary>
    /// Размеры тензора для каждого измерения
    /// </summary>
    public ulong[] Shape { get; set; } = Array.Empty<ulong>();

    /// <summary>
    /// Тип данных тензора
    /// </summary>
    public GGUFTensorType Type { get; set; }

    /// <summary>
    /// Смещение в файле относительно начала данных
    /// </summary>
    public ulong Offset { get; set; }

    /// <summary>
    /// Общее количество элементов в тензоре
    /// </summary>
    public ulong ElementCount
    {
        get
        {
            if (Shape.Length == 0) return 0;
            ulong count = 1;
            foreach (var dim in Shape)
                count *= dim;
            return count;
        }
    }

    /// <summary>
    /// Размер в байтах (для квантованных типов)
    /// </summary>
    public ulong SizeInBytes
    {
        get
        {
            ulong elements = ElementCount;
            return Type switch
            {
                GGUFTensorType.F32 => elements * 4,
                GGUFTensorType.F16 => elements * 2,
                GGUFTensorType.Q4_0 => (elements / 32) * 18, // 32 элемента = 16 байт данных + 2 байта scale
                GGUFTensorType.Q4_1 => (elements / 32) * 20,
                GGUFTensorType.Q5_0 => (elements / 32) * 22,
                GGUFTensorType.Q5_1 => (elements / 32) * 24,
                GGUFTensorType.Q8_0 => (elements / 32) * 34,
                GGUFTensorType.Q8_1 => (elements / 32) * 36,
                // K-quants: 256 elements per block
                GGUFTensorType.Q2_K => (elements / 256) * 82,   // 82 bytes per block
                GGUFTensorType.Q3_K => (elements / 256) * 110,  // ~110 bytes per block
                GGUFTensorType.Q4_K => (elements / 256) * 144,  // 144 bytes per block
                GGUFTensorType.Q5_K => (elements / 256) * 176,  // 176 bytes per block
                GGUFTensorType.Q6_K => (elements / 256) * 210,  // 210 bytes per block
                GGUFTensorType.I8 => elements,
                GGUFTensorType.I16 => elements * 2,
                GGUFTensorType.I32 => elements * 4,
                _ => elements * 4 // По умолчанию как F32
            };
        }
    }

    public override string ToString()
    {
        var shape = string.Join(" × ", Shape);
        return $"{Name}: {Type} [{shape}] = {ElementCount:N0} elements ({SizeInBytes / 1024.0 / 1024.0:F2} MB)";
    }
}
