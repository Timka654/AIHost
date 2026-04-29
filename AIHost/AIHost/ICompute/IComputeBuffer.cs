namespace AIHost.ICompute;

/// <summary>
/// Абстракция над памятью (VRAM). Скрывает, что это — CUdeviceptr или VkBuffer.
/// </summary>
public interface IComputeBuffer : IDisposable
{
    /// <summary>
    /// Размер буфера в байтах
    /// </summary>
    ulong Size { get; }

    /// <summary>
    /// Тип буфера
    /// </summary>
    BufferType Type { get; }

    /// <summary>
    /// Тип данных элементов
    /// </summary>
    DataType ElementType { get; }

    /// <summary>
    /// Получить указатель на буфер (зависит от провайдера)
    /// </summary>
    IntPtr GetPointer();

    /// <summary>
    /// Записать данные в буфер
    /// </summary>
    void Write<T>(T[] data) where T : unmanaged;

    /// <summary>
    /// Прочитать данные из буфера
    /// </summary>
    T[] Read<T>() where T : unmanaged;

    /// <summary>
    /// Read a contiguous slice without transferring the full buffer.
    /// </summary>
    T[] ReadRange<T>(ulong byteOffset, int elementCount) where T : unmanaged;
}

public enum BufferType
{
    Uniform,
    Storage,
    Vertex,
    Index,
    Texture,
    Other
}

/// <summary>
/// Типы данных для AI-инференса (GGUF-совместимые)
/// </summary>
public enum DataType
{
    F32,   // float32
    F16,   // float16
    Q4_0,  // 4-bit quantized (block size 32)
    Q4_1,  // 4-bit quantized (block size 32, with min value)
    Q5_0,  // 5-bit quantized (block size 32)
    Q5_1,  // 5-bit quantized (block size 32, with min value)
    Q8_0,  // 8-bit quantized (block size 32)
    Q8_1,  // 8-bit quantized (block size 32, with min value)
    Q2_K,  // 2-bit quantized (K-quant, block size 256)
    Q3_K,  // 3-bit quantized (K-quant, block size 256)
    Q4_K,  // 4-bit quantized (K-quant, block size 256)
    Q5_K,  // 5-bit quantized (K-quant, block size 256)
    Q6_K,  // 6-bit quantized (K-quant, block size 256)
    Q8_K,  // 8-bit quantized (K-quant, block size 256)
    I8,    // int8
    I16,   // int16
    I32,   // int32
}