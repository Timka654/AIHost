using AIHost.ICompute;

namespace AIHost.Compute;

/// <summary>
/// Тензор в GPU памяти с метаданными
/// </summary>
public class Tensor : IDisposable
{
    public IComputeBuffer Buffer { get; }
    public TensorShape Shape { get; }
    public DataType DataType { get; }
    public string? Name { get; set; }

    private bool _disposed;

    public Tensor(IComputeBuffer buffer, TensorShape shape, DataType dataType, string? name = null)
    {
        Buffer = buffer;
        Shape = shape;
        DataType = dataType;
        Name = name;
    }

    /// <summary>
    /// Создать пустой тензор заданной формы
    /// </summary>
    public static Tensor Create(IComputeDevice device, TensorShape shape, DataType dataType, string? name = null)
    {
        ulong sizeInBytes = (ulong)shape.TotalElements * GetDataTypeSize(dataType);
        var buffer = device.CreateBuffer(sizeInBytes, BufferType.Storage, dataType);
        return new Tensor(buffer, shape, dataType, name);
    }

    /// <summary>
    /// Создать тензор из данных
    /// </summary>
    public static Tensor FromData(IComputeDevice device, float[] data, TensorShape shape, string? name = null)
    {
        var tensor = Create(device, shape, DataType.F32, name);
        tensor.Buffer.Write(data);
        return tensor;
    }

    /// <summary>
    /// Прочитать данные из тензора
    /// </summary>
    public float[] ReadData()
    {
        if (DataType != DataType.F32)
            throw new InvalidOperationException("ReadData supports only F32 tensors");

        return Buffer.Read<float>();
    }

    /// <summary>
    /// Read a single row from a 2-D F32 tensor without transferring the full buffer.
    /// </summary>
    public float[] ReadRow(int rowIndex)
    {
        if (DataType != DataType.F32)
            throw new InvalidOperationException("ReadRow supports only F32 tensors");
        if (Shape.Rank != 2)
            throw new InvalidOperationException("ReadRow requires a 2-D tensor");

        int cols = Shape.Dimensions[1];
        ulong byteOffset = (ulong)(rowIndex * cols * sizeof(float));
        return Buffer.ReadRange<float>(byteOffset, cols);
    }

    public void Dispose()
    {
        if (_disposed) return;
        Buffer.Dispose();
        _disposed = true;
    }

    public override string ToString()
    {
        return Name != null 
            ? $"{Name}: {DataType} {Shape}" 
            : $"{DataType} {Shape}";
    }

    private static ulong GetDataTypeSize(DataType dataType)
    {
        return dataType switch
        {
            DataType.F32 => 4,
            DataType.F16 => 2,
            DataType.I32 => 4,
            DataType.I16 => 2,
            DataType.I8 => 1,
            _ => throw new ArgumentException($"Unknown data type size: {dataType}")
        };
    }
}
