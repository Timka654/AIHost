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
    /// Прочитать данные из тензора с авто-конвертацией F16→F32
    /// </summary>
    public float[] ReadF32()
    {
        if (DataType == DataType.F32)
            return Buffer.Read<float>();

        if (DataType == DataType.F16)
        {
            var raw = Buffer.Read<ushort>();
            var result = new float[raw.Length];
            for (int i = 0; i < raw.Length; i++)
                result[i] = HalfToFloat(raw[i]);
            return result;
        }

        throw new InvalidOperationException($"ReadF32: unsupported dtype {DataType}");
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

    /// <summary>IEEE 754 binary16 → float conversion (no intrinsics).</summary>
    private static float HalfToFloat(ushort h)
    {
        uint sign = (uint)((h >> 15) & 1);
        uint exp = (uint)((h >> 10) & 0x1F);
        uint mant = (uint)(h & 0x3FF);

        if (exp == 0)
        {
            // Subnormal / zero
            if (mant == 0) return sign == 0 ? 0f : -0f;
            // Normalize subnormal
            int shift = 10;
            while ((mant & 0x400) == 0) { mant <<= 1; shift--; }
            exp = (uint)(1 - shift + 127);
            mant = (mant & 0x3FF) << 13;
        }
        else if (exp == 0x1F)
        {
            // Inf / NaN
            exp = 0xFF;
            mant <<= 13;
        }
        else
        {
            exp = exp + 127 - 15;
            mant <<= 13;
        }

        uint f32 = (sign << 31) | (exp << 23) | mant;
        return BitConverter.ToSingle(BitConverter.GetBytes(f32), 0);
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
