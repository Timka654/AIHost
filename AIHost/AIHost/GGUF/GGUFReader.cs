using System.Text;

namespace AIHost.GGUF;

/// <summary>
/// Парсер GGUF файлов (GGML Universal Format)
/// </summary>
public class GGUFReader : IDisposable
{
    private readonly FileStream _fileStream;
    private readonly BinaryReader _reader;
    private bool _disposed;

    public GGUFHeader Header { get; private set; } = null!;
    public GGUFMetadata Metadata { get; private set; } = new();
    public List<GGUFTensorInfo> Tensors { get; private set; } = new();
    public ulong DataOffset { get; private set; }

    public GGUFReader(string filePath)
    {
        if (!File.Exists(filePath))
            throw new FileNotFoundException($"GGUF file not found: {filePath}");

        _fileStream = new FileStream(filePath, FileMode.Open, FileAccess.Read, FileShare.Read);
        _reader = new BinaryReader(_fileStream);
    }

    /// <summary>
    /// Загрузить и распарсить GGUF файл
    /// </summary>
    public void Load()
    {
        // Читаем заголовок
        Header = ReadHeader();
        if (!Header.IsValid())
            throw new InvalidDataException($"Invalid GGUF file: magic=0x{Header.Magic:X8}, version={Header.Version}");

        Console.WriteLine($"Loaded {Header}");

        // Читаем метаданные
        ReadMetadata();
        Console.WriteLine($"Metadata: {Metadata}");

        // Читаем информацию о тензорах
        ReadTensorInfo();
        Console.WriteLine($"Tensors: {Tensors.Count} total");

        // Вычисляем смещение данных (выравнивание по 32 байта)
        DataOffset = AlignOffset((ulong)_fileStream.Position, 32);
        Console.WriteLine($"Data starts at offset: 0x{DataOffset:X}");
    }

    private GGUFHeader ReadHeader()
    {
        return new GGUFHeader
        {
            Magic = _reader.ReadUInt32(),
            Version = _reader.ReadUInt32(),
            TensorCount = _reader.ReadUInt64(),
            MetadataCount = _reader.ReadUInt64()
        };
    }

    private void ReadMetadata()
    {
        for (ulong i = 0; i < Header.MetadataCount; i++)
        {
            string key = ReadString();
            var valueType = (GGUFValueType)_reader.ReadUInt32();
            object value = ReadValue(valueType);
            Metadata.Add(key, value);
        }
    }

    private void ReadTensorInfo()
    {
        for (ulong i = 0; i < Header.TensorCount; i++)
        {
            var tensor = new GGUFTensorInfo
            {
                Name = ReadString(),
                Dimensions = _reader.ReadUInt32()
            };

            // Читаем размеры
            tensor.Shape = new ulong[tensor.Dimensions];
            for (uint j = 0; j < tensor.Dimensions; j++)
            {
                tensor.Shape[j] = _reader.ReadUInt64();
            }

            tensor.Type = (GGUFTensorType)_reader.ReadUInt32();
            tensor.Offset = _reader.ReadUInt64();

            Tensors.Add(tensor);
        }
    }

    private string ReadString()
    {
        ulong length = _reader.ReadUInt64();
        byte[] bytes = _reader.ReadBytes((int)length);
        return Encoding.UTF8.GetString(bytes);
    }

    private object ReadValue(GGUFValueType type)
    {
        return type switch
        {
            GGUFValueType.UInt8 => _reader.ReadByte(),
            GGUFValueType.Int8 => _reader.ReadSByte(),
            GGUFValueType.UInt16 => _reader.ReadUInt16(),
            GGUFValueType.Int16 => _reader.ReadInt16(),
            GGUFValueType.UInt32 => _reader.ReadUInt32(),
            GGUFValueType.Int32 => _reader.ReadInt32(),
            GGUFValueType.Float32 => _reader.ReadSingle(),
            GGUFValueType.Bool => _reader.ReadByte() != 0,
            GGUFValueType.String => ReadString(),
            GGUFValueType.Array => ReadArray(),
            GGUFValueType.UInt64 => _reader.ReadUInt64(),
            GGUFValueType.Int64 => _reader.ReadInt64(),
            GGUFValueType.Float64 => _reader.ReadDouble(),
            _ => throw new NotSupportedException($"Unsupported value type: {type}")
        };
    }

    private object ReadArray()
    {
        var elementType = (GGUFValueType)_reader.ReadUInt32();
        ulong length = _reader.ReadUInt64();

        switch (elementType)
        {
            case GGUFValueType.String:
            {
                var arr = new string[length];
                for (ulong i = 0; i < length; i++) arr[i] = ReadString();
                return arr;
            }
            case GGUFValueType.Float32:
            {
                var arr = new float[length];
                for (ulong i = 0; i < length; i++) arr[i] = _reader.ReadSingle();
                return arr;
            }
            case GGUFValueType.Int32:
            {
                var arr = new int[length];
                for (ulong i = 0; i < length; i++) arr[i] = _reader.ReadInt32();
                return arr;
            }
            case GGUFValueType.UInt32:
            {
                var arr = new uint[length];
                for (ulong i = 0; i < length; i++) arr[i] = _reader.ReadUInt32();
                return arr;
            }
            case GGUFValueType.Int8:
            {
                var arr = new sbyte[length];
                for (ulong i = 0; i < length; i++) arr[i] = _reader.ReadSByte();
                return arr;
            }
            default:
            {
                var arr = new object[length];
                for (ulong i = 0; i < length; i++) arr[i] = ReadValue(elementType);
                return arr;
            }
        }
    }

    /// <summary>
    /// Прочитать данные тензора
    /// </summary>
    public byte[] ReadTensorData(GGUFTensorInfo tensor)
    {
        _fileStream.Seek((long)(DataOffset + tensor.Offset), SeekOrigin.Begin);
        byte[] data = new byte[tensor.SizeInBytes];
        
        int bytesRead = 0;
        int totalToRead = data.Length;
        while (bytesRead < totalToRead)
        {
            int read = _fileStream.Read(data, bytesRead, totalToRead - bytesRead);
            if (read == 0)
                throw new EndOfStreamException($"Unexpected end of stream while reading tensor '{tensor.Name}'");
            bytesRead += read;
        }
        
        return data;
    }

    private static ulong AlignOffset(ulong offset, ulong alignment)
    {
        return (offset + alignment - 1) & ~(alignment - 1);
    }

    public void Dispose()
    {
        if (_disposed) return;
        _reader?.Dispose();
        _fileStream?.Dispose();
        _disposed = true;
    }
}
