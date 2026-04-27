namespace AIHost.GGUF;

/// <summary>
/// Заголовок GGUF файла
/// </summary>
public class GGUFHeader
{
    /// <summary>
    /// Magic number (должен быть "GGUF")
    /// </summary>
    public uint Magic { get; set; }

    /// <summary>
    /// Версия формата GGUF
    /// </summary>
    public uint Version { get; set; }

    /// <summary>
    /// Количество тензоров
    /// </summary>
    public ulong TensorCount { get; set; }

    /// <summary>
    /// Количество полей метаданных
    /// </summary>
    public ulong MetadataCount { get; set; }

    public const uint GGUF_MAGIC = 0x46554747; // "GGUF" в little-endian
    public const uint GGUF_VERSION = 3;

    public bool IsValid()
    {
        return Magic == GGUF_MAGIC && Version == GGUF_VERSION;
    }

    public override string ToString()
    {
        return $"GGUF v{Version}: {TensorCount} tensors, {MetadataCount} metadata entries";
    }
}
