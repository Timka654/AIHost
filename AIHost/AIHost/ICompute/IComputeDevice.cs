namespace AIHost.ICompute;

/// <summary>
/// Фабрика, которая создает буферы и ядра, а также управляет очередями команд.
/// </summary>
public interface IComputeDevice : IDisposable
{
    /// <summary>
    /// Имя провайдера (Vulkan, ROCm, CUDA и т.д.)
    /// </summary>
    string ProviderName { get; }

    /// <summary>
    /// Версия API провайдера
    /// </summary>
    string ApiVersion { get; }

    /// <summary>
    /// Создать буфер памяти
    /// </summary>
    /// <param name="requireDeviceLocal">
    /// When true, throws <see cref="InsufficientVramException"/> if the buffer cannot be placed
    /// in dedicated device memory (VRAM). Use false for buffers that are acceptable in shared/system RAM.
    /// </param>
    IComputeBuffer CreateBuffer(ulong size, BufferType type, DataType elementType = DataType.F32, bool requireDeviceLocal = false);

    /// <summary>
    /// Создать ядро из исходного кода
    /// </summary>
    IComputeKernel CreateKernel(string source, string entryPoint);

    /// <summary>
    /// Создать ядро из файла
    /// </summary>
    IComputeKernel CreateKernelFromFile(string filePath, string entryPoint);

    /// <summary>
    /// Создать очередь команд
    /// </summary>
    IComputeCommandQueue CreateCommandQueue();

    /// <summary>
    /// Синхронизировать очередь команд
    /// </summary>
    void Synchronize();
}
