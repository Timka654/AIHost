namespace AIHost.ICompute;

/// <summary>
/// Базовый класс для реализации провайдера вычислений
/// </summary>
public abstract class ComputeProviderBase : IComputeDevice
{
    public abstract string ProviderName { get; }
    public abstract string ApiVersion { get; }

    public abstract IComputeBuffer CreateBuffer(ulong size, BufferType type, DataType elementType = DataType.F32, bool requireDeviceLocal = false);
    public abstract IComputeKernel CreateKernel(string source, string entryPoint);
    public abstract IComputeKernel CreateKernelFromFile(string filePath, string entryPoint);
    public abstract IComputeCommandQueue CreateCommandQueue();
    public abstract void Synchronize();
    public abstract void Dispose();
}

/// <summary>
/// Базовый класс для реализации буфера
/// </summary>
public abstract class ComputeBufferBase : IComputeBuffer
{
    public abstract ulong Size { get; }
    public abstract BufferType Type { get; }
    public abstract DataType ElementType { get; }
    public abstract IntPtr GetPointer();
    public abstract void Write<T>(T[] data) where T : unmanaged;
    public abstract T[] Read<T>() where T : unmanaged;
    public abstract void Dispose();
}

/// <summary>
/// Базовый класс для реализации ядра
/// </summary>
public abstract class ComputeKernelBase : IComputeKernel
{
    public abstract string Name { get; }
    public virtual KernelArgumentType[] ArgumentTypes { get; protected set; } = Array.Empty<KernelArgumentType>();
    public abstract void SetArgument(int index, object value);
    public abstract void Dispatch(uint[] globalWorkSize, uint[]? localWorkSize = null);
    public abstract void Compile();
    public abstract void Dispose();
}

/// <summary>
/// Базовый класс для реализации очереди команд
/// </summary>
public abstract class ComputeCommandQueueBase : IComputeCommandQueue
{
    public abstract void WriteBuffer(IComputeBuffer buffer, ulong offset, byte[] data);
    public abstract void ReadBuffer(IComputeBuffer buffer, ulong offset, byte[] data);
    public abstract void Dispatch(IComputeKernel kernel, uint[] globalWorkSize, uint[]? localWorkSize = null);
    public abstract void Flush();
    public virtual void InsertMemoryBarrier() { } // no-op for non-Vulkan backends
    public abstract void Dispose();
}
