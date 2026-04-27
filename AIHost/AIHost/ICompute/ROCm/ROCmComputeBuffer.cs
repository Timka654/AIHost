namespace AIHost.ICompute.ROCm;

/// <summary>
/// ROCm/HIP GPU memory buffer
/// TODO: Implement with hipMalloc/hipMemcpy
/// </summary>
public unsafe class ROCmComputeBuffer : IComputeBuffer
{
    private void* _devicePtr;
    private bool _disposed;

    public ulong Size { get; }
    public BufferType Type { get; }
    public DataType DataType { get; }
    public DataType ElementType => DataType;

    public ROCmComputeBuffer(ulong size, BufferType type, DataType dataType)
    {
        Size = size;
        Type = type;
        DataType = dataType;

        // TODO: Allocate device memory
        // hipError_t err = hipMalloc(&_devicePtr, size);
        // if (err != hipSuccess) throw new Exception($"hipMalloc failed: {err}");
        
        throw new NotImplementedException("ROCm buffer not implemented");
    }

    public void Write<T>(T[] data) where T : unmanaged
    {
        // TODO: Copy host to device
        // hipMemcpy(_devicePtr, data, sizeof(T) * data.Length, hipMemcpyHostToDevice);
        throw new NotImplementedException();
    }

    public T[] Read<T>() where T : unmanaged
    {
        // TODO: Copy device to host
        // T[] result = new T[Size / sizeof(T)];
        // hipMemcpy(result, _devicePtr, Size, hipMemcpyDeviceToHost);
        // return result;
        throw new NotImplementedException();
    }

    public nint GetPointer()
    {
        return (nint)_devicePtr;
    }

    public void Dispose()
    {
        if (_disposed) return;
        
        // TODO: Free device memory
        // if (_devicePtr != null)
        //     hipFree(_devicePtr);
        
        _disposed = true;
    }
}
