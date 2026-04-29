namespace AIHost.ICompute.ROCm;

/// <summary>
/// ROCm/HIP GPU memory buffer
/// </summary>
public unsafe class ROCmComputeBuffer : IComputeBuffer
{
    private IntPtr _devicePtr;
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

        // Allocate device memory
        HipApi.CheckError(HipApi.hipMalloc(out _devicePtr, size), "hipMalloc");
    }

    public void Write<T>(T[] data) where T : unmanaged
    {
        if (_disposed)
            throw new ObjectDisposedException(nameof(ROCmComputeBuffer));

        var size = (ulong)(sizeof(T) * data.Length);
        if (size > Size)
            throw new ArgumentException($"Data size {size} exceeds buffer size {Size}");

        fixed (T* ptr = data)
        {
            HipApi.CheckError(
                HipApi.hipMemcpy(_devicePtr, (IntPtr)ptr, size, HipApi.HipMemcpyKind.HostToDevice),
                "hipMemcpy HostToDevice");
        }
    }

    public T[] ReadRange<T>(ulong byteOffset, int elementCount) where T : unmanaged
    {
        int start = (int)(byteOffset / (ulong)sizeof(T));
        return Read<T>().Skip(start).Take(elementCount).ToArray();
    }

    public T[] Read<T>() where T : unmanaged
    {
        if (_disposed)
            throw new ObjectDisposedException(nameof(ROCmComputeBuffer));

        var count = (int)(Size / (ulong)sizeof(T));
        var result = new T[count];

        fixed (T* ptr = result)
        {
            HipApi.CheckError(
                HipApi.hipMemcpy((IntPtr)ptr, _devicePtr, Size, HipApi.HipMemcpyKind.DeviceToHost),
                "hipMemcpy DeviceToHost");
        }

        return result;
    }

    public nint GetPointer()
    {
        return _devicePtr;
    }

    public void Dispose()
    {
        if (_disposed) return;

        if (_devicePtr != IntPtr.Zero)
        {
            HipApi.hipFree(_devicePtr); // Don't throw in Dispose
            _devicePtr = IntPtr.Zero;
        }

        _disposed = true;
    }
}
