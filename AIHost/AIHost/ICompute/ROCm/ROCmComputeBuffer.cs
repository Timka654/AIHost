namespace AIHost.ICompute.ROCm;

/// <summary>
/// ROCm/HIP GPU memory buffer
/// </summary>
public unsafe class ROCmComputeBuffer : ComputeBufferBase
{
    private IntPtr _devicePtr;
    private bool _disposed;

    public override ulong Size { get; }
    public override BufferType Type { get; }
    public override DataType ElementType { get; }

    public ROCmComputeBuffer(ulong size, BufferType type, DataType dataType)
    {
        Size = size;
        Type = type;
        ElementType = dataType;

        // Allocate device memory
        HipApi.CheckError(HipApi.hipMalloc(out _devicePtr, size), "hipMalloc");
        TrackAllocate(size);
    }

    public override void Write<T>(T[] data)
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

    public override T[] ReadRange<T>(ulong byteOffset, int elementCount)
    {
        if (_disposed) throw new ObjectDisposedException(nameof(ROCmComputeBuffer));
        var result = new T[elementCount];
        var byteCount = (ulong)(sizeof(T) * elementCount);
        fixed (T* ptr = result)
        {
            HipApi.CheckError(
                HipApi.hipMemcpy((IntPtr)ptr, _devicePtr + (nint)byteOffset, byteCount,
                                 HipApi.HipMemcpyKind.DeviceToHost),
                "hipMemcpy ReadRange");
        }
        return result;
    }

    public override T[] Read<T>()
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

    public override IntPtr GetPointer()
    {
        return _devicePtr;
    }

    public override void Dispose()
    {
        if (_disposed) return;

        if (_devicePtr != IntPtr.Zero)
        {
            HipApi.hipFree(_devicePtr); // Don't throw in Dispose
            _devicePtr = IntPtr.Zero;
        }

        TrackFree(Size);
        _disposed = true;
    }
}
