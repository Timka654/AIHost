using System.Runtime.InteropServices;

namespace AIHost.ICompute.CUDA;

/// <summary>
/// CUDA compute buffer implementation
/// </summary>
public unsafe class CudaComputeBuffer : IComputeBuffer
{
    private IntPtr _devicePtr;
    private readonly ulong _size;
    private readonly BufferType _bufferType;
    private readonly DataType _elementType;
    private bool _disposed;

    public ulong Size => _size;
    public BufferType Type => _bufferType;
    public DataType ElementType => _elementType;

    public CudaComputeBuffer(ulong size, BufferType type, DataType elementType)
    {
        _size = size;
        _bufferType = type;
        _elementType = elementType;

        var error = CudaApi.Malloc(out _devicePtr, size);
        CudaApi.CheckError(error, "cudaMalloc");
    }

    public IntPtr GetPointer()
    {
        if (_disposed)
            throw new ObjectDisposedException(nameof(CudaComputeBuffer));
        return _devicePtr;
    }

    public void CopyFromHost(IntPtr hostPtr, ulong size)
    {
        if (_disposed)
            throw new ObjectDisposedException(nameof(CudaComputeBuffer));

        var error = CudaApi.Memcpy(_devicePtr, hostPtr, size, CudaMemcpyKind.HostToDevice);
        CudaApi.CheckError(error, "cudaMemcpy (Host->Device)");
    }

    public void CopyToHost(IntPtr hostPtr, ulong size)
    {
        if (_disposed)
            throw new ObjectDisposedException(nameof(CudaComputeBuffer));

        var error = CudaApi.Memcpy(hostPtr, _devicePtr, size, CudaMemcpyKind.DeviceToHost);
        CudaApi.CheckError(error, "cudaMemcpy (Device->Host)");
    }

    public void Write<T>(T[] data) where T : unmanaged
    {
        if (_disposed)
            throw new ObjectDisposedException(nameof(CudaComputeBuffer));

        ulong size = (ulong)(data.Length * Marshal.SizeOf<T>());
        fixed (T* ptr = data)
        {
            CopyFromHost((IntPtr)ptr, size);
        }
    }

    public T[] ReadRange<T>(ulong byteOffset, int elementCount) where T : unmanaged
    {
        int elementSize = Marshal.SizeOf<T>();
        int start = (int)(byteOffset / (ulong)elementSize);
        return Read<T>().Skip(start).Take(elementCount).ToArray();
    }

    public T[] Read<T>() where T : unmanaged
    {
        if (_disposed)
            throw new ObjectDisposedException(nameof(CudaComputeBuffer));

        int elementSize = Marshal.SizeOf<T>();
        int count = (int)(_size / (ulong)elementSize);
        T[] result = new T[count];

        fixed (T* ptr = result)
        {
            CopyToHost((IntPtr)ptr, _size);
        }

        return result;
    }

    public void Dispose()
    {
        if (_disposed) return;

        if (_devicePtr != IntPtr.Zero)
        {
            CudaApi.Free(_devicePtr);
            _devicePtr = IntPtr.Zero;
        }

        _disposed = true;
    }
}
