namespace AIHost.ICompute.CUDA;

/// <summary>
/// CUDA command queue (stream) implementation
/// </summary>
public unsafe class CudaComputeCommandQueue : IComputeCommandQueue
{
    private IntPtr _stream;
    private bool _disposed;

    public CudaComputeCommandQueue()
    {
        var error = CudaApi.StreamCreate(out _stream);
        CudaApi.CheckError(error, "cudaStreamCreate");
    }

    public void Synchronize()
    {
        if (_disposed)
            throw new ObjectDisposedException(nameof(CudaComputeCommandQueue));

        var error = CudaApi.StreamSynchronize(_stream);
        CudaApi.CheckError(error, "cudaStreamSynchronize");
    }

    internal IntPtr GetStream() => _stream;

    public void WriteBuffer(IComputeBuffer buffer, ulong offset, byte[] data)
    {
        if (_disposed)
            throw new ObjectDisposedException(nameof(CudaComputeCommandQueue));

        var cudaBuffer = (CudaComputeBuffer)buffer;
        fixed (byte* ptr = data)
        {
            var dstPtr = IntPtr.Add(cudaBuffer.GetPointer(), (int)offset);
            var error = CudaApi.Memcpy(dstPtr, (IntPtr)ptr, (ulong)data.Length, CudaMemcpyKind.HostToDevice);
            CudaApi.CheckError(error, "WriteBuffer");
        }
    }

    public void ReadBuffer(IComputeBuffer buffer, ulong offset, byte[] data)
    {
        if (_disposed)
            throw new ObjectDisposedException(nameof(CudaComputeCommandQueue));

        var cudaBuffer = (CudaComputeBuffer)buffer;
        fixed (byte* ptr = data)
        {
            var srcPtr = IntPtr.Add(cudaBuffer.GetPointer(), (int)offset);
            var error = CudaApi.Memcpy((IntPtr)ptr, srcPtr, (ulong)data.Length, CudaMemcpyKind.DeviceToHost);
            CudaApi.CheckError(error, "ReadBuffer");
        }
    }

    public void Dispatch(IComputeKernel kernel, uint[] globalWorkSize, uint[]? localWorkSize = null)
    {
        // Delegate to kernel's Execute method
        // This is a simplified implementation
        throw new NotSupportedException("Use CudaComputeKernel.Execute() directly");
    }

    public void Flush()
    {
        Synchronize();
    }

    public void Dispose()
    {
        if (_disposed) return;

        if (_stream != IntPtr.Zero)
        {
            CudaApi.StreamDestroy(_stream);
            _stream = IntPtr.Zero;
        }

        _disposed = true;
    }
}
