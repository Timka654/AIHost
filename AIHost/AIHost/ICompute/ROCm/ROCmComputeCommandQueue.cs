namespace AIHost.ICompute.ROCm;

/// <summary>
/// ROCm/HIP command queue using HIP streams
/// </summary>
public unsafe class ROCmComputeCommandQueue : ComputeCommandQueueBase
{
    private IntPtr _stream;
    private bool _disposed;

    public ROCmComputeCommandQueue()
    {
        // Create HIP stream
        HipApi.CheckError(HipApi.hipStreamCreate(out _stream), "hipStreamCreate");
    }

    public override void Dispatch(IComputeKernel kernel, uint[] globalWorkSize, uint[]? localWorkSize)
    {
        if (_disposed)
            throw new ObjectDisposedException(nameof(ROCmComputeCommandQueue));

        // Route dispatch to this queue's HIP stream so work is ordered correctly.
        if (kernel is ROCmComputeKernel rocmKernel)
            rocmKernel.SetStream(_stream);

        kernel.Dispatch(globalWorkSize, localWorkSize);
    }

    public override void WriteBuffer(IComputeBuffer buffer, ulong offset, byte[] data)
    {
        if (_disposed)
            throw new ObjectDisposedException(nameof(ROCmComputeCommandQueue));

        if (buffer is not ROCmComputeBuffer rocmBuffer)
            throw new ArgumentException("Buffer must be ROCmComputeBuffer", nameof(buffer));

        var ptr = rocmBuffer.GetPointer();
        fixed (byte* pData = data)
        {
            HipApi.CheckError(
                HipApi.hipMemcpyAsync(
                    ptr + (nint)offset,
                    (IntPtr)pData,
                    (ulong)data.Length,
                    HipApi.HipMemcpyKind.HostToDevice,
                    _stream),
                "hipMemcpyAsync HostToDevice");
        }
    }

    public override void ReadBuffer(IComputeBuffer buffer, ulong offset, byte[] data)
    {
        if (_disposed)
            throw new ObjectDisposedException(nameof(ROCmComputeCommandQueue));

        if (buffer is not ROCmComputeBuffer rocmBuffer)
            throw new ArgumentException("Buffer must be ROCmComputeBuffer", nameof(buffer));

        var ptr = rocmBuffer.GetPointer();
        fixed (byte* pData = data)
        {
            HipApi.CheckError(
                HipApi.hipMemcpyAsync(
                    (IntPtr)pData,
                    ptr + (nint)offset,
                    (ulong)data.Length,
                    HipApi.HipMemcpyKind.DeviceToHost,
                    _stream),
                "hipMemcpyAsync DeviceToHost");
        }
    }

    public override void Flush()
    {
        if (_disposed)
            throw new ObjectDisposedException(nameof(ROCmComputeCommandQueue));

        // Synchronize stream to ensure all commands complete
        HipApi.CheckError(HipApi.hipStreamSynchronize(_stream), "hipStreamSynchronize");
    }

    public override void Dispose()
    {
        if (_disposed) return;

        if (_stream != IntPtr.Zero)
        {
            HipApi.hipStreamSynchronize(_stream); // Don't throw in Dispose
            HipApi.hipStreamDestroy(_stream);
            _stream = IntPtr.Zero;
        }

        _disposed = true;
    }
}
