namespace AIHost.ICompute.ROCm;

/// <summary>
/// ROCm/HIP command queue (stream)
/// TODO: Implement with HIP streams
/// </summary>
public class ROCmComputeCommandQueue : IComputeCommandQueue
{
    private readonly ROCmComputeDevice _device;
    private bool _disposed;

    // TODO: Store hipStream_t
    // private IntPtr _stream;

    public ROCmComputeCommandQueue(ROCmComputeDevice device)
    {
        _device = device;
        
        // TODO: Create stream
        // hipStreamCreate(&_stream);
    }

    public void Dispatch(IComputeKernel kernel, uint[] globalWorkSize, uint[]? localWorkSize)
    {
        // TODO: Launch kernel on stream
        // Calculate grid and block dimensions
        // hipLaunchKernel(_function, gridDim, blockDim, args, 0, _stream);
        throw new NotImplementedException("ROCm kernel dispatch not implemented");
    }

    public void WriteBuffer(IComputeBuffer buffer, ulong offset, byte[] data)
    {
        throw new NotImplementedException();
        // TODO: hipMemcpy
    }

    public void ReadBuffer(IComputeBuffer buffer, ulong offset, byte[] data)
    {
        throw new NotImplementedException();
        // TODO: hipMemcpy
    }

    public void Flush()
    {
        // TODO: hipStreamSynchronize(_stream);
        throw new NotImplementedException();
    }

    public void Dispose()
    {
        if (_disposed) return;
        
        // TODO: Destroy stream
        // if (_stream != IntPtr.Zero)
        //     hipStreamDestroy(_stream);
        
        _disposed = true;
    }
}
