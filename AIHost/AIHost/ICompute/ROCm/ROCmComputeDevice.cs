using System.Runtime.InteropServices;

namespace AIHost.ICompute.ROCm;

/// <summary>
/// ROCm/HIP compute device implementation for AMD GPUs
/// TODO: Implement HIP API bindings and device management
/// </summary>
public class ROCmComputeDevice : IComputeDevice
{
    private bool _disposed;

    public string ProviderName => "ROCm/HIP";
    public string ApiVersion => "HIP 6.0"; // TODO: Get actual version from hipGetDeviceProperties

    public ROCmComputeDevice()
    {
        // TODO: Initialize HIP runtime
        // hipInit(0);
        // hipSetDevice(0);
        
        throw new NotImplementedException("ROCm provider not yet implemented. Requires HIP API bindings.");
    }

    public IComputeBuffer CreateBuffer(ulong size, BufferType type, DataType dataType)
    {
        throw new NotImplementedException();
        // TODO: Implement hipMalloc for buffer allocation
        // return new ROCmComputeBuffer(size, type, dataType);
    }

    public IComputeKernel CreateKernel(string source, string entryPoint)
    {
        throw new NotImplementedException();
        // TODO: Implement HIP kernel compilation from source
        // - Use hiprtcCompileProgram for runtime compilation
        // - Load compiled module with hipModuleLoad
        // return new ROCmComputeKernel(source, entryPoint, this);
    }

    public IComputeKernel CreateKernelFromFile(string filePath, string entryPoint)
    {
        throw new NotImplementedException();
        // TODO: Load and compile HIP kernel from file
    }

    public IComputeCommandQueue CreateCommandQueue()
    {
        throw new NotImplementedException();
        // TODO: Create HIP stream
        // return new ROCmComputeCommandQueue(this);
    }

    public void Synchronize()
    {
        // TODO: hipDeviceSynchronize();
        throw new NotImplementedException();
    }

    public void Dispose()
    {
        if (_disposed) return;
        
        // TODO: Cleanup HIP resources
        // hipDeviceReset();
        
        _disposed = true;
        GC.SuppressFinalize(this);
    }
}
