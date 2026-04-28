using AIHost.ICompute.CUDA;
using Xunit;

namespace AIHost.Tests;

public class NvrtcIntegrationTests : IDisposable
{
    private readonly CudaComputeDevice? _device;
    private readonly bool _cudaAvailable;

    public NvrtcIntegrationTests()
    {
        try
        {
            var devices = CudaComputeDevice.GetAvailableDevices();
            _cudaAvailable = devices.Length > 0;
            
            if (_cudaAvailable)
            {
                _device = new CudaComputeDevice(0);
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"CUDA initialization failed: {ex.Message}");
            _cudaAvailable = false;
        }
    }

    [Fact]
    public void NvrtcCompilation_SimpleKernel_Success()
    {
        if (!_cudaAvailable)
        {
            Console.WriteLine("CUDA not available, skipping test");
            return;
        }

        // Arrange
        string kernelSource = @"
extern ""C"" __global__ void simple_add(float* a, float* b, float* c, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        c[idx] = a[idx] + b[idx];
}";

        // Act & Assert (should not throw)
        using var kernel = _device!.CreateKernel(kernelSource, "simple_add");
        Assert.NotNull(kernel);
        
        Console.WriteLine("✓ NVRTC simple kernel compilation succeeded");
    }

    [Fact]
    public void NvrtcCompilation_MatmulKernel_Success()
    {
        if (!_cudaAvailable)
        {
            Console.WriteLine("CUDA not available, skipping test");
            return;
        }

        // Arrange
        string matmulPath = Path.Combine("Shaders", "CUDA", "matmul.cu");
        if (!File.Exists(matmulPath))
        {
            Console.WriteLine($"Kernel file not found: {matmulPath}");
            return;
        }

        // Act & Assert
        using var kernel = _device!.CreateKernelFromFile(matmulPath, "matmul");
        Assert.NotNull(kernel);
        
        Console.WriteLine("✓ NVRTC matmul kernel compilation succeeded");
    }

    [Fact]
    public void NvrtcCompilation_SoftmaxKernel_Success()
    {
        if (!_cudaAvailable)
        {
            Console.WriteLine("CUDA not available, skipping test");
            return;
        }

        // Arrange
        string softmaxPath = Path.Combine("Shaders", "CUDA", "softmax.cu");
        if (!File.Exists(softmaxPath))
        {
            Console.WriteLine($"Kernel file not found: {softmaxPath}");
            return;
        }

        // Act & Assert
        using var kernel = _device!.CreateKernelFromFile(softmaxPath, "softmax");
        Assert.NotNull(kernel);
        
        Console.WriteLine("✓ NVRTC softmax kernel compilation succeeded");
    }

    [Fact]
    public void NvrtcCompilation_ActivationsKernel_Success()
    {
        if (!_cudaAvailable)
        {
            Console.WriteLine("CUDA not available, skipping test");
            return;
        }

        // Arrange
        string activationsPath = Path.Combine("Shaders", "CUDA", "activations.cu");
        if (!File.Exists(activationsPath))
        {
            Console.WriteLine($"Kernel file not found: {activationsPath}");
            return;
        }

        // Act & Assert
        using var kernel = _device!.CreateKernelFromFile(activationsPath, "silu");
        Assert.NotNull(kernel);
        
        Console.WriteLine("✓ NVRTC silu kernel compilation succeeded");
    }

    [Fact]
    public void NvrtcCompilation_InvalidKernel_ThrowsException()
    {
        if (!_cudaAvailable)
        {
            Console.WriteLine("CUDA not available, skipping test");
            return;
        }

        // Arrange
        string invalidSource = @"
extern ""C"" __global__ void invalid_kernel()
{
    // Invalid CUDA code
    this_function_does_not_exist();
}";

        // Act & Assert
        Assert.Throws<CudaException>(() =>
        {
            using var kernel = _device!.CreateKernel(invalidSource, "invalid_kernel");
        });
        
        Console.WriteLine("✓ NVRTC correctly rejects invalid kernel");
    }

    public void Dispose()
    {
        _device?.Dispose();
    }
}
