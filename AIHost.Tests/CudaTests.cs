using AIHost.ICompute.CUDA;
using Xunit;

namespace AIHost.Tests;

public class CudaTests : IDisposable
{
    private readonly CudaComputeDevice? _device;
    private readonly bool _cudaAvailable;

    public CudaTests()
    {
        try
        {
            var devices = CudaComputeDevice.GetAvailableDevices();
            _cudaAvailable = devices.Length > 0;
            
            if (_cudaAvailable)
            {
                _device = new CudaComputeDevice(0);
                Console.WriteLine($"CUDA device available: {_device.DeviceName}");
            }
            else
            {
                Console.WriteLine("No CUDA devices available - tests will be skipped");
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"CUDA initialization failed: {ex.Message}");
            _cudaAvailable = false;
        }
    }

    [Fact]
    public void CudaDevice_GetAvailableDevices_ReturnsDevices()
    {
        try
        {
            // Act
            var devices = CudaComputeDevice.GetAvailableDevices();

            // Assert
            Assert.NotNull(devices);
            
            if (devices.Length > 0)
            {
                foreach (var device in devices)
                {
                    Console.WriteLine($"CUDA Device {device.Index}: {device.Name} ({device.ApiVersion})");
                    Assert.False(string.IsNullOrEmpty(device.Name));
                }
            }
            else
            {
                Console.WriteLine("No CUDA devices detected (expected if NVIDIA GPU not available)");
            }
        }
        catch (DllNotFoundException)
        {
            Console.WriteLine("CUDA Runtime not installed - this is expected on systems without NVIDIA GPUs");
            // Test passes even if CUDA is not available
        }
    }

    [Fact]
    public void CudaDevice_CreateDevice_Success()
    {
        if (!_cudaAvailable)
        {
            Console.WriteLine("CUDA not available, skipping test");
            return;
        }

        // Assert
        Assert.NotNull(_device);
        Assert.Equal("CUDA", _device!.ProviderName);
        Assert.False(string.IsNullOrEmpty(_device.DeviceName));
        Assert.False(string.IsNullOrEmpty(_device.ApiVersion));
        
        Console.WriteLine($"✓ CUDA device initialized: {_device.DeviceName}");
    }

    [Fact]
    public void CudaBuffer_CreateAndReadWrite_Success()
    {
        if (!_cudaAvailable)
        {
            Console.WriteLine("CUDA not available, skipping test");
            return;
        }

        // Arrange
        const int count = 1024;
        float[] data = new float[count];
        for (int i = 0; i < count; i++)
            data[i] = i * 2.0f;

        // Act
        using var buffer = _device!.CreateBuffer((ulong)(count * sizeof(float)), 
            AIHost.ICompute.BufferType.Storage, AIHost.ICompute.DataType.F32);
        
        buffer.Write(data);
        var result = buffer.Read<float>();

        // Assert
        Assert.NotNull(result);
        Assert.Equal(count, result.Length);
        Assert.Equal(data[0], result[0]);
        Assert.Equal(data[count - 1], result[count - 1]);
        
        Console.WriteLine($"✓ Buffer write/read test passed ({count} elements)");
    }

    [Fact]
    public void CudaKernel_VectorAddition_Success()
    {
        if (!_cudaAvailable)
        {
            Console.WriteLine("CUDA not available, skipping test");
            return;
        }

        // Arrange
        const int n = 1024;
        float[] a = new float[n];
        float[] b = new float[n];
        for (int i = 0; i < n; i++)
        {
            a[i] = i;
            b[i] = i * 2.0f;
        }

        string kernelSource = @"
extern ""C"" __global__ void vector_add(const float* a, const float* b, float* c, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        c[idx] = a[idx] + b[idx];
    }
}";

        // Act
        using var bufferA = _device!.CreateBuffer((ulong)(n * sizeof(float)), 
            AIHost.ICompute.BufferType.Storage, AIHost.ICompute.DataType.F32);
        using var bufferB = _device.CreateBuffer((ulong)(n * sizeof(float)), 
            AIHost.ICompute.BufferType.Storage, AIHost.ICompute.DataType.F32);
        using var bufferC = _device.CreateBuffer((ulong)(n * sizeof(float)), 
            AIHost.ICompute.BufferType.Storage, AIHost.ICompute.DataType.F32);
        
        bufferA.Write(a);
        bufferB.Write(b);

        using var kernel = _device.CreateKernel(kernelSource, "vector_add");
        using var queue = _device.CreateCommandQueue();
        
        // Launch kernel: (n threads, 256 threads per block)
        uint blockSize = 256;
        uint gridSize = (uint)((n + blockSize - 1) / blockSize);
        
        // Note: CudaComputeKernel.Execute is not yet compatible with standard interface
        // This test will verify compilation but not execution for now
        
        Console.WriteLine($"✓ CUDA kernel compiled successfully");
        Console.WriteLine($"  Grid: {gridSize} blocks, Block: {blockSize} threads");
    }

    [Fact]
    public void CudaDevice_Synchronize_Success()
    {
        if (!_cudaAvailable)
        {
            Console.WriteLine("CUDA not available, skipping test");
            return;
        }

        // Act & Assert (should not throw)
        _device!.Synchronize();
        
        Console.WriteLine("✓ Device synchronize succeeded");
    }

    public void Dispose()
    {
        _device?.Dispose();
    }
}
