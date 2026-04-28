using AIHost.Compute;
using AIHost.ICompute;
using AIHost.ICompute.Vulkan;
using Xunit;

namespace AIHost.Tests;

public class OptimizationTests : IDisposable
{
    private readonly VulkanComputeDevice _device;

    public OptimizationTests()
    {
        _device = new VulkanComputeDevice();
    }

    [Fact]
    public void BufferPool_RentAndReturn_ReusesBuffers()
    {
        // Arrange
        using var pool = new ComputeBufferPool(_device);
        const int bufferSize = 1024 * sizeof(float);

        // Act - Rent and return multiple times
        var buffer1 = pool.Rent((ulong)bufferSize);
        pool.Return(buffer1);

        var buffer2 = pool.Rent((ulong)bufferSize);
        pool.Return(buffer2);

        var stats = pool.GetStatistics();

        // Assert
        Assert.Equal(1L, stats.TotalAllocations); // Only 1 allocation (reused on 2nd rent)
        Assert.Equal(1L, stats.PoolHits); // Second rent should hit the pool
        Assert.Equal(1L, stats.PoolMisses); // First rent is a miss
        Assert.Equal(0.5, stats.HitRate); // 1 hit out of 2 rent operations
        
        Console.WriteLine($"✓ {stats}");
    }

    [Fact]
    public void BufferPool_MultipleRent_TracksActive()
    {
        // Arrange
        using var pool = new ComputeBufferPool(_device);
        const int bufferSize = 2048 * sizeof(float);

        // Act - Rent multiple buffers without returning
        var buffer1 = pool.Rent((ulong)bufferSize);
        var buffer2 = pool.Rent((ulong)bufferSize);
        var buffer3 = pool.Rent((ulong)bufferSize);

        var stats = pool.GetStatistics();

        // Assert
        Assert.Equal(3, stats.ActiveBufferCount);
        Assert.Equal(0, stats.PooledBufferCount); // Nothing returned yet
        
        // Cleanup
        pool.Return(buffer1);
        pool.Return(buffer2);
        pool.Return(buffer3);
        
        Console.WriteLine($"✓ Buffer pool tracks {stats.ActiveBufferCount} active buffers");
    }

    [Fact]
    public void BufferPool_Clear_DisposesAllBuffers()
    {
        // Arrange
        using var pool = new ComputeBufferPool(_device);
        
        var buffer = pool.Rent(1024);
        pool.Return(buffer);

        // Act
        pool.Clear();
        var stats = pool.GetStatistics();

        // Assert
        Assert.True(stats.PooledBufferCount == 0);
        Assert.True(stats.PooledMemoryBytes == 0);
        
        Console.WriteLine("✓ Buffer pool cleared successfully");
    }

    [Fact]
    public void KernelCache_GetOrCreate_CachesKernels()
    {
        // Arrange
        using var cache = new ComputeKernelCache(_device);
        
        string simpleShader = @"
            #version 450
            layout(local_size_x = 256) in;
            layout(binding = 0) buffer Input { float data[]; } input;
            layout(binding = 1) buffer Output { float data[]; } output;
            void main() {
                uint idx = gl_GlobalInvocationID.x;
                output.data[idx] = input.data[idx] * 2.0;
            }
        ";

        // Act - Get same kernel twice
        var kernel1 = cache.GetOrCreate(simpleShader, "main");
        var kernel2 = cache.GetOrCreate(simpleShader, "main");
        
        var stats = cache.GetStatistics();

        // Assert
        Assert.Same(kernel1, kernel2); // Should be the exact same instance
        Assert.Equal(1, stats.CacheMisses); // First call is a miss
        Assert.Equal(1, stats.CacheHits); // Second call is a hit
        Assert.Equal(1, stats.CachedKernelCount);
        Assert.Equal(0.5, stats.HitRate);
        
        Console.WriteLine($"✓ {stats}");
    }

    [Fact]
    public void KernelCache_DifferentKernels_StoresSeparately()
    {
        // Arrange
        using var cache = new ComputeKernelCache(_device);
        
        string shader1 = "#version 450\nlayout(local_size_x = 256) in;\nvoid main() {}";
        string shader2 = "#version 450\nlayout(local_size_x = 128) in;\nvoid main() {}";

        // Act
        var kernel1 = cache.GetOrCreate(shader1, "main");
        var kernel2 = cache.GetOrCreate(shader2, "main");
        
        var stats = cache.GetStatistics();

        // Assert
        Assert.NotSame(kernel1, kernel2);
        Assert.Equal(2, stats.CachedKernelCount);
        Assert.Equal(2, stats.CacheMisses);
        Assert.Equal(0, stats.CacheHits);
        
        Console.WriteLine($"✓ Kernel cache stores {stats.CachedKernelCount} different kernels");
    }

    [Fact]
    public async Task AsyncQueue_WriteAndRead_WorksAsynchronously()
    {
        // Arrange
        using var queue = _device.CreateCommandQueue();
        using var asyncQueue = new AsyncComputeQueue(queue);
        using var buffer = _device.CreateBuffer(1024 * sizeof(float), BufferType.Storage);

        float[] data = new float[1024];
        for (int i = 0; i < data.Length; i++)
            data[i] = i * 2.0f;

        byte[] bytes = new byte[data.Length * sizeof(float)];
        Buffer.BlockCopy(data, 0, bytes, 0, bytes.Length);

        // Act
        await asyncQueue.WriteBufferAsync(buffer, 0, bytes);
        await asyncQueue.FlushAsync();
        
        var resultBytes = await asyncQueue.ReadBufferAsync(buffer, 0, bytes.Length);
        float[] result = new float[data.Length];
        Buffer.BlockCopy(resultBytes, 0, result, 0, resultBytes.Length);

        // Assert
        Assert.Equal(data[0], result[0]);
        Assert.Equal(data[1023], result[1023]);
        
        Console.WriteLine("✓ Async queue write/read completed successfully");
    }

    [Fact]
    public void Profiler_RecordsOperations_CalculatesStatistics()
    {
        // Arrange
        var profiler = new ComputeProfiler();

        // Act - Simulate some operations
        using (profiler.Begin("MatMul"))
        {
            Thread.Sleep(10);
        }
        
        using (profiler.Begin("Softmax"))
        {
            Thread.Sleep(5);
        }
        
        using (profiler.Begin("MatMul"))
        {
            Thread.Sleep(10);
        }

        var results = profiler.GetResults();

        // Assert
        Assert.Equal(2, results.Length); // MatMul and Softmax
        
        var matmul = results.FirstOrDefault(r => r.Name == "MatMul");
        Assert.Equal(2, matmul.CallCount);
        Assert.True(matmul.TotalMilliseconds >= 20); // At least 20ms total
        
        var softmax = results.FirstOrDefault(r => r.Name == "Softmax");
        Assert.Equal(1, softmax.CallCount);
        Assert.True(softmax.TotalMilliseconds >= 5);
        
        Console.WriteLine("✓ Profiler statistics:");
        foreach (var result in results)
        {
            Console.WriteLine($"  {result}");
        }
    }

    [Fact]
    public void Profiler_GetSummary_ReturnsFormattedReport()
    {
        // Arrange
        var profiler = new ComputeProfiler();
        
        using (profiler.Begin("Operation1")) { Thread.Sleep(5); }
        using (profiler.Begin("Operation2")) { Thread.Sleep(3); }
        using (profiler.Begin("Operation1")) { Thread.Sleep(5); }

        // Act
        string summary = profiler.GetSummary();

        // Assert
        Assert.Contains("Compute Profiling Summary", summary);
        Assert.Contains("Operation1", summary);
        Assert.Contains("Operation2", summary);
        Assert.Contains("TOTAL", summary);
        
        Console.WriteLine(summary);
    }

    [Fact]
    public void Profiler_Clear_ResetsAllData()
    {
        // Arrange
        var profiler = new ComputeProfiler();
        using (profiler.Begin("Test")) { }

        // Act
        profiler.Clear();
        var results = profiler.GetResults();

        // Assert
        Assert.Empty(results);
        
        Console.WriteLine("✓ Profiler cleared successfully");
    }

    public void Dispose()
    {
        _device.Dispose();
    }
}
