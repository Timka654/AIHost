using AIHost.Compute;
using AIHost.ICompute;
using AIHost.ICompute.Vulkan;

namespace AIHost.Examples;

/// <summary>
/// Example demonstrating GPU optimization features: buffer pooling, kernel caching, and profiling
/// </summary>
public class OptimizationExample
{
    public static void Run()
    {
        Console.WriteLine("=== GPU Optimization Example ===\n");

        // 1. Setup device with buffer pool and kernel cache
        using var device = new VulkanComputeDevice();
        using var bufferPool = new ComputeBufferPool(device);
        using var kernelCache = new ComputeKernelCache(device);
        var profiler = new ComputeProfiler();

        Console.WriteLine($"Device: {device.DeviceName}");
        Console.WriteLine($"Provider: {device.ProviderName}\n");

        // 2. Demonstrate buffer pooling
        Console.WriteLine("=== Buffer Pooling Demo ===");
        DemoBufferPooling(bufferPool, profiler);

        // 3. Demonstrate kernel caching
        Console.WriteLine("\n=== Kernel Caching Demo ===");
        DemoKernelCaching(kernelCache, profiler);

        // 4. Print final statistics
        Console.WriteLine("\n" + bufferPool.GetStatistics());
        Console.WriteLine(kernelCache.GetStatistics());
        Console.WriteLine("\n" + profiler.GetSummary());
    }

    private static void DemoBufferPooling(ComputeBufferPool pool, ComputeProfiler profiler)
    {
        const int iterations = 10;
        const int bufferSize = 1024 * sizeof(float);

        Console.WriteLine($"Creating and returning {iterations} buffers...");

        for (int i = 0; i < iterations; i++)
        {
            using var scope = profiler.Begin("BufferRent");
            
            // Rent buffer (will reuse from pool after first allocation)
            var buffer = pool.Rent((ulong)bufferSize, BufferType.Storage, DataType.F32);
            
            // Simulate some work
            float[] data = new float[1024];
            for (int j = 0; j < data.Length; j++)
                data[j] = i * j * 0.1f;
            
            buffer.Write(data);
            var result = buffer.Read<float>();
            
            // Return buffer to pool
            pool.Return(buffer);
        }

        Console.WriteLine("✓ Buffer pooling demo complete");
        var stats = pool.GetStatistics();
        Console.WriteLine($"  Pool hit rate: {stats.HitRate:P1}");
        Console.WriteLine($"  Total allocations: {stats.TotalAllocations}");
    }

    private static void DemoKernelCaching(ComputeKernelCache cache, ComputeProfiler profiler)
    {
        const string simpleShader = @"
            #version 450
            layout(local_size_x = 256) in;
            layout(binding = 0) buffer Data { float values[]; } data;
            void main() {
                uint idx = gl_GlobalInvocationID.x;
                data.values[idx] *= 2.0;
            }
        ";

        Console.WriteLine("Compiling same kernel 5 times (should cache after first)...");

        for (int i = 0; i < 5; i++)
        {
            using var scope = profiler.Begin("KernelCompile");
            
            // Get kernel (will cache after first call)
            var kernel = cache.GetOrCreate(simpleShader, "main");
        }

        Console.WriteLine("✓ Kernel caching demo complete");
        var stats = cache.GetStatistics();
        Console.WriteLine($"  Cache hit rate: {stats.HitRate:P1}");
        Console.WriteLine($"  Cached kernels: {stats.CachedKernelCount}");
    }

    /// <summary>
    /// Demonstrate async compute operations
    /// </summary>
    public static async Task RunAsyncExample()
    {
        Console.WriteLine("=== Async Compute Example ===\n");

        using var device = new VulkanComputeDevice();
        using var queue = device.CreateCommandQueue();
        using var asyncQueue = new AsyncComputeQueue(queue);

        Console.WriteLine($"Device: {device.DeviceName}\n");

        // Create buffers
        using var buffer = device.CreateBuffer(1024 * sizeof(float), BufferType.Storage, DataType.F32);

        // Prepare data
        float[] data = new float[1024];
        for (int i = 0; i < data.Length; i++)
            data[i] = i * 0.5f;

        byte[] bytes = new byte[data.Length * sizeof(float)];
        Buffer.BlockCopy(data, 0, bytes, 0, bytes.Length);

        // Async write
        Console.WriteLine("Writing data asynchronously...");
        await asyncQueue.WriteBufferAsync(buffer, 0, bytes);
        await asyncQueue.FlushAsync();

        // Async read
        Console.WriteLine("Reading data asynchronously...");
        var resultBytes = await asyncQueue.ReadBufferAsync(buffer, 0, bytes.Length);

        float[] result = new float[data.Length];
        Buffer.BlockCopy(resultBytes, 0, result, 0, resultBytes.Length);

        Console.WriteLine($"✓ Async operations complete");
        Console.WriteLine($"  First value: {result[0]:F1} (expected {data[0]:F1})");
        Console.WriteLine($"  Last value: {result[1023]:F1} (expected {data[1023]:F1})");
    }
}

