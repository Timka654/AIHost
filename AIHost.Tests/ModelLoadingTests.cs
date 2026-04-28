using AIHost.GGUF;
using AIHost.ICompute.Vulkan;
using Xunit;

namespace AIHost.Tests;

public class ModelLoadingTests : IDisposable
{
    private readonly VulkanComputeDevice _device;
    private readonly string _modelPath;

    public ModelLoadingTests()
    {
        _device = new VulkanComputeDevice();
        _modelPath = Environment.GetEnvironmentVariable("TEST_MODEL_PATH") 
            ?? @"D:\User\Downloads\tinyllama-1.1b-chat-v1.0.Q2_K.gguf";
    }

    [Fact]
    public void LazyGGUFModel_LoadsMetadataWithoutTensors()
    {
        // Arrange & Act
        using var model = new LazyGGUFModel(_modelPath, _device, useMemoryMapping: false);

        // Assert
        Assert.NotNull(model.Metadata);
        Assert.NotNull(model.Tensors);
        Assert.True(model.TotalTensorCount > 0);
        Assert.Equal(0, model.LoadedTensorCount); // No tensors loaded yet
        
        Console.WriteLine($"Model metadata loaded: {model.TotalTensorCount} tensors available");
        Console.WriteLine($"✓ Lazy loading initialized without loading tensors");
    }

    [Fact]
    public void LazyGGUFModel_LoadsTensorOnDemand()
    {
        if (!File.Exists(_modelPath))
        {
            Console.WriteLine($"Model not found at {_modelPath}, skipping test");
            return;
        }

        // Arrange
        using var model = new LazyGGUFModel(_modelPath, _device, useMemoryMapping: false);
        Assert.Equal(0, model.LoadedTensorCount);

        // Act - load specific tensor
        var buffer = model.LoadTensor("token_embd.weight");

        // Assert
        Assert.NotNull(buffer);
        Assert.Equal(1, model.LoadedTensorCount);
        Assert.True(model.GetGPUMemoryUsageBytes() > 0);
        
        Console.WriteLine($"GPU memory usage: {model.GetGPUMemoryUsageBytes() / (1024 * 1024)}MB");
        Console.WriteLine("✓ Tensor loaded on-demand");
    }

    [Fact]
    public void LazyGGUFModel_CachesTensors()
    {
        if (!File.Exists(_modelPath))
        {
            Console.WriteLine($"Model not found at {_modelPath}, skipping test");
            return;
        }

        // Arrange
        using var model = new LazyGGUFModel(_modelPath, _device, useMemoryMapping: false);

        // Act - load same tensor twice
        var buffer1 = model.LoadTensor("token_embd.weight");
        var buffer2 = model.LoadTensor("token_embd.weight");

        // Assert - should return cached buffer
        Assert.Same(buffer1, buffer2);
        Assert.Equal(1, model.LoadedTensorCount);
        
        Console.WriteLine("✓ Tensor caching works");
    }

    [Fact]
    public void LazyGGUFModel_UnloadTensor_FreesMemory()
    {
        if (!File.Exists(_modelPath))
        {
            Console.WriteLine($"Model not found at {_modelPath}, skipping test");
            return;
        }

        // Arrange
        using var model = new LazyGGUFModel(_modelPath, _device, useMemoryMapping: false);
        model.LoadTensor("token_embd.weight");
        Assert.Equal(1, model.LoadedTensorCount);
        long memoryBefore = model.GetGPUMemoryUsageBytes();

        // Act
        model.UnloadTensor("token_embd.weight");

        // Assert
        Assert.Equal(0, model.LoadedTensorCount);
        Assert.Equal(0, model.GetGPUMemoryUsageBytes());
        
        Console.WriteLine($"Freed {memoryBefore / (1024 * 1024)}MB GPU memory");
        Console.WriteLine("✓ Tensor unloading works");
    }

    [Fact]
    public void LazyGGUFModel_LoadMultipleTensorsWithPredicate()
    {
        if (!File.Exists(_modelPath))
        {
            Console.WriteLine($"Model not found at {_modelPath}, skipping test");
            return;
        }

        // Arrange
        using var model = new LazyGGUFModel(_modelPath, _device, useMemoryMapping: false);

        // Act - load all tensors for layer 0
        var tensors = model.LoadTensors(t => t.Name.Contains("blk.0."));

        // Assert
        Assert.NotEmpty(tensors);
        Assert.True(model.LoadedTensorCount > 0);
        
        Console.WriteLine($"Loaded {tensors.Count} tensors for layer 0");
        Console.WriteLine("✓ Predicate-based tensor loading works");
    }

    [Fact]
    public void LazyGGUFModel_WithMemoryMapping_LoadsTensors()
    {
        if (!File.Exists(_modelPath))
        {
            Console.WriteLine($"Model not found at {_modelPath}, skipping test");
            return;
        }

        // Arrange & Act
        using var model = new LazyGGUFModel(_modelPath, _device, useMemoryMapping: true);
        
        if (!model.UseMemoryMapping)
        {
            Console.WriteLine("Memory mapping not available, skipping memory-mapped load test");
            return;
        }

        var buffer = model.LoadTensor("token_embd.weight");

        // Assert
        Assert.NotNull(buffer);
        Assert.Equal(1, model.LoadedTensorCount);
        
        Console.WriteLine("✓ Memory-mapped tensor loading works");
    }

    [Fact]
    public void LazyGGUFModel_UnloadAll_FreesAllMemory()
    {
        if (!File.Exists(_modelPath))
        {
            Console.WriteLine($"Model not found at {_modelPath}, skipping test");
            return;
        }

        // Arrange
        using var model = new LazyGGUFModel(_modelPath, _device, useMemoryMapping: false);
        model.LoadTensor("token_embd.weight");
        model.LoadTensor("output_norm.weight");
        int loadedBefore = model.LoadedTensorCount;
        Assert.True(loadedBefore > 0);

        // Act
        model.UnloadAllTensors();

        // Assert
        Assert.Equal(0, model.LoadedTensorCount);
        Assert.Equal(0, model.GetGPUMemoryUsageBytes());
        
        Console.WriteLine($"Unloaded {loadedBefore} tensors");
        Console.WriteLine("✓ Bulk unloading works");
    }

    public void Dispose()
    {
        _device?.Dispose();
    }
}
