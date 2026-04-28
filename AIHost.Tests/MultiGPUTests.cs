using AIHost.Compute;
using AIHost.ICompute.Vulkan;
using AIHost.Inference;
using Xunit;

namespace AIHost.Tests;

public class MultiGPUTests : IDisposable
{
    private readonly string _modelPath;

    public MultiGPUTests()
    {
        _modelPath = Environment.GetEnvironmentVariable("TEST_MODEL_PATH") 
            ?? @"D:\User\Downloads\tinyllama-1.1b-chat-v1.0.Q2_K.gguf";
    }

    [Fact]
    public void GetAvailableDevices_ReturnsDeviceList()
    {
        // Act
        var devices = VulkanComputeDevice.GetAvailableDevices();

        // Assert
        Assert.NotNull(devices);
        Assert.True(devices.Length > 0, "Should detect at least one Vulkan device");
        
        foreach (var device in devices)
        {
            Console.WriteLine($"Device {device.Index}: {device.Name} (API {device.ApiVersion}, Type: {device.DeviceType})");
            Assert.False(string.IsNullOrEmpty(device.Name));
            Assert.False(string.IsNullOrEmpty(device.ApiVersion));
        }
    }

    [Fact]
    public void VulkanComputeDevice_SelectSpecificDevice_Success()
    {
        // Arrange
        var devices = VulkanComputeDevice.GetAvailableDevices();
        if (devices.Length == 0)
        {
            Console.WriteLine("No Vulkan devices available, skipping test");
            return;
        }

        // Act - try to create device with explicit index
        using var device = new VulkanComputeDevice(0);

        // Assert
        Assert.Equal("Vulkan", device.ProviderName);
        Assert.Equal(0, device.DeviceIndex);
        Assert.False(string.IsNullOrEmpty(device.DeviceName));
        Console.WriteLine($"Selected device: {device.DeviceName}");
    }

    [Fact]
    public void MultiGPUInferenceEngine_InitializeWithSingleGPU_Success()
    {
        // Arrange
        var devices = VulkanComputeDevice.GetAvailableDevices();
        if (devices.Length == 0)
        {
            Console.WriteLine("No Vulkan devices available, skipping test");
            return;
        }

        if (!File.Exists(_modelPath))
        {
            Console.WriteLine($"Model not found at {_modelPath}, skipping test");
            return;
        }

        // Act
        using var engine = new MultiGPUInferenceEngine(new[] { 0 }, _modelPath);

        // Assert
        Assert.Equal(1, engine.DeviceCount);
        Console.WriteLine("✓ Multi-GPU engine initialized with single GPU");
    }

    [Fact]
    public void MultiGPUInferenceEngine_Generate_ProducesOutput()
    {
        // Arrange
        var devices = VulkanComputeDevice.GetAvailableDevices();
        if (devices.Length == 0)
        {
            Console.WriteLine("No Vulkan devices available, skipping test");
            return;
        }

        if (!File.Exists(_modelPath))
        {
            Console.WriteLine($"Model not found at {_modelPath}, skipping test");
            return;
        }

        using var engine = new MultiGPUInferenceEngine(new[] { 0 }, _modelPath);
        var config = new GenerationConfig
        {
            MaxNewTokens = 10,
            Temperature = 0.7f,
            TopK = 40,
            Seed = 42
        };

        // Act
        string result = engine.Generate("Hello", config);

        // Assert
        Assert.NotNull(result);
        Assert.Contains("Hello", result);
        Console.WriteLine($"Generated: {result}");
    }

    [Fact]
    public void MultiGPUInferenceEngine_InitializeWithMultipleGPUs_DistributesLayers()
    {
        // Arrange
        var devices = VulkanComputeDevice.GetAvailableDevices();
        if (devices.Length < 2)
        {
            Console.WriteLine($"Only {devices.Length} GPU(s) available, need 2+ for multi-GPU test");
            return;
        }

        if (!File.Exists(_modelPath))
        {
            Console.WriteLine($"Model not found at {_modelPath}, skipping test");
            return;
        }

        // Act - initialize with first two GPUs
        using var engine = new MultiGPUInferenceEngine(new[] { 0, 1 }, _modelPath);

        // Assert
        Assert.Equal(2, engine.DeviceCount);
        Console.WriteLine("✓ Multi-GPU engine initialized with 2 GPUs");
        
        // Note: Layer distribution is printed in constructor
        // This test verifies the engine initializes without errors
    }

    public void Dispose()
    {
        // Cleanup if needed
    }
}
