using Silk.NET.Vulkan;

namespace AIHost.ICompute.Vulkan;

/// <summary>
/// Провайдер вычислений на основе Vulkan.
/// Использует глобальный VulkanGlobalContext для разделения VkInstance и VkDevice
/// между несколькими экземплярами VulkanComputeDevice на одном физическом GPU.
/// </summary>
public unsafe class VulkanComputeDevice : ComputeProviderBase
{
    private readonly VulkanDeviceContext _deviceContext;
    private bool _disposed;

    public override string ProviderName => "Vulkan";
    public override string ApiVersion { get; }

    /// <summary>
    /// Информация об устройстве
    /// </summary>
    public string DeviceName { get; private set; } = string.Empty;
    public int DeviceIndex { get; private set; }

    /// <summary>
    /// Get information about all available Vulkan devices
    /// </summary>
    public static unsafe DeviceInfo[] GetAvailableDevices()
    {
        var vk = Vk.GetApi();
        var devices = new List<DeviceInfo>();

        try
        {
            // Create temporary instance
            var appNameBytes = System.Text.Encoding.UTF8.GetBytes("AIHost\0");
            var engineNameBytes = System.Text.Encoding.UTF8.GetBytes("AIHost Engine\0");
            
            Instance instance;
            fixed (byte* pAppName = appNameBytes)
            fixed (byte* pEngineName = engineNameBytes)
            {
                var appInfo = new ApplicationInfo
                {
                    SType = StructureType.ApplicationInfo,
                    PApplicationName = pAppName,
                    ApplicationVersion = Vk.MakeVersion(1, 0, 0),
                    PEngineName = pEngineName,
                    EngineVersion = Vk.MakeVersion(1, 0, 0),
                    ApiVersion = Vk.Version13
                };

                var createInfo = new InstanceCreateInfo
                {
                    SType = StructureType.InstanceCreateInfo,
                    PApplicationInfo = &appInfo
                };

                if (vk.CreateInstance(&createInfo, null, out instance) != Result.Success)
                    return Array.Empty<DeviceInfo>();
            }

            // Enumerate devices
            uint deviceCount = 0;
            vk.EnumeratePhysicalDevices(instance, &deviceCount, null);
            if (deviceCount > 0)
            {
                var physicalDevices = stackalloc PhysicalDevice[(int)deviceCount];
                vk.EnumeratePhysicalDevices(instance, &deviceCount, physicalDevices);

                for (int i = 0; i < deviceCount; i++)
                {
                    PhysicalDeviceProperties props;
                    vk.GetPhysicalDeviceProperties(physicalDevices[i], &props);
                    
                    uint apiVer = props.ApiVersion;
                    string apiVersion = $"{apiVer >> 22}.{(apiVer >> 12) & 0x3ff}.{apiVer & 0xfff}";
                    string name = System.Runtime.InteropServices.Marshal.PtrToStringAnsi((nint)props.DeviceName) ?? "Unknown";

                    devices.Add(new DeviceInfo
                    {
                        Index = i,
                        Name = name,
                        ApiVersion = apiVersion,
                        DeviceType = props.DeviceType.ToString()
                    });
                }
            }

            vk.DestroyInstance(instance, null);
        }
        finally
        {
            vk.Dispose();
        }

        return devices.ToArray();
    }

    public VulkanComputeDevice() : this(0)
    {
    }

    public VulkanComputeDevice(int deviceIndex)
    {
        DeviceIndex = deviceIndex;

        // Получаем глобальный контекст устройства (создаёт VkInstance и VkDevice при первом вызове)
        _deviceContext = VulkanGlobalContext.AcquireDeviceContext(deviceIndex, queueCount: 2);

        // Получение свойств устройства
        PhysicalDeviceProperties props;
        _deviceContext.Vk.GetPhysicalDeviceProperties(_deviceContext.PhysicalDevice, &props);
        
        uint apiVer = props.ApiVersion;
        ApiVersion = $"{apiVer >> 22}.{(apiVer >> 12) & 0x3ff}.{apiVer & 0xfff}";
        DeviceName = _deviceContext.DeviceName;
        
        Console.WriteLine($"Vulkan Device [{deviceIndex}]: {DeviceName}");
        Console.WriteLine($"API Version: {ApiVersion}");
    }

    public override IComputeBuffer CreateBuffer(ulong size, BufferType type, DataType elementType = DataType.F32, bool requireDeviceLocal = false)
    {
        return new VulkanComputeBuffer(_deviceContext, size, type, elementType, requireDeviceLocal);
    }

    public override IComputeKernel CreateKernel(string source, string entryPoint)
    {
        return new VulkanComputeKernel(_deviceContext, source, entryPoint);
    }

    public override IComputeKernel CreateKernelFromFile(string filePath, string entryPoint)
    {
        var source = File.ReadAllText(filePath);
        return new VulkanComputeKernel(_deviceContext, source, entryPoint);
    }

    public override IComputeCommandQueue CreateCommandQueue()
    {
        return new VulkanComputeCommandQueue(_deviceContext);
    }

    public override void Synchronize()
    {
        _deviceContext.Vk.DeviceWaitIdle(_deviceContext.Device);
    }

    public override void Dispose()
    {
        if (_disposed) return;

        VulkanGlobalContext.ReleaseDeviceContext(DeviceIndex);
        _disposed = true;
    }

    public override DeviceMemoryInfo GetMemoryInfo()
    {
        PhysicalDeviceMemoryProperties memProps;
        _deviceContext.Vk.GetPhysicalDeviceMemoryProperties(_deviceContext.PhysicalDevice, &memProps);

        ulong totalHeapSize = 0;
        ulong usedEstimate = 0;

        // Суммируем все heap'ы, которые являются DEVICE_LOCAL (VRAM)
        for (uint i = 0; i < memProps.MemoryHeapCount; i++)
        {
            var heap = memProps.MemoryHeaps[(int)i];
            // Проверяем, есть ли хотя бы один memory type в этом heap'е с DEVICE_LOCAL
            bool hasDeviceLocal = false;
            for (uint j = 0; j < memProps.MemoryTypeCount; j++)
            {
                if ((memProps.MemoryTypes[(int)j].HeapIndex == i) &&
                    (memProps.MemoryTypes[(int)j].PropertyFlags & MemoryPropertyFlags.DeviceLocalBit) != 0)
                {
                    hasDeviceLocal = true;
                    break;
                }
            }

            if (hasDeviceLocal)
            {
                totalHeapSize += heap.Size;
            }
        }

        // Используем трекер аллокаций как оценку used
        usedEstimate = (ulong)ComputeBufferBase.TotalAllocatedBytes;
        if (usedEstimate > totalHeapSize)
            usedEstimate = totalHeapSize;

        return new DeviceMemoryInfo
        {
            TotalBytes = (long)totalHeapSize,
            AvailableBytes = (long)(totalHeapSize - usedEstimate),
            UsedBytes = (long)usedEstimate,
            TrackedAllocatedBytes = ComputeBufferBase.TotalAllocatedBytes,
            SupportsNativeQuery = true
        };
    }

    internal VulkanDeviceContext DeviceContext => _deviceContext;
}

/// <summary>
/// Information about a Vulkan compute device
/// </summary>
public class DeviceInfo
{
    public int Index { get; set; }
    public string Name { get; set; } = string.Empty;
    public string ApiVersion { get; set; } = string.Empty;
    public string DeviceType { get; set; } = string.Empty;
}
