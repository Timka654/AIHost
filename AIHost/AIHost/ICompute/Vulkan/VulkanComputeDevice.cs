using Silk.NET.Vulkan;

namespace AIHost.ICompute.Vulkan;

/// <summary>
/// Провайдер вычислений на основе Vulkan
/// </summary>
public unsafe class VulkanComputeDevice : ComputeProviderBase
{
    private readonly Vk _vk;
    private readonly Instance _instance;
    private readonly PhysicalDevice _physicalDevice;
    private readonly Device _device;
    private readonly Queue _queue;
    private readonly uint _queueFamilyIndex;
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
        _vk = Vk.GetApi();

        // Создание Vulkan Instance
        var appNameBytes = System.Text.Encoding.UTF8.GetBytes("AIHost\0");
        var engineNameBytes = System.Text.Encoding.UTF8.GetBytes("AIHost Engine\0");
        
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

            if (_vk.CreateInstance(&createInfo, null, out _instance) != Result.Success)
                throw new InvalidOperationException("Failed to create Vulkan instance");
        }

        // Выбор физического устройства
        uint deviceCount = 0;
        _vk.EnumeratePhysicalDevices(_instance, &deviceCount, null);
        if (deviceCount == 0)
            throw new InvalidOperationException("No Vulkan physical devices found");

        if (deviceIndex < 0 || deviceIndex >= deviceCount)
            throw new ArgumentOutOfRangeException(nameof(deviceIndex), $"Device index {deviceIndex} out of range (0-{deviceCount - 1})");

        var physicalDevices = stackalloc PhysicalDevice[(int)deviceCount];
        _vk.EnumeratePhysicalDevices(_instance, &deviceCount, physicalDevices);
        _physicalDevice = physicalDevices[deviceIndex];
        DeviceIndex = deviceIndex;

        // Получение свойств устройства
        PhysicalDeviceProperties props;
        _vk.GetPhysicalDeviceProperties(_physicalDevice, &props);
        
        uint apiVer = props.ApiVersion;
        ApiVersion = $"{apiVer >> 22}.{(apiVer >> 12) & 0x3ff}.{apiVer & 0xfff}";
        DeviceName = System.Runtime.InteropServices.Marshal.PtrToStringAnsi((nint)props.DeviceName) ?? "Unknown";
        
        Console.WriteLine($"Vulkan Device [{deviceIndex}]: {DeviceName}");
        Console.WriteLine($"API Version: {ApiVersion}");

        // Поиск compute queue family
        uint queueFamilyCount = 0;
        _vk.GetPhysicalDeviceQueueFamilyProperties(_physicalDevice, &queueFamilyCount, null);
        var queueFamilies = stackalloc QueueFamilyProperties[(int)queueFamilyCount];
        _vk.GetPhysicalDeviceQueueFamilyProperties(_physicalDevice, &queueFamilyCount, queueFamilies);

        _queueFamilyIndex = uint.MaxValue;
        for (uint i = 0; i < queueFamilyCount; i++)
        {
            if ((queueFamilies[i].QueueFlags & QueueFlags.ComputeBit) != 0)
            {
                _queueFamilyIndex = i;
                break;
            }
        }

        if (_queueFamilyIndex == uint.MaxValue)
            throw new InvalidOperationException("No compute-capable queue family found");

        // Создание логического устройства
        float queuePriority = 1.0f;
        var queueCreateInfo = new DeviceQueueCreateInfo
        {
            SType = StructureType.DeviceQueueCreateInfo,
            QueueFamilyIndex = _queueFamilyIndex,
            QueueCount = 1,
            PQueuePriorities = &queuePriority
        };

        var deviceFeatures = new PhysicalDeviceFeatures();

        var deviceCreateInfo = new DeviceCreateInfo
        {
            SType = StructureType.DeviceCreateInfo,
            QueueCreateInfoCount = 1,
            PQueueCreateInfos = &queueCreateInfo,
            PEnabledFeatures = &deviceFeatures
        };

        if (_vk.CreateDevice(_physicalDevice, &deviceCreateInfo, null, out _device) != Result.Success)
            throw new InvalidOperationException("Failed to create Vulkan device");

        // Получение очереди
        _vk.GetDeviceQueue(_device, _queueFamilyIndex, 0, out _queue);
    }

    public override IComputeBuffer CreateBuffer(ulong size, BufferType type, DataType elementType = DataType.F32, bool requireDeviceLocal = false)
    {
        return new VulkanComputeBuffer(_vk, _device, _physicalDevice, _queue, _queueFamilyIndex, size, type, elementType, requireDeviceLocal);
    }

    public override IComputeKernel CreateKernel(string source, string entryPoint)
    {
        return new VulkanComputeKernel(_vk, _device, source, entryPoint);
    }

    public override IComputeKernel CreateKernelFromFile(string filePath, string entryPoint)
    {
        var source = File.ReadAllText(filePath);
        return new VulkanComputeKernel(_vk, _device, source, entryPoint);
    }

    public override IComputeCommandQueue CreateCommandQueue()
    {
        return new VulkanComputeCommandQueue(_vk, _device, _queue, _queueFamilyIndex);
    }

    public override void Synchronize()
    {
        _vk.DeviceWaitIdle(_device);
    }

    public override void Dispose()
    {
        if (_disposed) return;

        _vk.DeviceWaitIdle(_device);
        _vk.DestroyDevice(_device, null);
        _vk.DestroyInstance(_instance, null);
        _vk.Dispose();

        _disposed = true;
    }

    internal Vk VkApi => _vk;
    internal Device Device => _device;
    internal PhysicalDevice PhysicalDevice => _physicalDevice;
    internal Queue Queue => _queue;
    internal uint QueueFamilyIndex => _queueFamilyIndex;
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
