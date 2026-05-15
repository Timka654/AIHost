using Silk.NET.Vulkan;
using System.Collections.Concurrent;

namespace AIHost.ICompute.Vulkan;

/// <summary>
/// Глобальный контекст Vulkan с ленивым созданием VkInstance и VkDevice.
/// 
/// Принцип работы:
/// - Один VkInstance на весь процесс (лениво, при первом обращении)
/// - Один VkDevice на физический GPU (лениво, при первом запросе устройства)
/// - Несколько очередей команд на VkDevice для псевдо-параллельной обработки
/// - Reference counting: каждый VulkanComputeDevice увеличивает счётчик,
///   при обнулении ресурсы освобождаются
/// 
/// Это решает проблему ErrorDeviceLost при нескольких VulkanComputeDevice
/// на одном физическом GPU — теперь все используют один VkDevice.
/// </summary>
internal unsafe class VulkanGlobalContext
{
    // ── Глобальные ресурсы (один на процесс) ────────────────────────────────
    private static Vk? s_vk;
    private static Instance? s_instance;
    private static bool s_instanceCreated;
    private static readonly object s_instanceLock = new();

    private static readonly ILogger _logger = AppLogger.Create<VulkanGlobalContext>();

    // ── Per-physical-device контексты ────────────────────────────────────────
    private static readonly ConcurrentDictionary<int, VulkanDeviceContext> s_deviceContexts = new();

    // ── Vk API ──────────────────────────────────────────────────────────────

    /// <summary>Получить глобальный Vk API (лениво).</summary>
    public static Vk GetVk()
    {
        if (s_vk == null)
        {
            var vk = Vk.GetApi();
            Interlocked.CompareExchange(ref s_vk, vk, null);
        }
        return s_vk!;
    }

    // ── VkInstance ──────────────────────────────────────────────────────────

    /// <summary>Получить или создать глобальный VkInstance.</summary>
    public static Instance GetOrCreateInstance()
    {
        if (s_instanceCreated)
            return s_instance!.Value;

        lock (s_instanceLock)
        {
            if (s_instanceCreated)
                return s_instance!.Value;

            var vk = GetVk();

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
                    throw new InvalidOperationException("Failed to create Vulkan instance");
            }

            s_instance = instance;
            s_instanceCreated = true;
            _logger.LogDebug("[VulkanGlobal] Instance created");
            return instance;
        }
    }

    // ── Per-device контекст ─────────────────────────────────────────────────

    /// <summary>
    /// Получить или создать VulkanDeviceContext для указанного физического устройства.
    /// Увеличивает reference count.
    /// </summary>
    public static VulkanDeviceContext AcquireDeviceContext(int physicalDeviceIndex, int queueCount = 2)
    {
        if (s_deviceContexts.TryGetValue(physicalDeviceIndex, out var ctx))
        {
            ctx.AddRef();
            return ctx;
        }

        var instance = GetOrCreateInstance();
        var vk = GetVk();

        // Выбор физического устройства
        uint deviceCount = 0;
        vk.EnumeratePhysicalDevices(instance, &deviceCount, null);
        if (deviceCount == 0)
            throw new InvalidOperationException("No Vulkan physical devices found");

        if (physicalDeviceIndex < 0 || physicalDeviceIndex >= deviceCount)
            throw new ArgumentOutOfRangeException(nameof(physicalDeviceIndex),
                $"Device index {physicalDeviceIndex} out of range (0-{deviceCount - 1})");

        var physicalDevices = stackalloc PhysicalDevice[(int)deviceCount];
        vk.EnumeratePhysicalDevices(instance, &deviceCount, physicalDevices);
        var physicalDevice = physicalDevices[physicalDeviceIndex];

        // Получение свойств устройства
        PhysicalDeviceProperties props;
        vk.GetPhysicalDeviceProperties(physicalDevice, &props);
        string deviceName = System.Runtime.InteropServices.Marshal.PtrToStringAnsi((nint)props.DeviceName) ?? "Unknown";

        // Поиск compute queue family
        uint queueFamilyCount = 0;
        vk.GetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, null);
        var queueFamilies = stackalloc QueueFamilyProperties[(int)queueFamilyCount];
        vk.GetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, queueFamilies);

        uint queueFamilyIndex = uint.MaxValue;
        for (uint i = 0; i < queueFamilyCount; i++)
        {
            if ((queueFamilies[i].QueueFlags & QueueFlags.ComputeBit) != 0)
            {
                queueFamilyIndex = i;
                break;
            }
        }

        if (queueFamilyIndex == uint.MaxValue)
            throw new InvalidOperationException("No compute-capable queue family found");

        // Определяем максимальное количество очередей, которое поддерживает family
        uint maxQueueCount = queueFamilies[queueFamilyIndex].QueueCount;
        uint actualQueueCount = (uint)Math.Min(Math.Max(1, queueCount), (int)maxQueueCount);

        // Создание логического устройства с несколькими очередями
        var priorities = stackalloc float[(int)actualQueueCount];
        for (int i = 0; i < actualQueueCount; i++)
            priorities[i] = 1.0f;

        var queueCreateInfo = new DeviceQueueCreateInfo
        {
            SType = StructureType.DeviceQueueCreateInfo,
            QueueFamilyIndex = queueFamilyIndex,
            QueueCount = actualQueueCount,
            PQueuePriorities = priorities
        };

        var deviceFeatures = new PhysicalDeviceFeatures();

        var deviceCreateInfo = new DeviceCreateInfo
        {
            SType = StructureType.DeviceCreateInfo,
            QueueCreateInfoCount = 1,
            PQueueCreateInfos = &queueCreateInfo,
            PEnabledFeatures = &deviceFeatures
        };

        if (vk.CreateDevice(physicalDevice, &deviceCreateInfo, null, out var device) != Result.Success)
            throw new InvalidOperationException("Failed to create Vulkan device");

        // Получение всех очередей
        var queues = new Queue[actualQueueCount];
        for (int i = 0; i < actualQueueCount; i++)
        {
            vk.GetDeviceQueue(device, queueFamilyIndex, (uint)i, out queues[i]);
        }

        ctx = new VulkanDeviceContext
        {
            Vk = vk,
            Device = device,
            PhysicalDevice = physicalDevice,
            QueueFamilyIndex = queueFamilyIndex,
            Queues = queues,
            DeviceName = deviceName,
            PhysicalDeviceIndex = physicalDeviceIndex
        };
        ctx.SetInitialRefCount(1);

        s_deviceContexts[physicalDeviceIndex] = ctx;

        _logger.LogDebug($"[VulkanGlobal] Device context created: {deviceName} (idx={physicalDeviceIndex}, queues={actualQueueCount})");

        return ctx;
    }

    /// <summary>
    /// Освободить reference на контекст устройства.
    /// Когда счётчик достигает 0, VkDevice уничтожается.
    /// </summary>
    public static void ReleaseDeviceContext(int physicalDeviceIndex)
    {
        if (!s_deviceContexts.TryGetValue(physicalDeviceIndex, out var ctx))
            return;

        if (ctx.ReleaseRef() > 0)
            return;

        // Счётчик = 0, уничтожаем
        if (s_deviceContexts.TryRemove(physicalDeviceIndex, out ctx))
        {
            ctx.Vk.DeviceWaitIdle(ctx.Device);
            ctx.Vk.DestroyDevice(ctx.Device, null);
            _logger.LogDebug($"[VulkanGlobal] Device context destroyed: {ctx.DeviceName} (idx={physicalDeviceIndex})");
        }
    }

    /// <summary>
    /// Полностью уничтожить глобальный контекст (при завершении процесса).
    /// </summary>
    public static void DestroyAll()
    {
        // Уничтожаем все device контексты
        foreach (var kvp in s_deviceContexts)
        {
            var ctx = kvp.Value;
            ctx.Vk.DeviceWaitIdle(ctx.Device);
            ctx.Vk.DestroyDevice(ctx.Device, null);
        }
        s_deviceContexts.Clear();

        // Уничтожаем instance
        if (s_instanceCreated && s_instance.HasValue)
        {
            GetVk().DestroyInstance(s_instance.Value, null);
            s_instanceCreated = false;
            s_instance = null;
            _logger.LogDebug("[VulkanGlobal] Instance destroyed");
        }

        // Освобождаем Vk API
        var vk = s_vk;
        if (vk != null)
        {
            vk.Dispose();
            s_vk = null;
        }
    }
}

/// <summary>
/// Контекст одного логического Vulkan устройства.
/// Содержит VkDevice и массив очередей команд.
/// </summary>
internal sealed class VulkanDeviceContext
{
    public required Vk Vk { get; init; }
    public required Silk.NET.Vulkan.Device Device { get; init; }
    public required PhysicalDevice PhysicalDevice { get; init; }
    public required uint QueueFamilyIndex { get; init; }
    public required Queue[] Queues { get; init; }
    public required string DeviceName { get; init; }
    public required int PhysicalDeviceIndex { get; init; }

    private int _refCount;
    public int RefCount => _refCount;

    /// <summary>Количество очередей команд.</summary>
    public int QueueCount => Queues.Length;

    /// <summary>Получить очередь по индексу (циклически, thread-safe).</summary>
    public Queue GetQueue(int index) => Queues[index % Queues.Length];

    public void AddRef() => Interlocked.Increment(ref _refCount);
    public int ReleaseRef() => Interlocked.Decrement(ref _refCount);

    /// <summary>Установить начальное значение refcount (только при создании).</summary>
    internal void SetInitialRefCount(int count) => _refCount = count;
}
