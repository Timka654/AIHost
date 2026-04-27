using Silk.NET.Vulkan;
using System.Runtime.InteropServices;
using VkBuffer = Silk.NET.Vulkan.Buffer;

namespace AIHost.ICompute.Vulkan;

/// <summary>
/// Буфер вычислений для Vulkan
/// </summary>
internal unsafe class VulkanComputeBuffer : ComputeBufferBase
{
    private readonly Vk _vk;
    private readonly Device _device;
    private readonly ulong _size;
    private readonly BufferType _type;
    private readonly DataType _elementType;
    private VkBuffer _buffer;
    private DeviceMemory _memory;
    private IntPtr _mappedPointer;
    private bool _disposed;

    public override ulong Size => _size;
    public override BufferType Type => _type;
    public override DataType ElementType => _elementType;

    public VulkanComputeBuffer(Vk vk, Device device, PhysicalDevice physicalDevice, ulong size, BufferType type, DataType elementType)
    {
        _vk = vk;
        _device = device;
        _size = size;
        _type = type;
        _elementType = elementType;

        // Создание буфера
        var bufferInfo = new BufferCreateInfo
        {
            SType = StructureType.BufferCreateInfo,
            Size = size,
            Usage = BufferUsageFlags.StorageBufferBit | BufferUsageFlags.TransferSrcBit | BufferUsageFlags.TransferDstBit,
            SharingMode = SharingMode.Exclusive
        };

        if (_vk.CreateBuffer(device, &bufferInfo, null, out _buffer) != Result.Success)
            throw new InvalidOperationException("Failed to create Vulkan buffer");

        // Получение требований к памяти
        MemoryRequirements memRequirements;
        _vk.GetBufferMemoryRequirements(device, _buffer, &memRequirements);

        // Поиск подходящего типа памяти
        PhysicalDeviceMemoryProperties memProperties;
        _vk.GetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);

        uint memoryTypeIndex = FindMemoryType(
            memProperties,
            memRequirements.MemoryTypeBits,
            MemoryPropertyFlags.HostVisibleBit | MemoryPropertyFlags.HostCoherentBit
        );

        // Выделение памяти
        var allocInfo = new MemoryAllocateInfo
        {
            SType = StructureType.MemoryAllocateInfo,
            AllocationSize = memRequirements.Size,
            MemoryTypeIndex = memoryTypeIndex
        };

        if (_vk.AllocateMemory(device, &allocInfo, null, out _memory) != Result.Success)
            throw new InvalidOperationException("Failed to allocate Vulkan memory");

        // Привязка буфера к памяти
        _vk.BindBufferMemory(device, _buffer, _memory, 0);

        // Маппинг памяти
        void* data;
        _vk.MapMemory(device, _memory, 0, size, 0, &data);
        _mappedPointer = (IntPtr)data;
    }

    private static uint FindMemoryType(PhysicalDeviceMemoryProperties memProperties, uint typeFilter, MemoryPropertyFlags properties)
    {
        for (uint i = 0; i < memProperties.MemoryTypeCount; i++)
        {
            if ((typeFilter & (1 << (int)i)) != 0 &&
                (memProperties.MemoryTypes[(int)i].PropertyFlags & properties) == properties)
            {
                return i;
            }
        }

        throw new InvalidOperationException("Failed to find suitable memory type");
    }

    public override IntPtr GetPointer()
    {
        return _mappedPointer;
    }

    public override void Write<T>(T[] data)
    {
        int elementSize = Marshal.SizeOf<T>();
        ulong byteSize = (ulong)elementSize * (ulong)data.Length;

        if (byteSize > _size)
            throw new InvalidOperationException($"Buffer size {_size} is too small for {data.Length} elements");

        fixed (T* src = data)
        {
            System.Buffer.MemoryCopy(src, _mappedPointer.ToPointer(), (long)_size, (long)byteSize);
        }
    }

    public override T[] Read<T>()
    {
        int elementSize = Marshal.SizeOf<T>();
        int elementCount = (int)(_size / (ulong)elementSize);
        T[] result = new T[elementCount];

        fixed (T* dest = result)
        {
            System.Buffer.MemoryCopy(_mappedPointer.ToPointer(), dest, (long)_size, (long)_size);
        }

        return result;
    }

    public override void Dispose()
    {
        if (_disposed) return;

        _vk.UnmapMemory(_device, _memory);
        _vk.FreeMemory(_device, _memory, null);
        _vk.DestroyBuffer(_device, _buffer, null);

        _disposed = true;
    }

    internal VkBuffer VkBuffer => _buffer;
}
