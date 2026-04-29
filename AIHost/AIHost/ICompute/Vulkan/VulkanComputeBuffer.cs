using Silk.NET.Vulkan;
using System.Runtime.InteropServices;
using VkBuffer = Silk.NET.Vulkan.Buffer;

namespace AIHost.ICompute.Vulkan;

/// <summary>
/// Vulkan compute buffer with tiered memory allocation:
///   1. DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT  (ReBAR/SAM — VRAM, CPU-mappable, zero overhead)
///   2. DEVICE_LOCAL only                            (VRAM, fast GPU; Write/Read go through a staging buffer)
///   3. HOST_VISIBLE | HOST_COHERENT                 (system RAM fallback; always works)
/// </summary>
internal unsafe class VulkanComputeBuffer : ComputeBufferBase
{
    private readonly Vk _vk;
    private readonly Device _device;
    private readonly PhysicalDevice _physicalDevice;
    private readonly Queue _queue;
    private readonly uint _queueFamilyIndex;
    private readonly ulong _size;
    private readonly BufferType _type;
    private readonly DataType _elementType;

    private VkBuffer _buffer;
    private DeviceMemory _memory;
    private IntPtr _mappedPointer;   // non-null only for HOST_VISIBLE paths
    private bool _disposed;

    public override ulong Size => _size;
    public override BufferType Type => _type;
    public override DataType ElementType => _elementType;

    public VulkanComputeBuffer(
        Vk vk, Device device, PhysicalDevice physicalDevice,
        Queue queue, uint queueFamilyIndex,
        ulong size, BufferType type, DataType elementType,
        bool requireDeviceLocal = false)
    {
        _vk = vk;
        _device = device;
        _physicalDevice = physicalDevice;
        _queue = queue;
        _queueFamilyIndex = queueFamilyIndex;
        _size = size;
        _type = type;
        _elementType = elementType;

        var bufferInfo = new BufferCreateInfo
        {
            SType = StructureType.BufferCreateInfo,
            Size = size,
            Usage = BufferUsageFlags.StorageBufferBit
                  | BufferUsageFlags.TransferSrcBit
                  | BufferUsageFlags.TransferDstBit,
            SharingMode = SharingMode.Exclusive
        };

        if (_vk.CreateBuffer(device, &bufferInfo, null, out _buffer) != Result.Success)
            throw new InvalidOperationException("Failed to create Vulkan buffer");

        MemoryRequirements req;
        _vk.GetBufferMemoryRequirements(device, _buffer, &req);

        PhysicalDeviceMemoryProperties memProps;
        _vk.GetPhysicalDeviceMemoryProperties(physicalDevice, &memProps);

        // --- Tier 1: DEVICE_LOCAL + HOST_VISIBLE + HOST_COHERENT (ReBAR/SAM) ---
        if (TryFindMemoryType(memProps, req.MemoryTypeBits,
            MemoryPropertyFlags.DeviceLocalBit |
            MemoryPropertyFlags.HostVisibleBit |
            MemoryPropertyFlags.HostCoherentBit,
            out uint typeIdx))
        {
            AllocAndBind(req.Size, typeIdx);
            void* ptr;
            _vk.MapMemory(device, _memory, 0, size, 0, &ptr);
            _mappedPointer = (IntPtr)ptr;
            return;
        }

        // --- Tier 2: DEVICE_LOCAL only (fast VRAM; staging for CPU transfers) ---
        if (TryFindMemoryType(memProps, req.MemoryTypeBits,
            MemoryPropertyFlags.DeviceLocalBit,
            out typeIdx))
        {
            AllocAndBind(req.Size, typeIdx);
            // _mappedPointer stays IntPtr.Zero — Write/Read use staging
            return;
        }

        // --- Tier 3: HOST_VISIBLE + HOST_COHERENT (system RAM / shared GPU memory) ---
        if (requireDeviceLocal)
            throw new InsufficientVramException(size, "Vulkan");

        typeIdx = FindMemoryType(memProps, req.MemoryTypeBits,
            MemoryPropertyFlags.HostVisibleBit | MemoryPropertyFlags.HostCoherentBit);
        AllocAndBind(req.Size, typeIdx);
        void* ptr2;
        _vk.MapMemory(device, _memory, 0, size, 0, &ptr2);
        _mappedPointer = (IntPtr)ptr2;
    }

    private void AllocAndBind(ulong allocSize, uint memTypeIdx)
    {
        var allocInfo = new MemoryAllocateInfo
        {
            SType = StructureType.MemoryAllocateInfo,
            AllocationSize = allocSize,
            MemoryTypeIndex = memTypeIdx
        };

        if (_vk.AllocateMemory(_device, &allocInfo, null, out _memory) != Result.Success)
            throw new InvalidOperationException("Failed to allocate Vulkan memory");

        _vk.BindBufferMemory(_device, _buffer, _memory, 0);
    }

    // ── Public API ──────────────────────────────────────────────────────────────

    public override IntPtr GetPointer() => _mappedPointer;

    public override void Write<T>(T[] data)
    {
        int elementSize = Marshal.SizeOf<T>();
        ulong byteCount = (ulong)(elementSize * data.Length);

        if (byteCount > _size)
            throw new InvalidOperationException($"Data ({byteCount} B) exceeds buffer ({_size} B)");

        if (_mappedPointer != IntPtr.Zero)
        {
            // HOST_VISIBLE: direct CPU write
            fixed (T* src = data)
                System.Buffer.MemoryCopy(src, _mappedPointer.ToPointer(), (long)_size, (long)byteCount);
        }
        else
        {
            // DEVICE_LOCAL only: upload via staging buffer
            fixed (T* src = data)
                UploadStaging((byte*)src, byteCount);
        }
    }

    public override T[] Read<T>()
    {
        int elementSize = Marshal.SizeOf<T>();
        int count = (int)(_size / (ulong)elementSize);
        T[] result = new T[count];

        if (_mappedPointer != IntPtr.Zero)
        {
            fixed (T* dst = result)
                System.Buffer.MemoryCopy(_mappedPointer.ToPointer(), dst, (long)_size, (long)_size);
        }
        else
        {
            fixed (T* dst = result)
                DownloadStaging((byte*)dst, _size);
        }

        return result;
    }

    public override T[] ReadRange<T>(ulong byteOffset, int elementCount)
    {
        int elementSize = Marshal.SizeOf<T>();
        ulong byteCount = (ulong)(elementSize * elementCount);
        T[] result = new T[elementCount];

        if (_mappedPointer != IntPtr.Zero)
        {
            // HOST_VISIBLE: direct CPU read at offset
            fixed (T* dst = result)
                System.Buffer.MemoryCopy(
                    ((byte*)_mappedPointer.ToPointer()) + byteOffset,
                    dst, (long)byteCount, (long)byteCount);
        }
        else
        {
            // DEVICE_LOCAL: partial staging copy
            fixed (T* dst = result)
                DownloadStagingRange((byte*)dst, byteOffset, byteCount);
        }

        return result;
    }

    private void DownloadStagingRange(byte* dst, ulong srcByteOffset, ulong byteCount)
    {
        var (stagingBuf, stagingMem, stagingPtr) = CreateStagingBuffer(byteCount, upload: false);
        try
        {
            CopyBufferRange(_buffer, stagingBuf, srcByteOffset, 0, byteCount);
            System.Buffer.MemoryCopy(stagingPtr.ToPointer(), dst, (long)byteCount, (long)byteCount);
        }
        finally
        {
            _vk.UnmapMemory(_device, stagingMem);
            _vk.FreeMemory(_device, stagingMem, null);
            _vk.DestroyBuffer(_device, stagingBuf, null);
        }
    }

    private void CopyBufferRange(VkBuffer src, VkBuffer dst, ulong srcOffset, ulong dstOffset, ulong size)
    {
        var poolInfo = new CommandPoolCreateInfo
        {
            SType = StructureType.CommandPoolCreateInfo,
            Flags = CommandPoolCreateFlags.TransientBit,
            QueueFamilyIndex = _queueFamilyIndex
        };
        _vk.CreateCommandPool(_device, &poolInfo, null, out var cmdPool);

        var allocInfo = new CommandBufferAllocateInfo
        {
            SType = StructureType.CommandBufferAllocateInfo,
            CommandPool = cmdPool,
            Level = CommandBufferLevel.Primary,
            CommandBufferCount = 1
        };
        CommandBuffer cmdBuf;
        _vk.AllocateCommandBuffers(_device, &allocInfo, &cmdBuf);

        var beginInfo = new CommandBufferBeginInfo
        {
            SType = StructureType.CommandBufferBeginInfo,
            Flags = CommandBufferUsageFlags.OneTimeSubmitBit
        };
        _vk.BeginCommandBuffer(cmdBuf, &beginInfo);

        var region = new BufferCopy { SrcOffset = srcOffset, DstOffset = dstOffset, Size = size };
        _vk.CmdCopyBuffer(cmdBuf, src, dst, 1, &region);

        _vk.EndCommandBuffer(cmdBuf);

        var fenceInfo = new FenceCreateInfo { SType = StructureType.FenceCreateInfo };
        _vk.CreateFence(_device, &fenceInfo, null, out var fence);

        var submitInfo = new SubmitInfo
        {
            SType = StructureType.SubmitInfo,
            CommandBufferCount = 1,
            PCommandBuffers = &cmdBuf
        };
        _vk.QueueSubmit(_queue, 1, &submitInfo, fence);

        var fenceLocal = fence;
        _vk.WaitForFences(_device, 1, &fenceLocal, true, ulong.MaxValue);
        _vk.DestroyFence(_device, fence, null);
        _vk.FreeCommandBuffers(_device, cmdPool, 1, &cmdBuf);
        _vk.DestroyCommandPool(_device, cmdPool, null);
    }

    // ── Staging helpers ─────────────────────────────────────────────────────────

    private void UploadStaging(byte* src, ulong byteCount)
    {
        var (stagingBuf, stagingMem, stagingPtr) = CreateStagingBuffer(byteCount, upload: true);
        try
        {
            System.Buffer.MemoryCopy(src, stagingPtr.ToPointer(), (long)byteCount, (long)byteCount);
            CopyBuffer(stagingBuf, _buffer, byteCount);
        }
        finally
        {
            _vk.UnmapMemory(_device, stagingMem);
            _vk.FreeMemory(_device, stagingMem, null);
            _vk.DestroyBuffer(_device, stagingBuf, null);
        }
    }

    private void DownloadStaging(byte* dst, ulong byteCount)
    {
        var (stagingBuf, stagingMem, stagingPtr) = CreateStagingBuffer(byteCount, upload: false);
        try
        {
            CopyBuffer(_buffer, stagingBuf, byteCount);
            System.Buffer.MemoryCopy(stagingPtr.ToPointer(), dst, (long)byteCount, (long)byteCount);
        }
        finally
        {
            _vk.UnmapMemory(_device, stagingMem);
            _vk.FreeMemory(_device, stagingMem, null);
            _vk.DestroyBuffer(_device, stagingBuf, null);
        }
    }

    private (VkBuffer buf, DeviceMemory mem, IntPtr ptr) CreateStagingBuffer(ulong size, bool upload)
    {
        var usage = upload
            ? BufferUsageFlags.TransferSrcBit
            : BufferUsageFlags.TransferDstBit;

        var bufInfo = new BufferCreateInfo
        {
            SType = StructureType.BufferCreateInfo,
            Size = size,
            Usage = usage,
            SharingMode = SharingMode.Exclusive
        };

        _vk.CreateBuffer(_device, &bufInfo, null, out var stagingBuf);

        MemoryRequirements req;
        _vk.GetBufferMemoryRequirements(_device, stagingBuf, &req);

        PhysicalDeviceMemoryProperties memProps;
        _vk.GetPhysicalDeviceMemoryProperties(_physicalDevice, &memProps);

        uint typeIdx = FindMemoryType(memProps, req.MemoryTypeBits,
            MemoryPropertyFlags.HostVisibleBit | MemoryPropertyFlags.HostCoherentBit);

        var allocInfo = new MemoryAllocateInfo
        {
            SType = StructureType.MemoryAllocateInfo,
            AllocationSize = req.Size,
            MemoryTypeIndex = typeIdx
        };

        _vk.AllocateMemory(_device, &allocInfo, null, out var stagingMem);
        _vk.BindBufferMemory(_device, stagingBuf, stagingMem, 0);

        void* ptr;
        _vk.MapMemory(_device, stagingMem, 0, size, 0, &ptr);

        return (stagingBuf, stagingMem, (IntPtr)ptr);
    }

    private void CopyBuffer(VkBuffer src, VkBuffer dst, ulong size)
    {
        // One-shot command pool for the transfer
        var poolInfo = new CommandPoolCreateInfo
        {
            SType = StructureType.CommandPoolCreateInfo,
            Flags = CommandPoolCreateFlags.TransientBit,
            QueueFamilyIndex = _queueFamilyIndex
        };
        _vk.CreateCommandPool(_device, &poolInfo, null, out var cmdPool);

        var allocInfo = new CommandBufferAllocateInfo
        {
            SType = StructureType.CommandBufferAllocateInfo,
            CommandPool = cmdPool,
            Level = CommandBufferLevel.Primary,
            CommandBufferCount = 1
        };
        CommandBuffer cmdBuf;
        _vk.AllocateCommandBuffers(_device, &allocInfo, &cmdBuf);

        var beginInfo = new CommandBufferBeginInfo
        {
            SType = StructureType.CommandBufferBeginInfo,
            Flags = CommandBufferUsageFlags.OneTimeSubmitBit
        };
        _vk.BeginCommandBuffer(cmdBuf, &beginInfo);

        var region = new BufferCopy { Size = size };
        _vk.CmdCopyBuffer(cmdBuf, src, dst, 1, &region);

        _vk.EndCommandBuffer(cmdBuf);

        var fenceInfo = new FenceCreateInfo { SType = StructureType.FenceCreateInfo };
        _vk.CreateFence(_device, &fenceInfo, null, out var fence);

        var submitInfo = new SubmitInfo
        {
            SType = StructureType.SubmitInfo,
            CommandBufferCount = 1,
            PCommandBuffers = &cmdBuf
        };
        _vk.QueueSubmit(_queue, 1, &submitInfo, fence);

        var fenceLocal = fence;
        _vk.WaitForFences(_device, 1, &fenceLocal, true, ulong.MaxValue);
        _vk.DestroyFence(_device, fence, null);
        _vk.FreeCommandBuffers(_device, cmdPool, 1, &cmdBuf);
        _vk.DestroyCommandPool(_device, cmdPool, null);
    }

    // ── Memory type helpers ─────────────────────────────────────────────────────

    private static bool TryFindMemoryType(
        PhysicalDeviceMemoryProperties props, uint filter,
        MemoryPropertyFlags required, out uint index)
    {
        for (uint i = 0; i < props.MemoryTypeCount; i++)
        {
            if ((filter & (1u << (int)i)) != 0 &&
                (props.MemoryTypes[(int)i].PropertyFlags & required) == required)
            {
                index = i;
                return true;
            }
        }
        index = 0;
        return false;
    }

    private static uint FindMemoryType(
        PhysicalDeviceMemoryProperties props, uint filter, MemoryPropertyFlags required)
    {
        if (!TryFindMemoryType(props, filter, required, out uint idx))
            throw new InvalidOperationException($"No suitable Vulkan memory type found for flags {required}");
        return idx;
    }

    // ── Disposal ────────────────────────────────────────────────────────────────

    public override void Dispose()
    {
        if (_disposed) return;

        if (_mappedPointer != IntPtr.Zero)
            _vk.UnmapMemory(_device, _memory);

        _vk.FreeMemory(_device, _memory, null);
        _vk.DestroyBuffer(_device, _buffer, null);

        _disposed = true;
    }

    internal VkBuffer VkBuffer => _buffer;
}
