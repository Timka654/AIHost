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
    private ulong _size;
    private BufferType _type;
    private DataType _elementType;

    private VkBuffer _buffer;
    private DeviceMemory _memory;
    private IntPtr _mappedPointer;   // non-null only for HOST_VISIBLE paths
    private bool _disposed;

    public override ulong Size => _size;
    public override BufferType Type => _type;
    public override DataType ElementType => _elementType;

    public VulkanComputeBuffer(
        VulkanDeviceContext ctx,
        ulong size, BufferType type, DataType elementType,
        bool requireDeviceLocal = false)
        : this(ctx.Vk, ctx.Device, ctx.PhysicalDevice, ctx.GetQueue(0), ctx.QueueFamilyIndex,
              size, type, elementType, requireDeviceLocal)
    {
    }

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

        var createResult = _vk.CreateBuffer(device, &bufferInfo, null, out _buffer);
        if (createResult != Result.Success)
            throw new InvalidOperationException(
                $"Failed to create Vulkan buffer: {createResult} (size={size / (1024.0 * 1024.0):F1}MB, type={type}, elementType={elementType})");

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
            TrackAllocate(req.Size);
            return;
        }

        // --- Tier 2: DEVICE_LOCAL only (fast VRAM; staging for CPU transfers) ---
        if (TryFindMemoryType(memProps, req.MemoryTypeBits,
            MemoryPropertyFlags.DeviceLocalBit,
            out typeIdx))
        {
            try
            {
                AllocAndBind(req.Size, typeIdx);
                // _mappedPointer stays IntPtr.Zero — Write/Read use staging
                TrackAllocate(req.Size);
                return;
            }
            catch (InvalidOperationException)
            {
                // FIX: On APUs (Renoir), DEVICE_LOCAL heap is small (512MB–1GB).
                // 859 weight buffers fill it completely. Large temp allocations
                // (Dequantize 3.1 GB) fail here but have 25.8 GB free in HOST_VISIBLE.
                // Fall through to tier 3 instead of crashing.
            }
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
        TrackAllocate(req.Size);
    }

    private void AllocAndBind(ulong allocSize, uint memTypeIdx)
    {
        var allocInfo = new MemoryAllocateInfo
        {
            SType = StructureType.MemoryAllocateInfo,
            AllocationSize = allocSize,
            MemoryTypeIndex = memTypeIdx
        };

        var allocResult = _vk.AllocateMemory(_device, &allocInfo, null, out _memory);
        if (allocResult != Result.Success)
            throw new InvalidOperationException($"Failed to allocate Vulkan memory: {allocResult}");

        var bindResult = _vk.BindBufferMemory(_device, _buffer, _memory, 0);
        if (bindResult != Result.Success)
            throw new InvalidOperationException($"Failed to bind buffer memory: {bindResult}");
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
        // DeviceWaitIdle ensures the staging copy is fully visible to all queues
        // before any subsequent dispatch on a different queue reads this buffer.
        // Without this, a dispatch on queue 1 may read stale data if the staging
        // copy was submitted on queue 0 (different queue, no implicit ordering).
        _vk.DeviceWaitIdle(_device);
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

        if (_vk.CreateBuffer(_device, &bufInfo, null, out var stagingBuf) != Result.Success)
            throw new InvalidOperationException("Failed to create staging buffer");

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

        if (_vk.AllocateMemory(_device, &allocInfo, null, out var stagingMem) != Result.Success)
            throw new InvalidOperationException("Failed to allocate staging memory");

        if (_vk.BindBufferMemory(_device, stagingBuf, stagingMem, 0) != Result.Success)
            throw new InvalidOperationException("Failed to bind staging buffer memory");

        void* ptr;
        if (_vk.MapMemory(_device, stagingMem, 0, size, 0, &ptr) != Result.Success)
            throw new InvalidOperationException("Failed to map staging memory");

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
        if (IsArenaView) { _disposed = true; return; } // arena owns the memory

        // Get memory requirements BEFORE destroying the buffer
        MemoryRequirements req;
        _vk.GetBufferMemoryRequirements(_device, _buffer, &req);

        if (_mappedPointer != IntPtr.Zero)
            _vk.UnmapMemory(_device, _memory);

        _vk.DestroyBuffer(_device, _buffer, null);
        _vk.FreeMemory(_device, _memory, null);

        TrackFree(req.Size);

        _disposed = true;
    }

    internal VkBuffer VkBuffer => _buffer;
    internal ulong ArenaOffset { get; private set; }

    /// <summary>
    /// Create a view into an arena buffer. No memory allocation — just records
    /// the slice (buffer, offset, size) for descriptor binding via DescriptorBufferInfo.
    /// Dispose is no-op (arena owns the memory).
    /// </summary>
    internal static VulkanComputeBuffer CreateArenaView(VkBuffer arenaBuffer, ulong offset, ulong size, DataType dtype)
    {
        return new VulkanComputeBuffer
        {
            _buffer = arenaBuffer,
            _size = size,
            ArenaOffset = offset,
            _type = BufferType.Storage,
            _elementType = dtype,
        };
    }

    // Private ctor for arena views — skip all vkCreateBuffer/vkAllocateMemory
    private VulkanComputeBuffer() { }

    /// <summary>Skip dispose for arena views (memory owned by VulkanArenaAllocator).</summary>
    public bool IsArenaView => _memory.Handle == 0;
}
