using Silk.NET.Vulkan;
using System.Runtime.InteropServices;
using VkBuffer = Silk.NET.Vulkan.Buffer;

namespace AIHost.ICompute.Vulkan;

/// <summary>
/// Bump-pointer arena allocator для одного большого VkBuffer.
/// Заменяет сотни отдельных vkAllocateMemory/vkFreeMemory вызовов на
/// одну гигантскую аллокацию при старте + O(1) выдачу срезов.
///
/// Жизненный цикл per-frame памяти:
///   BeginFrame → Alloc × N → Flush → WaitForFences → Arena.Reset()
///
/// KV cache регион выделяется отдельно и не сбрасывается через Reset.
/// </summary>
internal unsafe class VulkanArenaAllocator : IDisposable
{
    private readonly Vk _vk;
    private readonly Device _device;
    private readonly PhysicalDevice _physicalDevice;
    private readonly ulong _totalSize;
    private readonly uint _memoryTypeIndex;

    private VkBuffer _arenaBuffer;
    private DeviceMemory _arenaMemory;
    private IntPtr _mappedPointer; // non-null for HOST_VISIBLE tier
    private bool _disposed;

    // ── Bump pointer ────────────────────────────────────────────────────────
    // _cursor points to the start of free space. Alloc moves it forward.
    // _kvCursor is a separate region at the start of the arena for KV cache
    // (persistent across frames — never reset).
    private long _cursor;
    private long _kvCursor;

    // ── Per-frame reset boundary ────────────────────────────────────────────
    private long _frameBase; // snapshot of _cursor before BeginFrame

    public ulong TotalSize => _totalSize;
    public ulong UsedBytes => (ulong)_cursor;
    public ulong FreeBytes => _totalSize - (ulong)_cursor;
    public VkBuffer ArenaBuffer => _arenaBuffer;
    public bool IsHostVisible => _mappedPointer != IntPtr.Zero;
    public IntPtr MappedPointer => _mappedPointer;

    /// <param name="ctx">Vulkan device context.</param>
    /// <param name="sizeBytes">Total arena size. Must be > 0.</param>
    /// <param name="kvCacheBytes">
    /// Bytes to reserve for KV cache at the start of the arena.
    /// This region is never reset — <see cref="KvAlloc"/> manages it separately.
    /// </param>
    public VulkanArenaAllocator(VulkanDeviceContext ctx, ulong sizeBytes, ulong kvCacheBytes = 0)
    {
        _vk = ctx.Vk;
        _device = ctx.Device;
        _physicalDevice = ctx.PhysicalDevice;
        _totalSize = sizeBytes;

        // ── Create the giant buffer ──────────────────────────────────────
        var bufferInfo = new BufferCreateInfo
        {
            SType = StructureType.BufferCreateInfo,
            Size = sizeBytes,
            Usage = BufferUsageFlags.StorageBufferBit
                  | BufferUsageFlags.TransferSrcBit
                  | BufferUsageFlags.TransferDstBit,
            SharingMode = SharingMode.Exclusive
        };

        fixed (VkBuffer* pBuf = &_arenaBuffer)
        {
            if (_vk.CreateBuffer(_device, &bufferInfo, null, pBuf) != Result.Success)
                throw new InvalidOperationException("VulkanArena: Failed to create arena buffer");
        }

        // ── Get memory requirements ──────────────────────────────────────
        MemoryRequirements req;
        _vk.GetBufferMemoryRequirements(_device, _arenaBuffer, &req);

        PhysicalDeviceMemoryProperties memProps;
        _vk.GetPhysicalDeviceMemoryProperties(_physicalDevice, &memProps);

        // Tier 1: DEVICE_LOCAL + HOST_VISIBLE + HOST_COHERENT (ReBAR)
        if (TryFindMemoryType(memProps, req.MemoryTypeBits,
            MemoryPropertyFlags.DeviceLocalBit |
            MemoryPropertyFlags.HostVisibleBit |
            MemoryPropertyFlags.HostCoherentBit,
            out _memoryTypeIndex))
        {
            AllocAndBind(req.Size, _memoryTypeIndex);
            void* ptr;
            _vk.MapMemory(_device, _arenaMemory, 0, req.Size, 0, &ptr);
            _mappedPointer = (IntPtr)ptr;
        }
        // Tier 2: DEVICE_LOCAL only
        else if (TryFindMemoryType(memProps, req.MemoryTypeBits,
            MemoryPropertyFlags.DeviceLocalBit,
            out _memoryTypeIndex))
        {
            AllocAndBind(req.Size, _memoryTypeIndex);
        }
        // Tier 3: HOST_VISIBLE fallback
        else
        {
            _memoryTypeIndex = FindMemoryType(memProps, req.MemoryTypeBits,
                MemoryPropertyFlags.HostVisibleBit | MemoryPropertyFlags.HostCoherentBit);
            AllocAndBind(req.Size, _memoryTypeIndex);
            void* ptr;
            _vk.MapMemory(_device, _arenaMemory, 0, req.Size, 0, &ptr);
            _mappedPointer = (IntPtr)ptr;
        }

        // Reserve KV cache region at the start
        _kvCursor = (long)kvCacheBytes;
        _cursor = _kvCursor;
        _frameBase = _cursor;
    }

    /// <summary>
    /// Allocate a slice of the arena. Thread-safe (Interlocked).
    /// Returns (offset, size) into <see cref="ArenaBuffer"/>.
    /// The caller is responsible for using the offset in VkDescriptorBufferInfo.
    /// </summary>
    public ArenaSlice Alloc(ulong size, ulong alignment = 256)
    {
        // Align to boundary
        ulong alignedOffset = (ulong)Interlocked.Add(ref _cursor, 0);
        ulong misalign = alignedOffset % alignment;
        ulong padding = misalign > 0 ? alignment - misalign : 0;
        ulong offset = (ulong)Interlocked.Add(ref _cursor, (long)(size + padding)) - size;

        if (offset + size > _totalSize)
        {
            // Rollback
            Interlocked.Exchange(ref _cursor, (long)(offset - padding));
            throw new InvalidOperationException(
                $"VulkanArena: out of memory. Requested {size} bytes (aligned {size+padding}), " +
                $"free {FreeBytes} of {TotalSize} bytes total.");
        }

        return new ArenaSlice(_arenaBuffer, offset, size);
    }

    /// <summary>
    /// Allocate from KV cache region (persistent, never reset).
    /// Simple bump pointer — KV cache grows monotonically during generation.
    /// </summary>
    public ArenaSlice KvAlloc(ulong size, ulong alignment = 256)
    {
        ulong misalign = (ulong)_kvCursor % alignment;
        ulong padding = misalign > 0 ? alignment - misalign : 0;
        ulong offset = (ulong)Interlocked.Add(ref _kvCursor, (long)(size + padding)) - size;

        if (offset + size > _totalSize)
            throw new InvalidOperationException(
                $"VulkanArena: KV cache overflow. Requested {size} bytes, " +
                $"free {(ulong)((long)_totalSize - _kvCursor)} of {TotalSize} bytes.");

        // Also advance _cursor past KV region so per-frame allocs don't overlap
        if (_kvCursor > _cursor)
            Interlocked.Exchange(ref _cursor, _kvCursor);

        return new ArenaSlice(_arenaBuffer, offset, size);
    }

    /// <summary>Snapshot cursor for BeginFrame.</summary>
    public void BeginFrame()
    {
        _frameBase = _cursor;
    }

    /// <summary>
    /// Reset per-frame allocations. KV cache region is NOT affected.
    /// Must be called after GPU fence signals completion.
    /// </summary>
    public void Reset()
    {
        _cursor = _frameBase;
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;

        if (_mappedPointer != IntPtr.Zero)
        {
            _vk.UnmapMemory(_device, _arenaMemory);
            _mappedPointer = IntPtr.Zero;
        }

        if (_arenaMemory.Handle != 0)
        {
            _vk.FreeMemory(_device, _arenaMemory, null);
            _arenaMemory = default;
        }

        if (_arenaBuffer.Handle != 0)
        {
            _vk.DestroyBuffer(_device, _arenaBuffer, null);
            _arenaBuffer = default;
        }
    }

    private void AllocAndBind(ulong allocSize, uint memTypeIdx)
    {
        var allocInfo = new MemoryAllocateInfo
        {
            SType = StructureType.MemoryAllocateInfo,
            AllocationSize = allocSize,
            MemoryTypeIndex = memTypeIdx
        };

        fixed (DeviceMemory* pMem = &_arenaMemory)
        {
            if (_vk.AllocateMemory(_device, &allocInfo, null, pMem) != Result.Success)
                throw new InvalidOperationException($"VulkanArena: vkAllocateMemory failed for {allocSize} bytes");
        }

        if (_vk.BindBufferMemory(_device, _arenaBuffer, _arenaMemory, 0) != Result.Success)
            throw new InvalidOperationException("VulkanArena: vkBindBufferMemory failed");
    }

    private static bool TryFindMemoryType(PhysicalDeviceMemoryProperties memProps, uint typeBits,
        MemoryPropertyFlags required, out uint index)
    {
        for (uint i = 0; i < memProps.MemoryTypeCount; i++)
        {
            if ((typeBits & (1u << (int)i)) != 0 &&
                (memProps.MemoryTypes[(int)i].PropertyFlags & required) == required)
            {
                index = i;
                return true;
            }
        }
        index = 0;
        return false;
    }

    private static uint FindMemoryType(PhysicalDeviceMemoryProperties memProps, uint typeBits,
        MemoryPropertyFlags required)
    {
        for (uint i = 0; i < memProps.MemoryTypeCount; i++)
        {
            if ((typeBits & (1u << (int)i)) != 0 &&
                (memProps.MemoryTypes[(int)i].PropertyFlags & required) == required)
                return i;
        }
        throw new InvalidOperationException("VulkanArena: no suitable memory type found");
    }

    /// <summary>Срез арены: (VkBuffer, offset, size).</summary>
    public readonly struct ArenaSlice
    {
        public readonly VkBuffer Buffer;
        public readonly ulong Offset;
        public readonly ulong Size;

        public ArenaSlice(VkBuffer buffer, ulong offset, ulong size)
        {
            Buffer = buffer;
            Offset = offset;
            Size = size;
        }
    }
}
