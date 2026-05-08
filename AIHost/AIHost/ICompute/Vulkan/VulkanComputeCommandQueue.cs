using Silk.NET.Vulkan;

namespace AIHost.ICompute.Vulkan;

/// <summary>
/// Очередь команд для Vulkan с пулом command buffers для поддержки
/// параллельных контекстов (thread-safe через слоты).
/// 
/// Каждый поток получает свой слот command buffer и свою очередь команд
/// из пула очередей VulkanDeviceContext. Это позволяет нескольким потокам
/// отправлять команды параллельно без ErrorDeviceLost.
/// </summary>
internal unsafe class VulkanComputeCommandQueue : ComputeCommandQueueBase
{
    private readonly Vk _vk;
    private readonly Device _device;
    private readonly VulkanDeviceContext _deviceContext;
    private readonly CommandPool _commandPool;

    /// <summary>Пул command buffers + fences для параллельного доступа.</summary>
    private readonly CommandBuffer[] _commandBuffers;
    private readonly Fence[] _fences;
    private readonly int _poolSize;

    /// <summary>Thread-local slot index — каждый поток использует свой слот.</summary>
    [ThreadStatic]
    private static int? t_slotIndex;

    /// <summary>Глобальный счётчик для распределения слотов между потоками.</summary>
    private static int s_nextSlot;

    /// <summary>Флаг, что текущий поток начал запись.</summary>
    [ThreadStatic]
    private static bool t_isRecording;

    private bool _disposed;

    internal VulkanComputeCommandQueue(VulkanDeviceContext ctx, int poolSize = 4)
    {
        _deviceContext = ctx;
        _vk = ctx.Vk;
        _device = ctx.Device;
        _poolSize = Math.Max(1, poolSize);

        // Создание command pool
        var poolInfo = new CommandPoolCreateInfo
        {
            SType = StructureType.CommandPoolCreateInfo,
            Flags = CommandPoolCreateFlags.ResetCommandBufferBit,
            QueueFamilyIndex = ctx.QueueFamilyIndex
        };

        if (_vk.CreateCommandPool(_device, &poolInfo, null, out _commandPool) != Result.Success)
            throw new InvalidOperationException("Failed to create command pool");

        // Выделение пула command buffers
        _commandBuffers = new CommandBuffer[_poolSize];
        _fences = new Fence[_poolSize];

        var allocInfo = new CommandBufferAllocateInfo
        {
            SType = StructureType.CommandBufferAllocateInfo,
            CommandPool = _commandPool,
            Level = CommandBufferLevel.Primary,
            CommandBufferCount = (uint)_poolSize
        };

        fixed (CommandBuffer* pCmdBuffers = _commandBuffers)
        {
            if (_vk.AllocateCommandBuffers(_device, &allocInfo, pCmdBuffers) != Result.Success)
                throw new InvalidOperationException("Failed to allocate command buffers");
        }

        // Создание fences для каждого слота
        for (int i = 0; i < _poolSize; i++)
        {
            var fenceInfo = new FenceCreateInfo
            {
                SType = StructureType.FenceCreateInfo,
                Flags = FenceCreateFlags.SignaledBit
            };

            if (_vk.CreateFence(_device, &fenceInfo, null, out _fences[i]) != Result.Success)
                throw new InvalidOperationException($"Failed to create fence for slot {i}");
        }
    }

    /// <summary>Получить или назначить слот для текущего потока.</summary>
    private int GetSlot()
    {
        if (t_slotIndex.HasValue)
            return t_slotIndex.Value;

        int slot = Interlocked.Increment(ref s_nextSlot) % _poolSize;
        t_slotIndex = slot;
        return slot;
    }

    /// <summary>Получить очередь команд для текущего потока (на основе слота).</summary>
    private Queue GetThreadQueue()
    {
        int slot = GetSlot();
        // Используем разные очереди из пула VulkanDeviceContext для разных потоков
        return _deviceContext.GetQueue(slot);
    }

    /// <summary>Начать запись в command buffer текущего слота.</summary>
    private CommandBuffer BeginSlotRecording()
    {
        int slot = GetSlot();
        var cb = _commandBuffers[slot];

        // Ожидаем завершения предыдущей работы на этом слоте
        fixed (Fence* pFence = &_fences[slot])
        {
            _vk.WaitForFences(_device, 1, pFence, true, ulong.MaxValue);
            _vk.ResetFences(_device, 1, pFence);
        }

        // Reset command buffer
        _vk.ResetCommandBuffer(cb, CommandBufferResetFlags.None);

        var beginInfo = new CommandBufferBeginInfo
        {
            SType = StructureType.CommandBufferBeginInfo,
            Flags = CommandBufferUsageFlags.OneTimeSubmitBit
        };

        if (_vk.BeginCommandBuffer(cb, &beginInfo) != Result.Success)
            throw new InvalidOperationException("Failed to begin command buffer");

        t_isRecording = true;
        return cb;
    }

    public override void WriteBuffer(IComputeBuffer buffer, ulong offset, byte[] data)
    {
        if (buffer is not VulkanComputeBuffer vkBuffer)
            throw new ArgumentException("Buffer must be a VulkanComputeBuffer");

        // Прямая запись в mapped memory — не требует command buffer
        unsafe
        {
            var ptr = vkBuffer.GetPointer();
            fixed (byte* src = data)
            {
                System.Buffer.MemoryCopy(src, (void*)((nint)ptr + (nint)offset), data.Length, data.Length);
            }
        }
    }

    public override void ReadBuffer(IComputeBuffer buffer, ulong offset, byte[] data)
    {
        if (buffer is not VulkanComputeBuffer vkBuffer)
            throw new ArgumentException("Buffer must be a VulkanComputeBuffer");

        // Прямое чтение из mapped memory — не требует command buffer
        unsafe
        {
            var ptr = vkBuffer.GetPointer();
            fixed (byte* dest = data)
            {
                System.Buffer.MemoryCopy((void*)((nint)ptr + (nint)offset), dest, data.Length, data.Length);
            }
        }
    }

    public override void Dispatch(IComputeKernel kernel, uint[] globalWorkSize, uint[]? localWorkSize = null)
    {
        if (kernel is not VulkanComputeKernel vkKernel)
            throw new ArgumentException("Kernel must be a VulkanComputeKernel");

        // Получаем command buffer для текущего потока (если ещё не начали)
        if (!t_isRecording)
            BeginSlotRecording();

        int slot = GetSlot();
        var cb = _commandBuffers[slot];

        // Allocate a fresh ring-slot descriptor set and update it with current buffer bindings.
        var descriptorSet = vkKernel.UpdateDescriptorSets();

        // Bind pipeline
        _vk.CmdBindPipeline(cb, PipelineBindPoint.Compute, vkKernel.Pipeline);

        // Bind the per-dispatch descriptor set
        _vk.CmdBindDescriptorSets(
            cb,
            PipelineBindPoint.Compute,
            vkKernel.PipelineLayout,
            0,
            1,
            &descriptorSet,
            0,
            null
        );

        // Dispatch compute
        uint groupCountX = localWorkSize != null && localWorkSize.Length > 0 && localWorkSize[0] > 0
            ? (globalWorkSize[0] + localWorkSize[0] - 1) / localWorkSize[0]
            : globalWorkSize[0];

        uint groupCountY = globalWorkSize.Length > 1
            ? (localWorkSize != null && localWorkSize.Length > 1 && localWorkSize[1] > 0
                ? (globalWorkSize[1] + localWorkSize[1] - 1) / localWorkSize[1]
                : globalWorkSize[1])
            : 1;

        uint groupCountZ = globalWorkSize.Length > 2
            ? (localWorkSize != null && localWorkSize.Length > 2 && localWorkSize[2] > 0
                ? (globalWorkSize[2] + localWorkSize[2] - 1) / localWorkSize[2]
                : globalWorkSize[2])
            : 1;

        _vk.CmdDispatch(cb, groupCountX, groupCountY, groupCountZ);
    }

    public override void InsertMemoryBarrier()
    {
        if (!t_isRecording)
            BeginSlotRecording();

        int slot = GetSlot();
        var cb = _commandBuffers[slot];

        // Compute → Compute barrier: all shader writes become visible to subsequent shaders.
        var barrier = new MemoryBarrier
        {
            SType = StructureType.MemoryBarrier,
            SrcAccessMask = AccessFlags.ShaderWriteBit,
            DstAccessMask = AccessFlags.ShaderReadBit
        };

        unsafe
        {
            _vk.CmdPipelineBarrier(
                cb,
                PipelineStageFlags.ComputeShaderBit,
                PipelineStageFlags.ComputeShaderBit,
                DependencyFlags.None,
                1, &barrier,
                0, null,
                0, null);
        }
    }

    public override void Flush()
    {
        if (!t_isRecording)
            return; // ничего не записывали

        t_isRecording = false;

        int slot = GetSlot();
        var cb = _commandBuffers[slot];
        var fence = _fences[slot];
        var queue = GetThreadQueue();

        // Завершение записи команд
        var endResult = _vk.EndCommandBuffer(cb);
        if (endResult != Result.Success)
            throw new InvalidOperationException($"vkEndCommandBuffer failed: {endResult}");

        // Отправка команд в очередь (используем очередь, назначенную этому потоку)
        var cmdBufferLocal = cb;
        var submitInfo = new SubmitInfo
        {
            SType = StructureType.SubmitInfo,
            CommandBufferCount = 1,
            PCommandBuffers = &cmdBufferLocal
        };

        var submitResult = _vk.QueueSubmit(queue, 1, &submitInfo, fence);
        if (submitResult != Result.Success)
            throw new InvalidOperationException($"vkQueueSubmit failed: {submitResult}");

        // Ожидаем завершения fence — это гарантирует, что последующий QueueSubmit
        // на этой же очереди не перезапишет текущий. Без этого ожидания два
        // параллельных Flush() на одной очереди вызывают ErrorDeviceLost.
        fixed (Fence* pFence = &_fences[slot])
        {
            _vk.WaitForFences(_device, 1, pFence, true, ulong.MaxValue);
        }
    }

    public override void Dispose()
    {
        if (_disposed) return;

        // Ожидаем завершения всех fences
        for (int i = 0; i < _poolSize; i++)
        {
            fixed (Fence* pFence = &_fences[i])
            {
                _vk.WaitForFences(_device, 1, pFence, true, ulong.MaxValue);
                _vk.DestroyFence(_device, _fences[i], null);
            }
        }

        fixed (CommandBuffer* pCmdBuffers = _commandBuffers)
        {
            _vk.FreeCommandBuffers(_device, _commandPool, (uint)_poolSize, pCmdBuffers);
        }
        _vk.DestroyCommandPool(_device, _commandPool, null);

        _disposed = true;
    }
}
