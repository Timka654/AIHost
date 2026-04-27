using Silk.NET.Vulkan;

namespace AIHost.ICompute.Vulkan;

/// <summary>
/// Очередь команд для Vulkan
/// </summary>
internal unsafe class VulkanComputeCommandQueue : ComputeCommandQueueBase
{
    private readonly Vk _vk;
    private readonly Device _device;
    private readonly Queue _queue;
    private readonly CommandPool _commandPool;
    private CommandBuffer _commandBuffer;
    private Fence _fence;
    private bool _isRecording;
    private bool _disposed;

    internal VulkanComputeCommandQueue(Vk vk, Device device, Queue queue, uint queueFamilyIndex)
    {
        _vk = vk;
        _device = device;
        _queue = queue;

        // Создание command pool
        var poolInfo = new CommandPoolCreateInfo
        {
            SType = StructureType.CommandPoolCreateInfo,
            Flags = CommandPoolCreateFlags.ResetCommandBufferBit,
            QueueFamilyIndex = queueFamilyIndex
        };

        if (_vk.CreateCommandPool(device, &poolInfo, null, out _commandPool) != Result.Success)
            throw new InvalidOperationException("Failed to create command pool");

        // Выделение command buffer
        var allocInfo = new CommandBufferAllocateInfo
        {
            SType = StructureType.CommandBufferAllocateInfo,
            CommandPool = _commandPool,
            Level = CommandBufferLevel.Primary,
            CommandBufferCount = 1
        };

        CommandBuffer cmdBuffer;
        if (_vk.AllocateCommandBuffers(device, &allocInfo, &cmdBuffer) != Result.Success)
            throw new InvalidOperationException("Failed to allocate command buffer");
        
        _commandBuffer = cmdBuffer;

        // Создание fence для синхронизации
        var fenceInfo = new FenceCreateInfo
        {
            SType = StructureType.FenceCreateInfo,
            Flags = FenceCreateFlags.SignaledBit
        };

        if (_vk.CreateFence(device, &fenceInfo, null, out _fence) != Result.Success)
            throw new InvalidOperationException("Failed to create fence");
    }

    private void BeginRecording()
    {
        if (_isRecording) return;

        // Reset command buffer before each recording
        _vk.ResetCommandBuffer(_commandBuffer, CommandBufferResetFlags.None);

        var beginInfo = new CommandBufferBeginInfo
        {
            SType = StructureType.CommandBufferBeginInfo,
            Flags = CommandBufferUsageFlags.OneTimeSubmitBit
        };

        if (_vk.BeginCommandBuffer(_commandBuffer, &beginInfo) != Result.Success)
            throw new InvalidOperationException("Failed to begin command buffer");

        _isRecording = true;
    }

    public override void WriteBuffer(IComputeBuffer buffer, ulong offset, byte[] data)
    {
        if (buffer is not VulkanComputeBuffer vkBuffer)
            throw new ArgumentException("Buffer must be a VulkanComputeBuffer");

        BeginRecording();

        // TODO: Реализовать копирование через staging buffer
        // Для упрощения используем прямую запись в mapped memory
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

        BeginRecording();

        // TODO: Реализовать копирование через staging buffer
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

        BeginRecording();

        // Обновить descriptor sets с текущими буферами
        vkKernel.UpdateDescriptorSets();

        // Bind pipeline
        _vk.CmdBindPipeline(_commandBuffer, PipelineBindPoint.Compute, vkKernel.Pipeline);

        // Bind descriptor sets
        var descriptorSet = vkKernel.DescriptorSet;
        _vk.CmdBindDescriptorSets(
            _commandBuffer,
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

        _vk.CmdDispatch(_commandBuffer, groupCountX, groupCountY, groupCountZ);
    }

    public override void Flush()
    {
        if (!_isRecording) return;

        // Завершение записи команд
        if (_vk.EndCommandBuffer(_commandBuffer) != Result.Success)
            throw new InvalidOperationException("Failed to end command buffer");

        _isRecording = false;

        // Ожидание завершения предыдущей команды
        var fenceLocal = _fence;
        _vk.WaitForFences(_device, 1, &fenceLocal, true, ulong.MaxValue);
        _vk.ResetFences(_device, 1, &fenceLocal);

        // Отправка команд в очередь
        var cmdBufferLocal = _commandBuffer;
        var submitInfo = new SubmitInfo
        {
            SType = StructureType.SubmitInfo,
            CommandBufferCount = 1,
            PCommandBuffers = &cmdBufferLocal
        };

        if (_vk.QueueSubmit(_queue, 1, &submitInfo, _fence) != Result.Success)
            throw new InvalidOperationException("Failed to submit command buffer");

        // Ожидание завершения
        _vk.WaitForFences(_device, 1, &fenceLocal, true, ulong.MaxValue);
    }

    public override void Dispose()
    {
        if (_disposed) return;

        var fenceLocal = _fence;
        _vk.WaitForFences(_device, 1, &fenceLocal, true, ulong.MaxValue);
        _vk.DestroyFence(_device, _fence, null);
        
        var cmdBufferLocal = _commandBuffer;
        _vk.FreeCommandBuffers(_device, _commandPool, 1, &cmdBufferLocal);
        _vk.DestroyCommandPool(_device, _commandPool, null);

        _disposed = true;
    }
}
