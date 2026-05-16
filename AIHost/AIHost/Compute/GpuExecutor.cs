using AIHost.ICompute;
using Microsoft.Extensions.Logging;
using System.Collections.Concurrent;
using System.Threading.Channels;

namespace AIHost.Compute;

/// <summary>
/// Provider-agnostic (Vulkan / CUDA / ROCm) dedicated GPU executor thread.
/// Принимает GpuTask через Channel, записывает dispatch'ы в командный буфер,
/// выполняет submit + fence wait только при FlushAndWait.
///
/// Жизненный цикл одного токена:
///   BeginLayer → [Dispatch × N + Barrier] → EndLayer → ... → FlushAndWait
///   = 1 fence wait на токен вместо 64+.
/// </summary>
internal sealed class GpuExecutor : IDisposable
{
    private readonly IComputeDevice _device;
    private readonly IComputeCommandQueue _queue;
    private readonly GpuLeasePool _leasePool;
    private readonly ILogger _logger;
    private readonly Channel<GpuTask> _taskChannel;
    private readonly Channel<GpuResult> _resultChannel;
    private readonly CancellationTokenSource _cts;
    private readonly Thread _workerThread;

    // Kernel cache — shared kernel objects (Vulkan/CUDA/ROCm will reuse)
    private readonly ConcurrentDictionary<string, IComputeKernel> _kernels = new();

    private long _totalTasksProcessed;
    private long _totalFlushes;
    public long TotalTasksProcessed => _totalTasksProcessed;
    public long TotalFlushes => _totalFlushes;

    private volatile bool _isFailed;
    private string? _lastError;
    public bool IsFailed => _isFailed;
    public string? LastError => _lastError;

    public ChannelWriter<GpuTask> Writer => _taskChannel.Writer;
    public ChannelReader<GpuResult> Reader => _resultChannel.Reader;

    public IReadOnlyDictionary<string, IComputeKernel> Kernels => _kernels;

    public GpuExecutor(IComputeDevice device, GpuLeasePool leasePool,
        ILogger? logger = null)
    {
        _device = device;
        // Executor owns its own queue — dedicated thread = dedicated Vulkan slot
        _queue = device.CreateCommandQueue();
        _leasePool = leasePool;
        _logger = logger ?? AppLogger.Create<GpuExecutor>();

        _taskChannel = Channel.CreateUnbounded<GpuTask>(
            new UnboundedChannelOptions { SingleReader = true, SingleWriter = false });
        _resultChannel = Channel.CreateUnbounded<GpuResult>(
            new UnboundedChannelOptions { SingleReader = false, SingleWriter = true });

        _cts = new CancellationTokenSource();
        _workerThread = new Thread(WorkerLoop)
        {
            Name = "GpuExecutor",
            IsBackground = true
        };
        _workerThread.Start();
    }

    private void WorkerLoop()
    {
        try
        {
            var reader = _taskChannel.Reader;
            while (!_cts.Token.IsCancellationRequested)
            {
                if (!reader.WaitToReadAsync(_cts.Token).AsTask().Result)
                    break;

                while (reader.TryRead(out var task))
                {
                    ProcessTask(task);
                }
            }
        }
        catch (OperationCanceledException) { }
        catch (AggregateException ex) when (ex.InnerException is OperationCanceledException) { }
        catch (Exception ex)
        {
            _isFailed = true;
            _lastError = ex.Message;
            _logger.LogError(ex, "[GpuExecutor] Worker loop crashed: {Error}", ex.Message);
            _resultChannel.Writer.TryWrite(GpuResult.Fail(ex.Message));
        }
    }

    private void ProcessTask(GpuTask task)
    {
        Interlocked.Increment(ref _totalTasksProcessed);

        switch (task.Type)
        {
            case GpuTaskType.BeginLayer:
                _logger.LogInformation("[DBG_EXEC] #{TaskId} BEGIN_LAYER layer={Layer}",
                    task.TaskId, task.LayerIndex);
                break;

            case GpuTaskType.EndLayer:
                _logger.LogInformation("[DBG_EXEC] #{TaskId} END_LAYER layer={Layer}",
                    task.TaskId, task.LayerIndex);
                break;

            case GpuTaskType.DispatchKernel:
                ProcessDispatch((DispatchKernelTask)task);
                break;

            case GpuTaskType.CopyBuffer:
                _logger.LogInformation("[DBG_EXEC] #{TaskId} COPY_BUF", task.TaskId);
                ProcessCopy((CopyBufferTask)task);
                break;

            case GpuTaskType.Barrier:
                _logger.LogInformation("[DBG_EXEC] #{TaskId} BARRIER", task.TaskId);
                _queue.InsertMemoryBarrier();
                break;

            case GpuTaskType.AcquireLease:
                ProcessAcquireLease((AcquireLeaseTask)task);
                break;

            case GpuTaskType.ReleaseLease:
                ProcessReleaseLease((ReleaseLeaseTask)task);
                break;

            case GpuTaskType.FlushAndWait:
                _logger.LogInformation("[DBG_EXEC] #{TaskId} FLUSH_AND_WAIT", task.TaskId);
                ProcessFlushAndWait((FlushAndWaitTask)task);
                break;
        }
    }

    private void ProcessDispatch(DispatchKernelTask task)
    {
        if (!_kernels.TryGetValue(task.KernelName, out var kernel))
        {
            throw new InvalidOperationException(
                $"[GpuExecutor] Kernel '{task.KernelName}' not registered. " +
                "Call RegisterKernel() before dispatching.");
        }

        _logger.LogInformation("[DBG_EXEC] #{TaskId} DISPATCH kernel={Kernel} tag={Tag} args={ArgCount}",
            task.TaskId, task.KernelName, task.OpTag ?? "?", task.ArgCount);

        // FIX: Clear stale buffer arguments from previous dispatches.
        // VulkanComputeKernel.SetArgument() only overwrites by index but does not
        // truncate _bufferArguments. If dispatch N has 3 args and dispatch N+1 has
        // 2 args, arg #3 from dispatch N persists. UpdateDescriptorSets() then writes
        // ALL _bufferArguments as descriptor updates, binding a disposed/stale buffer
        // to the GPU → ErrorDeviceLost on vkQueueSubmit (AMD RADV RENOIR).
        kernel.ClearArguments();

        for (int i = 0; i < task.ArgCount; i++)
        {
            var arg = task.Arguments[i];
            if (arg != null)
                kernel.SetArgument(i, arg);
        }

        _queue.Dispatch(kernel, task.Workgroups, null);
    }

    private void ProcessCopy(CopyBufferTask task)
    {
        if (task.Source == null || task.Destination == null) return;
        byte[] temp = new byte[task.ElementCount * sizeof(float)];
        _queue.ReadBuffer(task.Source, 0, temp);
        _queue.WriteBuffer(task.Destination, 0, temp);
    }

    private void ProcessAcquireLease(AcquireLeaseTask task)
    {
        // Lease acquired BEFORE dispatch (CPU side).
    }

    private void ProcessReleaseLease(ReleaseLeaseTask task)
    {
        // Lease returned AFTER FlushAndWait.
    }

    private void ProcessFlushAndWait(FlushAndWaitTask task)
    {
        try
        {
            // Reset all kernel dispatch rings BEFORE vkQueueSubmit.
            // Without this, dequant_q4k chunks fill descriptor ring (240+ slots),
            // causing ErrorDeviceLost when old descriptors are reused in-flight.
            foreach (var kv in _kernels)
            {
                if (kv.Value is AIHost.ICompute.Vulkan.VulkanComputeKernel vk)
                    vk.ResetDispatchRing();
            }

            _queue.Flush();
            Interlocked.Increment(ref _totalFlushes);

            _logger.LogInformation("[DBG_EXEC] #{TaskId} FLUSH_OK totalFlushes={N}", task.TaskId, _totalFlushes);

            task.Completion?.TrySetResult(GpuResult.Ok());
            _resultChannel.Writer.TryWrite(GpuResult.Ok());
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[DBG_EXEC] #{TaskId} FLUSH_FAILED: {Error}", task.TaskId, ex.Message);
            task.Completion?.TrySetResult(GpuResult.Fail(ex.Message));
            _resultChannel.Writer.TryWrite(GpuResult.Fail(ex.Message));
        }
    }

    /// <summary>
    /// Зарегистрировать ядро в кэше (вызывается из ComputeOps при инициализации).
    /// Ядра переиспользуются между вызовами — provider-specific (VulkanComputeKernel / CudaComputeKernel).
    /// </summary>
    public void RegisterKernel(string name, IComputeKernel kernel)
    {
        _logger.LogInformation("[DBG_EXEC] REGISTER kernel={Name}", name);
        _kernels[name] = kernel;
    }

    /// <summary>
    /// Асинхронно ожидать готовности GPU.
    /// </summary>
    public async Task<GpuResult> SubmitAndWaitAsync(CancellationToken ct = default)
    {
        var tcs = new TaskCompletionSource<GpuResult>();
        var task = FlushAndWaitTask.Create(tcs);
        await _taskChannel.Writer.WriteAsync(task, ct);
        return await tcs.Task;
    }

    public void Dispose()
    {
        _cts.Cancel();
        _taskChannel.Writer.TryComplete();

        if (_workerThread.IsAlive)
            _workerThread.Join(TimeSpan.FromSeconds(5));

        foreach (var kv in _kernels)
            kv.Value.Dispose();
        _kernels.Clear();

        _queue.Dispose();
        _leasePool.Dispose();
        _cts.Dispose();
    }
}
