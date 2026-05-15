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
                // Блокирующее ожидание асинхронно (через GetAwaiter — корректно для Thread)
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
            case GpuTaskType.EndLayer:
                // Command buffer is continuously recording between Begin/Flush
                break;

            case GpuTaskType.DispatchKernel:
                ProcessDispatch((DispatchKernelTask)task);
                break;

            case GpuTaskType.CopyBuffer:
                ProcessCopy((CopyBufferTask)task);
                break;

            case GpuTaskType.Barrier:
                _queue.InsertMemoryBarrier();
                break;

            case GpuTaskType.AcquireLease:
                ProcessAcquireLease((AcquireLeaseTask)task);
                break;

            case GpuTaskType.ReleaseLease:
                ProcessReleaseLease((ReleaseLeaseTask)task);
                break;

            case GpuTaskType.FlushAndWait:
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

        // Set arguments on the kernel object
        for (int i = 0; i < task.ArgCount; i++)
        {
            var arg = task.Arguments[i];
            if (arg != null)
                kernel.SetArgument(i, arg);
        }

        // Dispatch через command queue (NOT kernel.Dispatch — queue manages command buffer)
        _queue.Dispatch(kernel, task.Workgroups, null);
    }

    private void ProcessCopy(CopyBufferTask task)
    {
        if (task.Source == null || task.Destination == null) return;
        // Single float copy — simplest path through Read/Write
        // For bulk copies, use dedicated Copy kernel
        byte[] temp = new byte[task.ElementCount * sizeof(float)];
        _queue.ReadBuffer(task.Source, 0, temp);
        _queue.WriteBuffer(task.Destination, 0, temp);
    }

    private void ProcessAcquireLease(AcquireLeaseTask task)
    {
        // Lease acquired BEFORE dispatch (CPU side). Executor just tracks.
        // Actual Rent happens in caller (ComputeOps.EmitTask) for correctness.
    }

    private void ProcessReleaseLease(ReleaseLeaseTask task)
    {
        // Lease returned AFTER FlushAndWait. Executor just tracks.
        // Actual Return happens in caller after FlushAndWait.
    }

    private void ProcessFlushAndWait(FlushAndWaitTask task)
    {
        try
        {
            _queue.Flush();
            Interlocked.Increment(ref _totalFlushes);

            task.Completion?.TrySetResult(GpuResult.Ok());
            _resultChannel.Writer.TryWrite(GpuResult.Ok());
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[GpuExecutor] Flush failed");
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

        // Ждать завершения worker thread
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
