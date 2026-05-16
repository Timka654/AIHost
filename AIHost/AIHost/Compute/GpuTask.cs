using AIHost.ICompute;

namespace AIHost.Compute;

/// <summary>
/// Типы операций для Channel-based GPU pipeline.
/// </summary>
public enum GpuTaskType : byte
{
    BeginLayer,
    EndLayer,
    FlushAndWait,
    DispatchKernel,
    CopyBuffer,
    Barrier,
    AcquireLease,
    ReleaseLease,
}

/// <summary>
/// Базовая задача для GPU executor. Сериализуется через Channel<GpuTask>.
/// Использует object pooling (GpuTaskPool) для предотвращения аллокаций в hot path.
/// </summary>
public abstract class GpuTask
{
    private static int s_nextTaskId;

    public GpuTaskType Type { get; init; }
    public int LayerIndex { get; set; }
    public int TaskId { get; set; } // сквозной глобальный ID для трассировки

    /// <summary>Присвоить следующий глобальный ID (вызывается при создании задачи).</summary>
    protected void AssignTaskId() => TaskId = Interlocked.Increment(ref s_nextTaskId);

    /// <summary>Вернуть задачу в пул для переиспользования.</summary>
    public virtual void Reset() { TaskId = 0; }

    internal static class Pool<T> where T : GpuTask, new()
    {
        private static readonly System.Collections.Concurrent.ConcurrentQueue<T> _pool = new();
        public static T Rent()
        {
            if (_pool.TryDequeue(out var task))
            {
                task.Reset();
                return task;
            }
            return new T();
        }
        public static void Return(T task) => _pool.Enqueue(task);
    }
}

public sealed class BeginLayerTask : GpuTask
{
    public BeginLayerTask() => Type = GpuTaskType.BeginLayer;
    public static BeginLayerTask Create(int layerIndex)
    {
        var t = Pool<BeginLayerTask>.Rent();
        t.LayerIndex = layerIndex;
        t.AssignTaskId();
        return t;
    }
}

public sealed class EndLayerTask : GpuTask
{
    public EndLayerTask() => Type = GpuTaskType.EndLayer;
    public static EndLayerTask Create(int layerIndex)
    {
        var t = Pool<EndLayerTask>.Rent();
        t.LayerIndex = layerIndex;
        t.AssignTaskId();
        return t;
    }
}

public sealed class FlushAndWaitTask : GpuTask
{
    public FlushAndWaitTask() => Type = GpuTaskType.FlushAndWait;
    public System.Threading.Tasks.TaskCompletionSource<GpuResult>? Completion { get; set; }
    public static FlushAndWaitTask Create(System.Threading.Tasks.TaskCompletionSource<GpuResult>? comp = null)
    {
        var t = Pool<FlushAndWaitTask>.Rent();
        t.Completion = comp;
        t.AssignTaskId();
        return t;
    }
    public override void Reset() { Completion = null; TaskId = 0; }
}

public sealed class DispatchKernelTask : GpuTask
{
    public DispatchKernelTask() => Type = GpuTaskType.DispatchKernel;

    public string KernelName { get; set; } = "";
    // Fixed 12 argument slots (as in Vulkan bindings). Provider-agnostic.
    public IComputeBuffer?[] Arguments { get; } = new IComputeBuffer?[12];
    public int ArgCount { get; set; }
    public uint[] Workgroups { get; } = new uint[3];
    // Debug: имя операции для трассировки
    public string? OpTag { get; set; }

    public static DispatchKernelTask Create(string kernelName, uint[] workgroups)
    {
        var t = Pool<DispatchKernelTask>.Rent();
        t.KernelName = kernelName;
        t.ArgCount = 0;
        t.OpTag = null;
        Array.Clear(t.Arguments, 0, t.Arguments.Length);
        t.Workgroups[0] = workgroups.Length > 0 ? workgroups[0] : 1;
        t.Workgroups[1] = workgroups.Length > 1 ? workgroups[1] : 1;
        t.Workgroups[2] = workgroups.Length > 2 ? workgroups[2] : 1;
        t.AssignTaskId();
        return t;
    }

    public DispatchKernelTask AddArg(IComputeBuffer? buffer)
    {
        if (ArgCount < 12)
            Arguments[ArgCount++] = buffer;
        return this;
    }

    public override void Reset()
    {
        KernelName = "";
        ArgCount = 0;
        OpTag = null;
        Array.Clear(Arguments, 0, Arguments.Length);
        Workgroups[0] = Workgroups[1] = Workgroups[2] = 1;
        TaskId = 0;
    }
}

public sealed class CopyBufferTask : GpuTask
{
    public CopyBufferTask() => Type = GpuTaskType.CopyBuffer;
    public IComputeBuffer? Source { get; set; }
    public IComputeBuffer? Destination { get; set; }
    public uint ElementCount { get; set; }

    public static CopyBufferTask Create(IComputeBuffer? src, IComputeBuffer? dst, uint count)
    {
        var t = Pool<CopyBufferTask>.Rent();
        t.Source = src;
        t.Destination = dst;
        t.ElementCount = count;
        t.AssignTaskId();
        return t;
    }

    public override void Reset()
    {
        Source = null;
        Destination = null;
        ElementCount = 0;
        TaskId = 0;
    }
}

public sealed class BarrierTask : GpuTask
{
    public BarrierTask() => Type = GpuTaskType.Barrier;
    public static BarrierTask Create()
    {
        var t = Pool<BarrierTask>.Rent();
        t.AssignTaskId();
        return t;
    }
    public override void Reset() { TaskId = 0; }
}

public sealed class AcquireLeaseTask : GpuTask
{
    public AcquireLeaseTask() => Type = GpuTaskType.AcquireLease;
    public int Rows { get; set; }
    public int Cols { get; set; }
    public int LeaseSlot { get; set; } = -1;

    public static AcquireLeaseTask Create(int rows, int cols)
    {
        var t = Pool<AcquireLeaseTask>.Rent();
        t.Rows = rows;
        t.Cols = cols;
        t.LeaseSlot = -1;
        t.AssignTaskId();
        return t;
    }

    public override void Reset()
    {
        Rows = 0;
        Cols = 0;
        LeaseSlot = -1;
        TaskId = 0;
    }
}

public sealed class ReleaseLeaseTask : GpuTask
{
    public ReleaseLeaseTask() => Type = GpuTaskType.ReleaseLease;
    public GpuLease? Lease { get; set; }

    public static ReleaseLeaseTask Create(GpuLease lease)
    {
        var t = Pool<ReleaseLeaseTask>.Rent();
        t.Lease = lease;
        t.AssignTaskId();
        return t;
    }

    public override void Reset() { Lease = null; TaskId = 0; }
}

/// <summary>
/// Результат выполнения GPU операций.
/// </summary>
public readonly struct GpuResult
{
    public bool Success { get; init; }
    public string? Error { get; init; }
    public int LayersProcessed { get; init; }

    public static GpuResult Ok(int layers = 0) => new() { Success = true, LayersProcessed = layers };
    public static GpuResult Fail(string error) => new() { Success = false, Error = error };
}
