using AIHost.ICompute;
using System.Collections.Concurrent;

namespace AIHost.Compute;

/// <summary>
/// Пул pre-allocated GPU буферов (provider-agnostic: Vulkan / CUDA / ROCm).
/// Буферы распределены по size class (Small, Medium, Large) и выдаются через Rent/Return.
/// Lease не уничтожается при Reset — живёт пока не Return.
/// </summary>
public sealed class GpuLeasePool : IDisposable
{
    private readonly IComputeDevice _device;
    private bool _disposed;

    // Size classes (rows × cols in floats)
    public enum SizeClass { Small, Medium, Large }

    private const int SMALL_MAX_ELEMENTS = 128;          // row-vector 1×128
    private const int MEDIUM_MAX_ELEMENTS = 6144;         // 1×6144 (VD)
    private const int LARGE_MAX_ELEMENTS = 10_240;        // 1×10240 (CD) or larger
    private const int PRE_ALLOC_PER_CLASS = 16;

    private readonly ConcurrentQueue<GpuLease> _smallPool = new();
    private readonly ConcurrentQueue<GpuLease> _mediumPool = new();
    private readonly ConcurrentQueue<GpuLease> _largePool = new();

    // Track all created leases for disposal
    private readonly List<GpuLease> _allLeases = new();

    private int _totalRents;
    private int _totalReturns;

    public int TotalRents => _totalRents;
    public int TotalReturns => _totalReturns;
    public int SmallAvailable => _smallPool.Count;
    public int MediumAvailable => _mediumPool.Count;
    public int LargeAvailable => _largePool.Count;

    public GpuLeasePool(IComputeDevice device)
    {
        _device = device;

        // Pre-allocate buffers per size class
        for (int i = 0; i < PRE_ALLOC_PER_CLASS; i++)
        {
            PreAllocate(SizeClass.Small,  1, SMALL_MAX_ELEMENTS,  _smallPool);
            PreAllocate(SizeClass.Medium, 1, MEDIUM_MAX_ELEMENTS, _mediumPool);
            PreAllocate(SizeClass.Large,  1, LARGE_MAX_ELEMENTS,  _largePool);
        }
    }

    private void PreAllocate(SizeClass sizeClass, int rows, int cols, ConcurrentQueue<GpuLease> pool)
    {
        int totalElements = rows * cols;
        var buffer = AllocateBuffer(totalElements);
        var tensor = new Tensor(buffer, TensorShape.Matrix(rows, cols), DataType.F32,
            $"lease_{sizeClass}_{pool.Count}");
        var lease = new GpuLease(tensor, sizeClass);
        pool.Enqueue(lease);
        _allLeases.Add(lease);
    }

    /// <summary>
    /// Взять GPU-буфер из пула. Если пул пуст — аллоцируем новый.
    /// Provider-agnostic: работает с Vulkan / CUDA / ROCm через IComputeDevice.
    /// </summary>
    public GpuLease Rent(int rows, int cols)
    {
        int totalElements = rows * cols;
        SizeClass sizeClass = totalElements <= SMALL_MAX_ELEMENTS ? SizeClass.Small
            : totalElements <= MEDIUM_MAX_ELEMENTS ? SizeClass.Medium
            : SizeClass.Large;

        var pool = sizeClass switch
        {
            SizeClass.Small  => _smallPool,
            SizeClass.Medium => _mediumPool,
            _                => _largePool
        };

        if (pool.TryDequeue(out var lease))
        {
            // Verify buffer is large enough (dynamic prefill may exceed pre-alloc size)
            if (lease.Tensor.Shape.TotalElements >= totalElements)
            {
                Interlocked.Increment(ref _totalRents);
                return lease;
            }
            // Buffer too small — dispose and fall through to create new
            lease.Dispose();
        }

        // Pool empty — create new buffer
        return CreateNewLease(sizeClass, rows, cols);
    }

    private GpuLease CreateNewLease(SizeClass sizeClass, int rows, int cols)
    {
        int totalElements = rows * cols;
        var buffer = AllocateBuffer(totalElements);
        var tensor = new Tensor(buffer, TensorShape.Matrix(rows, cols), DataType.F32,
            $"lease_{sizeClass}_dyn");
        var lease = new GpuLease(tensor, sizeClass);
        _allLeases.Add(lease);
        Interlocked.Increment(ref _totalRents);
        return lease;
    }

    private IComputeBuffer AllocateBuffer(int totalElements)
    {
        uint bytes = (uint)(totalElements * sizeof(float));
        return _device.CreateBuffer(bytes, BufferType.Storage, DataType.F32);
    }

    /// <summary>
    /// Вернуть GPU-буфер в пул. Буфер становится доступным для следующего Rent того же size class.
    /// </summary>
    public void Return(GpuLease lease)
    {
        if (lease == null) return;
        Interlocked.Increment(ref _totalReturns);

        var pool = lease.SizeClass switch
        {
            SizeClass.Small  => _smallPool,
            SizeClass.Medium => _mediumPool,
            _                => _largePool
        };
        pool.Enqueue(lease);
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;

        foreach (var lease in _allLeases)
            lease.Dispose();
        _allLeases.Clear();

        while (_smallPool.TryDequeue(out var l)) l.Dispose();
        while (_mediumPool.TryDequeue(out var l)) l.Dispose();
        while (_largePool.TryDequeue(out var l)) l.Dispose();
    }
}

/// <summary>
/// Лизинг одного GPU-буфера. Оборачивает Tensor и возвращается в пул при Return.
/// </summary>
public sealed class GpuLease : IDisposable
{
    public Tensor Tensor { get; }
    public GpuLeasePool.SizeClass SizeClass { get; }
    public bool IsReturned { get; private set; }

    internal GpuLease(Tensor tensor, GpuLeasePool.SizeClass sizeClass)
    {
        Tensor = tensor;
        SizeClass = sizeClass;
    }

    /// <summary>Освободить lease без возврата в пул (на случай уничтожения).</summary>
    public void Dispose()
    {
        if (IsReturned) return;
        IsReturned = true;
        Tensor.Dispose();
    }
}
