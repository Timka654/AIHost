using AIHost.ICompute;
using System.Collections.Concurrent;

namespace AIHost.Compute;

/// <summary>
/// Memory pool for reusing compute buffers to reduce allocation overhead
/// </summary>
public class ComputeBufferPool : IDisposable
{
    private readonly IComputeDevice _device;
    private readonly ConcurrentDictionary<ulong, ConcurrentBag<IComputeBuffer>> _pools = new();
    private readonly HashSet<IComputeBuffer> _activeBuffers = new();
    private readonly object _lockObject = new();
    private bool _disposed;

    // Statistics
    private long _totalAllocations;
    private long _poolHits;
    private long _poolMisses;

    public ComputeBufferPool(IComputeDevice device)
    {
        _device = device ?? throw new ArgumentNullException(nameof(device));
    }

    /// <summary>
    /// Rent a buffer from the pool or create a new one
    /// </summary>
    public IComputeBuffer Rent(ulong size, BufferType type = BufferType.Storage, DataType elementType = DataType.F32)
    {
        if (_disposed)
            throw new ObjectDisposedException(nameof(ComputeBufferPool));

        // Round up to nearest power of 2 for better pooling
        ulong pooledSize = RoundToPowerOfTwo(size);
        
        if (_pools.TryGetValue(pooledSize, out var pool) && pool.TryTake(out var buffer))
        {
            Interlocked.Increment(ref _poolHits);
            lock (_lockObject)
            {
                _activeBuffers.Add(buffer);
            }
            return buffer;
        }

        // Create new buffer
        Interlocked.Increment(ref _poolMisses);
        Interlocked.Increment(ref _totalAllocations);
        
        var newBuffer = _device.CreateBuffer(pooledSize, type, elementType);
        lock (_lockObject)
        {
            _activeBuffers.Add(newBuffer);
        }
        return newBuffer;
    }

    /// <summary>
    /// Return a buffer to the pool for reuse
    /// </summary>
    public void Return(IComputeBuffer buffer)
    {
        if (_disposed || buffer == null)
            return;

        lock (_lockObject)
        {
            if (!_activeBuffers.Remove(buffer))
                return; // Buffer not from this pool
        }

        ulong size = RoundToPowerOfTwo(buffer.Size);
        var pool = _pools.GetOrAdd(size, _ => new ConcurrentBag<IComputeBuffer>());
        pool.Add(buffer);
    }

    /// <summary>
    /// Clear all pooled buffers and free memory
    /// </summary>
    public void Clear()
    {
        foreach (var pool in _pools.Values)
        {
            while (pool.TryTake(out var buffer))
            {
                buffer.Dispose();
            }
        }
        _pools.Clear();
    }

    /// <summary>
    /// Get pool statistics
    /// </summary>
    public PoolStatistics GetStatistics()
    {
        int pooledCount = 0;
        ulong pooledMemory = 0;

        foreach (var kvp in _pools)
        {
            int count = kvp.Value.Count;
            pooledCount += count;
            pooledMemory += kvp.Key * (ulong)count;
        }

        return new PoolStatistics
        {
            TotalAllocations = _totalAllocations,
            PoolHits = _poolHits,
            PoolMisses = _poolMisses,
            HitRate = (_poolHits + _poolMisses) > 0 ? (double)_poolHits / (_poolHits + _poolMisses) : 0,
            PooledBufferCount = pooledCount,
            PooledMemoryBytes = pooledMemory,
            ActiveBufferCount = _activeBuffers.Count
        };
    }

    private static ulong RoundToPowerOfTwo(ulong value)
    {
        if (value <= 0) return 1;
        value--;
        value |= value >> 1;
        value |= value >> 2;
        value |= value >> 4;
        value |= value >> 8;
        value |= value >> 16;
        value |= value >> 32;
        return value + 1;
    }

    public void Dispose()
    {
        if (_disposed) return;

        Clear();
        
        lock (_lockObject)
        {
            foreach (var buffer in _activeBuffers)
            {
                buffer.Dispose();
            }
            _activeBuffers.Clear();
        }

        _disposed = true;
    }
}

/// <summary>
/// Statistics for buffer pool performance
/// </summary>
public struct PoolStatistics
{
    public long TotalAllocations { get; set; }
    public long PoolHits { get; set; }
    public long PoolMisses { get; set; }
    public double HitRate { get; set; }
    public int PooledBufferCount { get; set; }
    public ulong PooledMemoryBytes { get; set; }
    public int ActiveBufferCount { get; set; }

    public override string ToString()
    {
        return $"Pool Stats: {PoolHits}/{TotalAllocations} hits ({HitRate:P1}), " +
               $"{PooledBufferCount} pooled buffers ({PooledMemoryBytes / (1024 * 1024)}MB), " +
               $"{ActiveBufferCount} active";
    }
}
