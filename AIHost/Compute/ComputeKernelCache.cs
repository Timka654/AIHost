using AIHost.ICompute;
using System.Collections.Concurrent;
using System.Security.Cryptography;
using System.Text;

namespace AIHost.Compute;

/// <summary>
/// Cache for compiled compute kernels to avoid recompilation
/// </summary>
public class ComputeKernelCache : IDisposable
{
    private readonly IComputeDevice _device;
    private readonly ConcurrentDictionary<string, IComputeKernel> _cache = new();
    private bool _disposed;

    // Statistics
    private long _cacheHits;
    private long _cacheMisses;

    public ComputeKernelCache(IComputeDevice device)
    {
        _device = device ?? throw new ArgumentNullException(nameof(device));
    }

    /// <summary>
    /// Get or create a kernel from source code
    /// </summary>
    public IComputeKernel GetOrCreate(string source, string entryPoint)
    {
        if (_disposed)
            throw new ObjectDisposedException(nameof(ComputeKernelCache));

        string key = GenerateKey(source, entryPoint);

        if (_cache.TryGetValue(key, out var kernel))
        {
            Interlocked.Increment(ref _cacheHits);
            return kernel;
        }

        Interlocked.Increment(ref _cacheMisses);
        
        // Create and cache new kernel
        var newKernel = _device.CreateKernel(source, entryPoint);
        _cache.TryAdd(key, newKernel);
        
        return newKernel;
    }

    /// <summary>
    /// Get or create a kernel from file
    /// </summary>
    public IComputeKernel GetOrCreateFromFile(string filePath, string entryPoint)
    {
        if (_disposed)
            throw new ObjectDisposedException(nameof(ComputeKernelCache));

        // Use file path + entry point as key
        string key = $"file:{filePath}:{entryPoint}";

        if (_cache.TryGetValue(key, out var kernel))
        {
            Interlocked.Increment(ref _cacheHits);
            return kernel;
        }

        Interlocked.Increment(ref _cacheMisses);
        
        var newKernel = _device.CreateKernelFromFile(filePath, entryPoint);
        _cache.TryAdd(key, newKernel);
        
        return newKernel;
    }

    /// <summary>
    /// Clear the cache and dispose all kernels
    /// </summary>
    public void Clear()
    {
        foreach (var kernel in _cache.Values)
        {
            kernel.Dispose();
        }
        _cache.Clear();
    }

    /// <summary>
    /// Get cache statistics
    /// </summary>
    public CacheStatistics GetStatistics()
    {
        long total = _cacheHits + _cacheMisses;
        return new CacheStatistics
        {
            CacheHits = _cacheHits,
            CacheMisses = _cacheMisses,
            HitRate = total > 0 ? (double)_cacheHits / total : 0,
            CachedKernelCount = _cache.Count
        };
    }

    private static string GenerateKey(string source, string entryPoint)
    {
        // Create hash of source code for cache key
        string combined = $"{source}:{entryPoint}";
        byte[] bytes = Encoding.UTF8.GetBytes(combined);
        byte[] hash = SHA256.HashData(bytes);
        return Convert.ToHexString(hash);
    }

    public void Dispose()
    {
        if (_disposed) return;

        Clear();
        _disposed = true;
    }
}

/// <summary>
/// Statistics for kernel cache performance
/// </summary>
public struct CacheStatistics
{
    public long CacheHits { get; set; }
    public long CacheMisses { get; set; }
    public double HitRate { get; set; }
    public int CachedKernelCount { get; set; }

    public override string ToString()
    {
        return $"Cache Stats: {CacheHits}/{CacheHits + CacheMisses} hits ({HitRate:P1}), " +
               $"{CachedKernelCount} cached kernels";
    }
}
