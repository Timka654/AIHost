using AIHost.ICompute;
using System.Collections.Concurrent;

namespace AIHost.Compute;

/// <summary>
/// Async wrapper for compute command queues to enable parallel execution
/// </summary>
public class AsyncComputeQueue : IDisposable
{
    private readonly IComputeCommandQueue _queue;
    private readonly ConcurrentQueue<ComputeCommand> _commands = new();
    private readonly SemaphoreSlim _semaphore = new(1, 1);
    private bool _disposed;

    public AsyncComputeQueue(IComputeCommandQueue queue)
    {
        _queue = queue ?? throw new ArgumentNullException(nameof(queue));
    }

    /// <summary>
    /// Enqueue a buffer write command
    /// </summary>
    public Task WriteBufferAsync(IComputeBuffer buffer, ulong offset, byte[] data)
    {
        if (_disposed)
            throw new ObjectDisposedException(nameof(AsyncComputeQueue));

        return Task.Run(() =>
        {
            _semaphore.Wait();
            try
            {
                _queue.WriteBuffer(buffer, offset, data);
            }
            finally
            {
                _semaphore.Release();
            }
        });
    }

    /// <summary>
    /// Enqueue a buffer read command
    /// </summary>
    public Task<byte[]> ReadBufferAsync(IComputeBuffer buffer, ulong offset, int size)
    {
        if (_disposed)
            throw new ObjectDisposedException(nameof(AsyncComputeQueue));

        return Task.Run(() =>
        {
            _semaphore.Wait();
            try
            {
                byte[] data = new byte[size];
                _queue.ReadBuffer(buffer, offset, data);
                return data;
            }
            finally
            {
                _semaphore.Release();
            }
        });
    }

    /// <summary>
    /// Enqueue a kernel dispatch command
    /// </summary>
    public Task DispatchAsync(IComputeKernel kernel, uint[] globalWorkSize, uint[]? localWorkSize = null)
    {
        if (_disposed)
            throw new ObjectDisposedException(nameof(AsyncComputeQueue));

        return Task.Run(() =>
        {
            _semaphore.Wait();
            try
            {
                _queue.Dispatch(kernel, globalWorkSize, localWorkSize);
            }
            finally
            {
                _semaphore.Release();
            }
        });
    }

    /// <summary>
    /// Execute all queued commands and wait for completion
    /// </summary>
    public async Task FlushAsync()
    {
        if (_disposed)
            throw new ObjectDisposedException(nameof(AsyncComputeQueue));

        await Task.Run(() =>
        {
            _semaphore.Wait();
            try
            {
                _queue.Flush();
            }
            finally
            {
                _semaphore.Release();
            }
        });
    }

    /// <summary>
    /// Synchronize and wait for all operations to complete (flush + wait)
    /// </summary>
    public async Task SynchronizeAsync()
    {
        if (_disposed)
            throw new ObjectDisposedException(nameof(AsyncComputeQueue));

        await _semaphore.WaitAsync();
        try
        {
            await Task.Run(() => _queue.Flush());
        }
        finally
        {
            _semaphore.Release();
        }
    }

    public void Dispose()
    {
        if (_disposed) return;

        _semaphore.Wait();
        try
        {
            _queue.Dispose();
        }
        finally
        {
            _semaphore.Release();
            _semaphore.Dispose();
        }

        _disposed = true;
    }
}

/// <summary>
/// Represents a queued compute command
/// </summary>
internal abstract class ComputeCommand
{
    public abstract void Execute(IComputeCommandQueue queue);
}

internal class WriteBufferCommand : ComputeCommand
{
    public IComputeBuffer Buffer { get; set; } = null!;
    public ulong Offset { get; set; }
    public byte[] Data { get; set; } = null!;

    public override void Execute(IComputeCommandQueue queue)
    {
        queue.WriteBuffer(Buffer, Offset, Data);
    }
}

internal class DispatchCommand : ComputeCommand
{
    public IComputeKernel Kernel { get; set; } = null!;
    public uint[] GlobalWorkSize { get; set; } = null!;
    public uint[]? LocalWorkSize { get; set; }

    public override void Execute(IComputeCommandQueue queue)
    {
        queue.Dispatch(Kernel, GlobalWorkSize, LocalWorkSize);
    }
}
