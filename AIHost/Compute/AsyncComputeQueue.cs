using AIHost.ICompute;
using System.Collections.Concurrent;

namespace AIHost.Compute;

/// <summary>
/// Async wrapper for compute command queues — serialises GPU operations
/// without blocking threadpool threads on the semaphore.
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

    public async Task WriteBufferAsync(IComputeBuffer buffer, ulong offset, byte[] data)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        await _semaphore.WaitAsync();
        try
        {
            await Task.Run(() => _queue.WriteBuffer(buffer, offset, data));
        }
        finally
        {
            _semaphore.Release();
        }
    }

    public async Task<byte[]> ReadBufferAsync(IComputeBuffer buffer, ulong offset, int size)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        await _semaphore.WaitAsync();
        try
        {
            return await Task.Run(() =>
            {
                byte[] data = new byte[size];
                _queue.ReadBuffer(buffer, offset, data);
                return data;
            });
        }
        finally
        {
            _semaphore.Release();
        }
    }

    public async Task DispatchAsync(IComputeKernel kernel, uint[] globalWorkSize, uint[]? localWorkSize = null)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        await _semaphore.WaitAsync();
        try
        {
            await Task.Run(() => _queue.Dispatch(kernel, globalWorkSize, localWorkSize));
        }
        finally
        {
            _semaphore.Release();
        }
    }

    public async Task FlushAsync()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

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

    public async Task SynchronizeAsync()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

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
        _disposed = true;

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
        => queue.WriteBuffer(Buffer, Offset, Data);
}

internal class DispatchCommand : ComputeCommand
{
    public IComputeKernel Kernel { get; set; } = null!;
    public uint[] GlobalWorkSize { get; set; } = null!;
    public uint[]? LocalWorkSize { get; set; }

    public override void Execute(IComputeCommandQueue queue)
        => queue.Dispatch(Kernel, GlobalWorkSize, LocalWorkSize);
}
