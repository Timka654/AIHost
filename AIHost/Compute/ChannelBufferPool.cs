using AIHost.ICompute;
using System.Threading.Channels;

namespace AIHost.Compute;

/// <summary>
/// VRAM buffer pool backed by <see cref="Channel{T}"/> — lock-free,
/// bounded, with built-in backpressure via BoundedChannelOptions.
///
/// In the hot inference path: <see cref="Rent"/> uses TryRead (non-blocking,
/// fallback-allocate on empty pool). <see cref="Return"/> uses TryWrite
/// (non-blocking, fallback-dispose on full pool). This guarantees the
/// GPU thread never waits for buffer availability.
/// </summary>
public sealed class ChannelBufferPool : IDisposable
{
    private readonly Channel<IComputeBuffer> _channel;
    private readonly IComputeDevice _device;
    private readonly ulong _bufferSize;
    private readonly BufferType _type;
    private readonly DataType _dtype;
    private readonly int _capacity;
    private bool _disposed;

    /// <param name="capacity">Maximum number of buffers to keep in the pool.</param>
    /// <param name="bufferSize">Size of each buffer in bytes.</param>
    /// <param name="device">GPU device for buffer creation.</param>
    public ChannelBufferPool(int capacity, ulong bufferSize, IComputeDevice device,
                             BufferType type = BufferType.Storage, DataType dtype = DataType.F32)
    {
        if (capacity < 1) throw new ArgumentOutOfRangeException(nameof(capacity));

        _capacity = capacity;
        _bufferSize = bufferSize;
        _device = device;
        _type = type;
        _dtype = dtype;

        var opts = new BoundedChannelOptions(capacity)
        {
            FullMode = BoundedChannelFullMode.Wait
        };
        _channel = Channel.CreateBounded<IComputeBuffer>(opts);

        // Pre-fill the pool
        for (int i = 0; i < capacity; i++)
        {
            var buf = CreateNew();
            if (!_channel.Writer.TryWrite(buf))
                buf.Dispose(); // should never happen with Wait mode
        }
    }

    /// <summary>
    /// Rent a buffer from the pool. Non-blocking in hot path:
    /// returns a pooled buffer if available, otherwise allocates a new one
    /// so the GPU thread is never blocked.
    /// </summary>
    public IComputeBuffer Rent()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (_channel.Reader.TryRead(out var buf))
            return buf;
        return CreateNew();
    }

    /// <summary>
    /// Return a buffer to the pool. Non-blocking: if the pool is full,
    /// the buffer is disposed immediately.
    /// </summary>
    public void Return(IComputeBuffer buffer)
    {
        if (_disposed || buffer == null) return;
        if (!_channel.Writer.TryWrite(buffer))
            buffer.Dispose();
    }

    /// <summary>Number of buffers currently available in the pool.</summary>
    public int Available => _channel.Reader.Count;

    /// <summary>Maximum pool capacity.</summary>
    public int Capacity => _capacity;

    /// <summary>Total VRAM held by this pool (bytes).</summary>
    public ulong TotalBytes => (ulong)(_capacity + Available) * _bufferSize;

    private IComputeBuffer CreateNew()
        => _device.CreateBuffer(_bufferSize, _type, _dtype);

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _channel.Writer.Complete();
        while (_channel.Reader.TryRead(out var buf))
            buf.Dispose();
    }
}
