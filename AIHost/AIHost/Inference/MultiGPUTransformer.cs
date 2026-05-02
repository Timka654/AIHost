using AIHost.Compute;
using AIHost.GGUF;
using AIHost.ICompute;
using AIHost.ICompute.Vulkan;

namespace AIHost.Inference;

/// <summary>
/// Holds one KVCache per GPU device.
/// Local layer indices (0-based per device) are used — each device owns its own cache.
/// </summary>
public class MultiDeviceKVCache : IDisposable
{
    private readonly KVCache[] _caches;
    private bool _disposed;

    public MultiDeviceKVCache(ComputeOps[] ops)
    {
        _caches = ops.Select(o => new KVCache(o)).ToArray();
    }

    public KVCache ForDevice(int devIndex) => _caches[devIndex];

    /// <summary>Sequence length (same on all devices after any forward pass).</summary>
    public int SequenceLength => _caches[0].SequenceLength;

    public void Clear()
    {
        foreach (var c in _caches) c.Clear();
    }

    public void Dispose()
    {
        if (_disposed) return;
        foreach (var c in _caches) c.Dispose();
        _disposed = true;
    }
}

/// <summary>
/// Splits a transformer model across multiple GPU devices using layer-wise parallelism.
///
/// Layout:
///   Device 0 : token_embd.weight + layers [0 .. split[0])
///   Device 1 : layers [split[0] .. split[1])
///   ...
///   Device N-1: layers [split[N-2] .. numLayers) + output_norm + output.weight
///
/// Cross-device activation transfer goes through CPU staging (ReadData → FromData).
/// For modest models (&lt;2 GB activation per token this is typically fast enough.
/// </summary>
public class MultiGPUTransformer : IDisposable
{
    private readonly IComputeDevice[] _devices;
    private readonly Transformer[] _transformers;
    private readonly int[] _deviceFirstLayer; // global layer index where each device starts
    private bool _disposed;

    public int DeviceCount => _devices.Length;
    public int TotalLayers => _transformers[0].LayerCount; // numLayers from metadata

    /// <summary>
    /// Create a multi-GPU transformer with automatic even layer distribution.
    /// </summary>
    public static MultiGPUTransformer Create(IGGUFModel model, int[] deviceIndices)
    {
        if (deviceIndices.Length == 0)
            throw new ArgumentException("Need at least one device");

        var devices = deviceIndices.Select(i => (IComputeDevice)new VulkanComputeDevice(i)).ToArray();
        return new MultiGPUTransformer(devices, model, layerSplit: null);
    }

    /// <summary>
    /// Create with explicit devices and optional layer-split boundaries.
    /// layerSplit[i] = global layer index where device i+1 starts.
    /// If null, layers are distributed evenly across devices.
    /// </summary>
    public MultiGPUTransformer(IComputeDevice[] devices, IGGUFModel model, int[]? layerSplit = null)
    {
        _devices = devices;
        int n = devices.Length;

        // Build per-device layer ranges
        var tempTransformer = new Transformer(devices[0], model);
        int numLayers = tempTransformer.LayerCount;
        tempTransformer.Dispose();

        _deviceFirstLayer = new int[n + 1]; // [n+1]: boundaries; last = numLayers
        if (layerSplit != null)
        {
            if (layerSplit.Length != n - 1)
                throw new ArgumentException($"layerSplit must have {n - 1} entries for {n} devices");
            _deviceFirstLayer[0] = 0;
            for (int i = 0; i < layerSplit.Length; i++)
                _deviceFirstLayer[i + 1] = layerSplit[i];
            _deviceFirstLayer[n] = numLayers;
        }
        else
        {
            // Even split
            for (int i = 0; i <= n; i++)
                _deviceFirstLayer[i] = (numLayers * i) / n;
        }

        // Create and load one Transformer per device
        _transformers = new Transformer[n];
        for (int d = 0; d < n; d++)
        {
            _transformers[d] = new Transformer(devices[d], model);
            bool isFirst = d == 0;
            bool isLast  = d == n - 1;
            _transformers[d].LoadWeightsPartial(
                globalFirstLayer: _deviceFirstLayer[d],
                globalLastLayer:  _deviceFirstLayer[d + 1],
                withEmbedding:    isFirst,
                withHead:         isLast);
        }

        Console.WriteLine($"[MultiGPU] {n} device(s), {numLayers} layers total");
        for (int d = 0; d < n; d++)
            Console.WriteLine($"  Device {d}: layers {_deviceFirstLayer[d]}..{_deviceFirstLayer[d + 1] - 1}");
    }

    // ── Forward pass ────────────────────────────────────────────────────────────

    /// <summary>
    /// Full forward pass across all GPUs.
    /// Returns logits tensor on the last device. Caller disposes it.
    /// </summary>
    public Tensor Forward(int[] tokenIds, uint startPosition, MultiDeviceKVCache? kvCache = null)
    {
        // 1. Embedding on device 0
        Tensor x = _transformers[0].ForwardEmbedding(tokenIds);
        int prevDevice = 0;

        // 2. Layers: each device runs its slice
        for (int d = 0; d < _devices.Length; d++)
        {
            if (d > 0)
            {
                // Transfer activation from previous device to device d (via CPU)
                x = TransferTensor(x, _transformers[d].Ops, "x_transfer");
                prevDevice = d;
            }
            x = _transformers[d].ForwardLayers(x, startPosition, kvCache?.ForDevice(d));
        }

        // 3. Head (output_norm + lm_head) on last device
        return _transformers[^1].ForwardHead(x);
    }

    /// <summary>
    /// Create a fresh MultiDeviceKVCache for this model's devices.
    /// </summary>
    public MultiDeviceKVCache CreateKVCache()
        => new MultiDeviceKVCache(_transformers.Select(t => t.Ops).ToArray());

    // ── Internals ───────────────────────────────────────────────────────────────

    /// <summary>
    /// Transfer a tensor between GPU devices via CPU staging buffer.
    /// src is disposed; returns a new tensor on dstOps.Device.
    /// </summary>
    private static Tensor TransferTensor(Tensor src, ComputeOps dstOps, string name)
    {
        float[] data = src.ReadData();
        var dst = Tensor.FromData(dstOps.Device, data, src.Shape, name);
        src.Dispose();
        return dst;
    }

    public void Dispose()
    {
        if (_disposed) return;
        foreach (var t in _transformers) t.Dispose();
        foreach (var d in _devices) d.Dispose();
        _disposed = true;
    }
}
