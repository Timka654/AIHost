using AIHost.Compute;
using AIHost.GGUF;
using AIHost.ICompute;

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
        => _caches = ops.Select(o => new KVCache(o)).ToArray();

    public KVCache ForDevice(int devIndex) => _caches[devIndex];

    /// <summary>Sequence length (same on all devices after any forward pass).</summary>
    public int SequenceLength => _caches[0].SequenceLength;

    public void Clear() { foreach (var c in _caches) c.Clear(); }

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
/// Each device gets its own IGGUFModel instance (same file, different device context)
/// so GPU buffers are allocated on the correct device.
/// Cross-device activation transfer goes through CPU staging (ReadData → FromData).
/// </summary>
public class MultiGPUTransformer : IDisposable
{
    private readonly IComputeDevice[] _devices;
    private readonly IGGUFModel[] _models;   // one per device
    private readonly Transformer[] _transformers;
    private readonly int[] _deviceFirstLayer; // global layer index where each device starts
    private bool _disposed;

    public int DeviceCount   => _devices.Length;
    public int TotalLayers   => _transformers[0].LayerCount;
    /// <summary>GGUF reader from device-0 model; use to load the tokenizer.</summary>
    public IGGUFModel PrimaryModel => _models[0];

    // ── Constructors ─────────────────────────────────────────────────────────

    /// <summary>
    /// Main constructor: factory creates one IGGUFModel per device so each device
    /// owns its GPU buffers.  Devices are already-created IComputeDevice instances.
    /// </summary>
    public MultiGPUTransformer(
        IComputeDevice[]          devices,
        Func<IComputeDevice, IGGUFModel> modelFactory,
        int[]?                    layerSplit = null)
    {
        _devices = devices;
        int n = devices.Length;

        // Create one model per device (each calls CreateBuffer on its own device)
        _models = devices.Select(modelFactory).ToArray();

        // Determine total layer count from first model
        var tempXfm = new Transformer(devices[0], _models[0]);
        int numLayers = tempXfm.LayerCount;
        tempXfm.Dispose();

        // Build per-device layer boundary array [0, split..., numLayers]
        _deviceFirstLayer = new int[n + 1];
        if (layerSplit != null)
        {
            if (layerSplit.Length != n - 1)
                throw new ArgumentException(
                    $"layerSplit needs {n - 1} entries for {n} devices, got {layerSplit.Length}");
            _deviceFirstLayer[0] = 0;
            for (int i = 0; i < layerSplit.Length; i++)
                _deviceFirstLayer[i + 1] = layerSplit[i];
            _deviceFirstLayer[n] = numLayers;
        }
        else
        {
            for (int i = 0; i <= n; i++)
                _deviceFirstLayer[i] = numLayers * i / n;
        }

        // Create and load one Transformer per device
        _transformers = new Transformer[n];
        for (int d = 0; d < n; d++)
        {
            _transformers[d] = new Transformer(devices[d], _models[d]);
            _transformers[d].LoadWeightsPartial(
                globalFirstLayer: _deviceFirstLayer[d],
                globalLastLayer:  _deviceFirstLayer[d + 1],
                withEmbedding:    d == 0,
                withHead:         d == n - 1);
        }

        Console.WriteLine($"[MultiGPU] {n} device(s), {numLayers} layers total");
        for (int d = 0; d < n; d++)
            Console.WriteLine($"  Device {d}: layers {_deviceFirstLayer[d]}..{_deviceFirstLayer[d + 1] - 1}" +
                              (d == 0 ? " + embedding" : "") + (d == n - 1 ? " + head" : ""));
    }

    // ── Forward pass ─────────────────────────────────────────────────────────

    /// <summary>
    /// Full forward pass across all GPUs.
    /// Returns logits tensor on the last device — caller disposes.
    /// </summary>
    public Tensor Forward(int[] tokenIds, uint startPosition, MultiDeviceKVCache? kvCache = null)
    {
        // 1. Embedding on device 0
        Tensor x = _transformers[0].ForwardEmbedding(tokenIds);

        // 2. Layers: each device runs its slice, transferring activations as needed
        for (int d = 0; d < _devices.Length; d++)
        {
            if (d > 0)
                x = TransferTensor(x, _transformers[d].Ops);

            x = _transformers[d].ForwardLayers(x, startPosition, kvCache?.ForDevice(d));
        }

        // 3. Head (output_norm + lm_head) on last device
        return _transformers[^1].ForwardHead(x);
    }

    /// <summary>Create a fresh KV-cache sized for this model's devices.</summary>
    public MultiDeviceKVCache CreateKVCache()
        => new(_transformers.Select(t => t.Ops).ToArray());

    // ── Helpers ───────────────────────────────────────────────────────────────

    /// <summary>
    /// CPU-staged transfer: read src from its GPU, upload to dstOps.Device.
    /// src is disposed; caller receives the new tensor.
    /// </summary>
    private static Tensor TransferTensor(Tensor src, ComputeOps dstOps)
    {
        float[] data = src.ReadData();
        var dst = Tensor.FromData(dstOps.Device, data, src.Shape, "x_transfer");
        src.Dispose();
        return dst;
    }

    public void Dispose()
    {
        if (_disposed) return;
        foreach (var t in _transformers) t.Dispose();
        foreach (var m in _models)      m.Dispose();
        foreach (var d in _devices)     d.Dispose();
        _disposed = true;
    }
}
