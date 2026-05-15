using AIHost.GGUF;
using AIHost.ICompute;
using AIHost.ICompute.Vulkan;
using AIHost.Inference;
using Microsoft.Extensions.Logging;
using System.Collections.Concurrent;
using System.Runtime.InteropServices;
using System.Threading.Channels;

namespace AIHost.Compute;

/// <summary>
/// Channel-only GPU compute API. All dispatches go through GpuExecutor
/// (dedicated thread) — one command buffer, one fence wait per token.
/// No fallback — USE_CHANNEL is always true.
/// </summary>
public class ComputeOps : IDisposable
{
    private readonly IComputeDevice _device;
    private readonly GpuExecutor _gpuExecutor;
    private readonly ChannelWriter<GpuTask> _taskWriter;
    private readonly ConcurrentDictionary<string, IComputeKernel> _kernelCache = new();
    private bool _disposed;

    // Layer tracking for BeginLayerTask
    private int _currentLayer;
    internal void SetCurrentLayer(int layer) => _currentLayer = layer;
    public bool UseChannelPipeline => true;

    internal bool _dbgLayer0 = true;
    private readonly ILogger<ComputeOps> _logger = AppLogger.Create<ComputeOps>();

    // ── Arena allocator ────────────────────────────────────────────────────────
    private VulkanArenaAllocator? _arena;
    public bool HasArena => _arena != null;

    // ── Persistent buffers ─────────────────────────────────────────────────────
    private readonly IComputeBuffer _persistentZeroOffset;
    private bool _persistentOffsetInitialized;
    private IComputeBuffer? _persistentSsmParams;
    private IComputeBuffer? _persistentRowIndex;

    public IComputeDevice Device => _device;

    public ComputeOps(IComputeDevice device, GpuLeasePool? leasePool = null,
        ILogger<ComputeOps>? logger = null)
    {
        _device = device;
        if (logger != null) _logger = logger;

        // Auto-create lease pool if not provided
        var pool = leasePool ?? new GpuLeasePool(device);
        _gpuExecutor = new GpuExecutor(device, pool, _logger);
        _taskWriter = _gpuExecutor.Writer;

        // Persistent zero-offset buffer for K-quant dequantization
        _persistentZeroOffset = _device.CreateBuffer(sizeof(uint), BufferType.Storage, DataType.I32);
        uint[] zeroOffset = { 0u };
        _persistentZeroOffset.Write(zeroOffset);
        _persistentOffsetInitialized = true;
    }

    // ── Kernel name → shader source mapping (all kernels must be registered here) ──
    private static readonly IReadOnlyDictionary<string, Func<string>> _kernelSources =
        new Dictionary<string, Func<string>>
        {
            ["copy"] = () => ComputeShaders.Copy,
            ["matmul_weights_t_f32"] = () => ComputeShaders.MatMulWeightsTF32,
            ["matmul_weights_f32"] = () => ComputeShaders.MatMulWeightsF32,
            ["matmul_f32"] = () => ComputeShaders.MatMulF32,
            ["elemwise_mul"] = () => ComputeShaders.ElementWiseMul,
            ["elemwise_add"] = () => ComputeShaders.ElementWiseAdd,
            ["silu"] = () => ComputeShaders.SiLU,
            ["sigmoid"] = () => ComputeShaders.Sigmoid,
            ["concat_axis0"] = () => ComputeShaders.ConcatAxis0,
            ["concat_axis1"] = () => ComputeShaders.ConcatAxis1,
            ["layer_norm"] = () => ComputeShaders.LayerNorm,
            ["softmax"] = () => ComputeShaders.Softmax,
            ["rowwise_softmax"] = () => ComputeShaders.RowwiseSoftmax,
            ["rope_full"] = () => ComputeShaders.RoPEFull,
            ["scale"] = () => ComputeShaders.Scale,
            ["slice_cols"] = () => ComputeShaders.SliceCols,
            ["deinterleave_q_gate"] = () => ComputeShaders.DeinterleaveQGate,
            ["scatter_cols"] = () => ComputeShaders.ScatterCols,
            ["repeat_kv_heads"] = () => ComputeShaders.RepeatKVHeads,
            ["repeat_columns"] = () => ComputeShaders.RepeatColumns,
            ["causal_mask"] = () => ComputeShaders.CausalMask,
            ["fused_mha_generate"] = () => ComputeShaders.FusedMHAGenerate,
            ["transpose"] = () => ComputeShaders.Transpose,
            ["embedding_lookup"] = () => ComputeShaders.EmbeddingLookup,
            ["ssm_gdn_decode"] = () => ComputeShaders.SsmGdnDecode,
            ["ssm_gdn_recur"] = () => ComputeShaders.SsmGdnRecur,
            ["dequant_q2k"] = () => ComputeShaders.DequantizeQ2K,
            ["dequant_q3k"] = () => ComputeShaders.DequantizeQ3K,
            ["dequant_q4k"] = () => ComputeShaders.DequantizeQ4K,
            ["dequant_q5k"] = () => ComputeShaders.DequantizeQ5K,
            ["dequant_q6k"] = () => ComputeShaders.DequantizeQ6K,
        };

    // ── Kernel factory (shared between ComputeOps cache and GpuExecutor) ──
    private IComputeKernel GetOrCreateKernel(string name, Func<string> source)
    {
        if (_kernelCache.TryGetValue(name, out var cached))
            return cached;

        var kernel = _device.CreateKernel(source(), "main");
        kernel.Compile();
        _kernelCache[name] = kernel;
        _gpuExecutor.RegisterKernel(name, kernel);
        return kernel;
    }

    // ── Batch infrastructure (channel-only) ────────────────────────────────────

    public void BeginBatch()
        => _taskWriter.TryWrite(BeginLayerTask.Create(_currentLayer));

    public void Flush()
    {
        // Check if GpuExecutor is already dead
        if (_gpuExecutor.IsFailed)
            throw new InvalidOperationException(
                $"[ComputeOps] GPU executor is in failed state: {_gpuExecutor.LastError}");

        var tcs = new System.Threading.Tasks.TaskCompletionSource<GpuResult>();
        var task = FlushAndWaitTask.Create(tcs);
        _taskWriter.TryWrite(task);

        // Timeout: 120 seconds should be enough for any reasonable dispatch
        if (!tcs.Task.Wait(TimeSpan.FromSeconds(120)))
            throw new TimeoutException("[ComputeOps] GPU flush timed out after 120s");

        var result = tcs.Task.Result;
        if (!result.Success)
            throw new InvalidOperationException(
                $"[ComputeOps] GPU flush failed: {result.Error}");

        _arena?.Reset();
    }

    private void MaybeFlush()
        => _taskWriter.TryWrite(BarrierTask.Create());

    private void EmitDispatch(string kernelName, IComputeBuffer?[] args, int argCount, uint[] workgroups)
    {
        // Lazy-init: ensure kernel is compiled and registered in GpuExecutor
        if (_kernelSources.TryGetValue(kernelName, out var source))
            GetOrCreateKernel(kernelName, source);
        else
        {
            _logger.LogWarning("[ComputeOps] EmitDispatch: unknown kernel '{Name}'", kernelName);
            // Still emit — GpuExecutor will throw with a clear message
        }

        var task = DispatchKernelTask.Create(kernelName, workgroups);
        for (int i = 0; i < argCount; i++)
            task.Arguments[i] = args[i];
        task.ArgCount = argCount;
        _taskWriter.TryWrite(task);
    }

    public void DeferExternal(IDisposable d) { /* no-op in channel mode */ }

    /// <summary>Insert a compute barrier (channel mode).</summary>
    public void InsertBarrier() => _taskWriter.TryWrite(BarrierTask.Create());

    /// <summary>Attach arena allocator to this ComputeOps instance.</summary>
    internal void AttachArena(VulkanArenaAllocator arena) => _arena = arena;

    /// <summary>Softmax without explicit rows parameter (infers from tensor shape).</summary>
    public void Softmax(Tensor tensor)
    {
        int rows = tensor.Shape.Rank >= 2 ? tensor.Shape[0] : 1;
        Softmax(tensor, rows);
    }

    /// <summary>ApplyRoPE with headDim (infers numHeads from tensor shape).</summary>
    public void ApplyRoPE(Tensor tensor, uint position, int headDim)
    {
        int totalDim = tensor.Shape.Rank >= 2 ? tensor.Shape[1] : tensor.Shape[0];
        int numHeads = totalDim / headDim;
        ApplyRoPEFull(tensor, position, numHeads, headDim);
    }

    /// <summary>ApplyRoPE compatibility shim.</summary>
    public void ApplyRoPE(Tensor tensor, uint position, int numHeads, int headDim)
        => ApplyRoPEFull(tensor, position, numHeads, headDim);

    /// <summary>
    /// Full-fused transformer layer (used by LlamaFormat).
    /// Channel-mode: emits all individual dispatches.
    /// </summary>
    public Tensor TransformerLayer(Tensor x,
        Tensor wAttnNorm, Tensor wQ, Tensor wK, Tensor wV, Tensor wAttnOut,
        Tensor wFfnNorm, Tensor wGate, Tensor wUp, Tensor wDown,
        int numHeads, uint position, KVCache? kvCache, int layerIdx)
    {
        int sl = x.Shape[0], dm = x.Shape[1];
        int hd = wQ.Shape[1] / numHeads;
        int kd = wK.Shape[1] / (wK.Shape[1] / hd);
        int vd = wV.Shape[1] / (wV.Shape[1] / hd);

        var xn = Clone(x, "an");
        LayerNorm(xn, wAttnNorm);

        var q = MatMulWeights(xn, wQ, "q");
        var k = MatMulWeights(xn, wK, "k");
        var v = MatMulWeights(xn, wV, "v");

        MaybeFlush();

        ApplyRoPEFull(q, position, numHeads, hd);
        ApplyRoPEFull(k, position, 1, kd);

        MaybeFlush();

        Tensor ao;
        if (kvCache != null)
        {
            kvCache.Add(layerIdx, k, v);
            Flush();
            var (ck, cv) = kvCache.Get(layerIdx);
            ao = MultiHeadAttention(q, ck!, cv!, numHeads, position, "ao");
        }
        else
        {
            ao = MultiHeadAttention(q, k, v, numHeads, position, "ao");
        }

        var ap = MatMulWeights(ao, wAttnOut, "ap");
        var x1 = Add(x, ap, "x1");

        MaybeFlush();

        var x1n = Clone(x1, "pn");
        LayerNorm(x1n, wFfnNorm);

        var fo = FeedForward(x1n, wGate, wUp, wDown, "fo");
        var ou = Add(x1, fo, "ou");

        return ou;
    }

    /// <summary>
    /// Embedding lookup from int[] token IDs + F32 weight tensor (convenience overload).
    /// </summary>
    public Tensor EmbeddingLookup(int[] tokenIds, Tensor weightF32, string? resultName = null)
    {
        int sl = tokenIds.Length, dim = weightF32.Shape[1];
        var tokenBuf = _device.CreateBuffer((ulong)(tokenIds.Length * sizeof(int)), BufferType.Storage, DataType.I32);
        tokenBuf.Write(tokenIds);

        var result = Tensor.Create(_device, TensorShape.Matrix(sl, dim), DataType.F32, resultName ?? "emb");
        var paramBytes = new byte[8];
        BitConverter.GetBytes((uint)sl).CopyTo(paramBytes, 0);
        BitConverter.GetBytes((uint)dim).CopyTo(paramBytes, 4);
        var paramsBuffer = _device.CreateBuffer(8, BufferType.Storage, DataType.I32);
        paramsBuffer.Write(paramBytes);

        EmitDispatch("embedding_lookup",
            [tokenBuf, weightF32.Buffer, result.Buffer, paramsBuffer], 4,
            [(uint)sl, 1, 1]);
        return result;
    }

    /// <summary>
    /// Embedding lookup from quantized tensor. Delegates to Dequantize + EmbeddingLookup.
    /// </summary>
    public unsafe Tensor EmbeddingLookupFromQuantized(int[] tokenIds, Tensor quantizedEmb, string? resultName = null)
    {
        int vocab = quantizedEmb.Shape[0], dim = quantizedEmb.Shape[1];
        var f32 = Dequantize(quantizedEmb);

        var tokenBuf = _device.CreateBuffer((ulong)(tokenIds.Length * sizeof(int)), BufferType.Storage, DataType.I32);
        tokenBuf.Write(tokenIds);

        int sl = tokenIds.Length;
        var result = Tensor.Create(_device, TensorShape.Matrix(sl, dim), DataType.F32, resultName ?? "emb");

        var paramBytes = new byte[8];
        BitConverter.GetBytes((uint)sl).CopyTo(paramBytes, 0);
        BitConverter.GetBytes((uint)dim).CopyTo(paramBytes, 4);
        var paramsBuffer = _device.CreateBuffer(8, BufferType.Storage, DataType.I32);
        paramsBuffer.Write(paramBytes);

        EmitDispatch("embedding_lookup",
            [tokenBuf, f32.Buffer, result.Buffer, paramsBuffer], 4,
            [(uint)sl, 1, 1]);

        return result;
    }

    public void AllocateArena(ulong sizeBytes, ulong kvCacheBytes = 0)
    {
        if (_device is IComputeDevice)
        {
            // Arena allocation through Vulkan-specific path via VulkanDeviceContext
            // For non-Vulkan providers, arena is skipped (fallback to CreateBuffer)
        }
    }

    internal Tensor? AllocTempTensor(TensorShape shape, string? name = null)
    {
        if (_arena == null) return null;
        try
        {
            int totalElements = shape.TotalElements;
            var slice = _arena.Alloc((ulong)(totalElements * sizeof(float)));
            var buffer = VulkanComputeBuffer.CreateArenaView(slice.Buffer, slice.Offset, slice.Size, DataType.F32);
            return new Tensor(buffer, shape, DataType.F32, name ?? "arena_temp");
        }
        catch { return null; }
    }


    // ═══════════════════════════════════════════════════════════════════════════
    //  MATMUL OPERATIONS
    // ═══════════════════════════════════════════════════════════════════════════

    public Tensor MatMulWeightsT(Tensor a, Tensor b, string? resultName = null)
    {
        int M = a.Shape[0], K = a.Shape[1], J = b.Shape[0];
        if (K != b.Shape[1]) throw new ArgumentException($"Shape mismatch: {K} vs {b.Shape[1]}");
        var result = Tensor.Create(_device, TensorShape.Matrix(M, J), DataType.F32, resultName);
        var paramsBuffer = MakeParamsBuffer((uint)M, (uint)K, (uint)J);
        EmitDispatch("matmul_weights_t_f32",
            [a.Buffer, b.Buffer, result.Buffer, paramsBuffer], 4,
            [(uint)((J + 15) / 16), (uint)((M + 15) / 16), 1]);
        return result;
    }

    public Tensor MatMulWeights(Tensor a, Tensor b, string? resultName = null)
    {
        int M = a.Shape[0], K = a.Shape[1], N = b.Shape[1];
        if (K != b.Shape[0]) throw new ArgumentException($"Shape mismatch");
        var result = Tensor.Create(_device, TensorShape.Matrix(M, N), DataType.F32, resultName);
        var paramsBuffer = MakeParamsBuffer((uint)M, (uint)K, (uint)N);
        EmitDispatch("matmul_weights_f32",
            [a.Buffer, b.Buffer, result.Buffer, paramsBuffer], 4,
            [(uint)((N + 15) / 16), (uint)((M + 15) / 16), 1]);
        return result;
    }

    public Tensor MatMul(Tensor a, Tensor b, string? resultName = null)
    {
        int M = a.Shape[0], K = a.Shape[1], N = b.Shape[1];
        if (K != b.Shape[0]) throw new ArgumentException($"Shape mismatch");
        var result = Tensor.Create(_device, TensorShape.Matrix(M, N), DataType.F32, resultName);
        var paramsBuffer = MakeParamsBuffer((uint)M, (uint)K, (uint)N);
        EmitDispatch("matmul_f32",
            [a.Buffer, b.Buffer, result.Buffer, paramsBuffer], 4,
            [(uint)((N + 15) / 16), (uint)((M + 15) / 16), 1]);
        return result;
    }

    public Tensor MatMulWeightsLarge(Tensor a, Tensor quantized, string? resultName = null)
    {
        // Fallback to standard path for now — large tensor support TBD
        return MatMulWeights(a, quantized, resultName);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //  ELEMENT-WISE OPERATIONS
    // ═══════════════════════════════════════════════════════════════════════════

    public Tensor Multiply(Tensor a, Tensor b, string? resultName = null)
    {
        if (a.Shape.TotalElements != b.Shape.TotalElements)
            throw new ArgumentException("Shape mismatch");
        var result = Tensor.Create(_device, a.Shape, DataType.F32, resultName);
        uint total = (uint)a.Shape.TotalElements;
        var paramsBuffer = MakeParamsBuffer(total);
        EmitDispatch("elemwise_mul",
            [a.Buffer, b.Buffer, result.Buffer, paramsBuffer], 4,
            [(total + 255) / 256, 1, 1]);
        return result;
    }

    public Tensor Add(Tensor a, Tensor b, string? resultName = null)
    {
        if (a.Shape.TotalElements != b.Shape.TotalElements)
            throw new ArgumentException("Shape mismatch");
        var result = Tensor.Create(_device, a.Shape, DataType.F32, resultName);
        uint total = (uint)a.Shape.TotalElements;
        var paramsBuffer = MakeParamsBuffer(total);
        EmitDispatch("elemwise_add",
            [a.Buffer, b.Buffer, result.Buffer, paramsBuffer], 4,
            [(total + 255) / 256, 1, 1]);
        return result;
    }

    public void SiLU(Tensor tensor)
    {
        uint total = (uint)tensor.Shape.TotalElements;
        var paramsBuffer = MakeParamsBuffer(total);
        EmitDispatch("silu",
            [tensor.Buffer, paramsBuffer], 2,
            [(total + 255) / 256, 1, 1]);
    }

    public void Sigmoid(Tensor tensor)
    {
        uint total = (uint)tensor.Shape.TotalElements;
        var paramsBuffer = MakeParamsBuffer(total);
        EmitDispatch("sigmoid",
            [tensor.Buffer, paramsBuffer], 2,
            [(total + 255) / 256, 1, 1]);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //  CONCAT
    // ═══════════════════════════════════════════════════════════════════════════

    public Tensor Concat(Tensor a, Tensor b, int axis, string? resultName = null)
    {
        return axis == 0 ? ConcatAxis0(a, b, resultName) : ConcatAxis1(a, b, resultName);
    }

    private Tensor ConcatAxis0(Tensor a, Tensor b, string? resultName)
    {
        int rows = a.Shape[0] + b.Shape[0], cols = a.Shape[1];
        var result = Tensor.Create(_device, TensorShape.Matrix(rows, cols), DataType.F32, resultName);
        uint total = (uint)a.Shape.TotalElements;
        uint bRows = (uint)b.Shape[0], bCols = (uint)b.Shape[1];
        var paramsBuffer = MakeParamsBuffer(total, bRows, bCols);
        EmitDispatch("concat_axis0",
            [a.Buffer, b.Buffer, result.Buffer, paramsBuffer], 4,
            [(total + 255) / 256, 1, 1]);
        return result;
    }

    private Tensor ConcatAxis1(Tensor a, Tensor b, string? resultName)
    {
        int rows = a.Shape[0], cols = a.Shape[1] + b.Shape[1];
        var result = Tensor.Create(_device, TensorShape.Matrix(rows, cols), DataType.F32, resultName);
        uint aCols = (uint)a.Shape[1], bCols = (uint)b.Shape[1];
        var paramsBuffer = MakeParamsBuffer(aCols, bCols);
        EmitDispatch("concat_axis1",
            [a.Buffer, b.Buffer, result.Buffer, paramsBuffer], 4,
            [aCols + bCols, 1, 1]); // single row approach
        return result;
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //  NORMALIZATION & SOFTMAX
    // ═══════════════════════════════════════════════════════════════════════════

    public void LayerNorm(Tensor tensor, Tensor weight, float eps = 1e-5f)
    {
        int rows = tensor.Shape.Rank >= 2 ? tensor.Shape[0] : 1;
        int cols = tensor.Shape.Rank >= 2 ? tensor.Shape[1] : tensor.Shape[0];
        var paramBytes = new byte[12];
        BitConverter.GetBytes((uint)rows).CopyTo(paramBytes, 0);
        BitConverter.GetBytes((uint)cols).CopyTo(paramBytes, 4);
        BitConverter.GetBytes(eps).CopyTo(paramBytes, 8);
        var paramsBuffer = _device.CreateBuffer(12, BufferType.Storage, DataType.I32);
        paramsBuffer.Write(paramBytes);
        EmitDispatch("layer_norm",
            [tensor.Buffer, weight.Buffer, paramsBuffer], 3,
            [(uint)rows, 1, 1]);
    }

    public void LayerNormVirtual(Tensor tensor, Tensor weight, int numRows, int numCols, float eps = 1e-5f)
    {
        var paramBytes = new byte[12];
        BitConverter.GetBytes((uint)numRows).CopyTo(paramBytes, 0);
        BitConverter.GetBytes((uint)numCols).CopyTo(paramBytes, 4);
        BitConverter.GetBytes(eps).CopyTo(paramBytes, 8);
        var paramsBuffer = _device.CreateBuffer(12, BufferType.Storage, DataType.I32);
        paramsBuffer.Write(paramBytes);
        EmitDispatch("layer_norm",
            [tensor.Buffer, weight.Buffer, paramsBuffer], 3,
            [(uint)numRows, 1, 1]);
    }

    public void Softmax(Tensor tensor, int rows)
    {
        uint total = (uint)tensor.Shape.TotalElements;
        var paramsBuffer = MakeParamsBuffer(total);
        EmitDispatch("softmax",
            [tensor.Buffer, paramsBuffer], 2,
            [(total + 255) / 256, 1, 1]);
    }

    public void RowwiseSoftmax(Tensor tensor)
    {
        int rows = tensor.Shape.Rank >= 2 ? tensor.Shape[0] : 1;
        int cols = tensor.Shape[1];
        var paramBytes = new byte[8]; // rows (4) + cols (4)
        BitConverter.GetBytes((uint)rows).CopyTo(paramBytes, 0);
        BitConverter.GetBytes((uint)cols).CopyTo(paramBytes, 4);
        var paramsBuffer = _device.CreateBuffer(8, BufferType.Storage, DataType.I32);
        paramsBuffer.Write(paramBytes);
        EmitDispatch("rowwise_softmax",
            [tensor.Buffer, paramsBuffer], 2,
            [(uint)rows, 1, 1]);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //  RoPE
    // ═══════════════════════════════════════════════════════════════════════════

    public void ApplyRoPEFull(Tensor tensor, uint startPosition, int numHeads, int headDim,
        float theta = 10000.0f, int ropeDim = 0)
    {
        int effectiveRopeDim = ropeDim > 0 ? ropeDim : headDim;
        int numPairs = effectiveRopeDim / 2;

        // Pack params: numHeads(4) + headDim(4) + ropeDim(4) + startPosition(4) + theta(4) = 20 bytes
        var paramBytes = new byte[20];
        BitConverter.GetBytes((uint)numHeads).CopyTo(paramBytes, 0);
        BitConverter.GetBytes((uint)headDim).CopyTo(paramBytes, 4);
        BitConverter.GetBytes((uint)effectiveRopeDim).CopyTo(paramBytes, 8);
        BitConverter.GetBytes(startPosition).CopyTo(paramBytes, 12);
        BitConverter.GetBytes(theta).CopyTo(paramBytes, 16);
        var paramsBuffer = _device.CreateBuffer(20, BufferType.Storage, DataType.I32);
        paramsBuffer.Write(paramBytes);

        uint total = (uint)(numHeads * numPairs);
        EmitDispatch("rope_full",
            [tensor.Buffer, paramsBuffer], 2,
            [(total + 255) / 256, 1, 1]);
    }

    public void RoPE(Tensor tensor, uint position, int numHeads, int headDim,
        float theta = 10000.0f, int ropeDim = 0)
    {
        ApplyRoPEFull(tensor, position, numHeads, headDim, theta, ropeDim);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //  SCALE
    // ═══════════════════════════════════════════════════════════════════════════

    public void Scale(Tensor tensor, float scale)
    {
        var paramBytes = new byte[8];
        BitConverter.GetBytes((uint)tensor.Shape.TotalElements).CopyTo(paramBytes, 0);
        BitConverter.GetBytes(scale).CopyTo(paramBytes, 4);
        var paramsBuffer = _device.CreateBuffer(8, BufferType.Storage, DataType.I32);
        paramsBuffer.Write(paramBytes);
        uint total = (uint)tensor.Shape.TotalElements;
        EmitDispatch("scale",
            [tensor.Buffer, paramsBuffer], 2,
            [(total + 255) / 256, 1, 1]);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //  SLICE & SCATTER
    // ═══════════════════════════════════════════════════════════════════════════

    public Tensor SliceCols(Tensor input, int startCol, int numCols, string? resultName = null)
    {
        int rows = input.Shape[0];
        var result = Tensor.Create(_device, TensorShape.Matrix(rows, numCols), DataType.F32, resultName);
        var paramBytes = new byte[12]; // rows(4) + cols(4) + startCol(4)
        BitConverter.GetBytes((uint)rows).CopyTo(paramBytes, 0);
        BitConverter.GetBytes((uint)numCols).CopyTo(paramBytes, 4);
        BitConverter.GetBytes((uint)startCol).CopyTo(paramBytes, 8);
        var paramsBuffer = _device.CreateBuffer(12, BufferType.Storage, DataType.I32);
        paramsBuffer.Write(paramBytes);
        EmitDispatch("slice_cols",
            [input.Buffer, result.Buffer, paramsBuffer], 3,
            [(uint)rows, 1, 1]);
        return result;
    }

    public Tensor DeinterleaveQGate(Tensor input, uint numHeads, uint qDim, string? resultName = null)
    {
        int sl = input.Shape[0];
        var result = Tensor.Create(_device, input.Shape, DataType.F32, resultName);
        uint totalCols = (uint)input.Shape[1];
        var paramBytes = new byte[8]; // numHeads(4) + qDim(4)
        BitConverter.GetBytes(numHeads).CopyTo(paramBytes, 0);
        BitConverter.GetBytes(qDim).CopyTo(paramBytes, 4);
        var paramsBuffer = _device.CreateBuffer(8, BufferType.Storage, DataType.I32);
        paramsBuffer.Write(paramBytes);
        EmitDispatch("deinterleave_q_gate",
            [input.Buffer, result.Buffer, paramsBuffer], 3,
            [(uint)sl, 1, 1]);
        return result;
    }

    public void ScatterCols(Tensor src, Tensor dst, int startCol)
    {
        uint srcRows = (uint)src.Shape[0], srcCols = (uint)src.Shape[1];
        var paramBytes = new byte[12];
        BitConverter.GetBytes(srcRows).CopyTo(paramBytes, 0);
        BitConverter.GetBytes(srcCols).CopyTo(paramBytes, 4);
        BitConverter.GetBytes((uint)startCol).CopyTo(paramBytes, 8);
        var paramsBuffer = _device.CreateBuffer(12, BufferType.Storage, DataType.I32);
        paramsBuffer.Write(paramBytes);
        EmitDispatch("scatter_cols",
            [src.Buffer, dst.Buffer, paramsBuffer], 3,
            [srcRows, 1, 1]);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //  KV-CACHE & ATTENTION
    // ═══════════════════════════════════════════════════════════════════════════

    public Tensor RepeatKVHeads(Tensor input, int numKVHeadsTarget, string? resultName = null)
    {
        int sl = input.Shape[0], kd = input.Shape[1];
        var result = Tensor.Create(_device, TensorShape.Matrix(sl, kd), DataType.F32, resultName);
        uint total = (uint)sl * (uint)kd;
        EmitDispatch("repeat_kv_heads",
            [input.Buffer, result.Buffer], 2,
            [(total + 255) / 256, 1, 1]);
        return result;
    }

    public Tensor RepeatColumns(Tensor input, int numRepeats, string? resultName = null)
    {
        int sl = input.Shape[0], cols = input.Shape[1];
        var result = Tensor.Create(_device, TensorShape.Matrix(sl, cols * numRepeats), DataType.F32, resultName);
        uint total = (uint)sl * (uint)cols;
        var paramBytes = new byte[4]; BitConverter.GetBytes((uint)numRepeats).CopyTo(paramBytes, 0);
        var paramsBuffer = _device.CreateBuffer(4, BufferType.Storage, DataType.I32);
        paramsBuffer.Write(paramBytes);
        EmitDispatch("repeat_columns",
            [input.Buffer, result.Buffer, paramsBuffer], 3,
            [(total + 255) / 256, 1, 1]);
        return result;
    }

    public void CausalMask(Tensor scores)
    {
        int rows = scores.Shape[0], cols = scores.Shape[1];
        var paramBytes = new byte[8];
        BitConverter.GetBytes((uint)rows).CopyTo(paramBytes, 0);
        BitConverter.GetBytes((uint)cols).CopyTo(paramBytes, 4);
        var paramsBuffer = _device.CreateBuffer(8, BufferType.Storage, DataType.I32);
        paramsBuffer.Write(paramBytes);
        uint total = (uint)(rows * cols);
        EmitDispatch("causal_mask",
            [scores.Buffer, paramsBuffer], 2,
            [(total + 255) / 256, 1, 1]);
    }

    public Tensor MultiHeadAttention(Tensor Q, Tensor K, Tensor V, int numHeads, uint position, string? resultName = null)
    {
        int sl = Q.Shape[0], hd = Q.Shape[1];
        var result = Tensor.Create(_device, TensorShape.Matrix(sl, hd), DataType.F32, resultName);


        var paramBytes = CopyBuffer(Q, K, V, numHeads, position, result);
        MaybeFlush();
        DispatchAndDeferAttention(Q, result);
        return result;
    }

    private byte[] CopyBuffer(Tensor Q, Tensor K, Tensor V, int numHeads, uint position, Tensor result)
    {
        int sl = Q.Shape[0], hd = Q.Shape[1], kd = K.Shape[1], vd = V.Shape[1];
        var paramBytes = PackAttentionParams(sl, hd, kd, vd, numHeads, position);
        var paramsBuffer = _device.CreateBuffer((uint)paramBytes.Length, BufferType.Storage, DataType.I32);
        paramsBuffer.Write(paramBytes);

        EmitDispatch("fused_mha_generate",
            [Q.Buffer, K.Buffer, V.Buffer, result.Buffer, paramsBuffer], 5,
            [(uint)((sl + 15) / 16) * (uint)numHeads, 1, 1]);
        return paramBytes;
    }

    private void DispatchAndDeferAttention(Tensor Q, Tensor result)
    {
        // No-op in channel mode — dispatch already emitted
    }

    private static byte[] PackAttentionParams(int sl, int hd, int kd, int vd, int numHeads, uint position)
    {
        var buf = new byte[24]; // 6 × uint32
        BitConverter.GetBytes((uint)sl).CopyTo(buf, 0);
        BitConverter.GetBytes((uint)hd).CopyTo(buf, 4);
        BitConverter.GetBytes((uint)kd).CopyTo(buf, 8);
        BitConverter.GetBytes((uint)vd).CopyTo(buf, 12);
        BitConverter.GetBytes((uint)numHeads).CopyTo(buf, 16);
        BitConverter.GetBytes(position).CopyTo(buf, 20);
        return buf;
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //  TRANSPOSE
    // ═══════════════════════════════════════════════════════════════════════════

    public Tensor Transpose(Tensor input, string? resultName = null)
    {
        int rows = input.Shape[0], cols = input.Shape[1];
        var result = Tensor.Create(_device, TensorShape.Matrix(cols, rows), DataType.F32, resultName);
        var paramBytes = new byte[8];
        BitConverter.GetBytes((uint)rows).CopyTo(paramBytes, 0);
        BitConverter.GetBytes((uint)cols).CopyTo(paramBytes, 4);
        var paramsBuffer = _device.CreateBuffer(8, BufferType.Storage, DataType.I32);
        paramsBuffer.Write(paramBytes);
        EmitDispatch("transpose",
            [input.Buffer, result.Buffer, paramsBuffer], 3,
            [(uint)((rows * cols + 255) / 256), 1, 1]);
        return result;
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //  DEQUANTIZATION
    // ═══════════════════════════════════════════════════════════════════════════

    public Tensor Dequantize(Tensor quantized, string? resultName = null)
    {
        int rows = quantized.Shape.Rank >= 2 ? quantized.Shape[0] : 1;
        int cols = quantized.Shape.Rank >= 2 ? quantized.Shape[1] : quantized.Shape[0];
        var result = Tensor.Create(_device, TensorShape.Matrix(rows, cols), DataType.F32, resultName);
        DequantizeInto(quantized, result);
        return result;
    }

    public void DequantizeInto(Tensor quantized, Tensor target)
    {
        if (target.DataType != DataType.F32)
            throw new ArgumentException("Target must be F32");

        // F32 weights — clone via copy shader (no dequant)
        if (quantized.DataType == DataType.F32)
        {
            uint total = (uint)quantized.Shape.TotalElements;
            var p = MakeParamsBuffer(total);
            EmitDispatch("copy", [quantized.Buffer, target.Buffer, p], 3, [(total + 255) / 256, 1, 1]);
            return;
        }

        string kernelName = quantized.DataType switch
        {
            DataType.Q2_K => "dequant_q2k",
            DataType.Q3_K => "dequant_q3k",
            DataType.Q4_K => "dequant_q4k",
            DataType.Q5_K => "dequant_q5k",
            DataType.Q6_K => "dequant_q6k",
            _ => throw new NotSupportedException($"DequantizeInto not supported for {quantized.DataType}")
        };

        GetOrCreateKernel(kernelName, () => quantized.DataType switch
        {
            DataType.Q2_K => ComputeShaders.DequantizeQ2K,
            DataType.Q3_K => ComputeShaders.DequantizeQ3K,
            DataType.Q4_K => ComputeShaders.DequantizeQ4K,
            DataType.Q5_K => ComputeShaders.DequantizeQ5K,
            DataType.Q6_K => ComputeShaders.DequantizeQ6K,
            _ => throw new NotSupportedException()
        });

        uint totalElements = (uint)quantized.Shape.TotalElements;
        uint totalGroups = (totalElements + 255) / 256;

        const uint MaxGroupsPerDispatch = 32768;
        const uint SuperblockElements = 256;

        if (totalGroups <= MaxGroupsPerDispatch)
        {
            // Single dispatch — zero offset
            EmitDispatch(kernelName,
                [quantized.Buffer, target.Buffer, _persistentZeroOffset], 3,
                [totalGroups, 1, 1]);
        }
        else
        {
            // Chunked dispatch — each chunk gets its own offset buffer
            uint groupsPerChunk = MaxGroupsPerDispatch;
            uint numChunks = (totalGroups + groupsPerChunk - 1) / groupsPerChunk;

            for (uint chunk = 0; chunk < numChunks; chunk++)
            {
                uint chunkStartGroup = chunk * groupsPerChunk;
                uint chunkEndGroup = Math.Min(chunkStartGroup + groupsPerChunk, totalGroups);
                uint chunkGroups = chunkEndGroup - chunkStartGroup;
                uint elementOffset = chunkStartGroup * SuperblockElements;

                var offsetBuf = _device.CreateBuffer(sizeof(uint), BufferType.Storage, DataType.I32);
                offsetBuf.Write(new[] { elementOffset });

                EmitDispatch(kernelName,
                    [quantized.Buffer, target.Buffer, offsetBuf], 3,
                    [chunkGroups, 1, 1]);

                // Chunked path needs barriers between chunks for correct offset reading
                MaybeFlush();
            }
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //  EMBEDDING
    // ═══════════════════════════════════════════════════════════════════════════

    public Tensor EmbeddingLookup(Tensor tokenIdx, Tensor weightF32, string? resultName = null)
    {
        int sl = tokenIdx.Shape.TotalElements, cols = weightF32.Shape[1];
        var result = Tensor.Create(_device, TensorShape.Matrix(sl, cols), DataType.F32, resultName);

        var paramBytes = new byte[8];
        BitConverter.GetBytes((uint)sl).CopyTo(paramBytes, 0);
        BitConverter.GetBytes((uint)cols).CopyTo(paramBytes, 4);
        var paramsBuffer = _device.CreateBuffer(8, BufferType.Storage, DataType.I32);
        paramsBuffer.Write(paramBytes);

        EmitDispatch("embedding_lookup",
            [tokenIdx.Buffer, weightF32.Buffer, result.Buffer, paramsBuffer], 4,
            [(uint)sl, 1, 1]);
        return result;
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //  CLONE (GPU memcpy)
    // ═══════════════════════════════════════════════════════════════════════════

    public Tensor Clone(Tensor input, string? resultName = null)
    {
        _logger.LogDebug("[DBG_OPS] Clone name={Name} shape={Shape}",
            resultName ?? "?", input.Shape);
        var result = Tensor.Create(_device, input.Shape, DataType.F32, resultName);
        uint total = (uint)input.Shape.TotalElements;
        var paramsBuffer = MakeParamsBuffer(total);
        EmitDispatch("copy",
            [input.Buffer, result.Buffer, paramsBuffer], 3,
            [(total + 255) / 256, 1, 1]);
        return result;
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //  FEED FORWARD
    // ═══════════════════════════════════════════════════════════════════════════

    public Tensor FeedForward(Tensor x, Tensor wGate, Tensor wUp, Tensor wDown, string? resultName = null)
    {
        int sl = x.Shape[0], dm = x.Shape[1], inter = wGate.Shape[1];
        var hidden = Tensor.Create(_device, TensorShape.Matrix(sl, inter), DataType.F32, "ffn_hidden");
        var gated = Tensor.Create(_device, TensorShape.Matrix(sl, inter), DataType.F32, "ffn_gated");

        // Gate projection: hidden = x @ wGate
        var paramsBuf1 = MakeParamsBuffer((uint)sl, (uint)dm, (uint)inter);
        EmitDispatch("matmul_weights_f32",
            [x.Buffer, wGate.Buffer, hidden.Buffer, paramsBuf1], 4,
            [(uint)((inter + 15) / 16), (uint)((sl + 15) / 16), 1]);

        MaybeFlush();

        // Up projection: gated = x @ wUp
        var paramsBuf2 = MakeParamsBuffer((uint)sl, (uint)dm, (uint)inter);
        EmitDispatch("matmul_weights_f32",
            [x.Buffer, wUp.Buffer, gated.Buffer, paramsBuf2], 4,
            [(uint)((inter + 15) / 16), (uint)((sl + 15) / 16), 1]);

        MaybeFlush();

        // SiLU activation
        SiLU(gated);

        MaybeFlush();

        // Gate: hidden *= gated
        // Multiply produces [sl, inter] result, then MatMulWeights with [inter, dm]
        var gatedResult = Multiply(hidden, gated, "ffn_gated_result");

        MaybeFlush();

        // Down projection: result = gatedResult @ wDown
        var result = Tensor.Create(_device, TensorShape.Matrix(sl, dm), DataType.F32, resultName);
        int interD = wDown.Shape[0];
        var paramsBuf3 = MakeParamsBuffer((uint)sl, (uint)interD, (uint)dm);
        EmitDispatch("matmul_weights_f32",
            [gatedResult.Buffer, wDown.Buffer, result.Buffer, paramsBuf3], 4,
            [(uint)((dm + 15) / 16), (uint)((sl + 15) / 16), 1]);

        return result;
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //  SSM GATED DELTA NET
    // ═══════════════════════════════════════════════════════════════════════════

    public void SsmGdnDecode(
        Tensor xNorm, Tensor convW, Tensor convState, Tensor wQKV,
        Tensor wZ, Tensor wBeta, Tensor wAlpha, Tensor dtBias, Tensor ssA,
        Tensor scratch, Tensor ssmState, Tensor ssmNorm, Tensor output,
        uint rowIndex = 0u,
        uint ssmDModel = 5120, uint ssmHVD = 128, uint ssmNVH = 48,
        uint ssmNKH = 16, uint ssmKD = 2048, uint ssmVD = 6144, uint ssmCD = 10240,
        uint debugLayer = uint.MaxValue)
    {
        uint convGroups = ssmCD / ssmHVD;
        uint vGroups = ssmNVH;

        // Lazy-init persistent SSM UBO
        if (_persistentSsmParams == null)
        {
            _persistentSsmParams = _device.CreateBuffer(32, BufferType.Storage, DataType.I32);
            _persistentSsmParams.Write(new[] { ssmDModel, ssmHVD, ssmNVH, ssmNKH, ssmKD, ssmVD, ssmCD, 0u });
        }
        if (_persistentRowIndex == null)
            _persistentRowIndex = _device.CreateBuffer(sizeof(uint), BufferType.Storage, DataType.I32);
        _persistentRowIndex.Write(new[] { rowIndex });

        // Part 1: conv1d + projections (ssm_gdn_decode)
        EmitDispatch("ssm_gdn_decode",
            [xNorm.Buffer, convW.Buffer, convState.Buffer, wQKV.Buffer,
             wZ.Buffer, wBeta.Buffer, wAlpha.Buffer, dtBias.Buffer,
             ssA.Buffer, scratch.Buffer, _persistentRowIndex, _persistentSsmParams], 12,
            [convGroups, 1, 1]);

        MaybeFlush();

        // Part 2: recurrence (ssm_gdn_recur)
        EmitDispatch("ssm_gdn_recur",
            [scratch.Buffer, ssmState.Buffer, ssmNorm.Buffer,
             output.Buffer, _persistentSsmParams], 5,
            [vGroups, 1, 1]);

        MaybeFlush();
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //  UTILITY
    // ═══════════════════════════════════════════════════════════════════════════

    private IComputeBuffer MakeParamsBuffer(uint a, uint b = 0, uint c = 0)
    {
        int count = 1 + (b > 0 ? 1 : 0) + (c > 0 ? 1 : 0);
        uint[] data = count switch
        {
            1 => [a],
            2 => [a, b],
            _ => [a, b, c]
        };
        var buf = _device.CreateBuffer((ulong)(data.Length * sizeof(uint)), BufferType.Storage, DataType.I32);
        buf.Write(data);
        return buf;
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;

        _taskWriter.TryComplete();

        foreach (var kv in _kernelCache)
            kv.Value.Dispose();
        _kernelCache.Clear();

        if (_persistentOffsetInitialized)
            _persistentZeroOffset.Dispose();
        _persistentSsmParams?.Dispose();
        _persistentRowIndex?.Dispose();

        _gpuExecutor.Dispose();
        _arena?.Dispose();
    }
}
