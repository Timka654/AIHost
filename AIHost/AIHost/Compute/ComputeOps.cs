using AIHost.GGUF;
using AIHost.ICompute;
using AIHost.ICompute.Vulkan;
using AIHost.Inference;
using Microsoft.Extensions.Logging;
using System.Buffers;
using System.Collections.Concurrent;
using System.Runtime.InteropServices;
using System.Threading.Channels;

namespace AIHost.Compute;

public class ComputeOps : IDisposable
{
    private readonly IComputeDevice _device;
    private readonly GpuExecutor _gpuExecutor;
    private readonly ChannelWriter<GpuTask> _taskWriter;
    private readonly ConcurrentDictionary<string, IComputeKernel> _kernelCache = new();
    private bool _disposed;

    private int _currentLayer;
    internal void SetCurrentLayer(int layer) => _currentLayer = layer;
    public bool UseChannelPipeline => true;

    internal bool _dbgLayer0 = true;
    private readonly ILogger<ComputeOps> _logger = AppLogger.Create<ComputeOps>();

    private VulkanArenaAllocator? _arena;
    public bool HasArena => _arena != null;

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

        var pool = leasePool ?? new GpuLeasePool(device);
        _gpuExecutor = new GpuExecutor(device, pool, _logger);
        _taskWriter = _gpuExecutor.Writer;

        _persistentZeroOffset = _device.CreateBuffer(sizeof(uint), BufferType.Storage, DataType.I32);
        uint[] zeroOffset = { 0u };
        _persistentZeroOffset.Write(zeroOffset);
        _persistentOffsetInitialized = true;
    }

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

    public void BeginBatch()
        => _taskWriter.TryWrite(BeginLayerTask.Create(_currentLayer));

    public void Flush()
    {
        if (_gpuExecutor.IsFailed)
            throw new InvalidOperationException(
                $"[ComputeOps] GPU executor is in failed state: {_gpuExecutor.LastError}");

        var tcs = new System.Threading.Tasks.TaskCompletionSource<GpuResult>();
        var task = FlushAndWaitTask.Create(tcs);
        _logger.LogInformation("[DBG_OPS] #{TaskId} EMIT FLUSH", task.TaskId);
        _taskWriter.TryWrite(task);

        if (!tcs.Task.Wait(TimeSpan.FromSeconds(120)))
            throw new TimeoutException("[ComputeOps] GPU flush timed out after 120s");

        var result = tcs.Task.Result;
        if (!result.Success)
            throw new InvalidOperationException(
                $"[ComputeOps] GPU flush failed: {result.Error}");

        _logger.LogInformation("[DBG_OPS] #{TaskId} FLUSH_COMPLETE", task.TaskId);

        FlushDeferred();
        _arena?.Reset();
    }

    private void ResetAllKernelDispatchRings()
    {
        foreach (var kv in _kernelCache)
        {
            if (kv.Value is VulkanComputeKernel vk)
                vk.ResetDispatchRing();
        }
    }

    private void MaybeFlush()
        => _taskWriter.TryWrite(BarrierTask.Create());

    /// <summary>Emit dispatch with optional operation tag for tracing.</summary>
    private void EmitDispatch(string kernelName, IComputeBuffer?[] args, int argCount,
        uint[] workgroups, string? opTag = null)
    {
        if (_kernelSources.TryGetValue(kernelName, out var source))
            GetOrCreateKernel(kernelName, source);
        else
            _logger.LogWarning("[ComputeOps] EmitDispatch: unknown kernel '{Name}'", kernelName);

        var task = DispatchKernelTask.Create(kernelName, workgroups);
        for (int i = 0; i < argCount; i++)
            task.Arguments[i] = args[i];
        task.ArgCount = argCount;
        task.OpTag = opTag;

        _logger.LogInformation("[DBG_OPS] #{TaskId} EMIT kernel={Kernel} tag={Tag} args={ArgCount}",
            task.TaskId, kernelName, opTag ?? "?", argCount);
        _taskWriter.TryWrite(task);
    }

    private readonly List<IDisposable> _deferred = new();

    public void DeferExternal(IDisposable d)
    {
        if (d != null)
        {
            lock (_deferred) { _deferred.Add(d); }
            _logger.LogInformation("[DBG_OPS] DEFER type={Type}", d.GetType().Name);
        }
    }

    private void FlushDeferred()
    {
        IDisposable[] items;
        lock (_deferred)
        {
            if (_deferred.Count == 0) return;
            items = _deferred.ToArray();
            _deferred.Clear();
        }
        _logger.LogInformation("[DBG_OPS] FLUSH_DEFERRED count={N}", items.Length);
        foreach (var d in items)
        {
            try { d.Dispose(); }
            catch (Exception ex) { _logger.LogWarning(ex, "[ComputeOps] Deferred dispose failed"); }
        }
    }

    public void InsertBarrier() => _taskWriter.TryWrite(BarrierTask.Create());
    internal void AttachArena(VulkanArenaAllocator arena) => _arena = arena;

    public void Softmax(Tensor tensor)
    {
        int rows = tensor.Shape.Rank >= 2 ? tensor.Shape[0] : 1;
        Softmax(tensor, rows);
    }

    public void ApplyRoPE(Tensor tensor, uint position, int headDim)
    {
        int totalDim = tensor.Shape.Rank >= 2 ? tensor.Shape[1] : tensor.Shape[0];
        int numHeads = totalDim / headDim;
        ApplyRoPEFull(tensor, position, numHeads, headDim);
    }

    public void ApplyRoPE(Tensor tensor, uint position, int numHeads, int headDim)
        => ApplyRoPEFull(tensor, position, numHeads, headDim);

    public Tensor TransformerLayer(Tensor x,
        Tensor wAttnNorm, Tensor wQ, Tensor wK, Tensor wV, Tensor wAttnOut,
        Tensor wFfnNorm, Tensor wGate, Tensor wUp, Tensor wDown,
        int numHeads, uint position, KVCache? kvCache, int layerIdx)
    {
        int sl = x.Shape[0], dm = x.Shape[1];
        int hd = wQ.Shape[1] / numHeads;
        int kd = wK.Shape[1] / (wK.Shape[1] / hd);
        int vd = wV.Shape[1] / (wV.Shape[1] / hd);

        var xn = Clone(x, "an");           LayerNorm(xn, wAttnNorm);
        var q = MatMulWeights(xn, wQ, "q"); var k = MatMulWeights(xn, wK, "k");
        var v = MatMulWeights(xn, wV, "v");
        MaybeFlush();

        ApplyRoPEFull(q, position, numHeads, hd);
        ApplyRoPEFull(k, position, 1, kd);
        MaybeFlush();

        Tensor ao;
        if (kvCache != null)
        {
            kvCache.Add(layerIdx, k, v); Flush();
            var (ck, cv) = kvCache.Get(layerIdx);
            ao = MultiHeadAttention(q, ck!, cv!, numHeads, position, "ao");
        }
        else
        {
            ao = MultiHeadAttention(q, k, v, numHeads, position, "ao");
        }

        var ap = MatMulWeights(ao, wAttnOut, "ap");
        var x1 = Add(x, ap, "x1"); MaybeFlush();

        var x1n = Clone(x1, "pn"); LayerNorm(x1n, wFfnNorm);
        var fo = FeedForward(x1n, wGate, wUp, wDown, "fo");
        var ou = Add(x1, fo, "ou");
        return ou;
    }

    public Tensor EmbeddingLookup(int[] tokenIds, Tensor weightF32, string? resultName = null)
    {
        int sl = tokenIds.Length, dim = weightF32.Shape[1];
        var tokenBuf = _device.CreateBuffer((ulong)(tokenIds.Length * sizeof(int)), BufferType.Storage, DataType.I32);
        tokenBuf.Write(tokenIds);
        DeferExternal(tokenBuf); // FIX: temp buffer must be disposed after Flush
        var result = Tensor.Create(_device, TensorShape.Matrix(sl, dim), DataType.F32, resultName ?? "emb");
        var paramBytes = new byte[8];
        BitConverter.GetBytes((uint)sl).CopyTo(paramBytes, 0);
        BitConverter.GetBytes((uint)dim).CopyTo(paramBytes, 4);
        var paramsBuffer = _device.CreateBuffer(8, BufferType.Storage, DataType.I32);
        paramsBuffer.Write(paramBytes);
        DeferExternal(paramsBuffer); // FIX: temp buffer must be disposed after Flush
        EmitDispatch("embedding_lookup",
            [tokenBuf, weightF32.Buffer, result.Buffer, paramsBuffer], 4,
            [(uint)sl, 1, 1], "emb_lookup");
        return result;
    }

    public unsafe Tensor EmbeddingLookupFromQuantized(int[] tokenIds, Tensor quantizedEmb, string? resultName = null)
    {
        var f32 = Dequantize(quantizedEmb);
        DeferExternal(f32); // FIX: dequantized token_embd F32 (up to 5 GB) must be disposed after Flush
        var tokenBuf = _device.CreateBuffer((ulong)(tokenIds.Length * sizeof(int)), BufferType.Storage, DataType.I32);
        tokenBuf.Write(tokenIds);
        DeferExternal(tokenBuf); // FIX: temp buffer must be disposed after Flush
        int sl = tokenIds.Length;
        var result = Tensor.Create(_device, TensorShape.Matrix(sl, quantizedEmb.Shape[1]), DataType.F32, resultName ?? "emb");
        var paramBytes = new byte[8];
        BitConverter.GetBytes((uint)sl).CopyTo(paramBytes, 0);
        BitConverter.GetBytes((uint)quantizedEmb.Shape[1]).CopyTo(paramBytes, 4);
        var paramsBuffer = _device.CreateBuffer(8, BufferType.Storage, DataType.I32);
        paramsBuffer.Write(paramBytes);
        DeferExternal(paramsBuffer); // FIX: temp buffer must be disposed after Flush
        EmitDispatch("embedding_lookup",
            [tokenBuf, f32.Buffer, result.Buffer, paramsBuffer], 4,
            [(uint)sl, 1, 1], "emb_q_lookup");
        return result;
    }

    public void AllocateArena(ulong sizeBytes, ulong kvCacheBytes = 0) { }

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

    // ═══════════════════════ MATMUL ═══════════════════════

    public Tensor MatMulWeightsT(Tensor a, Tensor b, string? resultName = null)
    {
        int M = a.Shape[0], K = a.Shape[1], J = b.Shape[0];
        if (K != b.Shape[1]) throw new ArgumentException($"Shape mismatch: {K} vs {b.Shape[1]}");
        var result = Tensor.Create(_device, TensorShape.Matrix(M, J), DataType.F32, resultName);
        var paramsBuffer = MakeParamsBuffer((uint)M, (uint)K, (uint)J);
        EmitDispatch("matmul_weights_t_f32",
            [a.Buffer, b.Buffer, result.Buffer, paramsBuffer], 4,
            [(uint)((J + 15) / 16), (uint)((M + 15) / 16), 1], "mm_wt");
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
            [(uint)((N + 15) / 16), (uint)((M + 15) / 16), 1], "mm_w");
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
            [(uint)((N + 15) / 16), (uint)((M + 15) / 16), 1], "mm");
        return result;
    }

    public Tensor MatMulWeightsLarge(Tensor a, Tensor quantized, string? resultName = null)
        => MatMulWeights(a, quantized, resultName);

    // ═══════════════════════ ELEMENT-WISE ═══════════════════════

    public Tensor Multiply(Tensor a, Tensor b, string? resultName = null)
    {
        if (a.Shape.TotalElements != b.Shape.TotalElements)
            throw new ArgumentException("Shape mismatch");
        var result = Tensor.Create(_device, a.Shape, DataType.F32, resultName);
        uint total = (uint)a.Shape.TotalElements;
        var paramsBuffer = MakeParamsBuffer(total);
        EmitDispatch("elemwise_mul",
            [a.Buffer, b.Buffer, result.Buffer, paramsBuffer], 4,
            [(total + 255) / 256, 1, 1], "mul");
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
            [(total + 255) / 256, 1, 1], "add");
        return result;
    }

    public void SiLU(Tensor tensor)
    {
        uint total = (uint)tensor.Shape.TotalElements;
        var paramsBuffer = MakeParamsBuffer(total);
        EmitDispatch("silu",
            [tensor.Buffer, paramsBuffer], 2,
            [(total + 255) / 256, 1, 1], "silu");
    }

    public void Sigmoid(Tensor tensor)
    {
        uint total = (uint)tensor.Shape.TotalElements;
        var paramsBuffer = MakeParamsBuffer(total);
        EmitDispatch("sigmoid",
            [tensor.Buffer, paramsBuffer], 2,
            [(total + 255) / 256, 1, 1], "sigmoid");
    }

    // ═══════════════════════ CONCAT ═══════════════════════

    public Tensor Concat(Tensor a, Tensor b, int axis, string? resultName = null)
        => axis == 0 ? ConcatAxis0(a, b, resultName) : ConcatAxis1(a, b, resultName);

    private Tensor ConcatAxis0(Tensor a, Tensor b, string? resultName)
    {
        int rows = a.Shape[0] + b.Shape[0], cols = a.Shape[1];
        var result = Tensor.Create(_device, TensorShape.Matrix(rows, cols), DataType.F32, resultName);
        uint total = (uint)a.Shape.TotalElements;
        uint bRows = (uint)b.Shape[0], bCols = (uint)b.Shape[1];
        var paramsBuffer = MakeParamsBuffer(total, bRows, bCols);
        EmitDispatch("concat_axis0",
            [a.Buffer, b.Buffer, result.Buffer, paramsBuffer], 4,
            [(total + 255) / 256, 1, 1], "cat0");
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
            [aCols + bCols, 1, 1], "cat1");
        return result;
    }

    // ═══════════════════════ NORM & SOFTMAX ═══════════════════════

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
        DeferExternal(paramsBuffer); // FIX: temp buffer must be disposed after Flush
        EmitDispatch("layer_norm",
            [tensor.Buffer, weight.Buffer, paramsBuffer], 3,
            [(uint)rows, 1, 1], "ln");
    }

    public void LayerNormVirtual(Tensor tensor, Tensor weight, int numRows, int numCols, float eps = 1e-5f)
    {
        var paramBytes = new byte[12];
        BitConverter.GetBytes((uint)numRows).CopyTo(paramBytes, 0);
        BitConverter.GetBytes((uint)numCols).CopyTo(paramBytes, 4);
        BitConverter.GetBytes(eps).CopyTo(paramBytes, 8);
        var paramsBuffer = _device.CreateBuffer(12, BufferType.Storage, DataType.I32);
        paramsBuffer.Write(paramBytes);
        DeferExternal(paramsBuffer); // FIX: temp buffer must be disposed after Flush
        EmitDispatch("layer_norm",
            [tensor.Buffer, weight.Buffer, paramsBuffer], 3,
            [(uint)numRows, 1, 1], "ln_virt");
    }

    public void Softmax(Tensor tensor, int rows)
    {
        uint total = (uint)tensor.Shape.TotalElements;
        var paramsBuffer = MakeParamsBuffer(total);
        EmitDispatch("softmax",
            [tensor.Buffer, paramsBuffer], 2,
            [(total + 255) / 256, 1, 1], "softmax");
    }

    public void RowwiseSoftmax(Tensor tensor)
    {
        int rows = tensor.Shape.Rank >= 2 ? tensor.Shape[0] : 1;
        int cols = tensor.Shape[1];
        var paramBytes = new byte[8];
        BitConverter.GetBytes((uint)rows).CopyTo(paramBytes, 0);
        BitConverter.GetBytes((uint)cols).CopyTo(paramBytes, 4);
        var paramsBuffer = _device.CreateBuffer(8, BufferType.Storage, DataType.I32);
        paramsBuffer.Write(paramBytes);
        DeferExternal(paramsBuffer); // FIX: temp buffer must be disposed after Flush
        EmitDispatch("rowwise_softmax",
            [tensor.Buffer, paramsBuffer], 2,
            [(uint)rows, 1, 1], "row_softmax");
    }

    // ═══════════════════════ RoPE ═══════════════════════

    public void ApplyRoPEFull(Tensor tensor, uint startPosition, int numHeads, int headDim,
        float theta = 10000.0f, int ropeDim = 0)
    {
        int effectiveRopeDim = ropeDim > 0 ? ropeDim : headDim;
        int numPairs = effectiveRopeDim / 2;
        var paramBytes = new byte[20];
        BitConverter.GetBytes((uint)numHeads).CopyTo(paramBytes, 0);
        BitConverter.GetBytes((uint)headDim).CopyTo(paramBytes, 4);
        BitConverter.GetBytes((uint)effectiveRopeDim).CopyTo(paramBytes, 8);
        BitConverter.GetBytes(startPosition).CopyTo(paramBytes, 12);
        BitConverter.GetBytes(theta).CopyTo(paramBytes, 16);
        var paramsBuffer = _device.CreateBuffer(20, BufferType.Storage, DataType.I32);
        paramsBuffer.Write(paramBytes);
        DeferExternal(paramsBuffer); // FIX: temp buffer must be disposed after Flush
        uint total = (uint)(numHeads * numPairs);
        EmitDispatch("rope_full",
            [tensor.Buffer, paramsBuffer], 2,
            [(total + 255) / 256, 1, 1], "rope_full");
    }

    public void RoPE(Tensor tensor, uint position, int numHeads, int headDim,
        float theta = 10000.0f, int ropeDim = 0)
        => ApplyRoPEFull(tensor, position, numHeads, headDim, theta, ropeDim);

    // ═══════════════════════ SCALE ═══════════════════════

    public void Scale(Tensor tensor, float scale)
    {
        var paramBytes = new byte[8];
        BitConverter.GetBytes((uint)tensor.Shape.TotalElements).CopyTo(paramBytes, 0);
        BitConverter.GetBytes(scale).CopyTo(paramBytes, 4);
        var paramsBuffer = _device.CreateBuffer(8, BufferType.Storage, DataType.I32);
        paramsBuffer.Write(paramBytes);
        DeferExternal(paramsBuffer); // FIX: temp buffer must be disposed after Flush
        uint total = (uint)tensor.Shape.TotalElements;
        EmitDispatch("scale",
            [tensor.Buffer, paramsBuffer], 2,
            [(total + 255) / 256, 1, 1], "scale");
    }

    // ═══════════════════════ SLICE & SCATTER ═══════════════════════

    public Tensor SliceCols(Tensor input, int startCol, int numCols, string? resultName = null)
    {
        int rows = input.Shape[0];
        var result = Tensor.Create(_device, TensorShape.Matrix(rows, numCols), DataType.F32, resultName);
        var paramBytes = new byte[12];
        BitConverter.GetBytes((uint)rows).CopyTo(paramBytes, 0);
        BitConverter.GetBytes((uint)numCols).CopyTo(paramBytes, 4);
        BitConverter.GetBytes((uint)startCol).CopyTo(paramBytes, 8);
        var paramsBuffer = _device.CreateBuffer(12, BufferType.Storage, DataType.I32);
        paramsBuffer.Write(paramBytes);
        DeferExternal(paramsBuffer); // FIX: temp buffer must be disposed after Flush
        EmitDispatch("slice_cols",
            [input.Buffer, result.Buffer, paramsBuffer], 3,
            [(uint)rows, 1, 1], "slice_cols");
        return result;
    }

    public Tensor DeinterleaveQGate(Tensor input, uint numHeads, uint qDim, string? resultName = null)
    {
        int sl = input.Shape[0];
        var result = Tensor.Create(_device, input.Shape, DataType.F32, resultName);
        uint totalCols = (uint)input.Shape[1];
        var paramBytes = new byte[8];
        BitConverter.GetBytes(numHeads).CopyTo(paramBytes, 0);
        BitConverter.GetBytes(qDim).CopyTo(paramBytes, 4);
        var paramsBuffer = _device.CreateBuffer(8, BufferType.Storage, DataType.I32);
        paramsBuffer.Write(paramBytes);
        DeferExternal(paramsBuffer); // FIX: temp buffer must be disposed after Flush
        EmitDispatch("deinterleave_q_gate",
            [input.Buffer, result.Buffer, paramsBuffer], 3,
            [(uint)sl, 1, 1], "deint_q_gate");
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
        DeferExternal(paramsBuffer); // FIX: temp buffer must be disposed after Flush
        EmitDispatch("scatter_cols",
            [src.Buffer, dst.Buffer, paramsBuffer], 3,
            [srcRows, 1, 1], "scatter_cols");
    }

    // ═══════════════════════ KV-CACHE & ATTENTION ═══════════════════════

    public Tensor RepeatKVHeads(Tensor input, int numKVHeadsTarget, string? resultName = null)
    {
        int sl = input.Shape[0], kd = input.Shape[1];
        var result = Tensor.Create(_device, TensorShape.Matrix(sl, kd), DataType.F32, resultName);
        uint total = (uint)sl * (uint)kd;
        EmitDispatch("repeat_kv_heads",
            [input.Buffer, result.Buffer], 2,
            [(total + 255) / 256, 1, 1], "rep_kv_heads");
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
        DeferExternal(paramsBuffer); // FIX: temp buffer must be disposed after Flush
        EmitDispatch("repeat_columns",
            [input.Buffer, result.Buffer, paramsBuffer], 3,
            [(total + 255) / 256, 1, 1], "rep_cols");
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
        DeferExternal(paramsBuffer); // FIX: temp buffer must be disposed after Flush
        uint total = (uint)(rows * cols);
        EmitDispatch("causal_mask",
            [scores.Buffer, paramsBuffer], 2,
            [(total + 255) / 256, 1, 1], "causal_mask");
    }

    public Tensor MultiHeadAttention(Tensor Q, Tensor K, Tensor V, int numHeads, uint position, string? resultName = null)
    {
        int sl = Q.Shape[0], hd = Q.Shape[1];
        var result = Tensor.Create(_device, TensorShape.Matrix(sl, hd), DataType.F32, resultName);
        CopyBuffer(Q, K, V, numHeads, position, result);
        MaybeFlush();
        return result;
    }

    private byte[] CopyBuffer(Tensor Q, Tensor K, Tensor V, int numHeads, uint position, Tensor result)
    {
        int sl = Q.Shape[0], hd = Q.Shape[1], kd = K.Shape[1], vd = V.Shape[1];
        var paramBytes = PackAttentionParams(sl, hd, kd, vd, numHeads, position);
        var paramsBuffer = _device.CreateBuffer((uint)paramBytes.Length, BufferType.Storage, DataType.I32);
        paramsBuffer.Write(paramBytes);
        DeferExternal(paramsBuffer); // FIX: temp buffer must be disposed after Flush
        EmitDispatch("fused_mha_generate",
            [Q.Buffer, K.Buffer, V.Buffer, result.Buffer, paramsBuffer], 5,
            [(uint)((sl + 15) / 16) * (uint)numHeads, 1, 1], "fused_mha");
        return paramBytes;
    }

    private static byte[] PackAttentionParams(int sl, int hd, int kd, int vd, int numHeads, uint position)
    {
        var buf = new byte[24];
        BitConverter.GetBytes((uint)sl).CopyTo(buf, 0);
        BitConverter.GetBytes((uint)hd).CopyTo(buf, 4);
        BitConverter.GetBytes((uint)kd).CopyTo(buf, 8);
        BitConverter.GetBytes((uint)vd).CopyTo(buf, 12);
        BitConverter.GetBytes((uint)numHeads).CopyTo(buf, 16);
        BitConverter.GetBytes(position).CopyTo(buf, 20);
        return buf;
    }

    // ═══════════════════════ TRANSPOSE ═══════════════════════

    public Tensor Transpose(Tensor input, string? resultName = null)
    {
        int rows = input.Shape[0], cols = input.Shape[1];
        var result = Tensor.Create(_device, TensorShape.Matrix(cols, rows), DataType.F32, resultName);
        var paramBytes = new byte[8];
        BitConverter.GetBytes((uint)rows).CopyTo(paramBytes, 0);
        BitConverter.GetBytes((uint)cols).CopyTo(paramBytes, 4);
        var paramsBuffer = _device.CreateBuffer(8, BufferType.Storage, DataType.I32);
        paramsBuffer.Write(paramBytes);
        DeferExternal(paramsBuffer); // FIX: temp buffer must be disposed after Flush
        EmitDispatch("transpose",
            [input.Buffer, result.Buffer, paramsBuffer], 3,
            [(uint)((rows * cols + 255) / 256), 1, 1], "transpose");
        return result;
    }

    // ═══════════════════════ DEQUANTIZATION ═══════════════════════

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

        if (quantized.DataType == DataType.F32)
        {
            uint total = (uint)quantized.Shape.TotalElements;
            var p = MakeParamsBuffer(total);
            EmitDispatch("copy", [quantized.Buffer, target.Buffer, p], 3,
                [(total + 255) / 256, 1, 1], "deq_copy_f32");
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
        // FIX: AMD iGPU (RADV RENOIR) cannot handle 32768 workgroups per dispatch
        // without triggering GPU timeout → ErrorDeviceLost. Reduce to 4096 groups
        // per chunk (1M elements) to stay within TDR limits.
        const uint MaxGroupsPerDispatch = 4096;
        const uint SuperblockElements = 256;

        if (totalGroups <= MaxGroupsPerDispatch)
        {
            EmitDispatch(kernelName,
                [quantized.Buffer, target.Buffer, _persistentZeroOffset], 3,
                [totalGroups, 1, 1], "deq");
        }
        else
        {
            uint groupsPerChunk = MaxGroupsPerDispatch;
            uint numChunks = (totalGroups + groupsPerChunk - 1) / groupsPerChunk;
            // FIX: Break giant dequant into sub-batches of 16 chunks with actual Flush().
            // MaybeFlush() only inserts vkCmdPipelineBarrier — all 93 chunks stay in
            // one command buffer → ErrorDeviceLost at vkQueueSubmit.
            // Real Flush() submits, waits for fence, and resets descriptor ring.
            // FIX: Flush after EVERY chunk on AMD iGPUs.
            // Any command buffer accumulation >16K workgroups causes GPU
            // timeout → ErrorDeviceLost (RADV RENOIR, iGPU).
            const uint ChunksPerSubBatch = 1;
            for (uint chunk = 0; chunk < numChunks; chunk++)
            {
                uint chunkStartGroup = chunk * groupsPerChunk;
                uint chunkEndGroup = Math.Min(chunkStartGroup + groupsPerChunk, totalGroups);
                uint chunkGroups = chunkEndGroup - chunkStartGroup;
                uint elementOffset = chunkStartGroup * SuperblockElements;
                var offsetBuf = _device.CreateBuffer(sizeof(uint), BufferType.Storage, DataType.I32);
                offsetBuf.Write(new[] { elementOffset });
                DeferExternal(offsetBuf);
                EmitDispatch(kernelName,
                    [quantized.Buffer, target.Buffer, offsetBuf], 3,
                    [chunkGroups, 1, 1], "deq_chunk");

                // Flush every ChunksPerSubBatch chunks
                if ((chunk + 1) % ChunksPerSubBatch == 0 && chunk + 1 < numChunks)
                    Flush();
            }
        }
    }

    // ═══════════════════════ EMBEDDING ═══════════════════════

    public Tensor EmbeddingLookup(Tensor tokenIdx, Tensor weightF32, string? resultName = null)
    {
        int sl = tokenIdx.Shape.TotalElements, cols = weightF32.Shape[1];
        var result = Tensor.Create(_device, TensorShape.Matrix(sl, cols), DataType.F32, resultName);
        var paramBytes = new byte[8];
        BitConverter.GetBytes((uint)sl).CopyTo(paramBytes, 0);
        BitConverter.GetBytes((uint)cols).CopyTo(paramBytes, 4);
        var paramsBuffer = _device.CreateBuffer(8, BufferType.Storage, DataType.I32);
        paramsBuffer.Write(paramBytes);
        DeferExternal(paramsBuffer); // FIX: temp buffer must be disposed after Flush
        EmitDispatch("embedding_lookup",
            [tokenIdx.Buffer, weightF32.Buffer, result.Buffer, paramsBuffer], 4,
            [(uint)sl, 1, 1], "emb_lookup_t");
        return result;
    }

    // ═══════════════════════ CLONE ═══════════════════════

    public Tensor Clone(Tensor input, string? resultName = null)
    {
        var result = Tensor.Create(_device, input.Shape, DataType.F32, resultName);
        uint total = (uint)input.Shape.TotalElements;
        var paramsBuffer = MakeParamsBuffer(total);
        EmitDispatch("copy",
            [input.Buffer, result.Buffer, paramsBuffer], 3,
            [(total + 255) / 256, 1, 1], "clone");
        return result;
    }

    // ═══════════════════════ FEED FORWARD ═══════════════════════

    public Tensor FeedForward(Tensor x, Tensor wGate, Tensor wUp, Tensor wDown, string? resultName = null)
    {
        int sl = x.Shape[0], dm = x.Shape[1], inter = wGate.Shape[1];
        var hidden = Tensor.Create(_device, TensorShape.Matrix(sl, inter), DataType.F32, "ffn_hidden");
        var gated = Tensor.Create(_device, TensorShape.Matrix(sl, inter), DataType.F32, "ffn_gated");

        var paramsBuf1 = MakeParamsBuffer((uint)sl, (uint)dm, (uint)inter);
        EmitDispatch("matmul_weights_f32",
            [x.Buffer, wGate.Buffer, hidden.Buffer, paramsBuf1], 4,
            [(uint)((inter + 15) / 16), (uint)((sl + 15) / 16), 1], "ffn_gate");
        MaybeFlush();

        var paramsBuf2 = MakeParamsBuffer((uint)sl, (uint)dm, (uint)inter);
        EmitDispatch("matmul_weights_f32",
            [x.Buffer, wUp.Buffer, gated.Buffer, paramsBuf2], 4,
            [(uint)((inter + 15) / 16), (uint)((sl + 15) / 16), 1], "ffn_up");
        MaybeFlush();

        SiLU(gated);
        MaybeFlush();

        var gatedResult = Multiply(hidden, gated, "ffn_gated_result");
        MaybeFlush();

        var result = Tensor.Create(_device, TensorShape.Matrix(sl, dm), DataType.F32, resultName);
        int interD = wDown.Shape[0];
        var paramsBuf3 = MakeParamsBuffer((uint)sl, (uint)interD, (uint)dm);
        EmitDispatch("matmul_weights_f32",
            [gatedResult.Buffer, wDown.Buffer, result.Buffer, paramsBuf3], 4,
            [(uint)((dm + 15) / 16), (uint)((sl + 15) / 16), 1], "ffn_down");
        return result;
    }

    // ═══════════════════════ SSM GATED DELTA NET ═══════════════════════

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

        if (_persistentSsmParams == null)
        {
            _persistentSsmParams = _device.CreateBuffer(32, BufferType.Storage, DataType.I32);
            _persistentSsmParams.Write(new[] { ssmDModel, ssmHVD, ssmNVH, ssmNKH, ssmKD, ssmVD, ssmCD, 0u });
        }
        if (_persistentRowIndex == null)
            _persistentRowIndex = _device.CreateBuffer(sizeof(uint), BufferType.Storage, DataType.I32);
        _persistentRowIndex.Write(new[] { rowIndex });

        EmitDispatch("ssm_gdn_decode",
            [xNorm.Buffer, convW.Buffer, convState.Buffer, wQKV.Buffer,
             wZ.Buffer, wBeta.Buffer, wAlpha.Buffer, dtBias.Buffer,
             ssA.Buffer, scratch.Buffer, _persistentRowIndex, _persistentSsmParams], 12,
            [convGroups, 1, 1], "ssm_decode");
        MaybeFlush();

        EmitDispatch("ssm_gdn_recur",
            [scratch.Buffer, ssmState.Buffer, ssmNorm.Buffer,
             output.Buffer, _persistentSsmParams], 5,
            [vGroups, 1, 1], "ssm_recur");
        MaybeFlush();
    }

    // ═══════════════════════ UTILITY ═══════════════════════

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
        // FIX: Register for deferred disposal after Flush completes
        DeferExternal(buf);
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
