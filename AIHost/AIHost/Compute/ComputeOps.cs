using AIHost.GGUF;
using AIHost.ICompute;
using AIHost.ICompute.Vulkan;
using AIHost.Inference;
using Microsoft.Extensions.Logging;
using System.Collections.Concurrent;
using System.Runtime.InteropServices;

namespace AIHost.Compute;

/// <summary>
/// Высокоуровневый API для тензорных операций на GPU
/// </summary>
public class ComputeOps : IDisposable
{
    private readonly IComputeDevice _device;
    private readonly IComputeCommandQueue _queue;
    private readonly Dictionary<string, IComputeKernel> _kernelCache = new();
    private bool _disposed;

    // Batch mode: all dispatches are recorded into one command buffer; no fence wait
    // between ops. A single Flush() at the end submits everything and waits once.
    // Deferred disposals accumulate here and are released after the flush.
    private bool _batchMode;
    private readonly List<IDisposable> _deferred = [];
    internal bool _dbgLayer0 = true;
    private readonly ILogger<ComputeOps> _logger = AppLogger.Create<ComputeOps>();

    // ── Arena allocator (replaces ChannelBufferPool) ──────────────────────────
    // One giant VkBuffer allocated once per model lifetime. Per-frame temp tensors
    // are sliced from it via bump-pointer. Reset() is called after Flush() when
    // the GPU fence signals completion — zero fragmentation, O(1), no vkAllocateMemory.
    private VulkanArenaAllocator? _arena;
    public bool HasArena => _arena != null;

    // ── Persistent offset buffer for K-quant dequantization ──────────────────
    // CRITICAL FIX: On AMD RADV, creating and destroying a small buffer for every
    // DequantizeQ*K call causes GPUVM faults (PERMISSION_FAULTS with CLIENT_ID=TCP).
    // The Texture Cache Parser caches descriptor data and may attempt to write to
    // a freed buffer's GPUVM address when a new buffer is allocated at the same address.
    //
    // Solution: create ONE persistent offset buffer at construction time and reuse it
    // for ALL K-quant dequantization dispatches. The buffer is never disposed until
    // ComputeOps itself is disposed.
    //
    // For chunked DequantizeInto, each chunk still gets its own offset buffer because
    // different chunks need different elementOffset values simultaneously (they are
    // dispatched sequentially, but the buffer content must not change between
    // SetArgument and the actual GPU read). However, the single-dispatch fast path
    // in DequantizeInto also uses the persistent zero-offset buffer.
    private readonly IComputeBuffer _persistentZeroOffset;
    private bool _persistentOffsetInitialized;

    // OPTIMIZATION: Persistent SSM UBO buffers — eliminated 3× CreateBuffer/Defer per token.
    // ssmParams = [DMODEL, HVD, NVH, NKH, KD, VD, CD, 0] (32 bytes), shared by decode (binding=11) and recur (binding=4).
    // rowIndex = [uint] (4 bytes), written per token before dispatch.
    // Same pattern as _persistentZeroOffset: created once, reused for all SsmGdnDecode calls.
    private IComputeBuffer? _persistentSsmParams;
    private IComputeBuffer? _persistentRowIndex;

    public IComputeDevice Device => _device;

    public ComputeOps(IComputeDevice device)
    {
        _device = device;
        _queue = device.CreateCommandQueue();

        // Create persistent zero-offset buffer for K-quant dequantization.
        // This buffer is initialized once and reused for all DequantizeQ*K calls
        // and the DequantizeInto fast path. It is NEVER disposed until ComputeOps
        // itself is disposed, eliminating the create/dispose cycle that triggers
        // GPUVM faults on AMD RADV.
        _persistentZeroOffset = _device.CreateBuffer(sizeof(uint), BufferType.Storage, DataType.I32);
        uint[] zeroOffset = { 0u };
        _persistentZeroOffset.Write(zeroOffset);
        _persistentOffsetInitialized = true;
    }


    // ── Batch infrastructure ─────────────────────────────────────────────────

    /// <summary>Enters batch mode: GPU ops accumulate in one command buffer.</summary>
    public void BeginBatch() => _batchMode = true;

    /// <summary>
    /// Submits all accumulated GPU work in a single fence wait, then releases
    /// all deferred buffers and tensors. Exits batch mode.
    /// Reduces ~500 fence waits/token to 1 per layer.
    /// </summary>
    public void Flush()
    {
        var _ts = GlobalProfiler.Start();
        try
        {
            _queue.Flush();
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[DBG_OPS] Flush failed — resetting kernel dispatch rings");
            // Reset dispatch rings even on failure so next batch starts clean
            foreach (var kernel in _kernelCache.Values)
                if (kernel is VulkanComputeKernel vk)
                    vk.ResetDispatchRing();
            _batchMode = false;
            throw; // rethrow — caller must handle ErrorDeviceLost
        }
        GlobalProfiler.End(_ts, "ComputeOps.Flush");
        foreach (var d in _deferred) d.Dispose();
        _deferred.Clear();
        _batchMode = false;
        // Rewind each kernel's descriptor ring so next batch starts from slot 0.
        foreach (var kernel in _kernelCache.Values)
            if (kernel is VulkanComputeKernel vk)
                vk.ResetDispatchRing();
        // Reclaim arena temp memory — GPU fence already signaled by _queue.Flush()
        _arena?.Reset();
    }

    /// <summary>In batch mode: insert a compute barrier (no submit). Otherwise flush.</summary>
    private void MaybeFlush()
    {
        if (_batchMode)
        {
            _logger.LogDebug("[DBG_OPS] MaybeFlush batchMode barrier");
            _queue.InsertMemoryBarrier();
        }
        else
        {
            _logger.LogDebug("[DBG_OPS] MaybeFlush non-batch flush");
            try
            {
                _queue.Flush();
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "[DBG_OPS] MaybeFlush failed — resetting dispatch rings");
                // Reset dispatch rings even on failure so next batch starts clean
                foreach (var kernel in _kernelCache.Values)
                    if (kernel is VulkanComputeKernel vk)
                        vk.ResetDispatchRing();
                throw; // rethrow — caller must handle ErrorDeviceLost
            }
            // Reset dispatch ring after every non-batch flush.
            // In non-batch mode, _dispatchIndex grows unboundedly and after 64 dispatches
            // starts overwriting descriptor sets that the GPU may still have cached in
            // the Texture Cache Parser (TCP). This causes GPUVM faults on AMD RADV.
            // DeviceWaitIdle inside _queue.Flush() guarantees the GPU has fully completed
            // all work referencing the descriptor sets, so it's safe to reset now.
            foreach (var kernel in _kernelCache.Values)
                if (kernel is VulkanComputeKernel vk)
                    vk.ResetDispatchRing();
        }
    }


    /// <summary>In batch mode: defer disposal until Flush(). Otherwise dispose now.</summary>
    private void Defer(IDisposable d)
    {
        if (_batchMode)
        {
            _logger.LogDebug("[DBG_OPS] Defer batchMode add");
            _deferred.Add(d);
        }
        else
        {
            _logger.LogDebug("[DBG_OPS] Defer non-batch dispose now");
            d.Dispose();
        }
    }

    /// <summary>
    /// Publicly defer a tensor for disposal after the current batch flush.
    /// Use this from Transformer when tensors are passed into TransformerLayer
    /// and must stay alive until the layer's GPU work completes.
    /// </summary>
    public void DeferExternal(IDisposable d) => Defer(d);

    #region Matrix Operations

    // Max F32 bytes per single GPU allocation for large-weight chunked projection.
    // Models with huge vocabularies (e.g. Qwen 248K) exceed VRAM with a full dequant.
    private const long MaxF32AllocationBytes = 1L * 1024 * 1024 * 1024; // 1 GB

    /// <summary>
    /// Transposed weight matmul: C[M,J] = A[M,K] @ B^T
    /// where B is F32 stored GGUF column-major [J,K]: B[j,k]=data[j+k*J].
    /// Used when the weight is stored as its transpose (e.g. Qwen3.5 attn_gate).
    /// </summary>
    public Tensor MatMulWeightsT(Tensor a, Tensor b, string? resultName = null)
    {
        if (a.Shape.Rank != 2 || b.Shape.Rank != 2)
            throw new ArgumentException("MatMulWeightsT requires 2D tensors");

        int M = a.Shape[0];
        int K = a.Shape[1];
        int J = b.Shape[0]; // output cols = b rows (transposed)

        if (K != b.Shape[1])
            throw new ArgumentException(
                $"MatMulWeightsT: A.cols={K} must equal B.cols={b.Shape[1]} (B stored transposed)");

        var result = Tensor.Create(_device, TensorShape.Matrix(M, J), DataType.F32, resultName);

        uint[] paramsData = { (uint)M, (uint)K, (uint)J };
        var paramsBuffer = _device.CreateBuffer((ulong)(paramsData.Length * sizeof(uint)),
                                                 BufferType.Storage, DataType.I32);
        paramsBuffer.Write(paramsData);

        var kernel = GetOrCreateKernel("matmul_weights_t_f32", () => ComputeShaders.MatMulWeightsTF32);
        kernel.SetArgument(0, a.Buffer);
        kernel.SetArgument(1, b.Buffer);
        kernel.SetArgument(2, result.Buffer);
        kernel.SetArgument(3, paramsBuffer);

        uint[] globalWorkSize = { (uint)((J + 15) / 16), (uint)((M + 15) / 16) };
        _queue.Dispatch(kernel, globalWorkSize, null);
        MaybeFlush();

        Defer(paramsBuffer);
        return result;
    }

    /// <summary>
    /// MatMulWeights with automatic chunking for tensors whose dequantized F32 size
    /// exceeds MaxF32AllocationBytes (e.g. lm_head for 248K-vocab models).
    /// Falls back to standard MatMulWeights when the tensor is small enough.
    /// Chunking splits the output-vocabulary axis into strips, dequantizes each strip
    /// separately (staying under the 1 GB single-allocation limit), and assembles the
    /// full logit tensor at the end.
    /// </summary>
    public Tensor MatMulWeightsLarge(Tensor a, Tensor quantized, string? resultName = null)
    {
        int M = a.Shape[0], K = a.Shape[1];
        int N = quantized.Shape[1]; // vocab / output dim

        long fullF32  = (long)quantized.Shape.TotalElements * sizeof(float);

        if (fullF32 <= MaxF32AllocationBytes)
            return MatMulWeights(a, Dequantize(quantized), resultName); // small enough — direct

        // Chunk along the N (vocabulary) dimension.
        // Each chunk row is `K` elements; bytes-per-row in the raw quantized buffer =
        // totalRawBytes / N (since rows are vocab items when ne[1]=N).
        int chunkRows   = (int)(MaxF32AllocationBytes / ((long)K * sizeof(float)));
        chunkRows       = Math.Max(256, chunkRows & ~255); // round down to multiple of 256
        int numChunks   = (N + chunkRows - 1) / chunkRows;

        long bytesPerRow = (long)quantized.Buffer.Size / N;

        var result = Tensor.Create(_device, TensorShape.Matrix(M, N), DataType.F32, resultName ?? "logits");

        for (int c = 0; c < numChunks; c++)
        {
            int start  = c * chunkRows;
            int end    = Math.Min(start + chunkRows, N);
            int actual = end - start;

            // Read quantized bytes for this vocab slice (CPU → already in mmap'd weight buffer)
            ulong byteStart = (ulong)(start * bytesPerRow);
            int   byteCount = (int)(actual * bytesPerRow);
            var rawBytes = quantized.Buffer.ReadRange<byte>(byteStart, byteCount);

            // Upload chunk as a temporary quantized tensor (same dtype, shape [K, actual])
            var chunkBuf    = _device.CreateBuffer((ulong)rawBytes.Length, ICompute.BufferType.Storage, DataType.I8);
            chunkBuf.Write(rawBytes);
            // FIX: shape must be [actual, K] (vocab_entries × model_dim), NOT [K, actual].
            // Dequantize processes the FIRST dimension as "rows", and each weight row
            // is one vocabulary entry with K dimensions.  The old [K, actual] shape
            // told the dequant shader to process K=5120 rows of actual elements,
            // massively overrunning the chunk buffer and producing garbage logits.
            var chunkTensor = new Tensor(chunkBuf, TensorShape.Matrix(actual, K), quantized.DataType, "w_chunk");

            // Dequantize → [actual, K] row-major F32.
            var chunkF32    = Dequantize(chunkTensor);
            chunkTensor.Dispose();

            // ── CPU REFERENCE for first chunk: verify Dequantize + MatMul correctness ──
            if (c == 0 && actual > 0)
            {
                try
                {
                    int refV = Math.Min(actual, 16);
                    var hRead = a.Buffer.ReadRange<float>(0, K);   // hidden state row 0
                    var gRead = chunkF32.Buffer.ReadRange<float>(0, K * refV); // GPU output rows
                    var cpuRef = new float[refV];
                    float[] wRow = new float[K];
                    for (int v = 0; v < refV; v++)
                    {
                        DequantizeRowCpu(quantized.DataType, rawBytes, v * (int)bytesPerRow, wRow, 0, K);
                        float dot = 0f;
                        for (int j = 0; j < K; j++) dot += hRead[j] * wRow[j];
                        cpuRef[v] = dot;
                    }
                    // Compare CPU ref vs GPU chunkF32 (dequantized weight) — compute dot on GPU data
                    var gpuRef = new float[refV];
                    for (int v = 0; v < refV; v++)
                    {
                        float dot = 0f;
                        for (int j = 0; j < K; j++) dot += hRead[j] * gRead[v * K + j]; // [actual,K] layout
                        gpuRef[v] = dot;
                    }
                    _logger.LogError("[DIAG_CHUNK0_REF] refV={RefV} cpuTop=[{CpuTop}] gpuMatch=[{GpuMatch}]",
                        refV,
                        string.Join(",", cpuRef.Select((x,i)=>(x,i)).OrderByDescending(p=>p.x).Take(5).Select(p=>$"{p.i}={p.Item1:F3}")),
                        string.Join(",", cpuRef.Zip(gpuRef,(c,g)=>Math.Abs(c-g)<0.001f?"✓":$"{(c-g):F3}")));
                }
                catch (Exception ex) { _logger.LogError("[DIAG_CHUNK0_REF] err={Err}", ex.Message); }
            }

            // FIX: MatMulWeightsT expects B in GGUF column-major layout (B[j,k]=data[j+k*J]).
            // chunkF32 is row-major (data[j*K + k]) from Dequantize, causing NaN logits.
            // Transpose chunkF32 to [K, actual] then use standard MatMul (row-major for both).
            var chunkF32T = Transpose(chunkF32);
            chunkF32.Dispose();
            var partialLogits = MatMul(a, chunkF32T, "partial_logits");
            chunkF32T.Dispose();

            // Scatter partial [M, actual] into result [M, N] at column offset `start`
            ScatterCols(result, partialLogits, start);
            partialLogits.Dispose();
        }

        return result;
    }

    /// <summary>
    /// Matrix multiply where B is a GGUF weight stored in column-major order.
    /// GGUF stores weight[k, n] at data[k + n * K] (ne[0]=K is innermost dim).
    /// A: [M × K], B_gguf: [K × N] (column-major), C: [M × N]
    /// </summary>
    public Tensor MatMulWeights(Tensor a, Tensor b, string? resultName = null)
    {
        if (a.Shape.Rank != 2 || b.Shape.Rank != 2)
            throw new ArgumentException("MatMulWeights requires 2D tensors");

        int M = a.Shape[0];
        int K = a.Shape[1];
        int N = b.Shape[1];

        if (K != b.Shape[0])
            throw new ArgumentException($"Incompatible shapes for MatMulWeights: [{M}×{K}] × [{b.Shape[0]}×{N}]");

        var result = Tensor.Create(_device, TensorShape.Matrix(M, N), DataType.F32, resultName);

        uint[] paramsData = { (uint)M, (uint)K, (uint)N };
        var paramsBuffer = _device.CreateBuffer((ulong)(paramsData.Length * sizeof(uint)), BufferType.Storage, DataType.I32);
        paramsBuffer.Write(paramsData);

        var kernel = GetOrCreateKernel("matmul_weights_f32", () => ComputeShaders.MatMulWeightsF32);
        kernel.SetArgument(0, a.Buffer);
        kernel.SetArgument(1, b.Buffer);
        kernel.SetArgument(2, result.Buffer);
        kernel.SetArgument(3, paramsBuffer);

        uint[] globalWorkSize = {
            (uint)((N + 15) / 16),
            (uint)((M + 15) / 16)
        };

        _queue.Dispatch(kernel, globalWorkSize, null);
        MaybeFlush();

        Defer(paramsBuffer);
        return result;
    }

    /// <summary>
    /// Matrix multiplication: C = A × B
    /// A: [M × K], B: [K × N], C: [M × N]
    /// </summary>
    public Tensor MatMul(Tensor a, Tensor b, string? resultName = null)
    {
        if (a.Shape.Rank != 2 || b.Shape.Rank != 2)
            throw new ArgumentException("MatMul requires 2D tensors");

        int M = a.Shape[0];
        int K = a.Shape[1];
        int N = b.Shape[1];

        if (K != b.Shape[0])
            throw new ArgumentException($"Incompatible shapes for MatMul: [{M}×{K}] × [{b.Shape[0]}×{N}]");

        var result = Tensor.Create(_device, TensorShape.Matrix(M, N), DataType.F32, resultName);

        // Создаём buffer для параметров M, K, N
        uint[] paramsData = { (uint)M, (uint)K, (uint)N };
        var paramsBuffer = _device.CreateBuffer((ulong)(paramsData.Length * sizeof(uint)), BufferType.Storage, DataType.I32);
        paramsBuffer.Write(paramsData);

        var kernel = GetOrCreateKernel("matmul_f32", () => ComputeShaders.MatMulF32);
        kernel.SetArgument(0, a.Buffer);
        kernel.SetArgument(1, b.Buffer);
        kernel.SetArgument(2, result.Buffer);
        kernel.SetArgument(3, paramsBuffer);
        
        uint[] globalWorkSize = { 
            (uint)((N + 15) / 16), 
            (uint)((M + 15) / 16) 
        };
        
        _queue.Dispatch(kernel, globalWorkSize, null);
        MaybeFlush();

        Defer(paramsBuffer);

        return result;
    }

    #endregion

    #region Element-wise Operations

    /// <summary>
    /// Element-wise multiplication: C = A ⊙ B
    /// </summary>
    public Tensor Multiply(Tensor a, Tensor b, string? resultName = null)
    {
        if (a.Shape.TotalElements != b.Shape.TotalElements)
            throw new ArgumentException("Tensors must have same total elements for element-wise ops");

        var result = Tensor.Create(_device, a.Shape, DataType.F32, resultName);

        uint[] paramsData = { (uint)a.Shape.TotalElements };
        var paramsBuffer = _device.CreateBuffer((ulong)(paramsData.Length * sizeof(uint)), BufferType.Storage, DataType.I32);
        paramsBuffer.Write(paramsData);

        var kernel = GetOrCreateKernel("elemwise_mul", () => ComputeShaders.ElementWiseMul);
        kernel.SetArgument(0, a.Buffer);
        kernel.SetArgument(1, b.Buffer);
        kernel.SetArgument(2, result.Buffer);
        kernel.SetArgument(3, paramsBuffer);

        uint[] globalWorkSize = { (uint)((a.Shape.TotalElements + 255) / 256) };
        _queue.Dispatch(kernel, globalWorkSize, null);
        MaybeFlush();

        Defer(paramsBuffer);

        return result;
    }

    /// <summary>
    /// Element-wise addition: C = A + B
    /// </summary>
    public Tensor Add(Tensor a, Tensor b, string? resultName = null)
    {
        if (a.Shape.TotalElements != b.Shape.TotalElements)
            throw new ArgumentException("Tensors must have same total elements for element-wise ops");

        var result = Tensor.Create(_device, a.Shape, DataType.F32, resultName);

        uint[] paramsData = { (uint)a.Shape.TotalElements };
        var paramsBuffer = _device.CreateBuffer((ulong)(paramsData.Length * sizeof(uint)), BufferType.Storage, DataType.I32);
        paramsBuffer.Write(paramsData);

        var kernel = GetOrCreateKernel("elemwise_add", () => ComputeShaders.ElementWiseAdd);
        kernel.SetArgument(0, a.Buffer);
        kernel.SetArgument(1, b.Buffer);
        kernel.SetArgument(2, result.Buffer);
        kernel.SetArgument(3, paramsBuffer);

        uint[] globalWorkSize = { (uint)((a.Shape.TotalElements + 255) / 256) };
        _queue.Dispatch(kernel, globalWorkSize, null);
        MaybeFlush();

        Defer(paramsBuffer);

        return result;
    }

    /// <summary>
    /// Concatenate two tensors along axis 1 (sequence dimension)
    /// A: [dim0 × dim1_a], B: [dim0 × dim1_b] → C: [dim0 × (dim1_a + dim1_b)]
    /// </summary>
    public Tensor Concat(Tensor a, Tensor b, int axis = 1, string? resultName = null)
    {
        if (a.Shape.Rank != 2 || b.Shape.Rank != 2)
            throw new ArgumentException("Currently only 2D tensor concat is supported");

        if (axis == 0)
        {
            // Concatenate along rows (axis 0): [rows_a, cols] + [rows_b, cols] = [rows_a + rows_b, cols]
            if (a.Shape[1] != b.Shape[1])
                throw new ArgumentException($"Columns must match for axis=0 concat: {a.Shape[1]} != {b.Shape[1]}");

            int rows_a = a.Shape[0];
            int rows_b = b.Shape[0];
            int cols = a.Shape[1];
            int total_rows = rows_a + rows_b;

            var result = Tensor.Create(_device, TensorShape.Matrix(total_rows, cols), DataType.F32, resultName);

            uint[] paramsData = { (uint)rows_a, (uint)rows_b, (uint)cols };
            var paramsBuffer = _device.CreateBuffer((ulong)(paramsData.Length * sizeof(uint)), BufferType.Storage, DataType.I32);
            paramsBuffer.Write(paramsData);

            var kernel = GetOrCreateKernel("concat_axis0", () => ComputeShaders.ConcatAxis0);
            kernel.SetArgument(0, a.Buffer);
            kernel.SetArgument(1, b.Buffer);
            kernel.SetArgument(2, result.Buffer);
            kernel.SetArgument(3, paramsBuffer);

            uint[] globalWorkSize = { (uint)((result.Shape.TotalElements + 255) / 256) };
            _queue.Dispatch(kernel, globalWorkSize, null);
            MaybeFlush();

            Defer(paramsBuffer);
            return result;
        }
        else if (axis == 1)
        {
            // Concatenate along columns (axis 1): [rows, cols_a] + [rows, cols_b] = [rows, cols_a + cols_b]
            if (a.Shape[0] != b.Shape[0])
                throw new ArgumentException($"Rows must match for axis=1 concat: {a.Shape[0]} != {b.Shape[0]}");

            int dim0 = a.Shape[0];
            int dim1_a = a.Shape[1];
            int dim1_b = b.Shape[1];
            int dim1_total = dim1_a + dim1_b;

            var result = Tensor.Create(_device, TensorShape.Matrix(dim0, dim1_total), DataType.F32, resultName);

            uint[] paramsData = { (uint)dim0, (uint)dim1_a, (uint)dim1_b };
            var paramsBuffer = _device.CreateBuffer((ulong)(paramsData.Length * sizeof(uint)), BufferType.Storage, DataType.I32);
            paramsBuffer.Write(paramsData);

            var kernel = GetOrCreateKernel("concat_axis1", () => ComputeShaders.ConcatAxis1);
            kernel.SetArgument(0, a.Buffer);
            kernel.SetArgument(1, b.Buffer);
            kernel.SetArgument(2, result.Buffer);
            kernel.SetArgument(3, paramsBuffer);

            uint[] globalWorkSize = { (uint)((result.Shape.TotalElements + 255) / 256) };
            _queue.Dispatch(kernel, globalWorkSize, null);
            MaybeFlush();

            Defer(paramsBuffer);
            return result;
        }
        else
        {
            throw new ArgumentException($"Unsupported concat axis: {axis}");
        }
    }

    /// <summary>
    /// In-place SiLU activation: x = x * sigmoid(x)
    /// </summary>
    public void SiLU(Tensor tensor)
    {
        uint[] paramsData = { (uint)tensor.Shape.TotalElements };
        var paramsBuffer = _device.CreateBuffer((ulong)(paramsData.Length * sizeof(uint)), BufferType.Storage, DataType.I32);
        paramsBuffer.Write(paramsData);

        var kernel = GetOrCreateKernel("silu", () => ComputeShaders.SiLU);
        kernel.SetArgument(0, tensor.Buffer);
        kernel.SetArgument(1, paramsBuffer);

        uint[] globalWorkSize = { (uint)((tensor.Shape.TotalElements + 255) / 256) };
        _queue.Dispatch(kernel, globalWorkSize, null);
        MaybeFlush();

        Defer(paramsBuffer);
    }

    /// <summary>
    /// In-place sigmoid activation: x = 1 / (1 + exp(-x))
    /// </summary>
    public void Sigmoid(Tensor tensor)
    {
        uint[] paramsData = { (uint)tensor.Shape.TotalElements };
        var paramsBuffer = _device.CreateBuffer((ulong)(paramsData.Length * sizeof(uint)), BufferType.Storage, DataType.I32);
        paramsBuffer.Write(paramsData);

        var kernel = GetOrCreateKernel("sigmoid", () => ComputeShaders.Sigmoid);
        kernel.SetArgument(0, tensor.Buffer);
        kernel.SetArgument(1, paramsBuffer);

        uint[] globalWorkSize = { (uint)((tensor.Shape.TotalElements + 255) / 256) };
        _queue.Dispatch(kernel, globalWorkSize, null);
        MaybeFlush();

        Defer(paramsBuffer);
    }

    #endregion

    #region Normalization

    /// <summary>
    /// Layer normalization: out = (x - mean(x)) / sqrt(var(x) + eps) * weight
    /// </summary>
    public void LayerNorm(Tensor tensor, Tensor weight, float eps = 1e-5f)
    {
        // For 1D tensor treat as single row; for 2D apply per-row (per-token) RMSNorm.
        int rows = tensor.Shape.Rank >= 2 ? tensor.Shape[0] : 1;
        int cols = tensor.Shape.Rank >= 2 ? tensor.Shape[1] : tensor.Shape[0];

        // Params layout must match shader struct: { uint rows; uint cols; float eps; }
        // Pack as raw bytes: rows (4 bytes) + cols (4 bytes) + eps (4 bytes)
        var paramsBuffer = _device.CreateBuffer(12, BufferType.Storage, DataType.I32);
        var paramBytes = new byte[12];
        BitConverter.GetBytes((uint)rows).CopyTo(paramBytes, 0);
        BitConverter.GetBytes((uint)cols).CopyTo(paramBytes, 4);
        BitConverter.GetBytes(eps).CopyTo(paramBytes, 8);
        paramsBuffer.Write(paramBytes);

        var kernel = GetOrCreateKernel("layer_norm", () => ComputeShaders.LayerNorm);
        kernel.SetArgument(0, tensor.Buffer);
        kernel.SetArgument(1, weight.Buffer);
        kernel.SetArgument(2, paramsBuffer);

        // One workgroup per row — each workgroup of 256 threads handles one token's vector
        _queue.Dispatch(kernel, new[] { (uint)rows }, null);
        MaybeFlush();

        Defer(paramsBuffer);
    }

    /// <summary>
    /// RMSNorm with explicit row/col override — used for per-head normalization.
    /// The tensor buffer is treated as (rows × cols) regardless of its logical shape.
    /// For per-head QK-norm on a [seqLen, numHeads×headDim] tensor, call with
    /// rows = seqLen×numHeads, cols = headDim and the un-tiled [headDim] weight.
    /// This normalizes each headDim-sized sub-vector independently.
    /// </summary>
    public void LayerNormVirtual(Tensor tensor, Tensor weight, int rows, int cols, float eps = 1e-5f)
    {
        var paramsBuffer = _device.CreateBuffer(12, BufferType.Storage, DataType.I32);
        var paramBytes = new byte[12];
        BitConverter.GetBytes((uint)rows).CopyTo(paramBytes, 0);
        BitConverter.GetBytes((uint)cols).CopyTo(paramBytes, 4);
        BitConverter.GetBytes(eps).CopyTo(paramBytes, 8);
        paramsBuffer.Write(paramBytes);
        var kernel = GetOrCreateKernel("layer_norm", () => ComputeShaders.LayerNorm);
        kernel.SetArgument(0, tensor.Buffer);
        kernel.SetArgument(1, weight.Buffer);
        kernel.SetArgument(2, paramsBuffer);
        _queue.Dispatch(kernel, new[] { (uint)rows }, null);
        MaybeFlush();
        Defer(paramsBuffer);
    }

    /// <summary>
    /// Softmax activation (in-place)
    /// </summary>
    public void Softmax(Tensor tensor)
    {
        uint[] paramsData = { (uint)tensor.Shape.TotalElements };
        var paramsBuffer = _device.CreateBuffer((ulong)(paramsData.Length * sizeof(uint)), BufferType.Storage, DataType.I32);
        paramsBuffer.Write(paramsData);

        var kernel = GetOrCreateKernel("softmax", () => ComputeShaders.Softmax);
        kernel.SetArgument(0, tensor.Buffer);
        kernel.SetArgument(1, paramsBuffer);

        uint[] globalWorkSize = { (uint)((tensor.Shape.TotalElements + 255) / 256) };
        _queue.Dispatch(kernel, globalWorkSize, null);
        MaybeFlush();

        Defer(paramsBuffer);
    }

    #endregion

    #region Position Encoding

    /// <summary>
    /// Применить RoPE (Rotary Position Embedding)
    /// </summary>
    public void ApplyRoPE(Tensor tensor, uint position, uint headDim, float theta = 10000.0f)
    {
        // Params: seq_len (uint), head_dim (uint), position (uint), theta (float)
        uint seqLen = (uint)(tensor.Shape.TotalElements / headDim);
        
        // Создаем буфер с mixed типами: 3 uint + 1 float
        byte[] paramsBytes = new byte[16]; // 4 * sizeof(uint/float)
        Buffer.BlockCopy(BitConverter.GetBytes(seqLen), 0, paramsBytes, 0, 4);
        Buffer.BlockCopy(BitConverter.GetBytes(headDim), 0, paramsBytes, 4, 4);
        Buffer.BlockCopy(BitConverter.GetBytes(position), 0, paramsBytes, 8, 4);
        Buffer.BlockCopy(BitConverter.GetBytes(theta), 0, paramsBytes, 12, 4);
        
        var paramsBuffer = _device.CreateBuffer(16, BufferType.Storage, DataType.I32);
        paramsBuffer.Write(paramsBytes);

        var kernel = GetOrCreateKernel("rope", () => ComputeShaders.RoPE);
        kernel.SetArgument(0, tensor.Buffer);
        kernel.SetArgument(1, paramsBuffer);

        uint[] globalWorkSize = { (uint)((headDim / 2 + 255) / 256) };
        _queue.Dispatch(kernel, globalWorkSize, null);
        MaybeFlush();

        Defer(paramsBuffer);
    }

    #endregion

    #region Dequantization

    /// <summary>
    /// Деквантизировать тензор Q2_K → F32
    /// Uses persistent zero-offset buffer (binding=2) to avoid GPUVM faults on AMD RADV.
    /// </summary>
    public Tensor DequantizeQ2K(Tensor quantized, string? resultName = null)
    {
        if (quantized.DataType != DataType.Q2_K)
            throw new ArgumentException("Input must be Q2_K tensor");

        var result = Tensor.Create(_device, quantized.Shape, DataType.F32, resultName);

        var kernel = GetOrCreateKernel("dequant_q2k", () => ComputeShaders.DequantizeQ2K);
        kernel.SetArgument(0, quantized.Buffer);
        kernel.SetArgument(1, result.Buffer);

        // CRITICAL FIX: Use persistent zero-offset buffer instead of creating a new one
        // every call. On AMD RADV, creating and destroying small buffers causes GPUVM
        // faults because the Texture Cache Parser (TCP) caches descriptor data and may
        // attempt to write to a freed buffer's GPUVM address.
        kernel.SetArgument(2, _persistentZeroOffset);

        uint[] globalWorkSize = { (uint)((quantized.Shape.TotalElements + 255) / 256) };
        _queue.Dispatch(kernel, globalWorkSize, null);
        MaybeFlush();

        return result;
    }

    /// <summary>
    /// Деквантизировать тензор Q3_K → F32
    /// Uses persistent zero-offset buffer (binding=2) to avoid GPUVM faults on AMD RADV.
    /// </summary>
    public Tensor DequantizeQ3K(Tensor quantized, string? resultName = null)
    {
        if (quantized.DataType != DataType.Q3_K)
            throw new ArgumentException("Input must be Q3_K tensor");

        var result = Tensor.Create(_device, quantized.Shape, DataType.F32, resultName);

        var kernel = GetOrCreateKernel("dequant_q3k", () => ComputeShaders.DequantizeQ3K);
        kernel.SetArgument(0, quantized.Buffer);
        kernel.SetArgument(1, result.Buffer);

        // CRITICAL FIX: Use persistent zero-offset buffer
        kernel.SetArgument(2, _persistentZeroOffset);

        uint[] globalWorkSize = { (uint)((quantized.Shape.TotalElements + 255) / 256) };
        _queue.Dispatch(kernel, globalWorkSize, null);
        MaybeFlush();

        return result;
    }

    /// <summary>
    /// Деквантизировать тензор Q4_K → F32
    /// Uses persistent zero-offset buffer (binding=2) to avoid GPUVM faults on AMD RADV.
    /// </summary>
    public Tensor DequantizeQ4K(Tensor quantized, string? resultName = null)
    {
        if (quantized.DataType != DataType.Q4_K)
            throw new ArgumentException("Input must be Q4_K tensor");

        var result = Tensor.Create(_device, quantized.Shape, DataType.F32, resultName);

        var kernel = GetOrCreateKernel("dequant_q4k", () => ComputeShaders.DequantizeQ4K);
        kernel.SetArgument(0, quantized.Buffer);
        kernel.SetArgument(1, result.Buffer);

        // CRITICAL FIX: Use persistent zero-offset buffer
        kernel.SetArgument(2, _persistentZeroOffset);

        uint[] globalWorkSize = { (uint)((quantized.Shape.TotalElements + 255) / 256) };
        _queue.Dispatch(kernel, globalWorkSize, null);
        MaybeFlush();

        return result;
    }

    /// <summary>
    /// Деквантизировать тензор Q5_K → F32
    /// Uses persistent zero-offset buffer (binding=2) to avoid GPUVM faults on AMD RADV.
    /// </summary>
    public Tensor DequantizeQ5K(Tensor quantized, string? resultName = null)
    {
        if (quantized.DataType != DataType.Q5_K)
            throw new ArgumentException("Input must be Q5_K tensor");

        var result = Tensor.Create(_device, quantized.Shape, DataType.F32, resultName);

        var kernel = GetOrCreateKernel("dequant_q5k", () => ComputeShaders.DequantizeQ5K);
        kernel.SetArgument(0, quantized.Buffer);
        kernel.SetArgument(1, result.Buffer);

        // CRITICAL FIX: Use persistent zero-offset buffer
        kernel.SetArgument(2, _persistentZeroOffset);

        uint[] globalWorkSize = { (uint)((quantized.Shape.TotalElements + 255) / 256) };
        _queue.Dispatch(kernel, globalWorkSize, null);
        MaybeFlush();

        return result;
    }

    /// <summary>
    /// Деквантизировать тензор Q6_K → F32
    /// Uses persistent zero-offset buffer (binding=2) to avoid GPUVM faults on AMD RADV.
    /// </summary>
    public Tensor DequantizeQ6K(Tensor quantized, string? resultName = null)
    {
        if (quantized.DataType != DataType.Q6_K)
            throw new ArgumentException("Input must be Q6_K tensor");

        var result = Tensor.Create(_device, quantized.Shape, DataType.F32, resultName);

        var kernel = GetOrCreateKernel("dequant_q6k", () => ComputeShaders.DequantizeQ6K);
        kernel.SetArgument(0, quantized.Buffer);
        kernel.SetArgument(1, result.Buffer);

        // CRITICAL FIX: Use persistent zero-offset buffer
        kernel.SetArgument(2, _persistentZeroOffset);

        uint[] globalWorkSize = { (uint)((quantized.Shape.TotalElements + 255) / 256) };
        _queue.Dispatch(kernel, globalWorkSize, null);
        MaybeFlush();

        return result;
    }


    // One-shot GPU-vs-CPU dequant verification per data type.
    private static readonly ConcurrentDictionary<DataType, bool> _dequantVerified = new();

    /// <summary>
    /// Универсальная деквантизация (автоопределение типа)
    /// </summary>
    public Tensor Dequantize(Tensor quantized, string? resultName = null)
    {
        return quantized.DataType switch
        {
            DataType.Q2_K => DequantizeQ2K(quantized, resultName),
            DataType.Q3_K => DequantizeQ3K(quantized, resultName),
            DataType.Q4_K => DequantizeQ4K(quantized, resultName),
            DataType.Q5_K => DequantizeQ5K(quantized, resultName),
            DataType.Q6_K => DequantizeQ6K(quantized, resultName),
            DataType.F32 => quantized,
            _ => throw new NotSupportedException($"Dequantization for {quantized.DataType} not implemented")
        };
    }

    /// <summary>
    /// Dequantize into a pre-allocated F32 tensor (avoids vkAllocateMemory per call).
    /// For large tensors, splits the dispatch into chunks of at most MaxGroupsPerDispatch
    /// workgroups to avoid GPUVM faults on AMD RADV caused by excessively long-running
    /// dispatches (>10 seconds) that trigger TDR or memory access violations.
    ///
    /// Each chunk passes an elementOffset via a small offset buffer (binding=2) so the
    /// shader computes gid = gl_GlobalInvocationID.x + offBuf.elementOffset, allowing
    /// the same input/output buffers to be reused across chunks without aliasing.
    ///
    /// Chunks are aligned to superblock boundaries (256 elements = 1 workgroup) to
    /// ensure correct byte offsets for quantized formats where bytes-per-element is
    /// fractional (e.g. Q4_K = 0.5 bytes/element).
    /// </summary>
    public void DequantizeInto(Tensor quantized, Tensor target)
    {
        string dequantTag = quantized.DataType switch
        {
            DataType.Q2_K => "Dequant.Q2K",
            DataType.Q3_K => "Dequant.Q3K",
            DataType.Q4_K => "Dequant.Q4K",
            DataType.Q5_K => "Dequant.Q5K",
            DataType.Q6_K => "Dequant.Q6K",
            _ => "Dequant."
        };
        var _ts = GlobalProfiler.Start();
        if (target.DataType != DataType.F32)
            throw new ArgumentException("Target must be F32");

        string kernelName = quantized.DataType switch
        {
            DataType.Q2_K => "dequant_q2k",
            DataType.Q3_K => "dequant_q3k",
            DataType.Q4_K => "dequant_q4k",
            DataType.Q5_K => "dequant_q5k",
            DataType.Q6_K => "dequant_q6k",
            _ => throw new NotSupportedException($"DequantizeInto not supported for {quantized.DataType}")
        };

        var kernel = GetOrCreateKernel(kernelName, ()=> quantized.DataType switch
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

        // AMD RADV can hang or fault on dispatches with >~65535 workgroups
        // (especially on large tensors like 89M elements = 348160 groups).
        // Split into chunks of at most MaxGroupsPerDispatch groups.
        const uint MaxGroupsPerDispatch = 32768;

        // Create per-chunk offset buffers (1 uint each) so each chunk gets its own
        // buffer. This avoids a CPU-GPU race condition where the CPU writes a new
        // elementOffset into a shared buffer while the GPU is still reading the old
        // value from its L2 cache (Texture Cache Parser on AMD RADV).
        // Binding=2: the shader reads offBuf.elementOffset and adds it to
        // gl_GlobalInvocationID.x to compute the global element index.
        const uint SuperblockElements = 256;
        uint groupsPerChunk = MaxGroupsPerDispatch;
        uint numChunks = (totalGroups + groupsPerChunk - 1) / groupsPerChunk;

        if (totalGroups <= MaxGroupsPerDispatch)
        {
            // Single dispatch — fast path, offset = 0
            // CRITICAL FIX: Use persistent zero-offset buffer instead of creating a new one.
            // On AMD RADV, creating and destroying small buffers causes GPUVM faults because
            // the Texture Cache Parser (TCP) caches descriptor data and may attempt to write
            // to a freed buffer's GPUVM address.
            kernel.SetArgument(0, quantized.Buffer);
            kernel.SetArgument(1, target.Buffer);
            kernel.SetArgument(2, _persistentZeroOffset);
            _queue.Dispatch(kernel, new[] { totalGroups }, null);

            if (_batchMode)
                _queue.InsertMemoryBarrier();
            else
            {
                _queue.Flush();
                foreach (var k in _kernelCache.Values)
                    if (k is VulkanComputeKernel vk)
                        vk.ResetDispatchRing();
            }
            GlobalProfiler.End(_ts, dequantTag);
        }

        else
        {
            var reusableOffsetBuf = _device.CreateBuffer(sizeof(uint), BufferType.Storage, DataType.I32);
            _logger.LogDebug("[DBG_OPS] DequantizeInto chunked: totalElements={Total} totalGroups={Groups} chunks={Chunks}",
                totalElements, totalGroups, numChunks);

            for (uint chunk = 0; chunk < numChunks; chunk++)
            {
                uint chunkStartGroup = chunk * groupsPerChunk;
                uint chunkEndGroup = Math.Min(chunkStartGroup + groupsPerChunk, totalGroups);
                uint chunkGroups = chunkEndGroup - chunkStartGroup;
                uint elementOffset = chunkStartGroup * SuperblockElements;
                uint[] offsetData = { elementOffset };
                reusableOffsetBuf.Write(offsetData);

                kernel.SetArgument(0, quantized.Buffer);
                kernel.SetArgument(1, target.Buffer);
                kernel.SetArgument(2, reusableOffsetBuf);
                _queue.Dispatch(kernel, new[] { chunkGroups }, null);

                _queue.Flush();
                foreach (var k in _kernelCache.Values)
                    if (k is VulkanComputeKernel vk) vk.ResetDispatchRing();
            }
            _logger.LogDebug("[DBG_OPS] DequantizeInto chunked done");
            Defer(reusableOffsetBuf);
            GlobalProfiler.End(_ts, dequantTag);
        }

        // FIX: One-shot GPU-vs-CPU dequant verification (once per data type).
        var dt = quantized.DataType;
        if (_dequantVerified.TryAdd(dt, true))
        {
            try
            {
                int check = Math.Min(256, (int)target.Shape.TotalElements);
                // Read only 1 superblock worth of raw bytes (≤256 elements), NOT entire buffer.
                int bytesPerBlock = dt switch {
                    DataType.Q2_K => 84, DataType.Q3_K => 110, DataType.Q4_K => 144,
                    DataType.Q5_K => 176, DataType.Q6_K => 210, _ => check * 4
                };
                var raw = quantized.Buffer.ReadRange<byte>(0, bytesPerBlock);
                var gpu = target.Buffer.ReadRange<float>(0, check);
                var cpu = new float[check];
                // FIX: pass check (not totalElements) — DequantizeRowCpu writes `dModel` floats into `cpu`,
                // and `cpu` only has `check` elements allocated.
                DequantizeRowCpu(dt, raw, 0, cpu, 0, check);
                // FIX: log raw bytes at the d/dmin offset for debugging GPU vs CPU data mismatch
                int dOff = dt == DataType.Q6_K ? 208 : 0;
                _logger.LogError("[DIAG_DEQUANT_{Dtype}] rawBytes[{Off}]={B0} rawBytes[{Off2}]={B1} rawBytes[0]={B2} rawBytes[1]={B3}",
                    dt, dOff, raw[dOff], dOff+1, raw[dOff+1], raw[0], raw[1]);
                int mismatches = 0; float maxErr = 0f;
                for (int i = 0; i < check; i++)
                {
                    float err = Math.Abs(gpu[i] - cpu[i]);
                    if (err > 0.01f) { mismatches++; if (err > maxErr) maxErr = err; }
                }
                if (mismatches > 0)
                {
                    var s = new string[Math.Min(8, check)];
                    for (int i = 0; i < s.Length; i++) s[i] = $"{i}:gpu={gpu[i]:F3} cpu={cpu[i]:F3}";
                    _logger.LogError("[DIAG_DEQUANT_{Dtype}] MISMATCH {M}/{N} maxErr={E:F4} [{S}]",
                        dt, mismatches, check, maxErr, string.Join(",", s));
                }
                else _logger.LogInformation("[DIAG_DEQUANT_{Dtype}] OK {N} elements", dt, check);
            }
            catch (Exception ex) { _logger.LogError("[DIAG_DEQUANT_{Dtype}] err={E}", dt, ex.Message); }
        }
    }

    /// <summary>
    /// Insert a pipeline barrier in the current batch (no-op in non-batch mode).
    /// Use after a group of independent dispatches that are all needed by subsequent ops.
    /// </summary>
    public void InsertBarrier()
    {
        if (_batchMode) _queue.InsertMemoryBarrier();
    }

    #endregion

    #region Attention Operations

    /// <summary>
    /// Transpose matrix: B = A^T
    /// A: [rows × cols], B: [cols × rows]
    /// </summary>
    public Tensor Transpose(Tensor input, string? resultName = null)
    {
        if (input.Shape.Rank != 2)
            throw new ArgumentException("Transpose requires 2D tensor");

        int rows = input.Shape[0];
        int cols = input.Shape[1];
        var result = Tensor.Create(_device, TensorShape.Matrix(cols, rows), DataType.F32, resultName);

        uint[] paramsData = { (uint)rows, (uint)cols };
        var paramsBuffer = _device.CreateBuffer((ulong)(paramsData.Length * sizeof(uint)), BufferType.Storage, DataType.I32);
        paramsBuffer.Write(paramsData);

        var kernel = GetOrCreateKernel("transpose", () => ComputeShaders.Transpose);
        kernel.SetArgument(0, input.Buffer);
        kernel.SetArgument(1, result.Buffer);
        kernel.SetArgument(2, paramsBuffer);

        uint[] globalWorkSize = { (uint)((cols + 15) / 16), (uint)((rows + 15) / 16) };
        _queue.Dispatch(kernel, globalWorkSize, null);
        MaybeFlush();

        Defer(paramsBuffer);
        return result;
    }

    /// <summary>
    /// Row-wise Softmax: apply softmax to each row independently
    /// </summary>
    public void RowwiseSoftmax(Tensor tensor)
    {
        if (tensor.Shape.Rank != 2)
            throw new ArgumentException("RowwiseSoftmax requires 2D tensor");

        int rows = tensor.Shape[0];
        int cols = tensor.Shape[1];

        uint[] paramsData = { (uint)rows, (uint)cols };
        var paramsBuffer = _device.CreateBuffer((ulong)(paramsData.Length * sizeof(uint)), BufferType.Storage, DataType.I32);
        paramsBuffer.Write(paramsData);

        var kernel = GetOrCreateKernel("rowwise_softmax", () => ComputeShaders.RowwiseSoftmax);
        kernel.SetArgument(0, tensor.Buffer);
        kernel.SetArgument(1, paramsBuffer);

        uint[] globalWorkSize = { (uint)rows };
        _queue.Dispatch(kernel, globalWorkSize, null);
        MaybeFlush();

        Defer(paramsBuffer);
    }

    /// <summary>
    /// Scale tensor by scalar: tensor *= scale
    /// </summary>
    public void Scale(Tensor tensor, float scale)
    {
        byte[] paramsBytes = new byte[8]; // uint size + float scale
        Buffer.BlockCopy(BitConverter.GetBytes((uint)tensor.Shape.TotalElements), 0, paramsBytes, 0, 4);
        Buffer.BlockCopy(BitConverter.GetBytes(scale), 0, paramsBytes, 4, 4);

        var paramsBuffer = _device.CreateBuffer(8, BufferType.Storage, DataType.I32);
        paramsBuffer.Write(paramsBytes);

        var kernel = GetOrCreateKernel("scale", () => ComputeShaders.Scale);
        kernel.SetArgument(0, tensor.Buffer);
        kernel.SetArgument(1, paramsBuffer);

        uint[] globalWorkSize = { (uint)((tensor.Shape.TotalElements + 255) / 256) };
        _queue.Dispatch(kernel, globalWorkSize, null);
        MaybeFlush();

        Defer(paramsBuffer);
    }

    /// <summary>
    /// Apply RoPE to a full Q or K projection tensor [seqLen, numHeads * headDim].
    /// Each (seq, head) pair is rotated at position startPosition + seq.
    /// </summary>
    public void ApplyRoPEFull(Tensor tensor, uint startPosition, int numHeads, int headDim, float theta = 10000.0f, int ropeDim = 0)
    {
        // ropeDim: number of head dimensions to rotate. 0 → use headDim (all pairs).
        // For models with partial RoPE (e.g. Qwen3.5/3.6 rope.dimension_count=64 in a 256-dim head),
        // only the first ropeDim dimensions per head are rotated; the rest are left unchanged.
        if (ropeDim <= 0) ropeDim = headDim;
        int seqLen = tensor.Shape[0];
        var paramsBuffer = _device.CreateBuffer(24, BufferType.Storage, DataType.I32);
        var p = new byte[24];
        BitConverter.GetBytes((uint)seqLen).CopyTo(p, 0);
        BitConverter.GetBytes((uint)numHeads).CopyTo(p, 4);
        BitConverter.GetBytes((uint)headDim).CopyTo(p, 8);
        BitConverter.GetBytes(startPosition).CopyTo(p, 12);
        BitConverter.GetBytes(theta).CopyTo(p, 16);
        BitConverter.GetBytes((uint)ropeDim).CopyTo(p, 20);
        paramsBuffer.Write(p);

        var kernel = GetOrCreateKernel("rope_full", () => ComputeShaders.RoPEFull);
        kernel.SetArgument(0, tensor.Buffer);
        kernel.SetArgument(1, paramsBuffer);

        uint total = (uint)(seqLen * numHeads * ropeDim / 2);
        _queue.Dispatch(kernel, new[] { (total + 255u) / 256u }, null);
        MaybeFlush();
        Defer(paramsBuffer);
    }

    /// <summary>
    /// Apply causal mask in-place to attention scores [seqLen_q × seqLen_k].
    /// Sets scores[i, j] = -inf where j > startPosition + i (future positions).
    /// Only has effect when seqLen_q > 1 (prefill).
    /// </summary>
    public void ApplyCausalMask(Tensor scores, uint startPosition)
    {
        int seqLen_q = scores.Shape[0];
        int seqLen_k = scores.Shape[1];
        if (seqLen_q <= 1) return;

        var paramsBuffer = _device.CreateBuffer(12, BufferType.Storage, DataType.I32);
        var p = new byte[12];
        BitConverter.GetBytes((uint)seqLen_q).CopyTo(p, 0);
        BitConverter.GetBytes((uint)seqLen_k).CopyTo(p, 4);
        BitConverter.GetBytes(startPosition).CopyTo(p, 8);
        paramsBuffer.Write(p);

        var kernel = GetOrCreateKernel("causal_mask", () => ComputeShaders.CausalMask);
        kernel.SetArgument(0, scores.Buffer);
        kernel.SetArgument(1, paramsBuffer);

        uint total = (uint)(seqLen_q * seqLen_k);
        _queue.Dispatch(kernel, new[] { (total + 255u) / 256u }, null);
        MaybeFlush();
        Defer(paramsBuffer);
    }

    /// <summary>
    /// Fused GQA attention for a single query token (seqLen=1).
    /// One GPU dispatch covers all numQHeads heads — avoids per-head tensor allocations.
    /// Q: [1, numQHeads*headDim], K/V: [kvSeqLen, numKvHeads*headDim]
    /// Returns: [1, numQHeads*headDim]
    /// </summary>
    public Tensor FusedMHAGenerate(Tensor Q, Tensor K, Tensor V,
        int numQHeads, int numKvHeads, int headDim, float scale, string? resultName = null)
    {
        int kvSeqLen = K.Shape[0];
        var result = Tensor.Create(_device, TensorShape.Matrix(1, numQHeads * headDim), DataType.F32, resultName);

        // Params: numQHeads, numKvHeads, headDim, kvSeqLen, scale (as float bits)
        var paramsBuffer = _device.CreateBuffer(20, BufferType.Storage, DataType.I32);
        var p = new byte[20];
        BitConverter.GetBytes((uint)numQHeads).CopyTo(p, 0);
        BitConverter.GetBytes((uint)numKvHeads).CopyTo(p, 4);
        BitConverter.GetBytes((uint)headDim).CopyTo(p, 8);
        BitConverter.GetBytes((uint)kvSeqLen).CopyTo(p, 12);
        BitConverter.GetBytes(scale).CopyTo(p, 16);
        paramsBuffer.Write(p);

        var kernel = GetOrCreateKernel("fused_mha_generate", () => ComputeShaders.FusedMHAGenerate);
        kernel.SetArgument(0, Q.Buffer);
        kernel.SetArgument(1, K.Buffer);
        kernel.SetArgument(2, V.Buffer);
        kernel.SetArgument(3, result.Buffer);
        kernel.SetArgument(4, paramsBuffer);

        // One workgroup per Q head, each with 256 threads
        _queue.Dispatch(kernel, new[] { (uint)numQHeads }, null);
        MaybeFlush();
        Defer(paramsBuffer);
        return result;
    }

    /// <summary>
    /// Correct multi-head / grouped-query attention computed per head.
    /// Each Q head attends to its corresponding KV head independently,
    /// then outputs are concatenated. This avoids the "mix-all-heads" bug
    /// where Q @ K^T sums contributions from all heads before softmax.
    ///
    /// Q: [seqLen, numQHeads * headDim]
    /// K: [kvSeqLen, numKvHeads * headDim]  (kvSeqLen may differ from seqLen with KV cache)
    /// V: [kvSeqLen, numKvHeads * headDim]
    /// </summary>
    public Tensor MultiHeadAttention(Tensor Q, Tensor K, Tensor V, int numHeads, uint startPosition = 0, string? resultName = null)
    {
        if (Q.Shape.Rank != 2 || K.Shape.Rank != 2 || V.Shape.Rank != 2)
            throw new ArgumentException("Q, K, V must be 2D tensors");

        int seqLen   = Q.Shape[0];
        int dModel_Q = Q.Shape[1];
        int kvDim    = K.Shape[1];
        int headDim  = dModel_Q / numHeads;
        int numKvHeads   = kvDim / headDim;
        float scale = 1.0f / MathF.Sqrt(headDim);

        // ── Fast path: seqLen=1 (generation) ────────────────────────────────
        // Single fused dispatch: one workgroup per Q head, no per-head allocs.
        if (seqLen == 1)
        {
            return FusedMHAGenerate(Q, K, V, numHeads, numKvHeads, headDim, scale,
                                    resultName ?? "attn_out");
        }

        // ── Slow path: seqLen>1 (prefill) — per-head loop ───────────────────
        var output = Tensor.Create(_device, TensorShape.Matrix(seqLen, dModel_Q), DataType.F32, resultName ?? "attn_out");

        for (int h = 0; h < numHeads; h++)
        {
            int kvHead = h / (numHeads / numKvHeads);

            var Q_h  = SliceCols(Q, h      * headDim, headDim, $"Q_{h}");
            var K_h  = SliceCols(K, kvHead * headDim, headDim, $"K_{h}");
            var V_h  = SliceCols(V, kvHead * headDim, headDim, $"V_{h}");
            var KT_h = Transpose(K_h, $"KT_{h}");

            var scores_h = MatMul(Q_h, KT_h, $"scores_{h}");
            Defer(KT_h); Defer(K_h); Defer(Q_h);

            Scale(scores_h, scale);
            ApplyCausalMask(scores_h, startPosition);
            RowwiseSoftmax(scores_h);

            var out_h = MatMul(scores_h, V_h, $"out_{h}");
            Defer(scores_h); Defer(V_h);

            ScatterCols(output, out_h, h * headDim);
            Defer(out_h);
        }

        return output;
    }

    /// <summary>
    /// Extract columns [colStart, colStart+colCount) from input [rows, srcCols] → [rows, colCount]
    /// </summary>
    public Tensor SliceCols(Tensor input, int colStart, int colCount, string? resultName = null)
    {
        int rows = input.Shape[0], srcCols = input.Shape[1];
        var result = Tensor.Create(_device, TensorShape.Matrix(rows, colCount), DataType.F32, resultName);
        var paramsBuffer = _device.CreateBuffer(16, BufferType.Storage, DataType.I32);
        var p = new byte[16];
        BitConverter.GetBytes((uint)rows).CopyTo(p, 0); BitConverter.GetBytes((uint)srcCols).CopyTo(p, 4);
        BitConverter.GetBytes((uint)colStart).CopyTo(p, 8); BitConverter.GetBytes((uint)colCount).CopyTo(p, 12);
        paramsBuffer.Write(p);
        var kernel = GetOrCreateKernel("slice_cols", () => ComputeShaders.SliceCols);
        kernel.SetArgument(0, input.Buffer); kernel.SetArgument(1, result.Buffer); kernel.SetArgument(2, paramsBuffer);
        _queue.Dispatch(kernel, new[] { ((uint)(rows * colCount) + 255u) / 256u }, null);
        MaybeFlush(); Defer(paramsBuffer);
        return result;
    }

    /// <summary>
    /// Rearranges Q+gate interleaved per head into [Q_all, gate_all] layout.
    /// Input : [sl, n_head * 2 * head_dim] — [Q_h0(hd), gate_h0(hd), Q_h1(hd), gate_h1(hd), ...]
    /// Output: [sl, n_head * 2 * head_dim] — [Q_h0, Q_h1, ..., Q_hn, gate_h0, ..., gate_hn]
    /// Use SliceCols(result, 0, n_head*head_dim) for Q and SliceCols(result, n_head*head_dim, n_head*head_dim) for gate.
    /// </summary>
    public Tensor DeinterleaveQGate(Tensor input, uint nHead, uint headDim, string? resultName = null)
    {
        var result = Tensor.Create(_device, input.Shape, DataType.F32, resultName);
        var paramsBuffer = _device.CreateBuffer(12, BufferType.Storage, DataType.I32);
        var p = new byte[12];
        BitConverter.GetBytes((uint)input.Shape[0]).CopyTo(p, 0);
        BitConverter.GetBytes(nHead).CopyTo(p, 4);
        BitConverter.GetBytes(headDim).CopyTo(p, 8);
        paramsBuffer.Write(p);
        var kernel = GetOrCreateKernel("deinterleave_q_gate", () => ComputeShaders.DeinterleaveQGate);
        kernel.SetArgument(0, input.Buffer);
        kernel.SetArgument(1, result.Buffer);
        kernel.SetArgument(2, paramsBuffer);
        _queue.Dispatch(kernel, new[] { ((uint)input.Shape.TotalElements + 255u) / 256u }, null);
        MaybeFlush();
        Defer(paramsBuffer);
        return result;
    }

    /// <summary>Write src[rows, colCount] into dst[rows, dstCols] at column offset colStart.</summary>
    public void ScatterCols(Tensor dst, Tensor src, int colStart)
    {
        int rows = src.Shape[0], colCount = src.Shape[1], dstCols = dst.Shape[1];
        var paramsBuffer = _device.CreateBuffer(16, BufferType.Storage, DataType.I32);
        var p = new byte[16];
        BitConverter.GetBytes((uint)rows).CopyTo(p, 0); BitConverter.GetBytes((uint)dstCols).CopyTo(p, 4);
        BitConverter.GetBytes((uint)colStart).CopyTo(p, 8); BitConverter.GetBytes((uint)colCount).CopyTo(p, 12);
        paramsBuffer.Write(p);
        var kernel = GetOrCreateKernel("scatter_cols", () => ComputeShaders.ScatterCols);
        kernel.SetArgument(0, src.Buffer); kernel.SetArgument(1, dst.Buffer); kernel.SetArgument(2, paramsBuffer);
        _queue.Dispatch(kernel, new[] { ((uint)(rows * colCount) + 255u) / 256u }, null);
        MaybeFlush(); Defer(paramsBuffer);
    }

    /// <summary>
    /// Correct GQA K/V head expansion.
    /// Repeats each headDim-sized block of columns repeatFactor times so that
    /// Q head h shares KV head (h / repeatFactor).
    /// Input: [rows, numKvHeads*headDim], Output: [rows, numQHeads*headDim]
    /// </summary>
    private Tensor RepeatKVHeads(Tensor input, int headDim, int repeatFactor, string? resultName = null)
    {
        if (input.Shape.Rank != 2)
            throw new ArgumentException("RepeatKVHeads requires 2D tensor");

        int rows    = input.Shape[0];
        int srcCols = input.Shape[1];
        int dstCols = srcCols * repeatFactor;

        var result = Tensor.Create(_device, TensorShape.Matrix(rows, dstCols), DataType.F32, resultName);

        var paramsBuffer = _device.CreateBuffer(16, BufferType.Storage, DataType.I32);
        var p = new byte[16];
        BitConverter.GetBytes((uint)rows).CopyTo(p, 0);
        BitConverter.GetBytes((uint)srcCols).CopyTo(p, 4);
        BitConverter.GetBytes((uint)headDim).CopyTo(p, 8);
        BitConverter.GetBytes((uint)repeatFactor).CopyTo(p, 12);
        paramsBuffer.Write(p);

        var kernel = GetOrCreateKernel("repeat_kv_heads", () => ComputeShaders.RepeatKVHeads);
        kernel.SetArgument(0, input.Buffer);
        kernel.SetArgument(1, result.Buffer);
        kernel.SetArgument(2, paramsBuffer);

        uint total = (uint)(rows * dstCols);
        _queue.Dispatch(kernel, new[] { (total + 255u) / 256u }, null);
        MaybeFlush();

        Defer(paramsBuffer);
        return result;
    }

    /// <summary>
    /// Repeat columns of a tensor by a factor (for GQA K/V expansion)
    /// Input: [rows, cols], Output: [rows, cols * repeatFactor]
    /// Each column is repeated repeatFactor times sequentially
    /// </summary>
    private Tensor RepeatColumns(Tensor input, int repeatFactor, string? resultName = null)
    {
        if (input.Shape.Rank != 2)
            throw new ArgumentException("RepeatColumns requires 2D tensor");

        int rows = input.Shape[0];
        int cols = input.Shape[1];
        int newCols = cols * repeatFactor;

        var result = Tensor.Create(_device, TensorShape.Matrix(rows, newCols), DataType.F32, resultName);

        var paramsBuffer = _device.CreateBuffer(12, BufferType.Storage, DataType.I32);
        var p = new byte[12];
        BitConverter.GetBytes((uint)rows).CopyTo(p, 0);
        BitConverter.GetBytes((uint)cols).CopyTo(p, 4);
        BitConverter.GetBytes((uint)repeatFactor).CopyTo(p, 8);
        paramsBuffer.Write(p);

        var kernel = GetOrCreateKernel("repeat_columns", () => ComputeShaders.RepeatColumns);
        kernel.SetArgument(0, input.Buffer);
        kernel.SetArgument(1, result.Buffer);
        kernel.SetArgument(2, paramsBuffer);

        uint total = (uint)(rows * newCols);
        _queue.Dispatch(kernel, new[] { (total + 255u) / 256u }, null);
        MaybeFlush();

        Defer(paramsBuffer);
        return result;
    }

    /// <summary>
    /// Feed-Forward Network: FFN(x) = W_down @ (SiLU(W_gate @ x) ⊙ W_up @ x)
    /// x: [seq_len × d_model]
    /// W_gate, W_up: [d_model × d_ff]
    /// W_down: [d_ff × d_model]
    /// </summary>
    public Tensor FeedForward(Tensor x, Tensor wGate, Tensor wUp, Tensor wDown, string? resultName = null)
    {
        if (x.Shape.Rank != 2)
            throw new ArgumentException("Input must be 2D tensor");

        // 1. Gate projection: gate = x @ W_gate  (weights are GGUF column-major)
        var gate = MatMulWeights(x, wGate, "ffn_gate");

        // 2. SiLU activation: gate = SiLU(gate)
        SiLU(gate);

        // 3. Up projection: up = x @ W_up
        var up = MatMulWeights(x, wUp, "ffn_up");

        // 4. Element-wise multiply: gate_up = gate ⊙ up
        var gateUp = Multiply(gate, up, "ffn_gate_up");
        Defer(gate);
        Defer(up);

        // 5. Down projection: output = gate_up @ W_down
        var output = MatMulWeights(gateUp, wDown, resultName ?? "ffn_output");
        Defer(gateUp);

        return output;
    }

    #endregion

    #region Transformer Layers

    /// <summary>
    /// Single transformer layer: x → Norm → Attention → Add → Norm → FFN → Add
    /// </summary>
    public Tensor TransformerLayer(
        Tensor x,
        Tensor attnNormWeight,
        Tensor wQ, Tensor wK, Tensor wV, Tensor wAttnOut,
        Tensor ffnNormWeight,
        Tensor wGate, Tensor wUp, Tensor wDown,
        int numHeads,
        uint position,
        KVCache? kvCache = null,
        int layerIdx = 0,
        float eps = 1e-5f,
        string? resultName = null)
    {
        int seqLen = x.Shape[0];
        int dModel = x.Shape[1];
        int headDim = dModel / numHeads;

        // 1. Attention block
        // 1.1 Pre-normalization (GPU copy — no CPU roundtrip)
        var xNorm = Clone(x, "attn_norm_input");
        LayerNorm(xNorm, attnNormWeight, eps);

        // 1.2 Q, K, V projections (GGUF column-major weights)
        var Q = MatMulWeights(xNorm, wQ, "Q");
        var K = MatMulWeights(xNorm, wK, "K");
        var V = MatMulWeights(xNorm, wV, "V");
        Defer(xNorm);

        // 1.3 RoPE — apply rotary embeddings to Q and K before attention
        int numKVHeads = K.Shape[1] / headDim;

        ApplyRoPEFull(Q, position, numHeads, headDim);
        ApplyRoPEFull(K, position, numKVHeads, headDim);

#if DEEP_DEBUG
        _logger.LogDebug($"  [TransLayer] Q={Q.Shape} K={K.Shape} V={V.Shape}");
#endif

        // 1.3 Multi-head attention with KV-cache
        Tensor attnOut;
        if (kvCache != null)
        {
#if DEEP_DEBUG
            _logger.LogDebug($"  [KVCache] layer={layerIdx} K={K.Shape} V={V.Shape}");
#endif
            kvCache.Add(layerIdx, K, V);

            var (cachedK, cachedV) = kvCache.Get(layerIdx);
            if (cachedK == null || cachedV == null)
                throw new InvalidOperationException($"Failed to retrieve cached K,V for layer {layerIdx}");
#if DEEP_DEBUG
            _logger.LogDebug($"  [KVCache] cached K={cachedK.Shape} V={cachedV.Shape} Q={Q.Shape}");
#endif
            attnOut = MultiHeadAttention(Q, cachedK, cachedV, numHeads, position, "attn_out");
            Defer(Q);
            // Don't dispose K, V - they're owned by cache
        }
        else
        {
            attnOut = MultiHeadAttention(Q, K, V, numHeads, position, "attn_out");
            Defer(Q);
            Defer(K);
            Defer(V);
        }

        // 1.4 Output projection (GGUF column-major weight)
        var attnProj = MatMulWeights(attnOut, wAttnOut, "attn_proj");
        Defer(attnOut);

        // 1.5 Residual connection
        var x1 = Add(x, attnProj, "x_after_attn");

        // Single-shot diagnostic: print attnProj for layer 0, seqLen<=3 only once
        if (_dbgLayer0 && layerIdx == 0 && seqLen <= 3)
        {
            var ap = attnProj.ReadData();
            var xraw = x.ReadData(); int lastRow = seqLen - 1;
            _logger.LogTrace("[L0] x[last,0..2]=[{X0:F5},{X1:F5},{X2:F5}]", xraw[lastRow*dModel], xraw[lastRow*dModel+1], xraw[lastRow*dModel+2]);
            _logger.LogTrace("[L0] attnProj[last,0..2]=[{A0:F5},{A1:F5},{A2:F5}] maxAbs={MaxAbs:F4}", ap[lastRow*dModel], ap[lastRow*dModel+1], ap[lastRow*dModel+2], ap.Max(Math.Abs));
        }

        Defer(attnProj);

        // 2. FFN block
        var x1Norm = Clone(x1, "ffn_norm_input");
        LayerNorm(x1Norm, ffnNormWeight, eps);

        var ffnOut = FeedForward(x1Norm, wGate, wUp, wDown, "ffn_out");
        Defer(x1Norm);

        // 2.3 Residual connection
        var output = Add(x1, ffnOut, resultName ?? "layer_output");

        if (_dbgLayer0 && layerIdx == 0 && seqLen <= 3)
        {
            var ff = ffnOut.ReadData(); int lr = seqLen - 1;
            _logger.LogTrace("[L0] ffnOut[last,0..2]=[{F0:F5},{F1:F5},{F2:F5}] maxAbs={MaxAbs:F4}", ff[lr*dModel], ff[lr*dModel+1], ff[lr*dModel+2], ff.Max(Math.Abs));
            var outp = output.ReadData();
            _logger.LogTrace("[L0] output[last,0..2]=[{O0:F5},{O1:F5},{O2:F5}] maxAbs={MaxAbs:F4}", outp[lr*dModel], outp[lr*dModel+1], outp[lr*dModel+2], outp.Max(Math.Abs));
            _dbgLayer0 = false;
        }
        Defer(x1);
        Defer(ffnOut);

        return output;
    }

    #endregion

    #region Kernel Management

    private IComputeKernel GetOrCreateKernel(string name, Func<string> source)
    {
        if (_kernelCache.TryGetValue(name, out var cached))
            return cached;

        var kernel = _device.CreateKernel(source(), "main");
        kernel.Compile();
        _kernelCache[name] = kernel;
        
        return kernel;
    }

    #endregion

    #region Embedding

    // Above this size (F32 bytes) dequantizing the whole embedding table is unsafe on
    // memory-limited GPUs. Use per-row extraction instead.
    private const long EmbeddingF32SizeThreshold = 512L * 1024 * 1024; // 512 MB

    // Max seqLen for CPU-side dequantization in EmbeddingLookupFromQuantized.
    // For sequences longer than this, GPU dequantization is used (with chunking).
    // CPU dequant avoids GPU staging copy + dispatch cycles that can trigger
    // ErrorDeviceLost on AMD RADV when the GPU is under memory pressure.
    private const int EmbeddingCpuDequantMaxSeqLen = 256;

    /// <summary>
    /// Embedding lookup that dequantizes only the rows needed for the current token IDs.
    /// For small embedding tables falls back to the standard full-dequant path.
    /// For large tables (e.g. 248K-vocab 27B models, 5 GB F32) reads only
    /// seqLen rows from the quantized buffer on CPU, concatenates them into a
    /// small scratch tensor, dequantizes that on GPU, and returns the result.
    ///
    /// CRITICAL FIX for AMD RADV: For small sequences (seqLen <= EmbeddingCpuDequantMaxSeqLen),
    /// performs full CPU-side dequantization to avoid GPU staging copy + dispatch cycles
    /// that can trigger ErrorDeviceLost (vkQueueSubmit failure).
    /// </summary>
    public Tensor EmbeddingLookupFromQuantized(int[] tokenIds, Tensor quantizedTable,
                                                string? resultName = null)
    {
        int dModel    = quantizedTable.Shape[0]; // ne[0] = embedding dim (innermost)
        int vocabSize = quantizedTable.Shape[1]; // ne[1] = vocab size

        long f32Size  = (long)dModel * vocabSize * sizeof(float);
        if (f32Size <= EmbeddingF32SizeThreshold)
        {
            // Small enough — dequantize full table, then do GPU lookup as usual
            var fullF32 = Dequantize(quantizedTable);
            var res     = EmbeddingLookup(tokenIds, fullF32, resultName);
            fullF32.Dispose();
            return res;
        }

        // Large embedding table: extract just the needed rows on CPU.
        int  seqLen      = tokenIds.Length;
        long bytesPerRow  = (long)quantizedTable.Buffer.Size / vocabSize;

        // Read one quantized row per token and pack them into a contiguous small buffer
        var packedBytes = new byte[seqLen * bytesPerRow];
        for (int i = 0; i < seqLen; i++)
        {
            int    tokenId    = tokenIds[i];
            ulong  byteOffset = (ulong)(tokenId * bytesPerRow);
            byte[] rowBytes   = quantizedTable.Buffer.ReadRange<byte>(byteOffset, (int)bytesPerRow);
            Buffer.BlockCopy(rowBytes, 0, packedBytes, (int)(i * bytesPerRow), (int)bytesPerRow);
        }

        // CRITICAL FIX: For small sequences, dequantize on CPU to avoid GPU staging
        // copy + dispatch cycles that trigger ErrorDeviceLost on AMD RADV.
        // The GPU dequant path (below) creates a temporary GPU buffer, uploads packed
        // quantized data, then dispatches a dequant shader — this sequence of
        // staging-copy → dispatch can cause vkQueueSubmit to fail with ErrorDeviceLost
        // when the GPU is under memory pressure from the large embedding table.
        if (seqLen <= EmbeddingCpuDequantMaxSeqLen)
        {
            return EmbeddingLookupFromQuantizedCpu(tokenIds, quantizedTable, packedBytes,
                                                    seqLen, dModel, bytesPerRow, resultName);
        }

        // Upload packed rows as a small quantized tensor shaped [seqLen, dModel].
        // The data layout is: row_token0 | row_token1 | ... so after dequantization
        // the result is exactly [seqLen, dModel] in row-major order.
        var smallBuf = _device.CreateBuffer((ulong)packedBytes.Length, BufferType.Storage, DataType.I8);
        smallBuf.Write(packedBytes);
        var smallQuantized = new Tensor(smallBuf, TensorShape.Matrix(seqLen, dModel),
                                         quantizedTable.DataType, "emb_rows");

        // Dequantize to F32; the resulting layout is [seqLen * dModel] contiguous,
        // which equals [seqLen, dModel] row-major — exactly the embedding tensor we need.
        var smallF32 = Dequantize(smallQuantized, resultName);
        smallQuantized.Dispose();
        return smallF32;
    }

    /// <summary>
    /// CPU-side dequantization of packed quantized embedding rows.
    /// Avoids GPU staging copies and dispatches that can trigger ErrorDeviceLost on AMD RADV.
    /// Implements all K-quant formats (Q2_K through Q6_K) inline in C#.
    /// </summary>
    private Tensor EmbeddingLookupFromQuantizedCpu(int[] tokenIds, Tensor quantizedTable,
        byte[] packedBytes, int seqLen, int dModel, long bytesPerRow, string? resultName)
    {
        var resultData = new float[seqLen * dModel];

        for (int i = 0; i < seqLen; i++)
        {
            int rowOffset = (int)(i * bytesPerRow);
            int destOffset = i * dModel;
            DequantizeRowCpu(quantizedTable.DataType, packedBytes, rowOffset, resultData, destOffset, dModel);
        }

        return Tensor.FromData(_device, resultData,
            new TensorShape(new[] { seqLen, dModel }), resultName ?? "embeddings");
    }

    /// <summary>CPU dequantization of one K-quant row (Q2_K..Q6_K, block size 256).</summary>
    public static void DequantizeRowCpu(DataType dtype, byte[] src, int srcOff,
        float[] dst, int dstOff, int dModel)
    {
        switch (dtype)
        {
            case DataType.Q2_K: DequantizeRowQ2K(src, srcOff, dst, dstOff, dModel); break;
            case DataType.Q3_K: DequantizeRowQ3K(src, srcOff, dst, dstOff, dModel); break;
            case DataType.Q4_K: DequantizeRowQ4K(src, srcOff, dst, dstOff, dModel); break;
            case DataType.Q5_K: DequantizeRowQ5K(src, srcOff, dst, dstOff, dModel); break;
            case DataType.Q6_K: DequantizeRowQ6K(src, srcOff, dst, dstOff, dModel); break;
            default:
                throw new NotSupportedException($"CPU dequant not supported for {dtype}");
        }
    }

    // ── K-quant CPU dequant helpers ──────────────────────────────────────
    // Block size = 256 elements for all K-quant formats.
    // Layouts match the GLSL shaders in QuantizationFormats.cs.

    // FIX: Subnormal F16 → F32: correct exponent is shift+103 (not 1-shift+127)
    //   F16 subnormal = (-1)^s * 2^(-14) * m/1024
    //   Normalize: shift m left until bit 10 (0x400) is set, count k = 10-shift
    //   exp = -14 - k + 127 = -14 - (10-shift) + 127 = shift + 103
    private static float F16ToF32(ushort h)
    {
        uint sign = (uint)((h >> 15) & 1);
        uint exp = (uint)((h >> 10) & 0x1F);
        uint mant = (uint)(h & 0x3FF);
        if (exp == 0)
        {
            if (mant == 0) return sign == 0 ? 0f : -0f;
            int shift = 10;
            while ((mant & 0x400) == 0) { mant <<= 1; shift--; }
            exp = (uint)(shift + 103);  // FIX: was "1 - shift + 127"
            mant = (mant & 0x3FF) << 13;
        }
        else if (exp == 0x1F)
        {
            exp = 0xFF;
            mant <<= 13;
        }
        else
        {
            exp = exp - 15 + 127;
            mant <<= 13;
        }
        uint bits = (sign << 31) | (exp << 23) | (mant & 0x7FFFFF);
        return BitConverter.Int32BitsToSingle((int)bits);
    }

    /// <summary>Q2_K: 256 elements, 84 bytes/block.</summary>
    private static void DequantizeRowQ2K(byte[] src, int srcOff, float[] dst, int dstOff, int dModel)
    {
        const int QK_K = 256;
        const int BLOCK_BYTES = 84;
        int numBlocks = (dModel + QK_K - 1) / QK_K;

        for (int b = 0; b < numBlocks; b++)
        {
            int blockOff = srcOff + b * BLOCK_BYTES;
            int elemBase = b * QK_K;
            int count = Math.Min(QK_K, dModel - elemBase);

            // scales[16] at offset 0, lower nibble = scale, upper nibble = min
            // d at offset 80 (f16), dmin at offset 82 (f16)
            ushort dRaw  = (ushort)(src[blockOff + 80] | (src[blockOff + 81] << 8));
            ushort dmRaw = (ushort)(src[blockOff + 82] | (src[blockOff + 83] << 8));
            float d    = F16ToF32(dRaw);
            float dmin = F16ToF32(dmRaw);

            // qs[64] at offset 16
            for (int e = 0; e < count; e++)
            {
                int sub = e / 16;
                byte scByte = src[blockOff + sub];
                float sc_lo = (scByte & 0x0F);
                float sc_hi = (scByte >> 4) & 0x0F;

                int chunk = e / 128;
                int local = e % 128;
                int qsIdx = chunk * 32 + (local % 32);
                int shift = (local / 32) * 2;
                int qv = (src[blockOff + 16 + qsIdx] >> shift) & 0x03;

                dst[dstOff + elemBase + e] = d * sc_lo * qv - dmin * sc_hi;
            }
        }
    }

    /// <summary>Q3_K: 256 elements, 110 bytes/block.</summary>
    private static void DequantizeRowQ3K(byte[] src, int srcOff, float[] dst, int dstOff, int dModel)
    {
        const int QK_K = 256;
        const int BLOCK_BYTES = 110;
        int numBlocks = (dModel + QK_K - 1) / QK_K;

        for (int b = 0; b < numBlocks; b++)
        {
            int blockOff = srcOff + b * BLOCK_BYTES;
            int elemBase = b * QK_K;
            int count = Math.Min(QK_K, dModel - elemBase);

            ushort dRaw = (ushort)(src[blockOff + 108] | (src[blockOff + 109] << 8));
            float d = F16ToF32(dRaw);

            for (int e = 0; e < count; e++)
            {
                int chunk = e / 128;
                int local = e % 128;
                int qsIdx = chunk * 32 + (local % 32);
                int shift = (local / 32) * 2;

                int low2 = (src[blockOff + 32 + qsIdx] >> shift) & 3;

                int hmIdx = local % 32;
                int hmBit = chunk * 4 + (local / 32);
                int high1 = (src[blockOff + hmIdx] >> hmBit) & 1;

                int qval = (low2 | (high1 << 2)) - 4;

                // scales[12] at offset 96
                int is_sc = e / 16;
                int k = is_sc % 4;
                int ag = is_sc / 4;
                int scOff = blockOff + 96;
                int tmpB = src[scOff + 8 + k];
                int rk;
                int scaleByte;
                if (ag == 0)
                {
                    rk = src[scOff + k];
                    scaleByte = (rk & 0x0F) | ((tmpB >> 0) & 3) << 4;
                }
                else if (ag == 1)
                {
                    rk = src[scOff + k + 4];
                    scaleByte = (rk & 0x0F) | ((tmpB >> 2) & 3) << 4;
                }
                else if (ag == 2)
                {
                    rk = src[scOff + k];
                    scaleByte = ((rk >> 4) & 0x0F) | ((tmpB >> 4) & 3) << 4;
                }
                else
                {
                    rk = src[scOff + k + 4];
                    scaleByte = ((rk >> 4) & 0x0F) | ((tmpB >> 6) & 3) << 4;
                }

                float dl = d * (scaleByte - 32);
                dst[dstOff + elemBase + e] = dl * qval;
            }
        }
    }

    /// <summary>Q4_K: 256 elements, 144 bytes/block.</summary>
    private static void DequantizeRowQ4K(byte[] src, int srcOff, float[] dst, int dstOff, int dModel)
    {
        const int QK_K = 256;
        const int BLOCK_BYTES = 144;
        int numBlocks = (dModel + QK_K - 1) / QK_K;

        for (int b = 0; b < numBlocks; b++)
        {
            int blockOff = srcOff + b * BLOCK_BYTES;
            int elemBase = b * QK_K;
            int count = Math.Min(QK_K, dModel - elemBase);

            ushort dRaw  = (ushort)(src[blockOff + 0] | (src[blockOff + 1] << 8));
            ushort dmRaw = (ushort)(src[blockOff + 2] | (src[blockOff + 3] << 8));
            float d    = F16ToF32(dRaw);
            float dmin = F16ToF32(dmRaw);

            // scales[12] at offset 4, qs[128] at offset 16
            for (int e = 0; e < count; e++)
            {
                int group = e / 64;
                int withinGroup = e % 64;
                int isUpper = (withinGroup >= 32) ? 1 : 0;
                int wgLower = withinGroup % 32;
                int scaleIndex = group * 2 + isUpper;

                // get_scale_min_k4
                float sc, mn;
                int scOff = blockOff + 4;
                if (scaleIndex < 4)
                {
                    sc = (src[scOff + scaleIndex] & 63);
                    mn = (src[scOff + scaleIndex + 4] & 63);
                }
                else
                {
                    int j = scaleIndex;
                    sc = ((src[scOff + j + 4] & 0x0F) | ((src[scOff + j - 4] >> 6) << 4));
                    mn = ((src[scOff + j + 4] >> 4) | ((src[scOff + j] >> 6) << 4));
                }

                int qsOff = blockOff + 16 + group * 32 + wgLower;
                int qsByte = src[qsOff];
                int q = (qsByte >> (isUpper * 4)) & 0x0F;

                dst[dstOff + elemBase + e] = d * sc * q - dmin * mn;
            }
        }
    }

    /// <summary>Q5_K: 256 elements, 176 bytes/block.</summary>
    private static void DequantizeRowQ5K(byte[] src, int srcOff, float[] dst, int dstOff, int dModel)
    {
        const int QK_K = 256;
        const int BLOCK_BYTES = 176;
        int numBlocks = (dModel + QK_K - 1) / QK_K;

        for (int b = 0; b < numBlocks; b++)
        {
            int blockOff = srcOff + b * BLOCK_BYTES;
            int elemBase = b * QK_K;
            int count = Math.Min(QK_K, dModel - elemBase);

            ushort dRaw  = (ushort)(src[blockOff + 0] | (src[blockOff + 1] << 8));
            ushort dmRaw = (ushort)(src[blockOff + 2] | (src[blockOff + 3] << 8));
            float d    = F16ToF32(dRaw);
            float dmin = F16ToF32(dmRaw);

            for (int e = 0; e < count; e++)
            {
                int sub = e / 32;
                float sc, mn;
                int scOff = blockOff + 4;
                if (sub < 4)
                {
                    sc = (src[scOff + sub] & 63);
                    mn = (src[scOff + sub + 4] & 63);
                }
                else
                {
                    int j = sub;
                    sc = ((src[scOff + j + 4] & 0x0F) | ((src[scOff + j - 4] >> 6) << 4));
                    mn = ((src[scOff + j + 4] >> 4) | ((src[scOff + j] >> 6) << 4));
                }

                int j5 = e / 64;
                int local = e % 64;
                int isUpper = (local >= 32) ? 1 : 0;
                int l = local % 32;
                int qlIdx = j5 * 32 + l;

                // Lower 4 bits from qs[128] at offset 48
                int ql = (src[blockOff + 48 + qlIdx] >> (isUpper * 4)) & 0x0F;

                // FIX: qlIdx ranges 0..127 across all j5 groups, but qh[] has only 32 bytes.
                //      Use l (local position 0..31) instead of qlIdx for qh indexing.
                int qhBit = j5 * 2 + isUpper;
                int qh = (src[blockOff + 16 + l] >> qhBit) & 1;

                int q = ql | (qh << 4);

                dst[dstOff + elemBase + e] = d * sc * q - dmin * mn;
            }
        }
    }

    /// <summary>Q6_K: 256 elements, 210 bytes/block.</summary>
    private static void DequantizeRowQ6K(byte[] src, int srcOff, float[] dst, int dstOff, int dModel)
    {
        const int QK_K = 256;
        const int BLOCK_BYTES = 210;
        int numBlocks = (dModel + QK_K - 1) / QK_K;

        for (int b = 0; b < numBlocks; b++)
        {
            int blockOff = srcOff + b * BLOCK_BYTES;
            int elemBase = b * QK_K;
            int count = Math.Min(QK_K, dModel - elemBase);

            ushort dRaw = (ushort)(src[blockOff + 208] | (src[blockOff + 209] << 8));
            float d = F16ToF32(dRaw);

            for (int e = 0; e < count; e++)
            {
                int blk_h = e / 128;
                int w = e % 128;
                int quarter = w / 32;
                int wq = w % 32;

                int qlExtra = ((quarter == 1 || quarter == 3) ? 32 : 0);
                int qlOff = blockOff + blk_h * 64 + qlExtra + wq;
                int isUpper = (quarter == 2 || quarter == 3) ? 1 : 0;
                int lower4 = (src[qlOff] >> (isUpper * 4)) & 0x0F;

                int qhOff = blockOff + 128 + blk_h * 32 + wq;
                int qhShift = quarter * 2;
                int upper2 = (src[qhOff] >> qhShift) & 0x03;

                int q = lower4 | (upper2 << 4);

                int sc = (sbyte)src[blockOff + 192 + (e / 16)];

                dst[dstOff + elemBase + e] = d * sc * (q - 32);
            }
        }
    }

    /// <summary>
    /// GPU embedding lookup: select rows from table by token IDs.
    /// table: [vocabSize × dModel], tokenIds: [seqLen] → result: [seqLen × dModel]
    /// </summary>
    public Tensor EmbeddingLookup(int[] tokenIds, Tensor table, string? resultName = null)
    {
        int seqLen = tokenIds.Length;
        int dModel = table.Shape[0]; // GGUF: ne[0] is innermost dim (embedding_dim), ne[1] is vocab_size

        var result = Tensor.Create(_device, TensorShape.Matrix(seqLen, dModel), DataType.F32, resultName);

        var tokenBuf = _device.CreateBuffer((ulong)(seqLen * sizeof(int)), BufferType.Storage, DataType.I32);
        tokenBuf.Write(tokenIds);

        uint[] paramsData = { (uint)seqLen, (uint)dModel };
        var paramsBuffer = _device.CreateBuffer((ulong)(paramsData.Length * sizeof(uint)), BufferType.Storage, DataType.I32);
        paramsBuffer.Write(paramsData);

        var kernel = GetOrCreateKernel("embedding_lookup", () => ComputeShaders.EmbeddingLookup);
        kernel.SetArgument(0, tokenBuf);
        kernel.SetArgument(1, table.Buffer);
        kernel.SetArgument(2, result.Buffer);
        kernel.SetArgument(3, paramsBuffer);

        uint total = (uint)(seqLen * dModel);
        _queue.Dispatch(kernel, new[] { (total + 255u) / 256u }, null);
        MaybeFlush();

        tokenBuf.Dispose();
        Defer(paramsBuffer);

        return result;
    }

    #endregion

    #region SSM/GDN Operations

    /// <summary>
    /// Gated Delta Net (SSM) decode for ONE token.
    /// Two dispatches with a memory barrier between them:
    ///   Part 1 (ssm_gdn_decode): computes z, beta, alpha, gate, qkv_mixed, conv1d, SiLU → scratch
    ///   Part 2 (ssm_gdn_recur):  reads scratch, L2 norm, Delta Net, Gated norm → output
    ///
    /// Dispatch: 48 workgroups × 128 threads for both parts.
    ///
    /// Input buffers:
    ///   xNorm [DMODEL] — normalized input for this token
    ///   convW [CONV_KERNEL * CONV_DIM] — conv1d weights
    ///   convState [CONV_STATE_LEN * CONV_DIM] — sliding window (in/out)
    ///   wQKV [DMODEL * CONV_DIM] — attn_qkv.weight
    ///   wZ [DMODEL * VALUE_DIM] — attn_gate.weight
    ///   wBeta [DMODEL * N_V_HEADS] — ssm_beta.weight
    ///   wAlpha [DMODEL * N_V_HEADS] — ssm_alpha.weight
    ///   dtBias [N_V_HEADS] — ssm_dt.bias
    ///   ssA [N_V_HEADS] — ssm_a (A_NOSCAN)
    ///   scratch [CONV_DIM + VALUE_DIM + N_V_HEADS*3] — scratch buffer (in/out)
    ///   ssmState [HEAD_V_DIM * HEAD_V_DIM * N_V_HEADS] — recurrent state (in/out)
    ///   ssmNorm [HEAD_V_DIM] — per-group RMSNorm weight
    ///
    /// Output:
    ///   output [VALUE_DIM] — SSM output for this token
    /// </summary>
    public void SsmGdnDecode(
        Tensor xNorm,
        Tensor convW,
        Tensor convState,
        Tensor wQKV,
        Tensor wZ,
        Tensor wBeta,
        Tensor wAlpha,
        Tensor dtBias,
        Tensor ssA,
        Tensor scratch,
        Tensor ssmState,
        Tensor ssmNorm,
        Tensor output,
        uint rowIndex = 0u,
        uint ssmDModel = 5120, // d_model
        uint ssmHVD = 128,     // head_v_dim
        uint ssmNVH = 48,      // n_v_heads
        uint ssmNKH = 16,      // n_k_heads
        uint ssmKD  = 2048,    // key_dim
        uint ssmVD  = 6144,    // value_dim
        uint ssmCD  = 10240,   // conv_dim
        uint debugLayer = uint.MaxValue)   // if 0/1/2 → trace scratch+state+output
    {
        bool dbg = debugLayer <= 2;
        uint convGroups = ssmCD / ssmHVD; // 10240/128 = 80
        uint vGroups    = ssmNVH;          // 48

        // OPTIMIZATION: Lazy-init persistent SSM UBO — eliminates 2× CreateBuffer(32) per token.
        // ssmParams are model-level constants — written once, reused for all layers/all tokens.
        if (_persistentSsmParams == null)
        {
            _persistentSsmParams = _device.CreateBuffer(32, BufferType.Storage, DataType.I32);
            _persistentSsmParams.Write(new[] { ssmDModel, ssmHVD, ssmNVH, ssmNKH, ssmKD, ssmVD, ssmCD, 0u });
        }
        // OPTIMIZATION: Persistent rowIndex buffer — written per token, reused instead of create+defer.
        if (_persistentRowIndex == null)
            _persistentRowIndex = _device.CreateBuffer(sizeof(uint), BufferType.Storage, DataType.I32);
        _persistentRowIndex.Write(new[] { rowIndex });

        var _tsDecode = GlobalProfiler.Start();

        // ── Part 1: conv1d ───────────────────────────────────────────────────
        var kernelDecode = GetOrCreateKernel("ssm_gdn_decode", () => ComputeShaders.SsmGdnDecode);
        kernelDecode.SetArgument(0, xNorm.Buffer);
        kernelDecode.SetArgument(1, convW.Buffer);
        kernelDecode.SetArgument(2, convState.Buffer);
        kernelDecode.SetArgument(3, wQKV.Buffer);
        kernelDecode.SetArgument(4, wZ.Buffer);
        kernelDecode.SetArgument(5, wBeta.Buffer);
        kernelDecode.SetArgument(6, wAlpha.Buffer);
        kernelDecode.SetArgument(7, dtBias.Buffer);
        kernelDecode.SetArgument(8, ssA.Buffer);
        kernelDecode.SetArgument(9, scratch.Buffer);
        kernelDecode.SetArgument(10, _persistentRowIndex);
        kernelDecode.SetArgument(11, _persistentSsmParams);

        // convGroups workgroups × 128 threads.
        _queue.Dispatch(kernelDecode, new[] { convGroups }, null);

        GlobalProfiler.End(_tsDecode, "SSM_GPU.Decode");

        // Memory barrier between part 1 and part 2
        _queue.InsertMemoryBarrier();

        var _tsRecur = GlobalProfiler.Start();

        // ── Part 2: recurrence ───────────────────────────────────────────────
        var kernelRecur = GetOrCreateKernel("ssm_gdn_recur", () => ComputeShaders.SsmGdnRecur);
        kernelRecur.SetArgument(0, scratch.Buffer);
        kernelRecur.SetArgument(1, ssmState.Buffer);
        kernelRecur.SetArgument(2, ssmNorm.Buffer);
        kernelRecur.SetArgument(3, output.Buffer);
        kernelRecur.SetArgument(4, _persistentSsmParams);

        // vGroups workgroups × 128 threads
        _queue.Dispatch(kernelRecur, new[] { vGroups }, null);

        GlobalProfiler.End(_tsRecur, "SSM_GPU.Recur");

        // Barrier after recur: ensures recur_N's writes (ssmState, output) are complete and visible
        // before the next decode_N+1 can start reading/writing scratch or ssmState.
        // Without this barrier, decode_{N+1} may race with recur_N over the shared scratch buffer
        // in batch mode (multiple tokens per GPU submit during prefill).
        _queue.InsertMemoryBarrier();

        MaybeFlush();

        // ── Debug tracing for layers 0-2 ────────────────────────────────────
        // MUST be AFTER MaybeFlush — GPU must finish before reading buffers.
        if (dbg && Formats.QwenDbgTrace.Once("ssm_phase", (int)debugLayer))
        {
            string p = $"L{debugLayer}";
            // scratch is arena-allocated and invalid after Flush → skip
            // state: [NVH][HVD][HVD] = [48*128*128] = 786432 floats, first 12 values of head 0
            Formats.QwenDbgTrace.Slice($"{p}_state_h0", ssmState, 0, 12);
            // output: [VD=6144], log first 8 and middle 8
            Formats.QwenDbgTrace.Slice($"{p}_out", output, 0, 8);
            Formats.QwenDbgTrace.Slice($"{p}_out_mid", output, (int)(ssmVD / 2), 8);
        }

        // OPTIMIZATION: Persistent buffers — no Defer needed (disposed in ComputeOps.Dispose).
    }

    #endregion

    #region Utility

    /// <summary>
    /// GPU-side tensor copy — avoids CPU roundtrip from ReadData/Write.
    /// </summary>
    public Tensor Clone(Tensor input, string? resultName = null)
    {
        _logger.LogDebug("[DBG_OPS] Clone start name={Name} shape={Shape} total={Total}",
            resultName ?? "?", input.Shape, input.Shape.TotalElements);

        var result = Tensor.Create(_device, input.Shape, DataType.F32, resultName);

        uint[] paramsData = { (uint)input.Shape.TotalElements };
        var paramsBuffer = _device.CreateBuffer(sizeof(uint), BufferType.Storage, DataType.I32);
        paramsBuffer.Write(paramsData);

        var kernel = GetOrCreateKernel("copy", () => ComputeShaders.Copy);
        kernel.SetArgument(0, input.Buffer);
        kernel.SetArgument(1, result.Buffer);
        kernel.SetArgument(2, paramsBuffer);

        uint total = (uint)input.Shape.TotalElements;
        _queue.Dispatch(kernel, new[] { (total + 255u) / 256u }, null);
        _logger.LogDebug("[DBG_OPS] Clone dispatch done, calling MaybeFlush");
        MaybeFlush();
        _logger.LogDebug("[DBG_OPS] Clone MaybeFlush done, deferring paramsBuffer");

        Defer(paramsBuffer);
        _logger.LogDebug("[DBG_OPS] Clone done name={Name}", resultName ?? "?");
        return result;
    }

    #endregion

    // ── Arena integration ─────────────────────────────────────────────────

    /// <summary>Attach an arena allocator (called once at model load).</summary>
    internal void AttachArena(VulkanArenaAllocator arena)
    {
        _arena = arena;
    }

    /// <summary>Begin inference frame — snapshot arena cursor for later Reset.</summary>
    internal void BeginArenaFrame()
    {
        _arena?.BeginFrame();
    }

    /// <summary>
    /// Allocate a temporary tensor from the arena. The tensor's buffer is an arena view
    /// (no vkAllocateMemory). Dispose is a no-op — arena reclaims memory on Reset().
    /// Returns null if arena is not attached (caller should fall back to Tensor.Create).
    /// </summary>
    internal Tensor? AllocTempTensor(TensorShape shape, string? name = null)
    {
        if (_arena == null) return null;

        ulong bytes = (ulong)shape.TotalElements * sizeof(float);
        var slice = _arena.Alloc(bytes);
        var buf = VulkanComputeBuffer.CreateArenaView(slice.Buffer, slice.Offset, slice.Size, DataType.F32);
        return new Tensor(buf, shape, DataType.F32, name);
    }

    public void Dispose()
    {
        if (_disposed) return;

        foreach (var kernel in _kernelCache.Values)
            kernel.Dispose();

        _kernelCache.Clear();

        // Dispose the persistent zero-offset buffer created in the constructor.
        // This buffer lives as long as ComputeOps itself to avoid the create/dispose
        // cycle that triggers GPUVM faults on AMD RADV.
        if (_persistentOffsetInitialized)
            _persistentZeroOffset.Dispose();

        // OPTIMIZATION: Dispose persistent SSM UBO buffers (created once, reused for all tokens).
        _persistentSsmParams?.Dispose();
        _persistentRowIndex?.Dispose();

        _queue.Dispose();
        _arena?.Dispose();
        _disposed = true;
    }

}







