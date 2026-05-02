using AIHost.GGUF;
using AIHost.ICompute;
using AIHost.ICompute.Vulkan;
using AIHost.Inference;
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

    public IComputeDevice Device => _device;

    public ComputeOps(IComputeDevice device)
    {
        _device = device;
        _queue = device.CreateCommandQueue();
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
        _queue.Flush();
        foreach (var d in _deferred) d.Dispose();
        _deferred.Clear();
        _batchMode = false;
        // Rewind each kernel's descriptor ring so next batch starts from slot 0.
        foreach (var kernel in _kernelCache.Values)
            if (kernel is VulkanComputeKernel vk)
                vk.ResetDispatchRing();
    }

    /// <summary>In batch mode: insert a compute barrier (no submit). Otherwise flush.</summary>
    private void MaybeFlush()
    {
        if (_batchMode)
            _queue.InsertMemoryBarrier();
        else
            _queue.Flush();
    }

    /// <summary>In batch mode: defer disposal until Flush(). Otherwise dispose now.</summary>
    private void Defer(IDisposable d)
    {
        if (_batchMode)
            _deferred.Add(d);
        else
            d.Dispose();
    }

    /// <summary>
    /// Publicly defer a tensor for disposal after the current batch flush.
    /// Use this from Transformer when tensors are passed into TransformerLayer
    /// and must stay alive until the layer's GPU work completes.
    /// </summary>
    public void DeferExternal(IDisposable d) => Defer(d);

    #region Matrix Operations

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

        var kernel = GetOrCreateKernel("matmul_weights_f32", ComputeShaders.MatMulWeightsF32);
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

        var kernel = GetOrCreateKernel("matmul_f32", ComputeShaders.MatMulF32);
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

        var kernel = GetOrCreateKernel("elemwise_mul", ComputeShaders.ElementWiseMul);
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

        var kernel = GetOrCreateKernel("elemwise_add", ComputeShaders.ElementWiseAdd);
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

            var kernel = GetOrCreateKernel("concat_axis0", ComputeShaders.ConcatAxis0);
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

            var kernel = GetOrCreateKernel("concat_axis1", ComputeShaders.ConcatAxis1);
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

        var kernel = GetOrCreateKernel("silu", ComputeShaders.SiLU);
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

        var kernel = GetOrCreateKernel("layer_norm", ComputeShaders.LayerNorm);
        kernel.SetArgument(0, tensor.Buffer);
        kernel.SetArgument(1, weight.Buffer);
        kernel.SetArgument(2, paramsBuffer);

        // One workgroup per row — each workgroup of 256 threads handles one token's vector
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

        var kernel = GetOrCreateKernel("softmax", ComputeShaders.Softmax);
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

        var kernel = GetOrCreateKernel("rope", ComputeShaders.RoPE);
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
    /// </summary>
    public Tensor DequantizeQ2K(Tensor quantized, string? resultName = null)
    {
        if (quantized.DataType != DataType.Q2_K)
            throw new ArgumentException("Input must be Q2_K tensor");

        var result = Tensor.Create(_device, quantized.Shape, DataType.F32, resultName);

        var kernel = GetOrCreateKernel("dequant_q2k", ComputeShaders.DequantizeQ2K);
        kernel.SetArgument(0, quantized.Buffer);
        kernel.SetArgument(1, result.Buffer);

        uint[] globalWorkSize = { (uint)((quantized.Shape.TotalElements + 255) / 256) };
        _queue.Dispatch(kernel, globalWorkSize, null);
        MaybeFlush();

        return result;
    }

    /// <summary>
    /// Деквантизировать тензор Q3_K → F32
    /// </summary>
    public Tensor DequantizeQ3K(Tensor quantized, string? resultName = null)
    {
        if (quantized.DataType != DataType.Q3_K)
            throw new ArgumentException("Input must be Q3_K tensor");

        var result = Tensor.Create(_device, quantized.Shape, DataType.F32, resultName);

        var kernel = GetOrCreateKernel("dequant_q3k", ComputeShaders.DequantizeQ3K);
        kernel.SetArgument(0, quantized.Buffer);
        kernel.SetArgument(1, result.Buffer);

        uint[] globalWorkSize = { (uint)((quantized.Shape.TotalElements + 255) / 256) };
        _queue.Dispatch(kernel, globalWorkSize, null);
        MaybeFlush();

        return result;
    }

    /// <summary>
    /// Деквантизировать тензор Q4_K → F32
    /// </summary>
    public Tensor DequantizeQ4K(Tensor quantized, string? resultName = null)
    {
        if (quantized.DataType != DataType.Q4_K)
            throw new ArgumentException("Input must be Q4_K tensor");

        var result = Tensor.Create(_device, quantized.Shape, DataType.F32, resultName);

        var kernel = GetOrCreateKernel("dequant_q4k", ComputeShaders.DequantizeQ4K);
        kernel.SetArgument(0, quantized.Buffer);
        kernel.SetArgument(1, result.Buffer);

        uint[] globalWorkSize = { (uint)((quantized.Shape.TotalElements + 255) / 256) };
        _queue.Dispatch(kernel, globalWorkSize, null);
        MaybeFlush();

        return result;
    }

    /// <summary>
    /// Деквантизировать тензор Q5_K → F32
    /// </summary>
    public Tensor DequantizeQ5K(Tensor quantized, string? resultName = null)
    {
        if (quantized.DataType != DataType.Q5_K)
            throw new ArgumentException("Input must be Q5_K tensor");

        var result = Tensor.Create(_device, quantized.Shape, DataType.F32, resultName);

        var kernel = GetOrCreateKernel("dequant_q5k", ComputeShaders.DequantizeQ5K);
        kernel.SetArgument(0, quantized.Buffer);
        kernel.SetArgument(1, result.Buffer);

        uint[] globalWorkSize = { (uint)((quantized.Shape.TotalElements + 255) / 256) };
        _queue.Dispatch(kernel, globalWorkSize, null);
        MaybeFlush();

        return result;
    }

    /// <summary>
    /// Деквантизировать тензор Q6_K → F32
    /// </summary>
    public Tensor DequantizeQ6K(Tensor quantized, string? resultName = null)
    {
        if (quantized.DataType != DataType.Q6_K)
            throw new ArgumentException("Input must be Q6_K tensor");

        var result = Tensor.Create(_device, quantized.Shape, DataType.F32, resultName);

        var kernel = GetOrCreateKernel("dequant_q6k", ComputeShaders.DequantizeQ6K);
        kernel.SetArgument(0, quantized.Buffer);
        kernel.SetArgument(1, result.Buffer);

        uint[] globalWorkSize = { (uint)((quantized.Shape.TotalElements + 255) / 256) };
        _queue.Dispatch(kernel, globalWorkSize, null);
        MaybeFlush();

        return result;
    }

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
    /// </summary>
    public void DequantizeInto(Tensor quantized, Tensor target)
    {
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

        string shaderSource = quantized.DataType switch
        {
            DataType.Q2_K => ComputeShaders.DequantizeQ2K,
            DataType.Q3_K => ComputeShaders.DequantizeQ3K,
            DataType.Q4_K => ComputeShaders.DequantizeQ4K,
            DataType.Q5_K => ComputeShaders.DequantizeQ5K,
            DataType.Q6_K => ComputeShaders.DequantizeQ6K,
            _ => throw new NotSupportedException()
        };

        var kernel = GetOrCreateKernel(kernelName, shaderSource);
        kernel.SetArgument(0, quantized.Buffer);
        kernel.SetArgument(1, target.Buffer);

        uint[] globalWorkSize = { (uint)((quantized.Shape.TotalElements + 255) / 256) };
        _queue.Dispatch(kernel, globalWorkSize, null);
        // No barrier here — independent weight dequants can run in parallel on GPU.
        // Caller must insert a barrier after ALL weight dequants are dispatched.
        if (!_batchMode) _queue.Flush();
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

        var kernel = GetOrCreateKernel("transpose", ComputeShaders.Transpose);
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

        var kernel = GetOrCreateKernel("rowwise_softmax", ComputeShaders.RowwiseSoftmax);
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

        var kernel = GetOrCreateKernel("scale", ComputeShaders.Scale);
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
    public void ApplyRoPEFull(Tensor tensor, uint startPosition, int numHeads, int headDim, float theta = 10000.0f)
    {
        int seqLen = tensor.Shape[0];
        var paramsBuffer = _device.CreateBuffer(20, BufferType.Storage, DataType.I32);
        var p = new byte[20];
        BitConverter.GetBytes((uint)seqLen).CopyTo(p, 0);
        BitConverter.GetBytes((uint)numHeads).CopyTo(p, 4);
        BitConverter.GetBytes((uint)headDim).CopyTo(p, 8);
        BitConverter.GetBytes(startPosition).CopyTo(p, 12);
        BitConverter.GetBytes(theta).CopyTo(p, 16);
        paramsBuffer.Write(p);

        var kernel = GetOrCreateKernel("rope_full", ComputeShaders.RoPEFull);
        kernel.SetArgument(0, tensor.Buffer);
        kernel.SetArgument(1, paramsBuffer);

        uint total = (uint)(seqLen * numHeads * headDim / 2);
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

        var kernel = GetOrCreateKernel("causal_mask", ComputeShaders.CausalMask);
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

        var kernel = GetOrCreateKernel("fused_mha_generate", ComputeShaders.FusedMHAGenerate);
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
        var kernel = GetOrCreateKernel("slice_cols", ComputeShaders.SliceCols);
        kernel.SetArgument(0, input.Buffer); kernel.SetArgument(1, result.Buffer); kernel.SetArgument(2, paramsBuffer);
        _queue.Dispatch(kernel, new[] { ((uint)(rows * colCount) + 255u) / 256u }, null);
        MaybeFlush(); Defer(paramsBuffer);
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
        var kernel = GetOrCreateKernel("scatter_cols", ComputeShaders.ScatterCols);
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

        var kernel = GetOrCreateKernel("repeat_kv_heads", ComputeShaders.RepeatKVHeads);
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

        var kernel = GetOrCreateKernel("repeat_columns", ComputeShaders.RepeatColumns);
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
        Console.WriteLine($"  [TransLayer] Q={Q.Shape} K={K.Shape} V={V.Shape}");
#endif

        // 1.3 Multi-head attention with KV-cache
        Tensor attnOut;
        if (kvCache != null)
        {
#if DEEP_DEBUG
            Console.WriteLine($"  [KVCache] layer={layerIdx} K={K.Shape} V={V.Shape}");
#endif
            kvCache.Add(layerIdx, K, V);

            var (cachedK, cachedV) = kvCache.Get(layerIdx);
            if (cachedK == null || cachedV == null)
                throw new InvalidOperationException($"Failed to retrieve cached K,V for layer {layerIdx}");
#if DEEP_DEBUG
            Console.WriteLine($"  [KVCache] cached K={cachedK.Shape} V={cachedV.Shape} Q={Q.Shape}");
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
            Console.WriteLine($"[L0] x[last,0..2]=[{xraw[lastRow*dModel]:F5},{xraw[lastRow*dModel+1]:F5},{xraw[lastRow*dModel+2]:F5}]");
            Console.WriteLine($"[L0] attnProj[last,0..2]=[{ap[lastRow*dModel]:F5},{ap[lastRow*dModel+1]:F5},{ap[lastRow*dModel+2]:F5}] maxAbs={ap.Max(Math.Abs):F4}");
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
            Console.WriteLine($"[L0] ffnOut[last,0..2]=[{ff[lr*dModel]:F5},{ff[lr*dModel+1]:F5},{ff[lr*dModel+2]:F5}] maxAbs={ff.Max(Math.Abs):F4}");
            var outp = output.ReadData();
            Console.WriteLine($"[L0] output[last,0..2]=[{outp[lr*dModel]:F5},{outp[lr*dModel+1]:F5},{outp[lr*dModel+2]:F5}] maxAbs={outp.Max(Math.Abs):F4}");
            _dbgLayer0 = false;
        }
        Defer(x1);
        Defer(ffnOut);

        return output;
    }

    #endregion

    #region Kernel Management

    private IComputeKernel GetOrCreateKernel(string name, string source)
    {
        if (_kernelCache.TryGetValue(name, out var cached))
            return cached;

        var kernel = _device.CreateKernel(source, "main");
        kernel.Compile();
        _kernelCache[name] = kernel;
        
        return kernel;
    }

    #endregion

    #region Embedding

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

        var kernel = GetOrCreateKernel("embedding_lookup", ComputeShaders.EmbeddingLookup);
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

    #region Utility

    /// <summary>
    /// GPU-side tensor copy — avoids CPU roundtrip from ReadData/Write.
    /// </summary>
    public Tensor Clone(Tensor input, string? resultName = null)
    {
        var result = Tensor.Create(_device, input.Shape, DataType.F32, resultName);

        uint[] paramsData = { (uint)input.Shape.TotalElements };
        var paramsBuffer = _device.CreateBuffer(sizeof(uint), BufferType.Storage, DataType.I32);
        paramsBuffer.Write(paramsData);

        var kernel = GetOrCreateKernel("copy", ComputeShaders.Copy);
        kernel.SetArgument(0, input.Buffer);
        kernel.SetArgument(1, result.Buffer);
        kernel.SetArgument(2, paramsBuffer);

        uint total = (uint)input.Shape.TotalElements;
        _queue.Dispatch(kernel, new[] { (total + 255u) / 256u }, null);
        MaybeFlush();

        Defer(paramsBuffer);
        return result;
    }

    #endregion

    public void Dispose()
    {
        if (_disposed) return;

        foreach (var kernel in _kernelCache.Values)
            kernel.Dispose();

        _kernelCache.Clear();
        _queue.Dispose();
        _disposed = true;
    }
}







