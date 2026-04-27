using AIHost.ICompute;

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

    public ComputeOps(IComputeDevice device)
    {
        _device = device;
        _queue = device.CreateCommandQueue();
    }

    #region Matrix Operations

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
        _queue.Flush();

        paramsBuffer.Dispose();

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
        _queue.Flush();

        paramsBuffer.Dispose();

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
        _queue.Flush();

        paramsBuffer.Dispose();

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
            _queue.Flush();

            paramsBuffer.Dispose();
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
            _queue.Flush();

            paramsBuffer.Dispose();
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
        _queue.Flush();

        paramsBuffer.Dispose();
    }

    #endregion

    #region Normalization

    /// <summary>
    /// Layer normalization: out = (x - mean(x)) / sqrt(var(x) + eps) * weight
    /// </summary>
    public void LayerNorm(Tensor tensor, Tensor weight, float eps = 1e-5f)
    {
        float[] paramsData = { (float)tensor.Shape.TotalElements, eps };
        var paramsBuffer = _device.CreateBuffer((ulong)(paramsData.Length * sizeof(float)), BufferType.Storage, DataType.F32);
        paramsBuffer.Write(paramsData);

        var kernel = GetOrCreateKernel("layer_norm", ComputeShaders.LayerNorm);
        kernel.SetArgument(0, tensor.Buffer);
        kernel.SetArgument(1, weight.Buffer);
        kernel.SetArgument(2, paramsBuffer);

        uint[] globalWorkSize = { (uint)((tensor.Shape.TotalElements + 255) / 256) };
        _queue.Dispatch(kernel, globalWorkSize, null);
        _queue.Flush();

        paramsBuffer.Dispose();
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
        _queue.Flush();

        paramsBuffer.Dispose();
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
        _queue.Flush();

        paramsBuffer.Dispose();
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
        _queue.Flush();

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
        _queue.Flush();

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
        _queue.Flush();

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
        _queue.Flush();

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
        _queue.Flush();

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
            DataType.F32 => quantized, // Уже F32
            _ => throw new NotSupportedException($"Dequantization for {quantized.DataType} not implemented")
        };
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
        _queue.Flush();

        paramsBuffer.Dispose();
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
        _queue.Flush();

        paramsBuffer.Dispose();
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
        _queue.Flush();

        paramsBuffer.Dispose();
    }

    /// <summary>
    /// Multi-head attention: Attention(Q, K, V) = Softmax(Q @ K^T / sqrt(d_k)) @ V
    /// Q, K, V: [seq_len × d_model]
    /// </summary>
    public Tensor MultiHeadAttention(Tensor Q, Tensor K, Tensor V, int numHeads, string? resultName = null)
    {
        Console.WriteLine($"  [MHA Debug] Q shape: {Q.Shape}, K shape: {K.Shape}, V shape: {V.Shape}");
        
        if (Q.Shape.Rank != 2 || K.Shape.Rank != 2 || V.Shape.Rank != 2)
            throw new ArgumentException("Q, K, V must be 2D tensors");

        int seqLen = Q.Shape[0];
        int dModel_Q = Q.Shape[1];
        int kvDim = K.Shape[1];
        
        // GQA (Grouped Query Attention): Q has more dimensions than K, V
        // This is used in models like TinyLlama for efficiency
        if (dModel_Q != kvDim)
        {
            // Calculate dimensions
            int headDimGQA = dModel_Q / numHeads;  // e.g., 2048 / 32 = 64
            int numKvHeads = kvDim / headDimGQA;    // e.g., 256 / 64 = 4
            int repeatFactor = numHeads / numKvHeads;  // e.g., 32 / 4 = 8
            
            // For GQA, we need to repeat K and V to match Q's dimensions
            // Each KV head is shared across multiple Q heads
            // Simplified approach: repeat K, V columns to match d_model
            var K_expanded = RepeatColumns(K, repeatFactor, "K_expanded");
            var V_expanded = RepeatColumns(V, repeatFactor, "V_expanded");
            
            // Now run standard attention with expanded K, V
            var KT = Transpose(K_expanded, "KT");
            var scores = MatMul(Q, KT, "attention_scores");
            KT.Dispose();
            K_expanded.Dispose();
            
            // Scale by 1/sqrt(d_k)
            float scale = 1.0f / MathF.Sqrt(headDimGQA);
            Scale(scores, scale);
            
            // Row-wise Softmax
            RowwiseSoftmax(scores);
            
            // Attention_weights @ V
            var output = MatMul(scores, V_expanded, resultName ?? "attention_output");
            scores.Dispose();
            V_expanded.Dispose();
            
            return output;
        }

        // Standard MHA (Multi-Head Attention) when Q, K, V have same dimensions
        int headDim = dModel_Q / numHeads;

        if (dModel_Q % numHeads != 0)
            throw new ArgumentException($"d_model ({dModel_Q}) must be divisible by num_heads ({numHeads})");

        // 1. Q @ K^T
        var KT_std = Transpose(K, "KT");
        var scores_std = MatMul(Q, KT_std, "attention_scores");
        KT_std.Dispose();

        // 2. Scale by 1/sqrt(d_k)
        float scale_std = 1.0f / MathF.Sqrt(headDim);
        Scale(scores_std, scale_std);

        // 3. Row-wise Softmax
        RowwiseSoftmax(scores_std);

        // 4. Attention_weights @ V
        var output_std = MatMul(scores_std, V, resultName ?? "attention_output");
        scores_std.Dispose();

        return output_std;
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

        // Simple CPU implementation - copy each column repeatFactor times
        float[] inputData = input.ReadData();
        float[] outputData = new float[rows * newCols];

        for (int r = 0; r < rows; r++)
        {
            for (int c = 0; c < cols; c++)
            {
                float value = inputData[r * cols + c];
                for (int rep = 0; rep < repeatFactor; rep++)
                {
                    int outCol = c * repeatFactor + rep;
                    outputData[r * newCols + outCol] = value;
                }
            }
        }

        result.Buffer.Write(outputData);
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

        // 1. Gate projection: gate = x @ W_gate
        var gate = MatMul(x, wGate, "ffn_gate");

        // 2. SiLU activation: gate = SiLU(gate)
        SiLU(gate);

        // 3. Up projection: up = x @ W_up
        var up = MatMul(x, wUp, "ffn_up");

        // 4. Element-wise multiply: gate_up = gate ⊙ up
        var gateUp = Multiply(gate, up, "ffn_gate_up");
        gate.Dispose();
        up.Dispose();

        // 5. Down projection: output = gate_up @ W_down
        var output = MatMul(gateUp, wDown, resultName ?? "ffn_output");
        gateUp.Dispose();

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
        Inference.KVCache? kvCache = null,
        int layerIdx = 0,
        float eps = 1e-5f,
        string? resultName = null)
    {
        int seqLen = x.Shape[0];
        int dModel = x.Shape[1];
        int headDim = dModel / numHeads;

        // 1. Attention block
        // 1.1 Pre-normalization
        var xNorm = Tensor.Create(_device, x.Shape, DataType.F32, "attn_norm_input");
        var xData = x.ReadData();
        xNorm.Buffer.Write(xData);
        LayerNorm(xNorm, attnNormWeight, eps);

        // 1.2 Q, K, V projections
        var Q = MatMul(xNorm, wQ, "Q");
        Console.WriteLine($"  [TransLayer Debug] After MatMul: Q shape = {Q.Shape}");
        var K = MatMul(xNorm, wK, "K");
        Console.WriteLine($"  [TransLayer Debug] After MatMul: K shape = {K.Shape}, wK shape = {wK.Shape}");
        var V = MatMul(xNorm, wV, "V");
        Console.WriteLine($"  [TransLayer Debug] After MatMul: V shape = {V.Shape}");
        xNorm.Dispose();

        // 1.3 Multi-head attention with KV-cache
        Tensor attnOut;
        if (kvCache != null)
        {
            Console.WriteLine($"  [KVCache Debug] Before Add: layer={layerIdx}, K shape={K.Shape}, V shape={V.Shape}");
            
            // Add current K, V to cache (will concatenate internally)
            kvCache.Add(layerIdx, K, V);
            
            // Get full cached K, V for attention
            var (cachedK, cachedV) = kvCache.Get(layerIdx);
            if (cachedK == null || cachedV == null)
                throw new InvalidOperationException($"Failed to retrieve cached K,V for layer {layerIdx}");
            
            Console.WriteLine($"  [KVCache Debug] After Get: cachedK shape={cachedK.Shape}, cachedV shape={cachedV.Shape}");
            Console.WriteLine($"  [KVCache Debug] Q shape={Q.Shape} for attention");
            
            // Use cached K, V for attention
            attnOut = MultiHeadAttention(Q, cachedK, cachedV, numHeads, "attn_out");
            Q.Dispose();
            // Don't dispose K, V - they're owned by cache
        }
        else
        {
            // Without cache: use K, V directly
            attnOut = MultiHeadAttention(Q, K, V, numHeads, "attn_out");
            Q.Dispose();
            K.Dispose();
            V.Dispose();
        }

        // 1.4 Output projection
        var attnProj = MatMul(attnOut, wAttnOut, "attn_proj");
        attnOut.Dispose();

        // 1.5 Residual connection
        var x1 = Add(x, attnProj, "x_after_attn");
        attnProj.Dispose();

        // 2. FFN block
        // 2.1 Pre-normalization
        var x1Norm = Tensor.Create(_device, x1.Shape, DataType.F32, "ffn_norm_input");
        var x1Data = x1.ReadData();
        x1Norm.Buffer.Write(x1Data);
        LayerNorm(x1Norm, ffnNormWeight, eps);

        // 2.2 Feed-forward network
        var ffnOut = FeedForward(x1Norm, wGate, wUp, wDown, "ffn_out");
        x1Norm.Dispose();

        // 2.3 Residual connection
        var output = Add(x1, ffnOut, resultName ?? "layer_output");
        x1.Dispose();
        ffnOut.Dispose();

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
        _queue.Flush();

        tokenBuf.Dispose();
        paramsBuffer.Dispose();

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
