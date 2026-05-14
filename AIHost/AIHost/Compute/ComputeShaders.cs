namespace AIHost.Compute;

/// <summary>
/// Shader source access with lazy loading from files
/// Falls back to inline source if file not found
/// </summary>
public static class ComputeShaders
{
    private const string DefaultProvider = "Vulkan";

    // Quantization shaders - cached once at startup (inline fallback if .glsl file missing)
    public static readonly string DequantizeQ2K = ShaderLoader.Load(DefaultProvider, "dequant_q2k");
    public static readonly string DequantizeQ3K = ShaderLoader.Load(DefaultProvider, "dequant_q3k");
    public static readonly string DequantizeQ4K = ShaderLoader.Load(DefaultProvider, "dequant_q4k");
    public static readonly string DequantizeQ5K = ShaderLoader.Load(DefaultProvider, "dequant_q5k");
    public static readonly string DequantizeQ6K = ShaderLoader.Load(DefaultProvider, "dequant_q6k");

    // Core operations
    public static readonly string MatMulF32 = ShaderLoader.Load(DefaultProvider, "matmul");

    // Weight matrices from GGUF are stored column-major (ne[0] is innermost/fastest dim).
    // This variant reads B as column-major: B[bRow, bCol] = data[bRow + bCol * K].
    // Weight matrices from GGUF use column-major layout: B[k, n] = data[k + n * K].
    // This variant replaces the row-major read B[k,n]=data[k*N+n] with the column-major
    // read B[k,n]=data[k + n*K], fixing out-of-bounds access and wrong computation
    // for non-square weight matrices (wK, wV, wGate, wUp, wDown, etc.).
    public static readonly string MatMulWeightsF32 = ShaderLoader.Load(DefaultProvider, "matmul_weights");
    /// <summary>
    /// C[M,J] = A[M,K] @ B^T where B[J,K] is GGUF col-major (B[j,k]=data[j+k*J]).
    /// Used for gated attention output projection when W_gate is stored transposed.
    /// </summary>
    // C[M,J] = A[M,K] @ B^T  where B[J,K] stored GGUF col-major: B[j,k]=data[j+k*J].
    // C[i,j] = sum_k  A[i*K+k] * B[j+k*J].
    // Used for gated attention output projection (attn_gate stored as W_o^T).
    public static readonly string MatMulWeightsTF32 = ShaderLoader.Load(DefaultProvider, "matmul_weights_t");
    public static readonly string Softmax = ShaderLoader.Load(DefaultProvider, "softmax");
    public static readonly string SiLU = ShaderLoader.Load(DefaultProvider, "silu");
    public static readonly string Sigmoid = ShaderLoader.Load(DefaultProvider, "sigmoid");
    public static readonly string ElementWiseAdd = ShaderLoader.Load(DefaultProvider, "add");
    public static readonly string ConcatAxis1 = ShaderLoader.Load(DefaultProvider, "concat_axis1");
    public static readonly string ConcatAxis0 = ShaderLoader.Load(DefaultProvider, "concat_axis0");

    // Additional operations

    // Row-wise RMSNorm (as used in LLaMA): each workgroup handles one row (token).
    // out[row,i] = x[row,i] / sqrt(mean(x[row]^2) + eps) * weight[i]
    public static readonly string LayerNorm = ShaderLoader.Load(DefaultProvider, "layernorm");
    public static readonly string ElementWiseMul = ShaderLoader.Load(DefaultProvider, "elementwise_mul");
    public static readonly string RoPE = ShaderLoader.Load(DefaultProvider, "rope");
    public static readonly string Transpose = ShaderLoader.Load(DefaultProvider, "transpose");
    public static readonly string RowwiseSoftmax = ShaderLoader.Load(DefaultProvider, "rowwise_softmax");
    public static readonly string Scale = ShaderLoader.Load(DefaultProvider, "scale");
    public static readonly string EmbeddingLookup = ShaderLoader.Load(DefaultProvider, "embedding_lookup");
    public static readonly string RoPEFull = ShaderLoader.Load(DefaultProvider, "rope_full");
    public static readonly string CausalMask = ShaderLoader.Load(DefaultProvider, "causal_mask");
    public static readonly string Copy = ShaderLoader.Load(DefaultProvider, "copy");
    // Each output element (row, outCol) gathers from srcCol = outCol / repeatFactor.
    public static readonly string RepeatColumns = ShaderLoader.Load(DefaultProvider, "repeat_columns");

    // Correct GQA K/V expansion: repeats groups of head_dim columns (not individual columns).
    // K[seq, numKvHeads*headDim] → K_expanded[seq, numQHeads*headDim]
    // Each KV head's headDim-block is repeated repeat_factor times.
    // GQA K/V expansion: repeat each head_dim block repeatFactor times.
    // output[row, j] = src[row, (j / (repeatFactor * headDim)) * headDim + j % headDim]
    // Params: rows, srcCols (=numKvHeads*headDim), headDim, repeatFactor
    public static readonly string RepeatKVHeads = ShaderLoader.Load(DefaultProvider, "repeat_kv_heads");

    // Extract a contiguous column slice from a 2D tensor.
    // Src: [rows, srcCols], colStart: first column index, colCount: number of columns
    // Dst: [rows, colCount]
    // Extract columns [colStart, colStart+colCount) from src[rows, srcCols] -> dst[rows, colCount]
    public static readonly string SliceCols = ShaderLoader.Load(DefaultProvider, "slice_cols");

    // Scatter (write) src [rows, colCount] into dst [rows, dstCols] starting at colStart.
    // Write src[rows, colCount] into dst[rows, dstCols] at column offset colStart.
    public static readonly string ScatterCols = ShaderLoader.Load(DefaultProvider, "scatter_cols");

    // Fused multi-head (GQA) attention for a SINGLE QUERY token (seqLen=1).
    // Avoids per-head SliceCols/ScatterCols/Transpose allocations.
    // Dispatch: globalWorkSize = { numQHeads } (one workgroup per Q head, 256 threads each)
    // Fused GQA attention for a single query token (seqLen=1).
    // One workgroup per Q head (gl_WorkGroupID.x = h), local_size_x = 256 threads.
    // Threads cooperate: round-robin over kvSeqLen for scores, per-dim for V sum.
    // Shared mem: 256 floats for reductions + 4096 floats for scores.
    // Max supported kvSeqLen = 4096 (increase shared array if needed).
    public static readonly string FusedMHAGenerate = ShaderLoader.Load(DefaultProvider, "fused_mha_generate");

    // ── SSM/GDN shaders ─────────────────────────────────────────────────────
    // Part 1: conv1d — computes z, beta, alpha, gate, qkv_mixed, conv1d, SiLU → scratch
    // Dispatch: 48 workgroups × 128 threads
    public static readonly string SsmGdnDecode = ShaderLoader.Load(DefaultProvider, "ssm_gdn_decode");

    // Part 2: recurrence — reads scratch, L2 norm, Delta Net, Gated norm → output
    // Dispatch: 48 workgroups × 128 threads
    public static readonly string SsmGdnRecur = ShaderLoader.Load(DefaultProvider, "ssm_gdn_recur");

    // De-interleave Q+gate from attn_q.weight output for TypeB (Qwen3Next/qwen35) attention layers.
    // Input : [sl, n_head * 2 * head_dim] — [Q_h0(hd), gate_h0(hd), Q_h1(hd), gate_h1(hd), ...]
    // Output: [sl, n_head * 2 * head_dim] — [Q_all(n_head*hd), gate_all(n_head*hd)]
    // Dispatch: ceil(sl * n_head * 2 * head_dim / 256) workgroups × 256 threads
    public static readonly string DeinterleaveQGate = ShaderLoader.Load(DefaultProvider, "deinterleave_q_gate");
}
