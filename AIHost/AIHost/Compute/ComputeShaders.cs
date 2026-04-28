namespace AIHost.Compute;

/// <summary>
/// Shader source access with lazy loading from files
/// Falls back to inline source if file not found
/// </summary>
public static class ComputeShaders
{
    private const string DefaultProvider = "Vulkan";

    // Quantization shaders - lazy loaded from files with inline fallbacks
    public static string DequantizeQ2K => TryLoadOrInline("dequant_q2k", QuantizationFormats.DequantizeQ2K_Correct);
    public static string DequantizeQ3K => TryLoadOrInline("dequant_q3k", QuantizationFormats.DequantizeQ3K_Correct);
    public static string DequantizeQ4K => TryLoadOrInline("dequant_q4k", QuantizationFormats.DequantizeQ4K_Correct);
    public static string DequantizeQ5K => TryLoadOrInline("dequant_q5k", QuantizationFormats.DequantizeQ5K_Correct);
    public static string DequantizeQ6K => TryLoadOrInline("dequant_q6k", QuantizationFormats.DequantizeQ6K_Correct);

    // Core operations - lazy loaded from files with inline fallbacks
    public static string MatMulF32 => TryLoadOrInline("matmul", _inlineMatMul);
    public static string Softmax => TryLoadOrInline("softmax", _inlineSoftmax);
    public static string SiLU => TryLoadOrInline("silu", _inlineSiLU);
    public static string ElementWiseAdd => TryLoadOrInline("add", _inlineAdd);
    public static string ConcatAxis1 => TryLoadOrInline("concat_axis1", _inlineConcat);
    public static string ConcatAxis0 => TryLoadOrInline("concat_axis0", _inlineConcatAxis0);
    
    // Additional operations - lazy loaded from files with inline fallbacks
    public static string LayerNorm => TryLoadOrInline("layernorm", _inlineLayerNorm);
    public static string ElementWiseMul => TryLoadOrInline("elementwise_mul", _inlineElementWiseMul);
    public static string RoPE => TryLoadOrInline("rope", _inlineRoPE);
    public static string Transpose => TryLoadOrInline("transpose", _inlineTranspose);
    public static string RowwiseSoftmax => TryLoadOrInline("rowwise_softmax", _inlineRowwiseSoftmax);
    public static string Scale => TryLoadOrInline("scale", _inlineScale);
    public static string EmbeddingLookup => TryLoadOrInline("embedding_lookup", _inlineEmbeddingLookup);
    public static string RoPEFull => TryLoadOrInline("rope_full", _inlineRoPEFull);
    public static string CausalMask => TryLoadOrInline("causal_mask", _inlineCausalMask);
    public static string Copy => TryLoadOrInline("copy", _inlineCopy);
    public static string RepeatColumns => TryLoadOrInline("repeat_columns", _inlineRepeatColumns);

    private static string TryLoadOrInline(string shaderName, string inlineSource)
    {
        try
        {
            return ShaderLoader.Load(DefaultProvider, shaderName);
        }
        catch (FileNotFoundException)
        {
            return inlineSource;
        }
    }

    // Inline fallbacks for backward compatibility
    private const string _inlineMatMul = @"
#version 450
layout(local_size_x = 16, local_size_y = 16) in;
layout(set = 0, binding = 0) readonly buffer MatrixA { float data[]; } A;
layout(set = 0, binding = 1) readonly buffer MatrixB { float data[]; } B;
layout(set = 0, binding = 2) buffer MatrixC { float data[]; } C;
layout(set = 0, binding = 3) readonly buffer Params { uint M; uint K; uint N; } params;
shared float tileA[16][16];
shared float tileB[16][16];
void main() {
    uint row = gl_GlobalInvocationID.y;
    uint col = gl_GlobalInvocationID.x;
    uint tileRow = gl_LocalInvocationID.y;
    uint tileCol = gl_LocalInvocationID.x;
    float sum = 0.0;
    uint numTiles = (params.K + 15u) / 16u;
    for (uint t = 0u; t < numTiles; t++) {
        uint aRow = row;
        uint aCol = t * 16u + tileCol;
        uint bRow = t * 16u + tileRow;
        uint bCol = col;
        // All threads must participate in shared memory loads to avoid garbage reads.
        // Out-of-bounds threads load 0.0 so they don't pollute the tile.
        tileA[tileRow][tileCol] = (aRow < params.M && aCol < params.K) ? A.data[aRow * params.K + aCol] : 0.0;
        tileB[tileRow][tileCol] = (bRow < params.K && bCol < params.N) ? B.data[bRow * params.N + bCol] : 0.0;
        barrier();
        for (uint k = 0u; k < 16u; k++) {
            sum += tileA[tileRow][k] * tileB[k][tileCol];
        }
        barrier();
    }
    // Only write result for valid output positions
    if (row < params.M && col < params.N) {
        C.data[row * params.N + col] = sum;
    }
}
";

    private const string _inlineSoftmax = @"
#version 450
layout(local_size_x = 256) in;
layout(set = 0, binding = 0) buffer InOutBuf { float data[]; } buf;
layout(set = 0, binding = 1) readonly buffer Params { uint size; } params;
shared float sMax[256];
shared float sSum[256];
void main() {
    uint gid = gl_GlobalInvocationID.x;
    uint lid = gl_LocalInvocationID.x;
    float val = (gid < params.size) ? buf.data[gid] : -1e38;
    sMax[lid] = val;
    barrier();
    for (uint s = 128u; s > 0u; s >>= 1u) {
        if (lid < s) sMax[lid] = max(sMax[lid], sMax[lid + s]);
        barrier();
    }
    float maxVal = sMax[0];
    if (gid < params.size) {
        val = exp(buf.data[gid] - maxVal);
        buf.data[gid] = val;
    } else {
        val = 0.0;
    }
    sSum[lid] = val;
    barrier();
    for (uint s = 128u; s > 0u; s >>= 1u) {
        if (lid < s) sSum[lid] += sSum[lid + s];
        barrier();
    }
    float sumVal = sSum[0];
    if (gid < params.size && sumVal > 0.0) {
        buf.data[gid] /= sumVal;
    }
}
";

    // Row-wise RMSNorm (as used in LLaMA): each workgroup handles one row (token).
    // out[row,i] = x[row,i] / sqrt(mean(x[row]^2) + eps) * weight[i]
    private const string _inlineLayerNorm = @"
#version 450
layout(local_size_x = 256) in;
layout(set = 0, binding = 0) buffer InOutBuf { float data[]; } buf;
layout(set = 0, binding = 1) readonly buffer WeightBuf { float data[]; } weight;
layout(set = 0, binding = 2) readonly buffer Params { uint rows; uint cols; float eps; } params;
shared float sharedMem[256];
void main() {
    uint row = gl_WorkGroupID.x;
    uint tid = gl_LocalInvocationID.x;
    if (row >= params.rows) return;
    uint rowOffset = row * params.cols;
    float localSumSq = 0.0;
    for (uint i = tid; i < params.cols; i += 256u) {
        float x = buf.data[rowOffset + i];
        localSumSq += x * x;
    }
    sharedMem[tid] = localSumSq;
    barrier();
    for (uint s = 128u; s > 0u; s >>= 1u) {
        if (tid < s) sharedMem[tid] += sharedMem[tid + s];
        barrier();
    }
    float rms = sqrt(sharedMem[0] / float(params.cols) + params.eps);
    barrier();
    for (uint i = tid; i < params.cols; i += 256u) {
        buf.data[rowOffset + i] = (buf.data[rowOffset + i] / rms) * weight.data[i];
    }
}
";

    private const string _inlineSiLU = @"
#version 450
layout(local_size_x = 256) in;
layout(set = 0, binding = 0) buffer InOutBuf { float data[]; } buf;
layout(set = 0, binding = 1) readonly buffer Params { uint size; } params;
void main() {
    uint gid = gl_GlobalInvocationID.x;
    if (gid >= params.size) return;
    float x = buf.data[gid];
    float sigmoid = 1.0 / (1.0 + exp(-x));
    buf.data[gid] = x * sigmoid;
}
";

    private const string _inlineElementWiseMul = @"
#version 450
layout(local_size_x = 256) in;
layout(set = 0, binding = 0) readonly buffer BufA { float data[]; } A;
layout(set = 0, binding = 1) readonly buffer BufB { float data[]; } B;
layout(set = 0, binding = 2) buffer BufC { float data[]; } C;
layout(set = 0, binding = 3) readonly buffer Params { uint size; } params;
void main() {
    uint gid = gl_GlobalInvocationID.x;
    if (gid >= params.size) return;
    C.data[gid] = A.data[gid] * B.data[gid];
}
";

    private const string _inlineAdd = @"
#version 450
layout(local_size_x = 256) in;
layout(set = 0, binding = 0) readonly buffer BufA { float data[]; } A;
layout(set = 0, binding = 1) readonly buffer BufB { float data[]; } B;
layout(set = 0, binding = 2) buffer BufC { float data[]; } C;
layout(set = 0, binding = 3) readonly buffer Params { uint size; } params;
void main() {
    uint gid = gl_GlobalInvocationID.x;
    if (gid >= params.size) return;
    C.data[gid] = A.data[gid] + B.data[gid];
}
";

    private const string _inlineConcat = @"
#version 450
layout(local_size_x = 256) in;
layout(set = 0, binding = 0) readonly buffer BufA { float data[]; } A;
layout(set = 0, binding = 1) readonly buffer BufB { float data[]; } B;
layout(set = 0, binding = 2) buffer BufC { float data[]; } C;
layout(set = 0, binding = 3) readonly buffer Params { 
    uint dim0; 
    uint dim1_a; 
    uint dim1_b; 
} params;
void main() {
    uint gid = gl_GlobalInvocationID.x;
    uint dim1_total = params.dim1_a + params.dim1_b;
    uint total_size = params.dim0 * dim1_total;
    
    if (gid >= total_size) return;
    
    uint i = gid / dim1_total;  // which row
    uint j = gid % dim1_total;  // position in row
    
    if (j < params.dim1_a) {
        // Copy from A
        C.data[gid] = A.data[i * params.dim1_a + j];
    } else {
        // Copy from B
        C.data[gid] = B.data[i * params.dim1_b + (j - params.dim1_a)];
    }
}
";

    private const string _inlineConcatAxis0 = @"
#version 450
layout(local_size_x = 256) in;
layout(set = 0, binding = 0) readonly buffer InputA { float a[]; };
layout(set = 0, binding = 1) readonly buffer InputB { float b[]; };
layout(set = 0, binding = 2) writeonly buffer Output { float result[]; };
layout(set = 0, binding = 3) readonly buffer Params {
    uint rows_a;
    uint rows_b;
    uint cols;
};
void main() {
    uint gid = gl_GlobalInvocationID.x;
    uint total_rows = rows_a + rows_b;
    uint total_elements = total_rows * cols;
    
    if (gid >= total_elements) return;
    
    uint row = gid / cols;
    uint col = gid % cols;
    
    if (row < rows_a) {
        result[gid] = a[row * cols + col];
    } else {
        uint b_row = row - rows_a;
        result[gid] = b[b_row * cols + col];
    }
}
";

    private const string _inlineRoPE = @"
#version 450
layout(local_size_x = 256) in;
layout(set = 0, binding = 0) buffer InOutBuf { float data[]; } buf;
layout(set = 0, binding = 1) readonly buffer ParamsBuffer {
    uint seq_len;
    uint head_dim;
    uint position;
    float theta;
} params;

void main() {
    uint pair_idx = gl_GlobalInvocationID.x;
    if (pair_idx >= params.head_dim / 2u) return;
    
    float freq = 1.0 / pow(params.theta, float(pair_idx * 2u) / float(params.head_dim));
    float angle = float(params.position) * freq;
    
    float cos_val = cos(angle);
    float sin_val = sin(angle);
    
    uint idx1 = pair_idx * 2u;
    uint idx2 = pair_idx * 2u + 1u;
    
    float x1 = buf.data[idx1];
    float x2 = buf.data[idx2];
    
    buf.data[idx1] = x1 * cos_val - x2 * sin_val;
    buf.data[idx2] = x1 * sin_val + x2 * cos_val;
}
";

    private const string _inlineTranspose = @"
#version 450
layout(local_size_x = 16, local_size_y = 16) in;
layout(set = 0, binding = 0) readonly buffer MatrixA { float data[]; } A;
layout(set = 0, binding = 1) buffer MatrixB { float data[]; } B;
layout(set = 0, binding = 2) readonly buffer Params { uint rows; uint cols; } params;
void main() {
    uint row = gl_GlobalInvocationID.y;
    uint col = gl_GlobalInvocationID.x;
    if (row >= params.rows || col >= params.cols) return;
    B.data[col * params.rows + row] = A.data[row * params.cols + col];
}
";

    private const string _inlineRowwiseSoftmax = @"
#version 450
layout(local_size_x = 256) in;
layout(set = 0, binding = 0) buffer InOutBuf { float data[]; } buf;
layout(set = 0, binding = 1) readonly buffer Params { uint rows; uint cols; } params;
shared float sharedMem[256];
void main() {
    uint row = gl_WorkGroupID.x;
    uint tid = gl_LocalInvocationID.x;
    if (row >= params.rows) return;
    
    uint rowOffset = row * params.cols;
    
    // Find max
    float localMax = -1e38;
    for (uint i = tid; i < params.cols; i += gl_WorkGroupSize.x) {
        localMax = max(localMax, buf.data[rowOffset + i]);
    }
    sharedMem[tid] = localMax;
    barrier();
    
    // Reduce max
    for (uint s = gl_WorkGroupSize.x / 2u; s > 0u; s >>= 1u) {
        if (tid < s) {
            sharedMem[tid] = max(sharedMem[tid], sharedMem[tid + s]);
        }
        barrier();
    }
    float rowMax = sharedMem[0];
    barrier();
    
    // Compute exp and sum
    float localSum = 0.0;
    for (uint i = tid; i < params.cols; i += gl_WorkGroupSize.x) {
        float val = exp(buf.data[rowOffset + i] - rowMax);
        buf.data[rowOffset + i] = val;
        localSum += val;
    }
    sharedMem[tid] = localSum;
    barrier();
    
    // Reduce sum
    for (uint s = gl_WorkGroupSize.x / 2u; s > 0u; s >>= 1u) {
        if (tid < s) {
            sharedMem[tid] += sharedMem[tid + s];
        }
        barrier();
    }
    float rowSum = sharedMem[0];
    barrier();
    
    // Normalize
    for (uint i = tid; i < params.cols; i += gl_WorkGroupSize.x) {
        buf.data[rowOffset + i] /= rowSum;
    }
}
";

    private const string _inlineScale = @"
#version 450
layout(local_size_x = 256) in;
layout(set = 0, binding = 0) buffer InOutBuf { float data[]; } buf;
layout(set = 0, binding = 1) readonly buffer Params { uint size; float scale; } params;
void main() {
    uint gid = gl_GlobalInvocationID.x;
    if (gid >= params.size) return;
    buf.data[gid] *= params.scale;
}
";

    private const string _inlineEmbeddingLookup = @"
#version 450
layout(local_size_x = 256) in;
layout(set = 0, binding = 0) readonly buffer TokenIds { int ids[]; } tokenIds;
layout(set = 0, binding = 1) readonly buffer EmbTable { float data[]; } table;
layout(set = 0, binding = 2) writeonly buffer Output { float data[]; } output_buf;
layout(set = 0, binding = 3) readonly buffer Params { uint seqLen; uint dModel; } params;
void main() {
    uint gid = gl_GlobalInvocationID.x;
    if (gid >= params.seqLen * params.dModel) return;
    uint seq = gid / params.dModel;
    uint dim = gid % params.dModel;
    int tokenId = tokenIds.ids[seq];
    if (tokenId < 0) return;
    output_buf.data[gid] = table.data[uint(tokenId) * params.dModel + dim];
}
";

    private const string _inlineRoPEFull = @"
#version 450
layout(local_size_x = 256) in;
layout(set = 0, binding = 0) buffer InOutBuf { float data[]; } buf;
layout(set = 0, binding = 1) readonly buffer Params {
    uint seqLen; uint numHeads; uint headDim; uint startPosition; float theta;
} params;
void main() {
    uint gid = gl_GlobalInvocationID.x;
    uint halfHead = params.headDim / 2u;
    uint pairsPerSeq = params.numHeads * halfHead;
    if (gid >= params.seqLen * pairsPerSeq) return;
    uint seq  = gid / pairsPerSeq;
    uint rem  = gid % pairsPerSeq;
    uint head = rem / halfHead;
    uint pair = rem % halfHead;
    uint pos = params.startPosition + seq;
    float freq = 1.0 / pow(params.theta, float(pair * 2u) / float(params.headDim));
    float angle = float(pos) * freq;
    float c = cos(angle), s = sin(angle);
    uint base = (seq * params.numHeads + head) * params.headDim + pair * 2u;
    float x1 = buf.data[base];
    float x2 = buf.data[base + 1u];
    buf.data[base]     = x1 * c - x2 * s;
    buf.data[base + 1u] = x1 * s + x2 * c;
}
";

    private const string _inlineCausalMask = @"
#version 450
layout(local_size_x = 256) in;
layout(set = 0, binding = 0) buffer InOutBuf { float data[]; } buf;
layout(set = 0, binding = 1) readonly buffer Params {
    uint seqLen_q; uint seqLen_k; uint startPosition;
} params;
void main() {
    uint gid = gl_GlobalInvocationID.x;
    if (gid >= params.seqLen_q * params.seqLen_k) return;
    uint i = gid / params.seqLen_k;
    uint j = gid % params.seqLen_k;
    if (j > params.startPosition + i) {
        buf.data[gid] = -1e38;
    }
}
";

    private const string _inlineCopy = @"
#version 450
layout(local_size_x = 256) in;
layout(set = 0, binding = 0) readonly buffer Src { float data[]; } src;
layout(set = 0, binding = 1) writeonly buffer Dst { float data[]; } dst;
layout(set = 0, binding = 2) readonly buffer Params { uint size; } params;
void main() {
    uint gid = gl_GlobalInvocationID.x;
    if (gid >= params.size) return;
    dst.data[gid] = src.data[gid];
}
";

    // Each output element (row, outCol) gathers from srcCol = outCol / repeatFactor.
    private const string _inlineRepeatColumns = @"
#version 450
layout(local_size_x = 256) in;
layout(set = 0, binding = 0) readonly buffer Src { float data[]; } src;
layout(set = 0, binding = 1) writeonly buffer Dst { float data[]; } dst;
layout(set = 0, binding = 2) readonly buffer Params { uint rows; uint cols; uint repeatFactor; } params;
void main() {
    uint gid = gl_GlobalInvocationID.x;
    uint newCols = params.cols * params.repeatFactor;
    if (gid >= params.rows * newCols) return;
    uint row    = gid / newCols;
    uint outCol = gid % newCols;
    uint srcCol = outCol / params.repeatFactor;
    dst.data[gid] = src.data[row * params.cols + srcCol];
}
";
}
