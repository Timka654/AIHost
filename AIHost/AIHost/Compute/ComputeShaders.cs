namespace AIHost.Compute;

/// <summary>
/// Shader source access with lazy loading from files
/// Falls back to inline source if file not found
/// </summary>
public static class ComputeShaders
{
    private const string DefaultProvider = "Vulkan";

    // Quantization shaders (inline for now, TODO: move to files)
    public const string DequantizeQ2K = QuantizationFormats.DequantizeQ2K_Correct;
    public const string DequantizeQ3K = QuantizationFormats.DequantizeQ3K_Correct;
    public const string DequantizeQ4K = QuantizationFormats.DequantizeQ4K_Correct;
    public const string DequantizeQ5K = QuantizationFormats.DequantizeQ5K_Correct;
    public const string DequantizeQ6K = QuantizationFormats.DequantizeQ6K_Correct;

    // Core operations - lazy loaded from files
    public static string MatMulF32 => TryLoadOrInline("matmul", _inlineMatMul);
    public static string Softmax => TryLoadOrInline("softmax", _inlineSoftmax);
    public static string SiLU => TryLoadOrInline("silu", _inlineSiLU);
    public static string ElementWiseAdd => TryLoadOrInline("add", _inlineAdd);
    public static string ConcatAxis1 => TryLoadOrInline("concat_axis1", _inlineConcat);
    public static string ConcatAxis0 => TryLoadOrInline("concat_axis0", _inlineConcatAxis0);

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
    if (row >= params.M || col >= params.N) return;
    float sum = 0.0;
    uint numTiles = (params.K + 15u) / 16u;
    for (uint t = 0u; t < numTiles; t++) {
        uint tileRow = gl_LocalInvocationID.y;
        uint tileCol = gl_LocalInvocationID.x;
        uint aRow = row;
        uint aCol = t * 16u + tileCol;
        uint bRow = t * 16u + tileRow;
        uint bCol = col;
        tileA[tileRow][tileCol] = (aCol < params.K) ? A.data[aRow * params.K + aCol] : 0.0;
        tileB[tileRow][tileCol] = (bRow < params.K) ? B.data[bRow * params.N + bCol] : 0.0;
        barrier();
        for (uint k = 0u; k < 16u; k++) {
            sum += tileA[tileRow][k] * tileB[k][tileCol];
        }
        barrier();
    }
    C.data[row * params.N + col] = sum;
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

    public const string LayerNorm = @"
#version 450
layout(local_size_x = 256) in;
layout(set = 0, binding = 0) buffer InOutBuf { float data[]; } buf;
layout(set = 0, binding = 1) readonly buffer WeightBuf { float data[]; } weight;
layout(set = 0, binding = 2) readonly buffer Params { uint size; float eps; } params;
shared float sMean[256];
shared float sVar[256];
void main() {
    uint gid = gl_GlobalInvocationID.x;
    uint lid = gl_LocalInvocationID.x;
    float val = (gid < params.size) ? buf.data[gid] : 0.0;
    sMean[lid] = val;
    barrier();
    for (uint s = 128u; s > 0u; s >>= 1u) {
        if (lid < s) sMean[lid] += sMean[lid + s];
        barrier();
    }
    float mean = sMean[0] / float(params.size);
    if (gid < params.size) val = buf.data[gid] - mean;
    else val = 0.0;
    sVar[lid] = val * val;
    barrier();
    for (uint s = 128u; s > 0u; s >>= 1u) {
        if (lid < s) sVar[lid] += sVar[lid + s];
        barrier();
    }
    float variance = sVar[0] / float(params.size);
    float stddev = sqrt(variance + params.eps);
    if (gid < params.size) {
        float normalized = (buf.data[gid] - mean) / stddev;
        buf.data[gid] = normalized * weight.data[gid];
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

    public const string ElementWiseMul = @"
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

    public const string RoPE = @"
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

    public const string Transpose = @"
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

    public const string RowwiseSoftmax = @"
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

    public const string Scale = @"
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
}
