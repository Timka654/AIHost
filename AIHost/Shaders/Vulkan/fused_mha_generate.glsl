#version 450
layout(local_size_x = 256) in;
layout(set = 0, binding = 0) readonly buffer QBuf   { float data[]; } q_buf;
layout(set = 0, binding = 1) readonly buffer KBuf   { float data[]; } k_buf;
layout(set = 0, binding = 2) readonly buffer VBuf   { float data[]; } v_buf;
layout(set = 0, binding = 3) buffer       OutBuf  { float data[]; } out_buf;
layout(set = 0, binding = 4) readonly buffer Params {
    uint numQHeads; uint numKvHeads; uint headDim; uint kvSeqLen; float scale;
} params;

shared float sharedMem[256];
shared float scores[4096];

void main() {
    uint h   = gl_WorkGroupID.x;          // Q head index (0..numQHeads-1)
    uint tid = gl_LocalInvocationID.x;    // thread index (0..255)

    uint repeatFactor = params.numQHeads / params.numKvHeads;
    uint kvH     = h / repeatFactor;
    uint kvStride = params.numKvHeads * params.headDim;
    uint q_base   = h    * params.headDim;
    uint kv_off   = kvH  * params.headDim;

    // -- Step 1: scaled dot-product Q . K^T --
    for (uint j = tid; j < params.kvSeqLen; j += 256u) {
        float s = 0.0;
        for (uint d = 0u; d < params.headDim; d++) {
            s += q_buf.data[q_base + d] *
                 k_buf.data[j * kvStride + kv_off + d];
        }
        scores[j] = s * params.scale;
    }
    barrier();

    // -- Step 2: stable softmax (parallel max then exp-sum) --
    float localMax = -1e38;
    for (uint j = tid; j < params.kvSeqLen; j += 256u)
        localMax = max(localMax, scores[j]);
    sharedMem[tid] = localMax;
    barrier();
    for (uint s = 128u; s > 0u; s >>= 1u) {
        if (tid < s) sharedMem[tid] = max(sharedMem[tid], sharedMem[tid + s]);
        barrier();
    }
    float rowMax = sharedMem[0];
    barrier();

    float localSum = 0.0;
    for (uint j = tid; j < params.kvSeqLen; j += 256u) {
        float e = exp(scores[j] - rowMax);
        scores[j] = e;
        localSum += e;
    }
    sharedMem[tid] = localSum;
    barrier();
    for (uint s = 128u; s > 0u; s >>= 1u) {
        if (tid < s) sharedMem[tid] += sharedMem[tid + s];
        barrier();
    }
    float rowSum = max(sharedMem[0], 0.000001);
    barrier();

    for (uint j = tid; j < params.kvSeqLen; j += 256u)
        scores[j] /= rowSum;
    barrier();

    // -- Step 3: weighted V sum --
    // Thread tid owns output dimension tid (tid < headDim only).
    if (tid < params.headDim) {
        float acc = 0.0;
        for (uint j = 0u; j < params.kvSeqLen; j++) {
            acc += scores[j] * v_buf.data[j * kvStride + kv_off + tid];
        }
        out_buf.data[h * params.headDim + tid] = acc;
    }
}