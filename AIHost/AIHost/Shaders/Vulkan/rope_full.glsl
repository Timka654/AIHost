#version 450
layout(local_size_x = 256) in;

layout(set = 0, binding = 0) buffer InOutBuf { float data[]; } buf;
layout(set = 0, binding = 1) readonly buffer Params {
    uint seqLen;
    uint numHeads;
    uint headDim;
    uint startPosition;
    float theta;
    uint ropeDim;   // how many dims per head to rotate (partial RoPE)
} params;

// Multi-head RoPE with optional partial rotation.
// For each token (seq) and each head, only the first ropeDim dimensions are rotated.
// Dimensions [ropeDim..headDim) are left unchanged.
// For Qwen3.5/3.6: ropeDim=64, headDim=128 — only 32 pairs rotated per head.
void main() {
    uint gid = gl_GlobalInvocationID.x;
    uint halfRope = params.ropeDim / 2u;
    uint pairsPerSeq = params.numHeads * halfRope;
    if (gid >= params.seqLen * pairsPerSeq) return;

    uint seq  = gid / pairsPerSeq;
    uint rem  = gid % pairsPerSeq;
    uint head = rem / halfRope;
    uint pair = rem % halfRope;

    uint pos = params.startPosition + seq;
    float freq = 1.0 / pow(params.theta, float(pair * 2u) / float(params.ropeDim));
    float angle = float(pos) * freq;
    float c = cos(angle);
    float s = sin(angle);

    // offset: token row * (numHeads * headDim) + head * headDim + pair*2
    uint base = (seq * params.numHeads + head) * params.headDim + pair * 2u;
    float x1 = buf.data[base];
    float x2 = buf.data[base + 1u];
    buf.data[base]      = x1 * c - x2 * s;
    buf.data[base + 1u] = x1 * s + x2 * c;
}
