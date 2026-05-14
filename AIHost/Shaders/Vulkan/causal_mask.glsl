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