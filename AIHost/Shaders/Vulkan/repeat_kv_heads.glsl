#version 450
layout(local_size_x = 256) in;
layout(set = 0, binding = 0) readonly buffer Src { float data[]; } src;
layout(set = 0, binding = 1) writeonly buffer Dst { float data[]; } dst;
layout(set = 0, binding = 2) readonly buffer Params {
    uint rows; uint srcCols; uint headDim; uint repeatFactor;
} params;
void main() {
    uint dstCols = params.srcCols * params.repeatFactor;
    uint gid = gl_GlobalInvocationID.x;
    if (gid >= params.rows * dstCols) return;
    uint row = gid / dstCols;
    uint j   = gid % dstCols;
    uint groupStride = params.repeatFactor * params.headDim;  // cols per kv-head in dst
    uint kvHead      = j / groupStride;
    uint dimInHead   = j % params.headDim;
    uint srcCol      = kvHead * params.headDim + dimInHead;
    dst.data[gid] = src.data[row * params.srcCols + srcCol];
}