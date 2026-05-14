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