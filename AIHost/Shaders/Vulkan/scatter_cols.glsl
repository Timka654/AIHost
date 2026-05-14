#version 450
layout(local_size_x = 256) in;
layout(set = 0, binding = 0) readonly buffer Src { float data[]; } src;
layout(set = 0, binding = 1) writeonly buffer Dst { float data[]; } dst;
layout(set = 0, binding = 2) readonly buffer Params { uint rows; uint dstCols; uint colStart; uint colCount; } params;
void main() {
    uint gid = gl_GlobalInvocationID.x;
    if (gid >= params.rows * params.colCount) return;
    uint row = gid / params.colCount;
    uint col = gid % params.colCount;
    dst.data[row * params.dstCols + params.colStart + col] = src.data[gid];
}