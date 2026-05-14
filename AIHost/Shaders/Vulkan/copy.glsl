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