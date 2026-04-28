#version 450
layout(local_size_x = 256) in;

layout(set = 0, binding = 0) buffer InOutBuf { float data[]; } buf;
layout(set = 0, binding = 1) readonly buffer Params { uint size; float scale; } params;

void main() {
    uint gid = gl_GlobalInvocationID.x;
    if (gid >= params.size) return;
    buf.data[gid] *= params.scale;
}
