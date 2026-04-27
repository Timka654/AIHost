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
