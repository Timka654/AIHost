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
