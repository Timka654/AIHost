#version 450
layout(local_size_x = 256) in;

layout(set = 0, binding = 0) readonly buffer BufA { float data[]; } A;
layout(set = 0, binding = 1) readonly buffer BufB { float data[]; } B;
layout(set = 0, binding = 2) buffer BufC { float data[]; } C;
layout(set = 0, binding = 3) readonly buffer Params { 
    uint dim0; 
    uint dim1_a; 
    uint dim1_b; 
} params;

void main() {
    uint gid = gl_GlobalInvocationID.x;
    uint dim1_total = params.dim1_a + params.dim1_b;
    uint total_size = params.dim0 * dim1_total;
    
    if (gid >= total_size) return;
    
    uint i = gid / dim1_total;  // which row
    uint j = gid % dim1_total;  // position in row
    
    if (j < params.dim1_a) {
        // Copy from A
        C.data[gid] = A.data[i * params.dim1_a + j];
    } else {
        // Copy from B
        C.data[gid] = B.data[i * params.dim1_b + (j - params.dim1_a)];
    }
}
