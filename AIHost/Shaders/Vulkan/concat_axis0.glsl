#version 450

// Concatenate two 2D tensors along axis 0 (rows)
// A: [rows_a, cols]
// B: [rows_b, cols]
// Result: [rows_a + rows_b, cols]

layout(local_size_x = 256) in;

layout(set = 0, binding = 0) readonly buffer InputA { float a[]; };
layout(set = 0, binding = 1) readonly buffer InputB { float b[]; };
layout(set = 0, binding = 2) writeonly buffer Output { float result[]; };
layout(set = 0, binding = 3) readonly buffer Params {
    uint rows_a;
    uint rows_b;
    uint cols;
};

void main() {
    uint gid = gl_GlobalInvocationID.x;
    uint total_rows = rows_a + rows_b;
    uint total_elements = total_rows * cols;
    
    if (gid >= total_elements) return;
    
    uint row = gid / cols;
    uint col = gid % cols;
    
    if (row < rows_a) {
        // Copy from A
        result[gid] = a[row * cols + col];
    } else {
        // Copy from B
        uint b_row = row - rows_a;
        result[gid] = b[b_row * cols + col];
    }
}
