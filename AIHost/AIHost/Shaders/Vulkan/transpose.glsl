#version 450
layout(local_size_x = 16, local_size_y = 16) in;

layout(set = 0, binding = 0) readonly buffer MatrixA { float data[]; } A;
layout(set = 0, binding = 1) buffer MatrixB { float data[]; } B;
layout(set = 0, binding = 2) readonly buffer Params { uint rows; uint cols; } params;

void main() {
    uint row = gl_GlobalInvocationID.y;
    uint col = gl_GlobalInvocationID.x;
    
    if (row >= params.rows || col >= params.cols) return;
    
    B.data[col * params.rows + row] = A.data[row * params.cols + col];
}
