// Matrix transpose kernel for ROCm/HIP
// B[cols, rows] = A[rows, cols]^T

extern "C" __global__ void transpose(
    const float* __restrict__ A,
    float* __restrict__ B,
    unsigned int rows,
    unsigned int cols)
{
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row >= rows || col >= cols)
        return;
    
    B[col * rows + row] = A[row * cols + col];
}
