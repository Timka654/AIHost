// Concatenate tensors along axis 0 (rows) for ROCm/HIP
// Concatenates A[rows_a, cols] and B[rows_b, cols] into C[rows_a + rows_b, cols]

extern "C" __global__ void concat_axis0(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ result,
    unsigned int rows_a,
    unsigned int rows_b,
    unsigned int cols)
{
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total_rows = rows_a + rows_b;
    unsigned int total_elements = total_rows * cols;
    
    if (gid >= total_elements)
        return;
    
    unsigned int row = gid / cols;
    unsigned int col = gid % cols;
    
    if (row < rows_a)
    {
        result[gid] = a[row * cols + col];
    }
    else
    {
        unsigned int b_row = row - rows_a;
        result[gid] = b[b_row * cols + col];
    }
}
