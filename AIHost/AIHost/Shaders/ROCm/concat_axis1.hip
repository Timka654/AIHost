// Concatenate tensors along axis 1 (columns) for ROCm/HIP
// Concatenates A[dim0, dim1_a] and B[dim0, dim1_b] into C[dim0, dim1_a + dim1_b]

extern "C" __global__ void concat_axis1(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    unsigned int dim0,
    unsigned int dim1_a,
    unsigned int dim1_b)
{
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int dim1_total = dim1_a + dim1_b;
    unsigned int total_size = dim0 * dim1_total;
    
    if (gid >= total_size)
        return;
    
    unsigned int i = gid / dim1_total;  // which row
    unsigned int j = gid % dim1_total;  // position in row
    
    if (j < dim1_a)
    {
        // Copy from A
        C[gid] = A[i * dim1_a + j];
    }
    else
    {
        // Copy from B
        C[gid] = B[i * dim1_b + (j - dim1_a)];
    }
}
