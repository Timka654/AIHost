// Element-wise addition kernel for ROCm/HIP
// C[i] = A[i] + B[i]

extern "C" __global__ void add(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    unsigned int size)
{
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (gid >= size)
        return;

    C[gid] = A[gid] + B[gid];
}
