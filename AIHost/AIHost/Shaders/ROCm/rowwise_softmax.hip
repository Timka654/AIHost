// Row-wise softmax kernel for ROCm/HIP
// Each block processes one row

extern "C" __global__ void rowwise_softmax(
    float* __restrict__ buf,
    unsigned int rows,
    unsigned int cols)
{
    unsigned int row = blockIdx.x;
    unsigned int tid = threadIdx.x;
    
    if (row >= rows)
        return;
    
    __shared__ float sharedMem[256];
    unsigned int rowOffset = row * cols;
    
    // Find max
    float localMax = -1e38f;
    for (unsigned int i = tid; i < cols; i += blockDim.x)
    {
        localMax = fmaxf(localMax, buf[rowOffset + i]);
    }
    sharedMem[tid] = localMax;
    __syncthreads();
    
    // Reduce max
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            sharedMem[tid] = fmaxf(sharedMem[tid], sharedMem[tid + s]);
        }
        __syncthreads();
    }
    float rowMax = sharedMem[0];
    __syncthreads();
    
    // Compute exp and sum
    float localSum = 0.0f;
    for (unsigned int i = tid; i < cols; i += blockDim.x)
    {
        float val = expf(buf[rowOffset + i] - rowMax);
        buf[rowOffset + i] = val;
        localSum += val;
    }
    sharedMem[tid] = localSum;
    __syncthreads();
    
    // Reduce sum
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            sharedMem[tid] += sharedMem[tid + s];
        }
        __syncthreads();
    }
    float rowSum = sharedMem[0];
    __syncthreads();
    
    // Normalize
    for (unsigned int i = tid; i < cols; i += blockDim.x)
    {
        buf[rowOffset + i] /= rowSum;
    }
}
