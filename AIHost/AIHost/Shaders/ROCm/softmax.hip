// Softmax kernel for ROCm/HIP
// Computes softmax in-place: exp(x - max(x)) / sum(exp(x - max(x)))

extern "C" __global__ void softmax(
    float* __restrict__ buf,
    unsigned int size)
{
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int lid = threadIdx.x;

    // Shared memory for reduction
    __shared__ float sMax[256];
    __shared__ float sSum[256];

    // Load and find max
    float val = (gid < size) ? buf[gid] : -1e38f;
    sMax[lid] = val;
    __syncthreads();

    // Reduce to find maximum
    for (unsigned int s = 128; s > 0; s >>= 1)
    {
        if (lid < s)
            sMax[lid] = fmaxf(sMax[lid], sMax[lid + s]);
        __syncthreads();
    }

    float maxVal = sMax[0];

    // Compute exp and sum
    if (gid < size)
    {
        val = expf(buf[gid] - maxVal);
        buf[gid] = val;
    }
    else
    {
        val = 0.0f;
    }

    sSum[lid] = val;
    __syncthreads();

    // Reduce to find sum
    for (unsigned int s = 128; s > 0; s >>= 1)
    {
        if (lid < s)
            sSum[lid] += sSum[lid + s];
        __syncthreads();
    }

    float sumVal = sSum[0];

    // Normalize
    if (gid < size && sumVal > 0.0f)
    {
        buf[gid] /= sumVal;
    }
}
