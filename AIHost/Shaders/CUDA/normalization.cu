// RMS Normalization
extern "C" __global__ void rms_norm(
    const float* input,
    const float* weight,
    float* output,
    int n,
    float eps)
{
    int idx = threadIdx.x;
    int stride = blockDim.x;
    
    // Compute mean square
    float sum_sq = 0.0f;
    for (int i = idx; i < n; i += stride)
    {
        float val = input[i];
        sum_sq += val * val;
    }
    
    // Reduce across block
    __shared__ float shared_sum[256];
    shared_sum[idx] = sum_sq;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (idx < s)
            shared_sum[idx] += shared_sum[idx + s];
        __syncthreads();
    }
    
    __shared__ float rms;
    if (idx == 0)
    {
        float mean_sq = shared_sum[0] / n;
        rms = rsqrtf(mean_sq + eps);
    }
    __syncthreads();
    
    // Normalize and scale
    for (int i = idx; i < n; i += stride)
    {
        output[i] = input[i] * rms * weight[i];
    }
}

// Layer Normalization
extern "C" __global__ void layer_norm(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int n,
    float eps)
{
    int idx = threadIdx.x;
    int stride = blockDim.x;
    
    // Compute mean
    float sum = 0.0f;
    for (int i = idx; i < n; i += stride)
    {
        sum += input[i];
    }
    
    __shared__ float shared_mean[256];
    shared_mean[idx] = sum;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (idx < s)
            shared_mean[idx] += shared_mean[idx + s];
        __syncthreads();
    }
    
    __shared__ float mean;
    if (idx == 0)
        mean = shared_mean[0] / n;
    __syncthreads();
    
    // Compute variance
    float sum_sq = 0.0f;
    for (int i = idx; i < n; i += stride)
    {
        float diff = input[i] - mean;
        sum_sq += diff * diff;
    }
    
    __shared__ float shared_var[256];
    shared_var[idx] = sum_sq;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (idx < s)
            shared_var[idx] += shared_var[idx + s];
        __syncthreads();
    }
    
    __shared__ float inv_std;
    if (idx == 0)
    {
        float variance = shared_var[0] / n;
        inv_std = rsqrtf(variance + eps);
    }
    __syncthreads();
    
    // Normalize, scale and shift
    for (int i = idx; i < n; i += stride)
    {
        float normalized = (input[i] - mean) * inv_std;
        output[i] = normalized * weight[i] + bias[i];
    }
}
