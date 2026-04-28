// Softmax: output[i] = exp(input[i] - max) / sum(exp(input - max))
extern "C" __global__ void softmax(
    const float* input,
    float* output,
    int n)
{
    __shared__ float shared_max;
    __shared__ float shared_sum;
    
    int idx = threadIdx.x;
    int stride = blockDim.x;
    
    // Find max (for numerical stability)
    float thread_max = -INFINITY;
    for (int i = idx; i < n; i += stride)
    {
        thread_max = fmaxf(thread_max, input[i]);
    }
    
    // Reduce max across block
    __shared__ float max_vals[256];
    max_vals[idx] = thread_max;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (idx < s)
            max_vals[idx] = fmaxf(max_vals[idx], max_vals[idx + s]);
        __syncthreads();
    }
    
    if (idx == 0)
        shared_max = max_vals[0];
    __syncthreads();
    
    // Compute exp and sum
    float thread_sum = 0.0f;
    for (int i = idx; i < n; i += stride)
    {
        float val = expf(input[i] - shared_max);
        output[i] = val;
        thread_sum += val;
    }
    
    // Reduce sum across block
    __shared__ float sum_vals[256];
    sum_vals[idx] = thread_sum;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (idx < s)
            sum_vals[idx] += sum_vals[idx + s];
        __syncthreads();
    }
    
    if (idx == 0)
        shared_sum = sum_vals[0];
    __syncthreads();
    
    // Normalize
    for (int i = idx; i < n; i += stride)
    {
        output[i] /= shared_sum;
    }
}

// Softmax for batched inputs (2D: [batch_size x seq_len])
extern "C" __global__ void softmax_batched(
    const float* input,
    float* output,
    int batch_size,
    int seq_len)
{
    int batch = blockIdx.x;
    if (batch >= batch_size) return;
    
    const float* in_ptr = input + batch * seq_len;
    float* out_ptr = output + batch * seq_len;
    
    __shared__ float shared_max;
    __shared__ float shared_sum;
    
    int idx = threadIdx.x;
    int stride = blockDim.x;
    
    // Find max
    float thread_max = -INFINITY;
    for (int i = idx; i < seq_len; i += stride)
    {
        thread_max = fmaxf(thread_max, in_ptr[i]);
    }
    
    __shared__ float max_vals[256];
    max_vals[idx] = thread_max;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (idx < s)
            max_vals[idx] = fmaxf(max_vals[idx], max_vals[idx + s]);
        __syncthreads();
    }
    
    if (idx == 0)
        shared_max = max_vals[0];
    __syncthreads();
    
    // Compute exp and sum
    float thread_sum = 0.0f;
    for (int i = idx; i < seq_len; i += stride)
    {
        float val = expf(in_ptr[i] - shared_max);
        out_ptr[i] = val;
        thread_sum += val;
    }
    
    __shared__ float sum_vals[256];
    sum_vals[idx] = thread_sum;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (idx < s)
            sum_vals[idx] += sum_vals[idx + s];
        __syncthreads();
    }
    
    if (idx == 0)
        shared_sum = sum_vals[0];
    __syncthreads();
    
    // Normalize
    for (int i = idx; i < seq_len; i += stride)
    {
        out_ptr[i] /= shared_sum;
    }
}
