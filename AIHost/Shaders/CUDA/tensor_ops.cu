// Concatenate two tensors along last dimension
extern "C" __global__ void concat(
    const float* A,
    const float* B,
    float* C,
    int size_a,
    int size_b)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = size_a + size_b;
    
    if (idx < total)
    {
        if (idx < size_a)
            C[idx] = A[idx];
        else
            C[idx] = B[idx - size_a];
    }
}

// Rope (Rotary Position Embedding)
extern "C" __global__ void rope(
    float* input,
    int seq_len,
    int head_dim,
    int position_offset)
{
    int pos = blockIdx.x;
    int dim = threadIdx.x * 2;
    
    if (pos < seq_len && dim + 1 < head_dim)
    {
        int idx = pos * head_dim + dim;
        float freq = 1.0f / powf(10000.0f, (float)dim / (float)head_dim);
        float angle = (pos + position_offset) * freq;
        
        float cos_val = cosf(angle);
        float sin_val = sinf(angle);
        
        float x0 = input[idx];
        float x1 = input[idx + 1];
        
        input[idx] = x0 * cos_val - x1 * sin_val;
        input[idx + 1] = x0 * sin_val + x1 * cos_val;
    }
}

// Transpose matrix: B = A^T (A is [M x N], B is [N x M])
extern "C" __global__ void transpose(
    const float* A,
    float* B,
    int M,
    int N)
{
    __shared__ float tile[32][33]; // +1 to avoid bank conflicts
    
    int x = blockIdx.x * 32 + threadIdx.x;
    int y = blockIdx.y * 32 + threadIdx.y;
    
    if (x < N && y < M)
        tile[threadIdx.y][threadIdx.x] = A[y * N + x];
    
    __syncthreads();
    
    x = blockIdx.y * 32 + threadIdx.x;
    y = blockIdx.x * 32 + threadIdx.y;
    
    if (x < M && y < N)
        B[y * M + x] = tile[threadIdx.x][threadIdx.y];
}
