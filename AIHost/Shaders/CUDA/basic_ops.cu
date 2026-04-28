// Simple vector addition kernel for testing
extern "C" __global__ void vector_add(const float* a, const float* b, float* c, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        c[idx] = a[idx] + b[idx];
    }
}

// Multiply vector by scalar
extern "C" __global__ void vector_scale(const float* input, float* output, float scale, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        output[idx] = input[idx] * scale;
    }
}

// Matrix multiplication kernel (simplified)
extern "C" __global__ void matmul(const float* A, const float* B, float* C, int M, int K, int N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N)
    {
        float sum = 0.0f;
        for (int k = 0; k < K; k++)
        {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}
