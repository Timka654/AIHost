// Matrix multiplication: C = A * B
// A: [M x K], B: [K x N], C: [M x N]
extern "C" __global__ void matmul(
    const float* A,
    const float* B,
    float* C,
    int M, int K, int N)
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

// Optimized matmul with shared memory
extern "C" __global__ void matmul_shared(
    const float* A,
    const float* B,
    float* C,
    int M, int K, int N)
{
    __shared__ float tileA[16][16];
    __shared__ float tileB[16][16];
    
    int row = blockIdx.y * 16 + threadIdx.y;
    int col = blockIdx.x * 16 + threadIdx.x;
    
    float sum = 0.0f;
    
    for (int t = 0; t < (K + 15) / 16; t++)
    {
        if (row < M && t * 16 + threadIdx.x < K)
            tileA[threadIdx.y][threadIdx.x] = A[row * K + t * 16 + threadIdx.x];
        else
            tileA[threadIdx.y][threadIdx.x] = 0.0f;
            
        if (col < N && t * 16 + threadIdx.y < K)
            tileB[threadIdx.y][threadIdx.x] = B[(t * 16 + threadIdx.y) * N + col];
        else
            tileB[threadIdx.y][threadIdx.x] = 0.0f;
            
        __syncthreads();
        
        for (int k = 0; k < 16; k++)
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
            
        __syncthreads();
    }
    
    if (row < M && col < N)
        C[row * N + col] = sum;
}
