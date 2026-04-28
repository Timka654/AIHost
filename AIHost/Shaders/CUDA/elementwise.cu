// Element-wise addition: C = A + B
extern "C" __global__ void add(
    const float* A,
    const float* B,
    float* C,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        C[idx] = A[idx] + B[idx];
    }
}

// Scalar addition: output = input + scalar
extern "C" __global__ void add_scalar(
    const float* input,
    float* output,
    float scalar,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        output[idx] = input[idx] + scalar;
    }
}

// Element-wise multiplication: C = A * B
extern "C" __global__ void mul(
    const float* A,
    const float* B,
    float* C,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        C[idx] = A[idx] * B[idx];
    }
}

// Scalar multiplication: output = input * scalar
extern "C" __global__ void mul_scalar(
    const float* input,
    float* output,
    float scalar,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        output[idx] = input[idx] * scalar;
    }
}
