// SiLU (Swish) activation: output = x * sigmoid(x)
extern "C" __global__ void silu(
    const float* input,
    float* output,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        float x = input[idx];
        output[idx] = x / (1.0f + expf(-x));
    }
}

// ReLU activation: output = max(0, x)
extern "C" __global__ void relu(
    const float* input,
    float* output,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        output[idx] = fmaxf(0.0f, input[idx]);
    }
}

// GELU activation (approximation)
extern "C" __global__ void gelu(
    const float* input,
    float* output,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        float x = input[idx];
        // GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
        float x3 = x * x * x;
        float inner = 0.7978845608f * (x + 0.044715f * x3);
        output[idx] = 0.5f * x * (1.0f + tanhf(inner));
    }
}
