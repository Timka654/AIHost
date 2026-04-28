// Embedding lookup kernel for ROCm/HIP
// output[seq, dim] = table[tokenIds[seq], dim]

extern "C" __global__ void embedding_lookup(
    const int* __restrict__ tokenIds,
    const float* __restrict__ table,
    float* __restrict__ output_buf,
    unsigned int seqLen,
    unsigned int dModel)
{
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (gid >= seqLen * dModel)
        return;
    
    unsigned int seq = gid / dModel;
    unsigned int dim = gid % dModel;
    
    int tokenId = tokenIds[seq];
    if (tokenId < 0)
        return;
    
    output_buf[gid] = table[tokenId * dModel + dim];
}
