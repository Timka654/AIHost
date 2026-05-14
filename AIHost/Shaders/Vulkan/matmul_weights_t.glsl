#version 450
layout(local_size_x = 16, local_size_y = 16) in;
layout(set = 0, binding = 0) readonly buffer MatrixA { float data[]; } A;
layout(set = 0, binding = 1) readonly buffer MatrixB { float data[]; } B;
layout(set = 0, binding = 2) buffer MatrixC          { float data[]; } C;
layout(set = 0, binding = 3) readonly buffer Params  { uint M; uint K; uint J; } params;
shared float tileA[16][16];
shared float tileB[16][16];
void main() {
    uint row     = gl_GlobalInvocationID.y;  // output row  (0..M-1)
    uint col     = gl_GlobalInvocationID.x;  // output col  (0..J-1)
    uint tileRow = gl_LocalInvocationID.y;
    uint tileCol = gl_LocalInvocationID.x;
    float sum    = 0.0;
    uint numTiles = (params.K + 15u) / 16u;
    for (uint t = 0u; t < numTiles; t++) {
        uint aRow = row;
        uint aCol = t * 16u + tileCol;  // k-index in A
        uint bCol = t * 16u + tileRow;  // k-index in B
        uint bRow = col;                // j-index in B (B[j,k]=data[j+k*J])
        tileA[tileRow][tileCol] = (aRow < params.M && aCol < params.K)
            ? A.data[aRow * params.K + aCol] : 0.0;
        // B^T: B[j,k] = data[j + k*J]
        tileB[tileRow][tileCol] = (bRow < params.J && bCol < params.K)
            ? B.data[bRow + bCol * params.J] : 0.0;
        barrier();
        for (uint k = 0u; k < 16u; k++) {
            sum += tileA[tileRow][k] * tileB[k][tileCol];
        }
        barrier();
    }
    if (row < params.M && col < params.J) {
        C.data[row * params.J + col] = sum;
    }
}