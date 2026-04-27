#version 450
layout(local_size_x = 16, local_size_y = 16) in;

layout(set = 0, binding = 0) readonly buffer MatrixA { float data[]; } A;
layout(set = 0, binding = 1) readonly buffer MatrixB { float data[]; } B;
layout(set = 0, binding = 2) buffer MatrixC { float data[]; } C;
layout(set = 0, binding = 3) readonly buffer Params { uint M; uint K; uint N; } params;

shared float tileA[16][16];
shared float tileB[16][16];

void main() {
    uint row = gl_GlobalInvocationID.y;
    uint col = gl_GlobalInvocationID.x;
    uint tileRow = gl_LocalInvocationID.y;
    uint tileCol = gl_LocalInvocationID.x;

    float sum = 0.0;
    uint numTiles = (params.K + 15u) / 16u;

    for (uint t = 0u; t < numTiles; t++) {
        uint aRow = row;
        uint aCol = t * 16u + tileCol;
        uint bRow = t * 16u + tileRow;
        uint bCol = col;

        // All threads must participate in shared memory loads.
        // Out-of-bounds threads write 0.0 so they don't corrupt the tile.
        tileA[tileRow][tileCol] = (aRow < params.M && aCol < params.K) ? A.data[aRow * params.K + aCol] : 0.0;
        tileB[tileRow][tileCol] = (bRow < params.K && bCol < params.N) ? B.data[bRow * params.N + bCol] : 0.0;

        barrier();

        for (uint k = 0u; k < 16u; k++) {
            sum += tileA[tileRow][k] * tileB[k][tileCol];
        }

        barrier();
    }

    if (row < params.M && col < params.N) {
        C.data[row * params.N + col] = sum;
    }
}
