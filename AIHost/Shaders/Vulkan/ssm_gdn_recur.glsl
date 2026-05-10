#version 450
layout(local_size_x = 128, local_size_y = 1, local_size_z = 1) in;
const uint HEAD_V_DIM = 128u;
const uint N_V_HEADS  = 48u;
const uint N_K_HEADS  = 16u;
const uint KEY_DIM    = 2048u;
const uint VALUE_DIM  = 6144u;
const uint CONV_DIM   = 10240u;
const float EPS       = 1e-5;
const float SCALE     = 0.08838834764831845;
layout(set = 0, binding = 0) readonly  buffer ScratchBuf { float data[]; } scratch;
layout(set = 0, binding = 1)          buffer SsmStateBuf { float data[]; } ssmState;
layout(set = 0, binding = 2) readonly  buffer SsmNormBuf { float data[]; } ssmNorm;
layout(set = 0, binding = 3) writeonly buffer OutputBuf  { float data[]; } outBuf;
shared float sharedMem[128];
float silu(float x) { return x / (1.0 + exp(-x)); }
void main() {
    uint n = gl_WorkGroupID.x;
    uint d = gl_LocalInvocationID.x;
    uint h = n % N_K_HEADS;
    uint qkvIdx = n * HEAD_V_DIM + d;
    float zVal = scratch.data[CONV_DIM + qkvIdx];
    float beta = scratch.data[CONV_DIM + VALUE_DIM + n];
    float gate = scratch.data[CONV_DIM + VALUE_DIM + N_V_HEADS + n];
    uint qkIdx = h * HEAD_V_DIM + d;
    float qConvVal = 0.0;
    float kConvVal = 0.0;
    if (qkIdx < KEY_DIM) {
        qConvVal = scratch.data[qkIdx];
        kConvVal = scratch.data[KEY_DIM + qkIdx];
    }
    float vConvVal = scratch.data[2u * KEY_DIM + qkvIdx];
    float localSumSqQ = qConvVal * qConvVal;
    float localSumSqK = kConvVal * kConvVal;
    sharedMem[d] = localSumSqQ;
    barrier();
    for (uint s = 64u; s > 0u; s >>= 1u) {
        if (d < s) sharedMem[d] += sharedMem[d + s];
        barrier();
    }
    float sumSqQ = sharedMem[0];
    barrier();
    sharedMem[d] = localSumSqK;
    barrier();
    for (uint s = 64u; s > 0u; s >>= 1u) {
        if (d < s) sharedMem[d] += sharedMem[d + s];
        barrier();
    }
    float sumSqK = sharedMem[0];
    barrier();
    float qNorm = qConvVal / sqrt(sumSqQ + EPS);
    float kNorm = kConvVal / sqrt(sumSqK + EPS);
    uint stateBase = n * HEAD_V_DIM * HEAD_V_DIM;
    float gExp = exp(gate);
    for (uint i = d; i < HEAD_V_DIM; i += 128u) {
        uint rowBase = stateBase + i * HEAD_V_DIM;
        for (uint j = 0u; j < HEAD_V_DIM; j++) ssmState.data[rowBase + j] *= gExp;
    }
    barrier();
    sharedMem[d] = kNorm;
    barrier();
    float sk = 0.0;
    for (uint i = 0u; i < HEAD_V_DIM; i++) sk += ssmState.data[stateBase + i * HEAD_V_DIM + d] * sharedMem[i];
    barrier();
    float dVec = (vConvVal - sk) * beta;
    sharedMem[d] = dVec;
    barrier();
    uint rowBaseD = stateBase + d * HEAD_V_DIM;
    for (uint j = 0u; j < HEAD_V_DIM; j++) ssmState.data[rowBaseD + j] += kNorm * sharedMem[j];
    barrier();
    sharedMem[d] = qNorm;
    barrier();
    float oVal = 0.0;
    for (uint i = 0u; i < HEAD_V_DIM; i++) oVal += ssmState.data[stateBase + i * HEAD_V_DIM + d] * sharedMem[i];
    oVal *= SCALE;
    float zSilu = silu(zVal);
    sharedMem[d] = oVal * oVal;
    barrier();
    for (uint s = 64u; s > 0u; s >>= 1u) {
        if (d < s) sharedMem[d] += sharedMem[d + s];
        barrier();
    }
    float sumSq = sharedMem[0];
    barrier();
    float rmsInv = 1.0 / sqrt(sumSq / float(HEAD_V_DIM) + EPS);
    float yVal = oVal * rmsInv * ssmNorm.data[d] * zSilu;
    outBuf.data[qkvIdx] = yVal;
}
