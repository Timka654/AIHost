#version 450
layout(local_size_x = 128, local_size_y = 1, local_size_z = 1) in;
const uint HEAD_V_DIM = 128u;
const uint N_V_HEADS  = 48u;
const uint N_K_HEADS  = 16u;
const uint KEY_DIM    = 2048u;
const uint VALUE_DIM  = 6144u;
const uint CONV_DIM   = 10240u;
const uint CONV_KERNEL = 4u;
const uint CONV_STATE_LEN = 3u;
const uint DMODEL     = 5120u;
layout(set = 0, binding = 0) readonly  buffer XNormBuf    { float data[]; } xNorm;
layout(set = 0, binding = 1) readonly  buffer ConvWBuf    { float data[]; } convW;
layout(set = 0, binding = 2)          buffer ConvStateBuf { float data[]; } convState;
layout(set = 0, binding = 3) readonly  buffer WQKVBuf     { float data[]; } wQKV;
layout(set = 0, binding = 4) readonly  buffer WZBuf       { float data[]; } wZ;
layout(set = 0, binding = 5) readonly  buffer WBetaBuf    { float data[]; } wBeta;
layout(set = 0, binding = 6) readonly  buffer WAlphaBuf   { float data[]; } wAlpha;
layout(set = 0, binding = 7) readonly  buffer DtBiasBuf   { float data[]; } dtBias;
layout(set = 0, binding = 8) readonly  buffer SsABuf      { float data[]; } ssA;
layout(set = 0, binding = 9)          buffer ScratchBuf   { float data[]; } scratch;
float silu(float x) { return x / (1.0 + exp(-x)); }
void main() {
    uint n = gl_WorkGroupID.x;
    uint d = gl_LocalInvocationID.x;
    uint qkvIdx = n * HEAD_V_DIM + d;
    float zVal = 0.0;
    for (uint k = 0u; k < DMODEL; k++) zVal += xNorm.data[k] * wZ.data[k + qkvIdx * DMODEL];
    float betaSum = 0.0;
    for (uint k = 0u; k < DMODEL; k++) betaSum += xNorm.data[k] * wBeta.data[k + n * DMODEL];
    float beta = 1.0 / (1.0 + exp(-betaSum));
    float alphaSum = 0.0;
    for (uint k = 0u; k < DMODEL; k++) alphaSum += xNorm.data[k] * wAlpha.data[k + n * DMODEL];
    float alpha = log(1.0 + exp(alphaSum + dtBias.data[n]));
    float gate = alpha * ssA.data[n];
    float qkvVal = 0.0;
    if (qkvIdx < CONV_DIM) {
        for (uint k = 0u; k < DMODEL; k++) qkvVal += xNorm.data[k] * wQKV.data[k + qkvIdx * DMODEL];
    }
    float convOut = 0.0;
    if (qkvIdx < CONV_DIM) {
        convOut += convState.data[0u * CONV_DIM + qkvIdx] * convW.data[0u * CONV_DIM + qkvIdx];
        convOut += convState.data[1u * CONV_DIM + qkvIdx] * convW.data[1u * CONV_DIM + qkvIdx];
        convOut += convState.data[2u * CONV_DIM + qkvIdx] * convW.data[2u * CONV_DIM + qkvIdx];
        convOut += qkvVal * convW.data[3u * CONV_DIM + qkvIdx];
    }
    float siluVal = silu(convOut);
    if (qkvIdx < CONV_DIM) scratch.data[qkvIdx] = siluVal;
    scratch.data[CONV_DIM + qkvIdx] = zVal;
    if (d == 0u) {
        scratch.data[CONV_DIM + VALUE_DIM + n] = beta;
        scratch.data[CONV_DIM + VALUE_DIM + N_V_HEADS + n] = gate;
    }
    if (qkvIdx < CONV_DIM) {
        convState.data[0u * CONV_DIM + qkvIdx] = convState.data[1u * CONV_DIM + qkvIdx];
        convState.data[1u * CONV_DIM + qkvIdx] = convState.data[2u * CONV_DIM + qkvIdx];
        convState.data[2u * CONV_DIM + qkvIdx] = qkvVal;
    }
}
