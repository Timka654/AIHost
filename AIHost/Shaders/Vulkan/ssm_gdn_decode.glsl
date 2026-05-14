#version 450
layout(local_size_x = 128, local_size_y = 1, local_size_z = 1) in;
const uint CONV_KERNEL = 4u;
const uint CONV_STATE_LEN = 3u;
// SSM params UBO (binding=11): DMODEL, HVD, NVH, NKH, KD, VD, CD (7 uints, pad 1)
layout(set = 0, binding = 11) readonly buffer SsmP { uint DMODEL; uint HVD; uint NVH; uint NKH; uint KD; uint VD; uint CD; } smp;
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
layout(set = 0, binding = 10) readonly buffer RowIdxBuf   { uint rowIndex; } rowIdx;
float silu(float x) { return x / (1.0 + exp(-x)); }
void main() {
    uint n = gl_WorkGroupID.x;
    uint d = gl_LocalInvocationID.x;
    uint DM = smp.DMODEL;
    uint base = rowIdx.rowIndex * DM;
    uint qkvIdx = n * smp.HVD + d;
    // Conv1d: uses all workgroups (CD/HVD groups).
    float qkvVal = 0.0;
    float convOut = 0.0;
    if (qkvIdx < smp.CD) {
        for (uint k = 0u; k < DM; k++) qkvVal += xNorm.data[base + k] * wQKV.data[k + qkvIdx * DM];
        // convW GGUF shape [CONV_KERNEL=4, CD], column-major: data[slot + ch * CONV_KERNEL]
        uint cwOff = qkvIdx * CONV_KERNEL;
        convOut += convState.data[0u * smp.CD + qkvIdx] * convW.data[cwOff + 0u];
        convOut += convState.data[1u * smp.CD + qkvIdx] * convW.data[cwOff + 1u];
        convOut += convState.data[2u * smp.CD + qkvIdx] * convW.data[cwOff + 2u];
        convOut += qkvVal * convW.data[cwOff + 3u];
        scratch.data[qkvIdx] = silu(convOut);
    }
    // z / beta / alpha / gate: only for first NVH workgroups.
    if (n < smp.NVH) {
        float zVal = 0.0;
        for (uint k = 0u; k < DM; k++) zVal += xNorm.data[base + k] * wZ.data[k + qkvIdx * DM];
        scratch.data[smp.CD + qkvIdx] = zVal;
        if (d == 0u) {
            float betaSum = 0.0;
            for (uint k = 0u; k < DM; k++) betaSum += xNorm.data[base + k] * wBeta.data[k + n * DM];
            scratch.data[smp.CD + smp.VD + n] = 1.0 / (1.0 + exp(-betaSum));
            float alphaSum = 0.0;
            for (uint k = 0u; k < DM; k++) alphaSum += xNorm.data[base + k] * wAlpha.data[k + n * DM];
            // dtBias[48] and ssA[48] — one per v_head, direct index (not modulo NKH=16)
            float alpha = log(1.0 + exp(alphaSum + dtBias.data[n]));
            scratch.data[smp.CD + smp.VD + smp.NVH + n] = alpha * ssA.data[n];
        }
    }
    if (qkvIdx < smp.CD) {
        convState.data[0u * smp.CD + qkvIdx] = convState.data[1u * smp.CD + qkvIdx];
        convState.data[1u * smp.CD + qkvIdx] = convState.data[2u * smp.CD + qkvIdx];
        convState.data[2u * smp.CD + qkvIdx] = qkvVal;
    }
}
