#version 450
layout(local_size_x = 128, local_size_y = 1, local_size_z = 1) in;
const float EPS   = 1e-5;
const float SCALE = 0.08838834764831845;
// SSM params UBO (binding=4): DMODEL, HVD, NVH, NKH, KD, VD, CD (7 uints, pad 1)
layout(set = 0, binding = 4) readonly buffer SsmP { uint DMODEL; uint HVD; uint NVH; uint NKH; uint KD; uint VD; uint CD; } smp;
layout(set = 0, binding = 0) readonly  buffer ScratchBuf { float data[]; } scratch;
layout(set = 0, binding = 1)          buffer SsmStateBuf { float data[]; } ssmState;
layout(set = 0, binding = 2) readonly  buffer SsmNormBuf { float data[]; } ssmNorm;
layout(set = 0, binding = 3) writeonly buffer OutputBuf  { float data[]; } outBuf;
shared float sharedMem[128];
float silu(float x) { return x / (1.0 + exp(-x)); }
void main() {
    uint n = gl_WorkGroupID.x;
    uint d = gl_LocalInvocationID.x;
    uint h = n / (smp.NVH / smp.NKH);   // repeat-interleave: v_head n -> k_head n/3
    uint qkvIdx = n * smp.HVD + d;
    uint qkIdx = h * smp.HVD + d;
    float zVal  = (qkvIdx < smp.VD)
        ? scratch.data[smp.CD + qkvIdx] : 0.0;
    float beta  = (n < smp.NVH)
        ? scratch.data[smp.CD + smp.VD + n] : 0.0;
    float gate  = (n < smp.NVH)
        ? scratch.data[smp.CD + smp.VD + smp.NVH + n] : 0.0;
    float qConvVal = (qkIdx < smp.KD) ? scratch.data[qkIdx] : 0.0;
    float kConvVal = (qkIdx < smp.KD) ? scratch.data[smp.KD + qkIdx] : 0.0;
    float vConvVal = (qkvIdx < smp.VD) ? scratch.data[2u * smp.KD + qkvIdx] : 0.0;
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
    uint stateBase = n * smp.HVD * smp.HVD;
    // Step 1: decay state S *= gExp.
    float gExp = exp(gate);
    for (uint i = d; i < smp.HVD; i += 128u) {
        uint rowBase = stateBase + i * smp.HVD;
        for (uint j = 0u; j < smp.HVD; j++) ssmState.data[rowBase + j] *= gExp;
    }
    barrier();
    // Step 2: compute sk = S_decayed @ k  (row d × cols j).
    sharedMem[d] = kNorm;
    barrier();
    float sk = 0.0;
    uint sdRowBase = stateBase + d * smp.HVD;
    for (uint j = 0u; j < smp.HVD; j++) sk += ssmState.data[sdRowBase + j] * sharedMem[j];
    barrier();
    float dVec = (vConvVal - sk) * beta;
    sharedMem[d] = dVec;
    barrier();
    uint rowBaseD = stateBase + d * smp.HVD;
    for (uint j = 0u; j < smp.HVD; j++) ssmState.data[rowBaseD + j] += kNorm * sharedMem[j];
    barrier();
    sharedMem[d] = qNorm;
    barrier();
    float oVal = 0.0;
    uint soRowBase = stateBase + d * smp.HVD;
    for (uint j = 0u; j < smp.HVD; j++) oVal += ssmState.data[soRowBase + j] * sharedMem[j];
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
    float rmsInv = 1.0 / sqrt(sumSq / float(smp.HVD) + EPS);
    // ssmNorm is [HVD] = [128], applied per-head (d is the intra-head index).
    float yVal = oVal * rmsInv * ssmNorm.data[d] * zSilu;
    outBuf.data[qkvIdx] = yVal;
}
