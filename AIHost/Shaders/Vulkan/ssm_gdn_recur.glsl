#version 450

// Gated Delta Net (SSM) decode step for ONE token — PART 2: recurrence.
// Reads from scratch (written by ssm_gdn_decode), computes L2 norm, Delta Net, Gated norm.
//
// Dispatch: 48 workgroups (one per V-head), 128 threads each.
// gl_WorkGroupID.x = n (V-head index 0..47)
// gl_LocalInvocationID.x = d (dimension within head 0..127)
//
// Input buffers:
//   binding 0: scratch [CONV_DIM + VALUE_DIM + N_V_HEADS*3] — from part 1
//     scratch[0..CONV_DIM-1] = SiLU(conv_out)
//     scratch[CONV_DIM..CONV_DIM+VALUE_DIM-1] = z
//     scratch[CONV_DIM+VALUE_DIM..CONV_DIM+VALUE_DIM+N_V_HEADS-1] = beta
//     scratch[CONV_DIM+VALUE_DIM+N_V_HEADS..CONV_DIM+VALUE_DIM+2*N_V_HEADS-1] = gate
//   binding 1: ssmState [HEAD_V_DIM * HEAD_V_DIM * N_V_HEADS] — recurrent state S
//   binding 2: ssmNorm [HEAD_V_DIM] — per-group RMSNorm weight
//
// Output buffers:
//   binding 3: output [VALUE_DIM] — SSM output for this token
//   binding 4: ssmState (in/out) — updated recurrent state

layout(local_size_x = 128, local_size_y = 1, local_size_z = 1) in;

const uint HEAD_V_DIM = 128u;
const uint N_V_HEADS  = 48u;
const uint N_K_HEADS  = 16u;
const uint KEY_DIM    = 2048u;
const uint VALUE_DIM  = 6144u;
const uint CONV_DIM   = 10240u;
const float EPS       = 1e-5;
const float SCALE     = 0.08838834764831845; // 1/sqrt(128)

layout(set = 0, binding = 0) readonly  buffer ScratchBuf   { float data[]; } scratch;
layout(set = 0, binding = 1) coherent buffer SsmStateBuf   { float data[]; } ssmState;
layout(set = 0, binding = 2) readonly  buffer SsmNormBuf   { float data[]; } ssmNorm;
layout(set = 0, binding = 3) writeonly buffer OutputBuf    { float data[]; } output;

shared float sharedMem[128];

float silu(float x) { return x / (1.0 + exp(-x)); }

void main() {
    uint n = gl_WorkGroupID.x;
    uint d = gl_LocalInvocationID.x;
    uint h = n % N_K_HEADS;

    // ── Read from scratch ───────────────────────────────────────────────────
    uint qkvIdx = n * HEAD_V_DIM + d;
    float zVal = scratch.data[CONV_DIM + qkvIdx];
    float beta = scratch.data[CONV_DIM + VALUE_DIM + n];
    float gate = scratch.data[CONV_DIM + VALUE_DIM + N_V_HEADS + n];

    // Read q_conv at key head h position d
    uint qkIdx = h * HEAD_V_DIM + d;
    float qConvVal = 0.0;
    float kConvVal = 0.0;
    if (qkIdx < KEY_DIM) {
        qConvVal = scratch.data[qkIdx];
        kConvVal = scratch.data[KEY_DIM + qkIdx];
    }

    // Read v_conv at our V-head n position d
    float vConvVal = scratch.data[2u * KEY_DIM + qkvIdx];

    // ── Step 7: L2 norm q_conv, k_conv ──────────────────────────────────────
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

    float normQ = sqrt(sumSqQ + EPS);
    float normK = sqrt(sumSqK + EPS);
    float qNorm = qConvVal / normQ;
    float kNorm = kConvVal / normK;

    // ── Step 8: Delta Net autoregressive ────────────────────────────────────
    uint stateBase = n * HEAD_V_DIM * HEAD_V_DIM;

    // 8a. Decay state: s[:,:,n] *= exp(gate[n])
    float gExp = exp(gate);
    for (uint i = d; i < HEAD_V_DIM; i += 128u) {
        uint rowBase = stateBase + i * HEAD_V_DIM;
        for (uint j = 0u; j < HEAD_V_DIM; j++)
            ssmState.data[rowBase + j] *= gExp;
    }
    barrier();

    // 8b. Broadcast kNorm, compute sk[d]
    sharedMem[d] = kNorm;
    barrier();
    float sk = 0.0;
    uint rowBaseD = stateBase + d * HEAD_V_DIM;
    for (uint j = 0u; j < HEAD_V_DIM; j++)
        sk += ssmState.data[rowBaseD + j] * sharedMem[j];
    barrier();

    // 8c. dVec[d] = (vConvVal - sk) * beta
    float dVec = (vConvVal - sk) * beta;

    // 8d. s[d, :, n] += kNorm[d] * dVec[:]
    sharedMem[d] = dVec;
    barrier();
    for (uint j = 0u; j < HEAD_V_DIM; j++)
        ssmState.data[rowBaseD + j] += kNorm * sharedMem[j];
    barrier();

    // 8e. Broadcast qNorm, compute output
    sharedMem[d] = qNorm;
    barrier();
    float oVal = 0.0;
    for (uint j = 0u; j < HEAD_V_DIM; j++)
        oVal += ssmState.data[rowBaseD + j] * sharedMem[j];
    oVal *= SCALE;

    // ── Step 9: Gated norm ──────────────────────────────────────────────────
    float zSilu = silu(zVal);
    sharedMem[d] = oVal * oVal;
    barrier();
    for (uint s = 64u; s > 0u; s >>= 1u) {
        if (d < s) sharedMem[d] += sharedMem[d + s];
        barrier();
    }
    float sumSq = sharedMem[0];
    barrier();
    float rms = sqrt(sumSq / float(HEAD_V_DIM) + EPS);
    float rmsInv = 1.0 / rms;
    float yVal = oVal * rmsInv * ssmNorm.data[d] * zSilu;

    // ── Write output ────────────────────────────────────────────────────────
    output.data[qkvIdx] = yVal;
}
