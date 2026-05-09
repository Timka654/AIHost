#version 450

// Gated Delta Net (SSM) decode step for ONE token — PART 1: conv1d.
// Computes: z, beta, alpha, gate, qkv_mixed, conv1d, SiLU → writes to scratch.
//
// Dispatch: 48 workgroups (one per V-head), 128 threads each.
// gl_WorkGroupID.x = n (V-head index 0..47)
// gl_LocalInvocationID.x = d (dimension within head 0..127)
//
// Input buffers:
//   binding 0: xNorm [DMODEL] — normalized input for this token
//   binding 1: convW [CONV_KERNEL * CONV_DIM] — conv1d weights
//   binding 2: convState [CONV_STATE_LEN * CONV_DIM] — sliding window (3 frames)
//   binding 3: wQKV [DMODEL * CONV_DIM] — attn_qkv.weight
//   binding 4: wZ [DMODEL * VALUE_DIM] — attn_gate.weight (for z)
//   binding 5: wBeta [DMODEL * N_V_HEADS] — ssm_beta.weight
//   binding 6: wAlpha [DMODEL * N_V_HEADS] — ssm_alpha.weight
//   binding 7: dtBias [N_V_HEADS] — ssm_dt.bias
//   binding 8: ssA [N_V_HEADS] — ssm_a (A_NOSCAN)
//
// Output buffers:
//   binding 9: scratch [CONV_DIM + VALUE_DIM + N_V_HEADS*3] — scratch for cross-workgroup exchange
//     scratch[0..CONV_DIM-1] = SiLU(conv_out)
//     scratch[CONV_DIM..CONV_DIM+VALUE_DIM-1] = z
//     scratch[CONV_DIM+VALUE_DIM..CONV_DIM+VALUE_DIM+N_V_HEADS-1] = beta
//     scratch[CONV_DIM+VALUE_DIM+N_V_HEADS..CONV_DIM+VALUE_DIM+2*N_V_HEADS-1] = gate
//   binding 10: convState (in/out) — updated sliding window

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

layout(set = 0, binding = 0)  readonly  buffer XNormBuf     { float data[]; } xNorm;
layout(set = 0, binding = 1)  readonly  buffer ConvWBuf     { float data[]; } convW;
layout(set = 0, binding = 2)  coherent  buffer ConvStateBuf  { float data[]; } convState;
layout(set = 0, binding = 3)  readonly  buffer WQKVBuf       { float data[]; } wQKV;
layout(set = 0, binding = 4)  readonly  buffer WZBuf         { float data[]; } wZ;
layout(set = 0, binding = 5)  readonly  buffer WBetaBuf      { float data[]; } wBeta;
layout(set = 0, binding = 6)  readonly  buffer WAlphaBuf     { float data[]; } wAlpha;
layout(set = 0, binding = 7)  readonly  buffer DtBiasBuf     { float data[]; } dtBias;
layout(set = 0, binding = 8)  readonly  buffer SsABuf        { float data[]; } ssA;
layout(set = 0, binding = 9)  coherent  buffer ScratchBuf    { float data[]; } scratch;
// convState (binding 2) is updated in-place

float silu(float x) { return x / (1.0 + exp(-x)); }

void main() {
    uint n = gl_WorkGroupID.x;
    uint d = gl_LocalInvocationID.x;
    uint qkvIdx = n * HEAD_V_DIM + d;

    // ── Step 1: z[n*128 + d] = xNorm @ wZ[:, n*128 + d] ────────────────────
    float zVal = 0.0;
    for (uint k = 0u; k < DMODEL; k++)
        zVal += xNorm.data[k] * wZ.data[k + qkvIdx * DMODEL];

    // ── Step 2: beta[n] = sigmoid(xNorm @ wBeta[:, n]) ──────────────────────
    float betaSum = 0.0;
    for (uint k = 0u; k < DMODEL; k++)
        betaSum += xNorm.data[k] * wBeta.data[k + n * DMODEL];
    float beta = 1.0 / (1.0 + exp(-betaSum));

    // ── Step 3: alpha[n] = softplus(xNorm @ wAlpha[:, n] + dtBias[n]) ───────
    float alphaSum = 0.0;
    for (uint k = 0u; k < DMODEL; k++)
        alphaSum += xNorm.data[k] * wAlpha.data[k + n * DMODEL];
    float alpha = log(1.0 + exp(alphaSum + dtBias.data[n]));

    // ── Step 4: gate[n] = alpha * ssA[n] ────────────────────────────────────
    float gate = alpha * ssA.data[n];

    // ── Step 5: qkv_mixed = xNorm @ wQKV → [CONV_DIM] ──────────────────────
    float qkvVal = 0.0;
    if (qkvIdx < CONV_DIM) {
        for (uint k = 0u; k < DMODEL; k++)
            qkvVal += xNorm.data[k] * wQKV.data[k + qkvIdx * DMODEL];
    }

    // ── Conv1d ──────────────────────────────────────────────────────────────
    // convW is GGUF column-major [CONV_KERNEL=4, CONV_DIM=10240]
    // Element (k,c) at index k + c * CONV_KERNEL
    float convOut = 0.0;
    if (qkvIdx < CONV_DIM) {
        convOut += convState.data[0u * CONV_DIM + qkvIdx] * convW.data[0u + qkvIdx * CONV_KERNEL];
        convOut += convState.data[1u * CONV_DIM + qkvIdx] * convW.data[1u + qkvIdx * CONV_KERNEL];
        convOut += convState.data[2u * CONV_DIM + qkvIdx] * convW.data[2u + qkvIdx * CONV_KERNEL];
        convOut += qkvVal * convW.data[3u + qkvIdx * CONV_KERNEL];
    }

    // ── SiLU ────────────────────────────────────────────────────────────────
    float siluVal = silu(convOut);

    // ── Write to scratch ────────────────────────────────────────────────────
    if (qkvIdx < CONV_DIM)
        scratch.data[qkvIdx] = siluVal;  // SiLU(conv_out)
    scratch.data[CONV_DIM + qkvIdx] = zVal;  // z
    if (d == 0u) {
        scratch.data[CONV_DIM + VALUE_DIM + n] = beta;  // beta
        scratch.data[CONV_DIM + VALUE_DIM + N_V_HEADS + n] = gate;  // gate
    }

    // ── Update conv state ───────────────────────────────────────────────────
    if (qkvIdx < CONV_DIM) {
        convState.data[0u * CONV_DIM + qkvIdx] = convState.data[1u * CONV_DIM + qkvIdx];
        convState.data[1u * CONV_DIM + qkvIdx] = convState.data[2u * CONV_DIM + qkvIdx];
        convState.data[2u * CONV_DIM + qkvIdx] = qkvVal;
    }
}
