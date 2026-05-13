#version 450
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) readonly  buffer InBuf     { float data[]; } inBuf;
layout(set = 0, binding = 1) writeonly buffer OutBuf    { float data[]; } outBuf;
layout(set = 0, binding = 2) readonly  buffer ParamsBuf { uint sl; uint n_head; uint head_dim; } params;

// Rearranges Q+gate interleaved per head into [Q_all, gate_all] layout.
// Input : [sl, n_head * 2 * head_dim]  — [Q_h0(hd), gate_h0(hd), Q_h1(hd), gate_h1(hd), ...]
// Output: [sl, n_head * 2 * head_dim]  — [Q_h0, Q_h1, ..., Q_hn, gate_h0, ..., gate_hn]
void main() {
    uint idx = gl_GlobalInvocationID.x;
    uint full_stride = params.n_head * params.head_dim * 2u;
    if (idx >= params.sl * full_stride) return;

    uint t        = idx / full_stride;
    uint tok_local = idx % full_stride;

    uint half_stride = params.n_head * params.head_dim;
    uint part  = tok_local / half_stride;   // 0 = Q part, 1 = gate part
    uint rem   = tok_local % half_stride;
    uint h     = rem / params.head_dim;
    uint d     = rem % params.head_dim;

    // Source index in interleaved layout: [Q_h0(hd), gate_h0(hd), Q_h1(hd), ...]
    uint in_local = h * (2u * params.head_dim) + part * params.head_dim + d;
    outBuf.data[idx] = inBuf.data[t * full_stride + in_local];
}
