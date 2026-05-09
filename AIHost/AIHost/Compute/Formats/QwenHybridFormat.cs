using System.Numerics;
using AIHost.Compute;
using AIHost.ICompute;
using AIHost.Inference;
using Microsoft.Extensions.Logging;
using System.Buffers;

namespace AIHost.Compute.Formats;

public class QwenHybridFormat : ITransformerFormat
{
    private readonly ILogger<QwenHybridFormat> _logger = AppLogger.Create<QwenHybridFormat>();

    public Tensor ApplyLayer(TransformerBase t, Tensor x, int li, uint pos, KVCache? kvc, SSMState? ss)
    {
        int g = t.GlobalLayer(li); var nm = t._nameMapper!;
        bool hc = t.HasWeight($"blk.{g}.attn_qkv.weight"), hs = t.HasWeight($"blk.{g}.attn_q.weight");
        if (!hc && !hs) return ApplyLayerSSMFallback(t, x, g, nm);
        if (hs) return ApplyLayerTypeB(t, x, g, li, pos, kvc);
        return ApplyLayerCombinedQKV(t, x, g, li, pos, kvc, nm, ss);
    }

    // ─── Type B ────────────────────────────────────────────────────────────
    static Tensor ApplyLayerTypeB(TransformerBase t, Tensor x, int g, int li, uint pos, KVCache? kvc)
    {
        var o = t.Ops; int sl = x.Shape[0]; bool b = sl > 1;
        var a = t.TempF32Named($"blk.{g}.attn_norm.weight"); var q = t.TempF32Named($"blk.{g}.attn_q.weight");
        var k = t.TempF32Named($"blk.{g}.attn_k.weight"); var v = t.TempF32Named($"blk.{g}.attn_v.weight");
        var wo = t.TempF32Named($"blk.{g}.attn_output.weight"); var p = t.TempF32Named($"blk.{g}.post_attention_norm.weight");
        var wg = t.TempF32Named($"blk.{g}.ffn_gate.weight"); var wu = t.TempF32Named($"blk.{g}.ffn_up.weight"); var wd = t.TempF32Named($"blk.{g}.ffn_down.weight");
        if (b) o.BeginBatch();
        var xn = o.Clone(x, "an"); o.LayerNorm(xn, a.tensor); if (!a.isScratch) o.DeferExternal(a.tensor);
        var rq = o.MatMulWeights(xn, q.tensor, "rq"); var K = o.MatMulWeights(xn, k.tensor, "K"); var V = o.MatMulWeights(xn, v.tensor, "V");
        if (!q.isScratch) o.DeferExternal(q.tensor); if (!k.isScratch) o.DeferExternal(k.tensor); if (!v.isScratch) o.DeferExternal(v.tensor);
        if (b) o.DeferExternal(xn); else xn.Dispose();
        int kd = K.Shape[1], qtd = rq.Shape[1], hd = 256;
        bool ig = t.HasWeight($"blk.{g}.attn_gate.weight");
        int qd = t.HasWeight($"blk.{g}.attn_output.weight") ? t._weightCache[$"blk.{g}.attn_output.weight"].Shape[0] : qtd;
        Tensor gq; int nqh;
        if (ig || qtd > qd) { var qp = o.SliceCols(rq, 0, qd, "qp"); var qg = o.SliceCols(rq, qd, qd, "qg"); if (b) o.DeferExternal(rq); else rq.Dispose(); o.SiLU(qg); gq = o.Multiply(qp, qg, "gq"); if (b) o.DeferExternal(qp); else qp.Dispose(); if (b) o.DeferExternal(qg); else qg.Dispose(); nqh = qd / hd; }
        else { gq = rq; nqh = qtd / hd; }
        var qn = t.GetOrBuildTiledNorm($"blk.{g}.attn_q_norm.weight", hd, qd); var kn = t.GetOrBuildTiledNorm($"blk.{g}.attn_k_norm.weight", hd, kd);
        if (qn != null) o.LayerNorm(gq, qn); if (kn != null) o.LayerNorm(K, kn);
        o.ApplyRoPEFull(gq, pos, nqh, hd, t._ropeFreqBase); o.ApplyRoPEFull(K, pos, kd / hd, hd, t._ropeFreqBase);
        Tensor ao; if (kvc != null) { kvc.Add(li, K, V); var (ck, cv) = kvc.Get(li); ao = o.MultiHeadAttention(gq, ck!, cv!, nqh, pos, "ao_B"); o.DeferExternal(gq); }
        else { ao = o.MultiHeadAttention(gq, K, V, nqh, pos, "ao_B"); o.DeferExternal(gq); o.DeferExternal(K); o.DeferExternal(V); }
        Tensor ap = wo.tensor.Shape[0] == ao.Shape[1] ? o.MatMulWeights(ao, wo.tensor, "ap_B") : o.MatMulWeightsT(ao, wo.tensor, "ap_B");
        if (!wo.isScratch) o.DeferExternal(wo.tensor); if (b) o.DeferExternal(ao); else ao.Dispose();
        var x1 = o.Add(x, ap, "xa_B"); if (b) o.DeferExternal(ap); else ap.Dispose();
        var x1n = o.Clone(x1, "pn"); o.LayerNorm(x1n, p.tensor); if (!p.isScratch) o.DeferExternal(p.tensor);
        var fo = o.FeedForward(x1n, wg.tensor, wu.tensor, wd.tensor, "ff_B"); if (!wg.isScratch) o.DeferExternal(wg.tensor); if (!wu.isScratch) o.DeferExternal(wu.tensor); if (!wd.isScratch) o.DeferExternal(wd.tensor);
        if (b) o.DeferExternal(x1n); else x1n.Dispose(); var ou = o.Add(x1, fo, "lo_B"); if (b) o.DeferExternal(x1); else x1.Dispose(); if (b) o.DeferExternal(fo); else fo.Dispose(); if (b) o.DeferExternal(x); else x.Dispose(); if (b) o.Flush(); return ou;
    }

    // ─── Type A ────────────────────────────────────────────────────────────
    static Tensor ApplyLayerCombinedQKV(TransformerBase t, Tensor x, int g, int li, uint pos, KVCache? kvc, TensorNameMapper nm, SSMState? ss)
    {
        var o = t.Ops; int sl = x.Shape[0]; bool b = sl > 1;
        var a = t.TempF32(nm.AttnNorm(g)); var qk = t.TempF32(nm.AttnQKV(g));
        string ak = t.HasWeight($"blk.{g}.ssm_out.weight") ? $"blk.{g}.ssm_out.weight" : nm.AttnOutput(g);
        var aoW = t.TempF32Named(ak); var fn = t.TempF32(nm.FfnNorm(g)); var wg = t.TempF32(nm.FfnGate(g)); var wu = t.TempF32(nm.FfnUp(g)); var wd = t.TempF32(nm.FfnDown(g));
        if (b) o.BeginBatch();
        var xn = o.Clone(x, "an"); o.LayerNorm(xn, a.tensor); if (!a.isScratch) o.DeferExternal(a.tensor);
        var qkv = o.MatMulWeights(xn, qk.tensor, "qkv"); if (!qk.isScratch) o.DeferExternal(qk.tensor);
        int hd = 256, qd = t._numHeads * hd, kd = t._numKVHeads * hd, nkh = t._numKVHeads;
        int tq = qkv.Shape[1], eb = qd + 2 * kd; bool hg = tq > eb; int qgd = hg ? tq - eb : 0;
        var Q = o.SliceCols(qkv, 0, qd, "Q"); var K = o.SliceCols(qkv, qd, kd, "K"); var V = o.SliceCols(qkv, qd + kd, kd, "V");
        Tensor gq; if (hg) { var qg = o.SliceCols(qkv, qd + 2 * kd, qgd, "qg"); o.SiLU(qg); var qt = o.Concat(qg, qg, 1, "qt2"); qt = o.Concat(qt, qg, 1, "qt3"); if (b) o.DeferExternal(qg); else qg.Dispose(); gq = o.Multiply(Q, qt, "gq"); if (b) o.DeferExternal(Q); else Q.Dispose(); if (b) o.DeferExternal(qt); else qt.Dispose(); } else { gq = Q; }
        if (b) o.DeferExternal(qkv); else qkv.Dispose();
        o.ApplyRoPEFull(gq, pos, t._numHeads, hd, t._ropeFreqBase); o.ApplyRoPEFull(K, pos, nkh, hd, t._ropeFreqBase);
        Tensor ao; if (kvc != null) { kvc.Add(li, K, V); var (ck, cv) = kvc.Get(li); ao = o.MultiHeadAttention(gq, ck!, cv!, t._numHeads, pos, "ao"); o.DeferExternal(gq); }
        else { ao = o.MultiHeadAttention(gq, K, V, t._numHeads, pos, "ao"); o.DeferExternal(gq); o.DeferExternal(K); o.DeferExternal(V); }
        Tensor ap = aoW.tensor.Shape[0] == ao.Shape[1] ? o.MatMulWeights(ao, aoW.tensor, "ap") : o.MatMulWeightsT(ao, aoW.tensor, "ap");
        if (!aoW.isScratch) o.DeferExternal(aoW.tensor); if (b) o.DeferExternal(ao); else ao.Dispose(); if (b) o.DeferExternal(xn); else xn.Dispose();
        var x1 = o.Add(x, ap, "xa"); if (b) o.DeferExternal(ap); else ap.Dispose();
        var x1n = o.Clone(x1, "pn"); o.LayerNorm(x1n, fn.tensor); if (!fn.isScratch) o.DeferExternal(fn.tensor);
        Tensor x2; if (ss != null && t.HasWeight($"blk.{g}.ssm_a")) { o.Flush(); var so = ApplySSMRecurrence(t, x1, x1n, g, ss, sl); x1.Dispose(); x2 = so; o.BeginBatch(); } else { x2 = x1; }
        var fo = o.FeedForward(x1n, wg.tensor, wu.tensor, wd.tensor, "ff"); if (!wg.isScratch) o.DeferExternal(wg.tensor); if (!wu.isScratch) o.DeferExternal(wu.tensor); if (!wd.isScratch) o.DeferExternal(wd.tensor);
        if (b) o.DeferExternal(x1n); else x1n.Dispose(); var ou = o.Add(x2, fo, "lo"); if (b) o.DeferExternal(x2); else x2.Dispose(); if (b) o.DeferExternal(fo); else fo.Dispose(); if (b) o.DeferExternal(x); else x.Dispose(); if (b) o.Flush(); return ou;
    }

    // ─── SSM ───────────────────────────────────────────────────────────────

    struct PT { public float[] z, beta, alpha, gate, qkvMixed; }

    static Tensor ApplySSMRecurrence(TransformerBase t, Tensor xr, Tensor xn, int g, SSMState ss, int sl)
    {
        const int HVD = 128, NVH = 48, NKH = 16, KD = NKH * HVD, VD = NVH * HVD, CD = 2 * KD + VD;
        int dm = xn.Shape[1];

        float[] Ld(string n) { var (tt, sc) = t.TempF32Named(n); var d = tt.ReadData(); if (!sc) tt.Dispose(); return d; }
        var xd = xn.ReadData();
        var Wdt = Ld($"blk.{g}.ssm_alpha.weight"); var Wb = Ld($"blk.{g}.ssm_beta.weight"); var ssA = Ld($"blk.{g}.ssm_a");
        var dtB = Ld($"blk.{g}.ssm_dt.bias"); var sNw = Ld($"blk.{g}.ssm_norm.weight"); var cW = Ld($"blk.{g}.ssm_conv1d.weight");
        var wZ = Ld($"blk.{g}.attn_gate.weight"); var wQ = Ld($"blk.{g}.attn_qkv.weight"); var wO = Ld($"blk.{g}.ssm_out.weight");

        // Transpose ONCE: GGUF row-major [dModel, dim] → MV needs [dim, dModel]
        var wZT = Tr(wZ, dm, VD); var WbT = Tr(Wb, dm, NVH); var WdtT = Tr(Wdt, dm, NVH);
        var wQT = Tr(wQ, dm, CD); var wOT = Tr(wO, VD, dm);

        var pt = new PT[sl];
        Parallel.For(0, sl, ti => { int oo = ti * dm; var p = new PT();
            p.z = MV(xd, oo, wZT, VD, dm); var rb = MV(xd, oo, WbT, NVH, dm); p.beta = new float[NVH]; for (int n = 0; n < NVH; n++) p.beta[n] = 1f / (1f + MathF.Exp(-rb[n]));
            var ra = MV(xd, oo, WdtT, NVH, dm); p.alpha = new float[NVH]; for (int n = 0; n < NVH; n++) p.alpha[n] = MathF.Log(1f + MathF.Exp(ra[n] + dtB[n]));
            p.gate = new float[NVH]; for (int n = 0; n < NVH; n++) p.gate[n] = p.alpha[n] * ssA[n];
            p.qkvMixed = MV(xd, oo, wQT, CD, dm); pt[ti] = p; });

        // Free transposed weights after Phase 1
        wZT = WbT = WdtT = wQT = null;

        var st = ss.GetLayer(g); var ys = new float[sl * VD]; float sc = 1f / MathF.Sqrt(HVD); const float ep = 1e-5f;
        for (int ti = 0; ti < sl; ti++) { var p = pt[ti];
            var cw = ss.UpdateConvState(g, p.qkvMixed); var cv = new float[CD];
            for (int c = 0; c < CD; c++) { float s = 0; s += cw[0 * CD + c] * cW[0 * CD + c]; s += cw[1 * CD + c] * cW[1 * CD + c]; s += cw[2 * CD + c] * cW[2 * CD + c]; s += cw[3 * CD + c] * cW[3 * CD + c]; cv[c] = s; }
            var qc = new float[KD]; var kc = new float[KD]; var vc = new float[VD];
            for (int i = 0; i < KD; i++) { float f = cv[i]; qc[i] = f / (1f + MathF.Exp(-f)); }
            for (int i = 0; i < KD; i++) { float f = cv[KD + i]; kc[i] = f / (1f + MathF.Exp(-f)); }
            for (int i = 0; i < VD; i++) { float f = cv[2 * KD + i]; vc[i] = f / (1f + MathF.Exp(-f)); }
            for (int h = 0; h < NKH; h++) { int b_ = h * HVD; float q2 = 0, k2 = 0; for (int d = 0; d < HVD; d++) { q2 += qc[b_ + d] * qc[b_ + d]; k2 += kc[b_ + d] * kc[b_ + d]; } float nq = MathF.Sqrt(q2 + ep), nk = MathF.Sqrt(k2 + ep); for (int d = 0; d < HVD; d++) { qc[b_ + d] /= nq; kc[b_ + d] /= nk; } }
            var yt = new float[VD];
            for (int n = 0; n < NVH; n++) { int h = n % NKH, sB = n * HVD * HVD, vB = n * HVD, kB = h * HVD, qB = h * HVD;
                float gE = MathF.Exp(p.gate[n]); for (int i = 0; i < HVD; i++) { int rB = sB + i * HVD; for (int j = 0; j < HVD; j++) st[rB + j] *= gE; }
                var sk = new float[HVD]; for (int d = 0; d < HVD; d++) { float sm = 0; int rB = sB + d * HVD; for (int j = 0; j < HVD; j++) sm += st[rB + j] * kc[kB + j]; sk[d] = sm; }
                var dv = new float[HVD]; for (int d = 0; d < HVD; d++) dv[d] = (vc[vB + d] - sk[d]) * p.beta[n];
                for (int i = 0; i < HVD; i++) { int rB = sB + i * HVD; float ki = kc[kB + i]; for (int j = 0; j < HVD; j++) st[rB + j] += ki * dv[j]; }
                for (int d = 0; d < HVD; d++) { float sm = 0; int rB = sB + d * HVD; for (int j = 0; j < HVD; j++) sm += st[rB + j] * qc[qB + j]; yt[vB + d] = sm * sc; } }
            for (int n = 0; n < NVH; n++) { int gB = n * HVD; float s2 = 0; for (int d = 0; d < HVD; d++) { float vv = yt[gB + d]; s2 += vv * vv; } float ri = 1f / MathF.Sqrt(s2 / HVD + 1e-5f); float zs = p.z[gB] / (1f + MathF.Exp(-p.z[gB])); for (int d = 0; d < HVD; d++) yt[gB + d] = yt[gB + d] * ri * sNw[d] * zs; }
            Buffer.BlockCopy(yt, 0, ys, ti * VD * sizeof(float), VD * sizeof(float)); }

        // Parallel output projection
        var rd = new float[sl * dm]; Parallel.For(0, sl, ti => { var rw = MV(ys, ti * VD, wOT, dm, VD); Buffer.BlockCopy(rw, 0, rd, ti * dm * sizeof(float), dm * sizeof(float)); });
        var xrd = xr.ReadData(); for (int i = 0; i < rd.Length; i++) rd[i] += xrd[i];
        return Tensor.FromData(t.Ops.Device, rd, new TensorShape(new[] { sl, dm }), "ssm_out");
    }

    /// GGUF row-major [dModel,dim2] → matvec-friendly [dim2,dModel]
    static float[] Tr(float[] w, int dm, int d2)
    { var r = new float[w.Length]; for (int k = 0; k < dm; k++) { int s = k * d2; for (int n = 0; n < d2; n++) r[n * dm + k] = w[s + n]; } return r; }

    /// MV with transposed weights: result[j] = sum_k x[off+k] * wt[j*dm+k]
    static float[] MV(float[] x, int off, float[] wt, int rdim, int dm)
    { var r = new float[rdim]; int vs = Vector<float>.Count;
        for (int j = 0; j < rdim; j++) { int wo = j * dm; float s = 0; int k = 0;
            if (vs > 1) { var vsu = Vector<float>.Zero; for (; k <= dm - vs; k += vs) vsu += new Vector<float>(x, off + k) * new Vector<float>(wt, wo + k); for (int z = 0; z < vs; z++) s += vsu[z]; }
            for (; k < dm; k++) s += x[off + k] * wt[wo + k]; r[j] = s; } return r; }

    // ─── Stubs ─────────────────────────────────────────────────────────────
    static Tensor ApplySSMDecodeGPU(TransformerBase t, Tensor xr, Tensor xn, int g, SSMState ss, ComputeOps o, int dm) => null!;
    static Tensor ApplyLayerSSMFallback(TransformerBase t, Tensor x, int g, TensorNameMapper nm) { var o = t.Ops; var xn = o.Clone(x, "an"); var (wa, sa) = t.TempF32(nm.AttnNorm(g)); o.LayerNorm(xn, wa); if (!sa) wa.Dispose(); Tensor ao; if (t.HasWeight(nm.AttnQKV(g))) { var (w, s) = t.TempF32(nm.AttnQKV(g)); var q = o.MatMulWeights(xn, w, "q_f"); if (!s) w.Dispose(); int qd = t._numHeads * 256, kd = t._numKVHeads * 256; var Q = o.SliceCols(q, 0, qd, "Q_f"); var K = o.SliceCols(q, qd, kd, "K_f"); var V = o.SliceCols(q, qd + kd, kd, "V_f"); q.Dispose(); o.ApplyRoPEFull(Q, 0, t._numHeads, 256, t._ropeFreqBase); o.ApplyRoPEFull(K, 0, t._numKVHeads, 256, t._ropeFreqBase); ao = o.MultiHeadAttention(Q, K, V, t._numHeads, 0, "a_f"); o.DeferExternal(Q); o.DeferExternal(K); o.DeferExternal(V); } else { var (wq, sq) = t.TempF32(nm.AttnQ(g)); var (wk, sk) = t.TempF32(nm.AttnK(g)); var (wv, sv) = t.TempF32(nm.AttnV(g)); var Q = o.MatMulWeights(xn, wq, "Q_f"); var K = o.MatMulWeights(xn, wk, "K_f"); var V = o.MatMulWeights(xn, wv, "V_f"); if (!sq) wq.Dispose(); if (!sk) wk.Dispose(); if (!sv) wv.Dispose(); o.ApplyRoPEFull(Q, 0, t._numHeads, 256, t._ropeFreqBase); o.ApplyRoPEFull(K, 0, t._numKVHeads, 256, t._ropeFreqBase); ao = o.MultiHeadAttention(Q, K, V, t._numHeads, 0, "a_f"); o.DeferExternal(Q); o.DeferExternal(K); o.DeferExternal(V); } xn.Dispose(); var (wo, so) = t.TempF32(nm.AttnOutput(g)); Tensor ap = wo.Shape[0] == ao.Shape[1] ? o.MatMulWeights(ao, wo, "ap_f") : o.MatMulWeightsT(ao, wo, "ap_f"); if (!so) wo.Dispose(); ao.Dispose(); var x1 = o.Add(x, ap, "xa_f"); ap.Dispose(); var x1n = o.Clone(x1, "pn_f"); var (wp, sp) = t.TempF32(nm.FfnNorm(g)); o.LayerNorm(x1n, wp); if (!sp) wp.Dispose(); var (wg, sg) = t.TempF32(nm.FfnGate(g)); var (wu, su) = t.TempF32(nm.FfnUp(g)); var (wd, sd) = t.TempF32(nm.FfnDown(g)); var fo = o.FeedForward(x1n, wg, wu, wd, "ff_f"); if (!sg) wg.Dispose(); if (!su) wu.Dispose(); if (!sd) wd.Dispose(); x1n.Dispose(); var ou = o.Add(x1, fo, "lo_f"); x1.Dispose(); fo.Dispose(); x.Dispose(); return ou; }
}
