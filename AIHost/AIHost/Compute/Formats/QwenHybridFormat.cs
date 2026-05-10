using AIHost.Compute;
using AIHost.ICompute;
using AIHost.Inference;
using Microsoft.Extensions.Logging;
using System.Numerics;

namespace AIHost.Compute.Formats;

public class QwenHybridFormat : ITransformerFormat
{
    private readonly ILogger<QwenHybridFormat> _logger = AppLogger.Create<QwenHybridFormat>();

    public Tensor ApplyLayer(TransformerBase t, Tensor x, int li, uint pos, KVCache? kvc, SSMState? ss)
    {
        var _ts = GlobalProfiler.Start();
        int g = t.GlobalLayer(li); var nm = t._nameMapper!;
        bool hc = t.HasWeight($"blk.{g}.attn_qkv.weight"), hs = t.HasWeight($"blk.{g}.attn_q.weight");
        Tensor result;
        if (!hc && !hs) result = Fallback(t, x, g, nm);
        else if (hs) result = TypeB(t, x, g, li, pos, kvc);
        else result = TypeA(t, x, g, li, pos, kvc, nm, ss);
        GlobalProfiler.End(_ts, "Qwen.ApplyLayer");
        return result;
    }


    // ─── Type A ────────────────────────────────────────────────────────
    static Tensor TypeA(TransformerBase t, Tensor x, int g, int li, uint pos, KVCache? kvc, TensorNameMapper nm, SSMState? ss)
    {
        var o = t.Ops; int sl = x.Shape[0]; bool b = sl > 1;
        t._logger.LogInformation("[TypeA] Layer {Layer} START sl={SeqLen}", g, sl);
        var _ts = GlobalProfiler.Start();

        // --- Load weights (dequant) ---
        var a = t.TempF32(nm.AttnNorm(g)); var qkw = t.TempF32(nm.AttnQKV(g));
        string ak = t.HasWeight($"blk.{g}.ssm_out.weight") ? $"blk.{g}.ssm_out.weight" : nm.AttnOutput(g);
        var aoW = t.TempF32Named(ak); var fn = t.TempF32(nm.FfnNorm(g));
        var wg = t.TempF32(nm.FfnGate(g)); var wu = t.TempF32(nm.FfnUp(g));
        var wd = t.TempF32(nm.FfnDown(g));
        GlobalProfiler.End(_ts, "LayerA.LoadW");

        if (b) o.BeginBatch();

        // --- Attn norm + QKV matmul ---
        var _ts2 = GlobalProfiler.Start();
        var xn = o.Clone(x, "an"); o.LayerNorm(xn, a.tensor);
        if (!a.isScratch) o.DeferExternal(a.tensor);
        var qkv = o.MatMulWeights(xn, qkw.tensor, "qkv");
        if (!qkw.isScratch) o.DeferExternal(qkw.tensor);
        GlobalProfiler.End(_ts2, "LayerA.AttnMatMul");

        int hd = 256, qd = t._numHeads * hd, kd = t._numKVHeads * hd;
        int tq = qkv.Shape[1], eb = qd + 2 * kd; bool hg = tq > eb; int qgd = hg ? tq - eb : 0;
        var Q = o.SliceCols(qkv, 0, qd, "Q"); var K = o.SliceCols(qkv, qd, kd, "K"); var V = o.SliceCols(qkv, qd + kd, kd, "V");
        Tensor gq; if (hg) { var qg = o.SliceCols(qkv, qd + 2 * kd, qgd, "qg"); o.SiLU(qg); var qt = o.Concat(qg, qg, 1, "qt2"); qt = o.Concat(qt, qg, 1, "qt3"); if (b) o.DeferExternal(qg); else qg.Dispose(); gq = o.Multiply(Q, qt, "gq"); if (b) o.DeferExternal(Q); else Q.Dispose(); if (b) o.DeferExternal(qt); else qt.Dispose(); } else { gq = Q; }
        if (b) o.DeferExternal(qkv); else qkv.Dispose();

        // --- RoPE + Attention ---
        var _ts3 = GlobalProfiler.Start();
        o.ApplyRoPEFull(gq, pos, t._numHeads, hd, t._ropeFreqBase);
        o.ApplyRoPEFull(K, pos, t._numKVHeads, hd, t._ropeFreqBase);
        Tensor ao; if (kvc != null) {
            kvc.Add(li, K, V); var (ck, cv) = kvc.Get(li);
            ao = o.MultiHeadAttention(gq, ck!, cv!, t._numHeads, pos, "ao");
            o.DeferExternal(gq);
        } else {
            ao = o.MultiHeadAttention(gq, K, V, t._numHeads, pos, "ao");
            o.DeferExternal(gq); o.DeferExternal(K); o.DeferExternal(V);
        }
        GlobalProfiler.End(_ts3, "LayerA.Attn");

        // --- Output projection ---
        var _ts4 = GlobalProfiler.Start();
        Tensor ap = aoW.tensor.Shape[0] == ao.Shape[1]
            ? o.MatMulWeights(ao, aoW.tensor, "ap")
            : o.MatMulWeightsT(ao, aoW.tensor, "ap");
        if (!aoW.isScratch) o.DeferExternal(aoW.tensor);
        if (b) o.DeferExternal(ao); else ao.Dispose();
        if (b) o.DeferExternal(xn); else xn.Dispose();
        GlobalProfiler.End(_ts4, "LayerA.OutProj");

        var x1 = o.Add(x, ap, "xa");
        if (b) o.DeferExternal(ap); else ap.Dispose();

        // --- FFN norm ---
        var _ts5 = GlobalProfiler.Start();
        var x1n = o.Clone(x1, "pn"); o.LayerNorm(x1n, fn.tensor);
        if (!fn.isScratch) o.DeferExternal(fn.tensor);
        GlobalProfiler.End(_ts5, "LayerA.FfnNorm");

        // --- SSM on GPU ---
        var _ts6 = GlobalProfiler.Start();
        Tensor x2;
        if (ss != null && t.HasWeight($"blk.{g}.ssm_a"))
        {
            try
            {
                x2 = SSM_Gpu(t, o, x1, x1n, g, ss, sl);
                x1.Dispose();
            }
            catch (Exception ex)
            {
                // Log and rethrow so the error is visible in docker logs
                t._logger.LogError(ex, "[SSM_GPU] Failed at layer {Layer}", g);
                throw;
            }
        }
        else { x2 = x1; }
        GlobalProfiler.End(_ts6, "LayerA.SSM");

        // --- FeedForward ---
        var _ts7 = GlobalProfiler.Start();
        var fo = o.FeedForward(x1n, wg.tensor, wu.tensor, wd.tensor, "ff");
        if (!wg.isScratch) o.DeferExternal(wg.tensor);
        if (!wu.isScratch) o.DeferExternal(wu.tensor);
        if (!wd.isScratch) o.DeferExternal(wd.tensor);
        GlobalProfiler.End(_ts7, "LayerA.FFN");

        if (b) o.DeferExternal(x1n); else x1n.Dispose();
        var ou = o.Add(x2, fo, "lo");
        if (b) o.DeferExternal(x2); else x2.Dispose();
        if (b) o.DeferExternal(fo); else fo.Dispose();
        if (b) o.DeferExternal(x); else x.Dispose();

        var _ts8 = GlobalProfiler.Start();
        if (b) o.Flush();
        GlobalProfiler.End(_ts8, "LayerA.Flush");

        return ou;
    }

    // ─── Type B ────────────────────────────────────────────────────────
    static Tensor TypeB(TransformerBase t, Tensor x, int g, int li, uint pos, KVCache? kvc)
    {
        var o = t.Ops; int sl = x.Shape[0]; bool b = sl > 1;
        var anW = t.TempF32Named($"blk.{g}.attn_norm.weight"); var qW = t.TempF32Named($"blk.{g}.attn_q.weight"); var kW = t.TempF32Named($"blk.{g}.attn_k.weight"); var vW = t.TempF32Named($"blk.{g}.attn_v.weight");
        var oW = t.TempF32Named($"blk.{g}.attn_output.weight"); var pW = t.TempF32Named($"blk.{g}.post_attention_norm.weight");
        var gW = t.TempF32Named($"blk.{g}.ffn_gate.weight"); var uW = t.TempF32Named($"blk.{g}.ffn_up.weight"); var dW = t.TempF32Named($"blk.{g}.ffn_down.weight");
        if (b) o.BeginBatch();
        var xn = o.Clone(x, "anB"); o.LayerNorm(xn, anW.tensor); if (!anW.isScratch) o.DeferExternal(anW.tensor);
        var rq = o.MatMulWeights(xn, qW.tensor, "rqB"); var K = o.MatMulWeights(xn, kW.tensor, "KB"); var V = o.MatMulWeights(xn, vW.tensor, "VB");
        if (!qW.isScratch) o.DeferExternal(qW.tensor); if (!kW.isScratch) o.DeferExternal(kW.tensor); if (!vW.isScratch) o.DeferExternal(vW.tensor);
        if (b) o.DeferExternal(xn); else xn.Dispose();
        int kd = K.Shape[1], qtd = rq.Shape[1], hd = 256; bool ig = t.HasWeight($"blk.{g}.attn_gate.weight");
        int qd = t.HasWeight($"blk.{g}.attn_output.weight") ? t._weightCache[$"blk.{g}.attn_output.weight"].Shape[0] : qtd;
        Tensor gq; int nqh;
        if (ig || qtd > qd) { var qp = o.SliceCols(rq, 0, qd, "qpB"); var qg = o.SliceCols(rq, qd, qd, "qgB"); if (b) o.DeferExternal(rq); else rq.Dispose(); o.SiLU(qg); gq = o.Multiply(qp, qg, "gqB"); if (b) o.DeferExternal(qp); else qp.Dispose(); if (b) o.DeferExternal(qg); else qg.Dispose(); nqh = qd / hd; } else { gq = rq; nqh = qtd / hd; }
        var qn = t.GetOrBuildTiledNorm($"blk.{g}.attn_q_norm.weight", hd, qd); var kn = t.GetOrBuildTiledNorm($"blk.{g}.attn_k_norm.weight", hd, kd);
        if (qn != null) o.LayerNorm(gq, qn); if (kn != null) o.LayerNorm(K, kn);
        o.ApplyRoPEFull(gq, pos, nqh, hd, t._ropeFreqBase); o.ApplyRoPEFull(K, pos, kd / hd, hd, t._ropeFreqBase);
        Tensor ao; if (kvc != null) { kvc.Add(li, K, V); var (ck, cv) = kvc.Get(li); ao = o.MultiHeadAttention(gq, ck!, cv!, nqh, pos, "aoB"); o.DeferExternal(gq); }
        else { ao = o.MultiHeadAttention(gq, K, V, nqh, pos, "aoB"); o.DeferExternal(gq); o.DeferExternal(K); o.DeferExternal(V); }
        Tensor ap = oW.tensor.Shape[0] == ao.Shape[1] ? o.MatMulWeights(ao, oW.tensor, "apB") : o.MatMulWeightsT(ao, oW.tensor, "apB");
        if (!oW.isScratch) o.DeferExternal(oW.tensor); if (b) o.DeferExternal(ao); else ao.Dispose();
        var x1 = o.Add(x, ap, "xaB"); if (b) o.DeferExternal(ap); else ap.Dispose();
        var x1n = o.Clone(x1, "pnB"); o.LayerNorm(x1n, pW.tensor); if (!pW.isScratch) o.DeferExternal(pW.tensor);
        var fo = o.FeedForward(x1n, gW.tensor, uW.tensor, dW.tensor, "ffB"); if (!gW.isScratch) o.DeferExternal(gW.tensor); if (!uW.isScratch) o.DeferExternal(uW.tensor); if (!dW.isScratch) o.DeferExternal(dW.tensor);
        if (b) o.DeferExternal(x1n); else x1n.Dispose(); var ou = o.Add(x1, fo, "loB"); if (b) o.DeferExternal(x1); else x1.Dispose(); if (b) o.DeferExternal(fo); else fo.Dispose(); if (b) o.DeferExternal(x); else x.Dispose();
        if (b) o.Flush(); return ou;
    }

    // ─── SSM recurrence on GPU ────────────────────────────────────────
    static Tensor SSM_Gpu(TransformerBase t, ComputeOps o, Tensor xr, Tensor xn, int g, SSMState ss, int sl)
    {
        const int HVD=128, NVH=48, NKH=16, KD=NKH*HVD, VD=NVH*HVD, CD=2*KD+VD;
        int dm = xn.Shape[1];

        // All SSM weights must be pre-dequantized (scratch) for GPU access.
        // TempF32Named returns (tensor, isScratch) — scratch tensors live on GPU.
        var (convW,  _) = t.TempF32Named($"blk.{g}.ssm_conv1d.weight");
        var (wQKV,   _) = t.TempF32Named($"blk.{g}.attn_qkv.weight");
        var (wZ,     _) = t.TempF32Named($"blk.{g}.attn_gate.weight");
        var (wBeta,  _) = t.TempF32Named($"blk.{g}.ssm_beta.weight");
        var (wAlpha, _) = t.TempF32Named($"blk.{g}.ssm_alpha.weight");
        var (dtBias, _) = t.TempF32Named($"blk.{g}.ssm_dt.bias");
        var (ssA,    _) = t.TempF32Named($"blk.{g}.ssm_a");

        // Get GPU scratch for norm weight
        var normF32 = t.GetOrBuildTiledNorm($"blk.{g}.ssm_norm.weight", HVD, VD);
        if (normF32 == null) { normF32 = t.TempF32Named($"blk.{g}.ssm_norm.weight").tensor; }

        // Output projection: always ssm_out.weight [dm, VD] (GGUF row-major).
        // CPU SSM transposes and matvecs; GPU uses MatMulWeightsT (A=[sl,VD], B=[dm,VD]^T)
        var (wOut,  _) = t.TempF32Named($"blk.{g}.ssm_out.weight");

        // Wrap GPU state buffers as Tensors for SsmGdnDecode
        var convStateBuf = ss.GetGpuConvBuffer(g);
        var convStateTensor = new Tensor(convStateBuf, new TensorShape(new[]{SSMState.CONV_STATE_DIM}), DataType.F32);
        var ssmStateBuf = ss.GetGpuStateBuffer(g);
        var ssmStateTensor = new Tensor(ssmStateBuf, new TensorShape(new[]{SSMState.STATE_DIM}), DataType.F32);

        // SsmGdnDecode writes per-token output [VD]; chain-concatenate into [sl, VD].
        var _tsP = GlobalProfiler.Start();
        t._logger.LogInformation("[SSM_GPU] Layer {Layer} sl={SeqLen} dm={DModel}", g, sl, dm);
        bool b = sl > 1; // batch mode same as TypeA
        var scratch = Tensor.Create(o.Device, new TensorShape(new[]{CD + VD + NVH*3}), DataType.F32, "ssm_scratch");
        Tensor? result = null;

        for (int ti = 0; ti < sl; ti++)
        {
            var _tsTi = GlobalProfiler.Start();
            var x1nRow = o.Clone(xn);
            var tokenOut = Tensor.Create(o.Device, TensorShape.Matrix(1, VD), DataType.F32, $"ssm_tok{ti}");

            o.SsmGdnDecode(
                x1nRow,
                convW, convStateTensor, wQKV, wZ, wBeta, wAlpha, dtBias, ssA,
                scratch,
                ssmStateTensor,
                normF32,
                tokenOut);

            GlobalProfiler.End(_tsTi, "SSM_GPU.Token");
            if (result == null)
                result = tokenOut;
            else
            {
                result = o.Concat(result, tokenOut, 0, "ssm_cat");
                // CRITICAL: in batch mode tokenOut is referenced by pending GPU dispatches
                if (b) o.DeferExternal(tokenOut); else tokenOut.Dispose();
            }
            // CRITICAL: in batch mode x1nRow is referenced by pending SsmGdnDecode dispatch
            if (sl > 1 && ti < sl - 1)
            {
                if (b) o.DeferExternal(x1nRow); else x1nRow.Dispose();
            }
        }
        GlobalProfiler.End(_tsP, "SSM_GPU.Total");
        result ??= Tensor.Create(o.Device, TensorShape.Matrix(1, VD), DataType.F32, "ssm_empty");

        var _tsP2 = GlobalProfiler.Start();
        var projected = o.MatMulWeights(result, wOut, "ssm_proj");
        var residual = o.Add(xr, projected, "ssm_out");
        // CRITICAL: in batch mode, GPU dispatches reference these buffers.
        // They will be disposed by the batch Flush() in the caller (TypeA).
        if (b) { o.DeferExternal(result); o.DeferExternal(projected); o.DeferExternal(scratch); }
        else { result.Dispose(); projected.Dispose(); scratch.Dispose(); }
        GlobalProfiler.End(_tsP2, "SSM_GPU.OutProj");
        t._logger.LogInformation("[SSM_GPU] Layer {Layer} done", g);
        return residual;
    }


    // ─── Fallback ──────────────────────────────────────────────────────
    static Tensor Fallback(TransformerBase t,Tensor x,int g,TensorNameMapper nm){var o=t.Ops;var xn=o.Clone(x,"an_fb");var(wa,sa)=t.TempF32(nm.AttnNorm(g));o.LayerNorm(xn,wa);if(!sa)wa.Dispose();Tensor ao;if(t.HasWeight(nm.AttnQKV(g))){var(w,s_)=t.TempF32(nm.AttnQKV(g));var q=o.MatMulWeights(xn,w,"q_fb");if(!s_)w.Dispose();int qd=t._numHeads*256,kd=t._numKVHeads*256;var Q=o.SliceCols(q,0,qd,"Q_fb");var K=o.SliceCols(q,qd,kd,"K_fb");var V=o.SliceCols(q,qd+kd,kd,"V_fb");q.Dispose();o.ApplyRoPEFull(Q,0,t._numHeads,256,t._ropeFreqBase);o.ApplyRoPEFull(K,0,t._numKVHeads,256,t._ropeFreqBase);ao=o.MultiHeadAttention(Q,K,V,t._numHeads,0,"a_fb");o.DeferExternal(Q);o.DeferExternal(K);o.DeferExternal(V);}else{var(wq,sq)=t.TempF32(nm.AttnQ(g));var(wk,sk)=t.TempF32(nm.AttnK(g));var(wv,sv)=t.TempF32(nm.AttnV(g));var Q=o.MatMulWeights(xn,wq,"Q_fb");var K=o.MatMulWeights(xn,wk,"K_fb");var V=o.MatMulWeights(xn,wv,"V_fb");if(!sq)wq.Dispose();if(!sk)wk.Dispose();if(!sv)wv.Dispose();o.ApplyRoPEFull(Q,0,t._numHeads,256,t._ropeFreqBase);o.ApplyRoPEFull(K,0,t._numKVHeads,256,t._ropeFreqBase);ao=o.MultiHeadAttention(Q,K,V,t._numHeads,0,"a_fb");o.DeferExternal(Q);o.DeferExternal(K);o.DeferExternal(V);}xn.Dispose();var(wo,so)=t.TempF32(nm.AttnOutput(g));Tensor ap=wo.Shape[0]==ao.Shape[1]?o.MatMulWeights(ao,wo,"ap_fb"):o.MatMulWeightsT(ao,wo,"ap_fb");if(!so)wo.Dispose();ao.Dispose();var x1=o.Add(x,ap,"xa_fb");ap.Dispose();var x1n=o.Clone(x1,"pn_fb");var(wp,sp)=t.TempF32(nm.FfnNorm(g));o.LayerNorm(x1n,wp);if(!sp)wp.Dispose();var(wg,sg)=t.TempF32(nm.FfnGate(g));var(wu,su)=t.TempF32(nm.FfnUp(g));var(wd,sd)=t.TempF32(nm.FfnDown(g));var fo=o.FeedForward(x1n,wg,wu,wd,"ff_fb");if(!sg)wg.Dispose();if(!su)wu.Dispose();if(!sd)wd.Dispose();x1n.Dispose();var ou=o.Add(x1,fo,"lo_fb");x1.Dispose();fo.Dispose();x.Dispose();return ou;}
}
