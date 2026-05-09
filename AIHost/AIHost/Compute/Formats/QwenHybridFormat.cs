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
        int g = t.GlobalLayer(li); var nm = t._nameMapper!;
        bool hc = t.HasWeight($"blk.{g}.attn_qkv.weight"), hs = t.HasWeight($"blk.{g}.attn_q.weight");
        if (!hc && !hs) return Fallback(t, x, g, nm);
        if (hs) return TypeB(t, x, g, li, pos, kvc);
        return TypeA(t, x, g, li, pos, kvc, nm, ss);
    }

    // Transpose GGUF row-major [dm, d2] → SIMD [d2, dm]
    static float[] Tr(float[] w, int dm, int d2) { var r = new float[w.Length]; for (int k = 0; k < dm; k++) { int s = k * d2; for (int n = 0; n < d2; n++) r[n * dm + k] = w[s + n]; } return r; }

    // SIMD matvec with TRANSPOSED weights: r[j] += x[off+k] * wT[j*dm+k]
    static void MV(Span<float> r, float[] x, int xo, float[] wT, int d2, int dm) { int vs = Vector<float>.Count;
        for (int j = 0; j < d2; j++) { int wo = j * dm; float s = 0; int k = 0;
            if (vs > 1) { var vsu = Vector<float>.Zero; for (; k <= dm - vs; k += vs) vsu += new Vector<float>(x, xo + k) * new Vector<float>(wT, wo + k); for (int z = 0; z < vs; z++) s += vsu[z]; }
            for (; k < dm; k++) s += x[xo + k] * wT[wo + k]; r[j] = s; } }

    // ─── Type A ────────────────────────────────────────────────────────
    static Tensor TypeA(TransformerBase t, Tensor x, int g, int li, uint pos, KVCache? kvc, TensorNameMapper nm, SSMState? ss)
    {
        var o = t.Ops; int sl = x.Shape[0]; bool b = sl > 1;
        var a = t.TempF32(nm.AttnNorm(g)); var qkw = t.TempF32(nm.AttnQKV(g));
        string ak = t.HasWeight($"blk.{g}.ssm_out.weight") ? $"blk.{g}.ssm_out.weight" : nm.AttnOutput(g);
        var aoW = t.TempF32Named(ak); var fn = t.TempF32(nm.FfnNorm(g)); var wg = t.TempF32(nm.FfnGate(g)); var wu = t.TempF32(nm.FfnUp(g)); var wd = t.TempF32(nm.FfnDown(g));
        if (b) o.BeginBatch();
        var xn = o.Clone(x, "an"); o.LayerNorm(xn, a.tensor); if (!a.isScratch) o.DeferExternal(a.tensor);
        var qkv = o.MatMulWeights(xn, qkw.tensor, "qkv"); if (!qkw.isScratch) o.DeferExternal(qkw.tensor);
        int hd = 256, qd = t._numHeads * hd, kd = t._numKVHeads * hd;
        int tq = qkv.Shape[1], eb = qd + 2 * kd; bool hg = tq > eb; int qgd = hg ? tq - eb : 0;
        var Q = o.SliceCols(qkv, 0, qd, "Q"); var K = o.SliceCols(qkv, qd, kd, "K"); var V = o.SliceCols(qkv, qd + kd, kd, "V");
        Tensor gq; if (hg) { var qg = o.SliceCols(qkv, qd + 2 * kd, qgd, "qg"); o.SiLU(qg); var qt = o.Concat(qg, qg, 1, "qt2"); qt = o.Concat(qt, qg, 1, "qt3"); if (b) o.DeferExternal(qg); else qg.Dispose(); gq = o.Multiply(Q, qt, "gq"); if (b) o.DeferExternal(Q); else Q.Dispose(); if (b) o.DeferExternal(qt); else qt.Dispose(); } else { gq = Q; }
        if (b) o.DeferExternal(qkv); else qkv.Dispose();
        o.ApplyRoPEFull(gq, pos, t._numHeads, hd, t._ropeFreqBase); o.ApplyRoPEFull(K, pos, t._numKVHeads, hd, t._ropeFreqBase);
        Tensor ao; if (kvc != null) { kvc.Add(li, K, V); var (ck, cv) = kvc.Get(li); ao = o.MultiHeadAttention(gq, ck!, cv!, t._numHeads, pos, "ao"); o.DeferExternal(gq); }
        else { ao = o.MultiHeadAttention(gq, K, V, t._numHeads, pos, "ao"); o.DeferExternal(gq); o.DeferExternal(K); o.DeferExternal(V); }
        Tensor ap = aoW.tensor.Shape[0] == ao.Shape[1] ? o.MatMulWeights(ao, aoW.tensor, "ap") : o.MatMulWeightsT(ao, aoW.tensor, "ap");
        if (!aoW.isScratch) o.DeferExternal(aoW.tensor); if (b) o.DeferExternal(ao); else ao.Dispose(); if (b) o.DeferExternal(xn); else xn.Dispose();
        var x1 = o.Add(x, ap, "xa"); if (b) o.DeferExternal(ap); else ap.Dispose();
        var x1n = o.Clone(x1, "pn"); o.LayerNorm(x1n, fn.tensor); if (!fn.isScratch) o.DeferExternal(fn.tensor);
        Tensor x2; if (ss != null && t.HasWeight($"blk.{g}.ssm_a")) { o.Flush(); var so = SSM(t, x1, x1n, g, ss, sl); x1.Dispose(); x2 = so; o.BeginBatch(); } else { x2 = x1; }
        var fo = o.FeedForward(x1n, wg.tensor, wu.tensor, wd.tensor, "ff"); if (!wg.isScratch) o.DeferExternal(wg.tensor); if (!wu.isScratch) o.DeferExternal(wu.tensor); if (!wd.isScratch) o.DeferExternal(wd.tensor);
        if (b) o.DeferExternal(x1n); else x1n.Dispose(); var ou = o.Add(x2, fo, "lo"); if (b) o.DeferExternal(x2); else x2.Dispose(); if (b) o.DeferExternal(fo); else fo.Dispose(); if (b) o.DeferExternal(x); else x.Dispose();
        if (b) o.Flush(); return ou;
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

    // ─── SSM recurrence (Gated Delta Net) ──────────────────────────────
    static Tensor SSM(TransformerBase t, Tensor xr, Tensor xn, int g, SSMState ss, int sl)
    {
        const int HVD=128,NVH=48,NKH=16,KD=NKH*HVD,VD=NVH*HVD,CD=2*KD+VD; int dm=xn.Shape[1];
        float[] Ld(string n){var(tt,sc)=t.TempF32Named(n);var d=tt.ReadData();if(!sc)tt.Dispose();return d;}
        var xd=xn.ReadData();
        var Wdt=Ld($"blk.{g}.ssm_alpha.weight");var Wb=Ld($"blk.{g}.ssm_beta.weight");var ssA=Ld($"blk.{g}.ssm_a");
        var dtB=Ld($"blk.{g}.ssm_dt.bias");var sNw=Ld($"blk.{g}.ssm_norm.weight");var cW=Ld($"blk.{g}.ssm_conv1d.weight");
        var wZ=Ld($"blk.{g}.attn_gate.weight");var wQ=Ld($"blk.{g}.attn_qkv.weight");var wO=Ld($"blk.{g}.ssm_out.weight");

        // Phase 1: parallel matvecs — transpose ONE weight at a time to avoid OOM
        var zA=new float[sl*VD];var bA=new float[sl*NVH];var aA=new float[sl*NVH];var gA=new float[sl*NVH];var qA=new float[sl*CD];

        // wZ → z
        { var wT=Tr(wZ,dm,VD);Parallel.For(0,sl,ti=>MV(zA.AsSpan(ti*VD,VD),xd,ti*dm,wT,VD,dm));wT=null;}

        // Wb → rawBeta → beta (sigmoid)
        { var wT=Tr(Wb,dm,NVH);Parallel.For(0,sl,ti=>MV(bA.AsSpan(ti*NVH,NVH),xd,ti*dm,wT,NVH,dm));wT=null;}
        Parallel.For(0,sl,ti=>{int o=ti*NVH;for(int n=0;n<NVH;n++)bA[o+n]=1f/(1f+MathF.Exp(-bA[o+n]));});

        // Wdt → rawAlpha → alpha (softplus+dt_bias)
        { var wT=Tr(Wdt,dm,NVH);Parallel.For(0,sl,ti=>MV(aA.AsSpan(ti*NVH,NVH),xd,ti*dm,wT,NVH,dm));wT=null;}
        Parallel.For(0,sl,ti=>{int o=ti*NVH;for(int n=0;n<NVH;n++)aA[o+n]=MathF.Log(1f+MathF.Exp(aA[o+n]+dtB[n]));});

        // alpha → gate = alpha * ssA
        Parallel.For(0,sl,ti=>{int o=ti*NVH;for(int n=0;n<NVH;n++)gA[o+n]=aA[o+n]*ssA[n];});

        // wQKV → qkvMixed
        { var wT=Tr(wQ,dm,CD);Parallel.For(0,sl,ti=>MV(qA.AsSpan(ti*CD,CD),xd,ti*dm,wT,CD,dm));wT=null;}

        // Phase 2: sequential conv1d + DeltaNet
        var st=ss.GetLayer(g);var ys=new float[sl*VD];float sc=1f/MathF.Sqrt(HVD);const float ep=1e-5f;
        for(int ti=0;ti<sl;ti++){int bt=ti*NVH,gt=ti*NVH,zt=ti*VD,qt=ti*CD;
            var z=new float[VD];Array.Copy(zA,zt,z,0,VD);var beta=new float[NVH];Array.Copy(bA,bt,beta,0,NVH);var gate=new float[NVH];Array.Copy(gA,gt,gate,0,NVH);var qm=new float[CD];Array.Copy(qA,qt,qm,0,CD);
            var cw=ss.UpdateConvState(g,qm);var cv=new float[CD];
            for(int c=0;c<CD;c++){float s=0;s+=cw[0*CD+c]*cW[0*CD+c];s+=cw[1*CD+c]*cW[1*CD+c];s+=cw[2*CD+c]*cW[2*CD+c];s+=cw[3*CD+c]*cW[3*CD+c];cv[c]=s;}
            var qc=new float[KD];var kc=new float[KD];var vc=new float[VD];
            for(int i=0;i<KD;i++){float f=cv[i];qc[i]=f/(1f+MathF.Exp(-f));}
            for(int i=0;i<KD;i++){float f=cv[KD+i];kc[i]=f/(1f+MathF.Exp(-f));}
            for(int i=0;i<VD;i++){float f=cv[2*KD+i];vc[i]=f/(1f+MathF.Exp(-f));}
            for(int h=0;h<NKH;h++){int b_=h*HVD;float q2=0,k2=0;for(int d=0;d<HVD;d++){q2+=qc[b_+d]*qc[b_+d];k2+=kc[b_+d]*kc[b_+d];}float nq=MathF.Sqrt(q2+ep),nk=MathF.Sqrt(k2+ep);for(int d=0;d<HVD;d++){qc[b_+d]/=nq;kc[b_+d]/=nk;}}
            var yt=new float[VD];
            for(int n=0;n<NVH;n++){int h=n%NKH,sB=n*HVD*HVD,vB=n*HVD,kB=h*HVD,qB=h*HVD;float gE=MathF.Exp(gate[n]);for(int i=0;i<HVD;i++){int rB=sB+i*HVD;for(int j=0;j<HVD;j++)st[rB+j]*=gE;}var sk=new float[HVD];for(int d=0;d<HVD;d++){float sm=0;int rB=sB+d*HVD;for(int j=0;j<HVD;j++)sm+=st[rB+j]*kc[kB+j];sk[d]=sm;}var dv=new float[HVD];for(int d=0;d<HVD;d++)dv[d]=(vc[vB+d]-sk[d])*beta[n];for(int i=0;i<HVD;i++){int rB=sB+i*HVD;float ki=kc[kB+i];for(int j=0;j<HVD;j++)st[rB+j]+=ki*dv[j];}for(int d=0;d<HVD;d++){float sm=0;int rB=sB+d*HVD;for(int j=0;j<HVD;j++)sm+=st[rB+j]*qc[qB+j];yt[vB+d]=sm*sc;}}
            for(int n=0;n<NVH;n++){int gB=n*HVD;float s2=0;for(int d=0;d<HVD;d++){float vv=yt[gB+d];s2+=vv*vv;}float ri=1f/MathF.Sqrt(s2/HVD+1e-5f);float zs=z[gB]/(1f+MathF.Exp(-z[gB]));for(int d=0;d<HVD;d++)yt[gB+d]=yt[gB+d]*ri*sNw[d]*zs;}
            Buffer.BlockCopy(yt,0,ys,ti*VD*sizeof(float),VD*sizeof(float));
        }

        // Phase 3: parallel output projection
        var rd=new float[sl*dm];
        { var wT=Tr(wO,VD,dm);Parallel.For(0,sl,ti=>MV(rd.AsSpan(ti*dm,dm),ys,ti*VD,wT,dm,VD));wT=null;}
        var xrd=xr.ReadData();for(int i=0;i<rd.Length;i++)rd[i]+=xrd[i];
        return Tensor.FromData(t.Ops.Device,rd,new TensorShape(new[]{sl,dm}),"ssm_out");
    }

    // ─── Fallback ──────────────────────────────────────────────────────
    static Tensor Fallback(TransformerBase t,Tensor x,int g,TensorNameMapper nm){var o=t.Ops;var xn=o.Clone(x,"an_fb");var(wa,sa)=t.TempF32(nm.AttnNorm(g));o.LayerNorm(xn,wa);if(!sa)wa.Dispose();Tensor ao;if(t.HasWeight(nm.AttnQKV(g))){var(w,s_)=t.TempF32(nm.AttnQKV(g));var q=o.MatMulWeights(xn,w,"q_fb");if(!s_)w.Dispose();int qd=t._numHeads*256,kd=t._numKVHeads*256;var Q=o.SliceCols(q,0,qd,"Q_fb");var K=o.SliceCols(q,qd,kd,"K_fb");var V=o.SliceCols(q,qd+kd,kd,"V_fb");q.Dispose();o.ApplyRoPEFull(Q,0,t._numHeads,256,t._ropeFreqBase);o.ApplyRoPEFull(K,0,t._numKVHeads,256,t._ropeFreqBase);ao=o.MultiHeadAttention(Q,K,V,t._numHeads,0,"a_fb");o.DeferExternal(Q);o.DeferExternal(K);o.DeferExternal(V);}else{var(wq,sq)=t.TempF32(nm.AttnQ(g));var(wk,sk)=t.TempF32(nm.AttnK(g));var(wv,sv)=t.TempF32(nm.AttnV(g));var Q=o.MatMulWeights(xn,wq,"Q_fb");var K=o.MatMulWeights(xn,wk,"K_fb");var V=o.MatMulWeights(xn,wv,"V_fb");if(!sq)wq.Dispose();if(!sk)wk.Dispose();if(!sv)wv.Dispose();o.ApplyRoPEFull(Q,0,t._numHeads,256,t._ropeFreqBase);o.ApplyRoPEFull(K,0,t._numKVHeads,256,t._ropeFreqBase);ao=o.MultiHeadAttention(Q,K,V,t._numHeads,0,"a_fb");o.DeferExternal(Q);o.DeferExternal(K);o.DeferExternal(V);}xn.Dispose();var(wo,so)=t.TempF32(nm.AttnOutput(g));Tensor ap=wo.Shape[0]==ao.Shape[1]?o.MatMulWeights(ao,wo,"ap_fb"):o.MatMulWeightsT(ao,wo,"ap_fb");if(!so)wo.Dispose();ao.Dispose();var x1=o.Add(x,ap,"xa_fb");ap.Dispose();var x1n=o.Clone(x1,"pn_fb");var(wp,sp)=t.TempF32(nm.FfnNorm(g));o.LayerNorm(x1n,wp);if(!sp)wp.Dispose();var(wg,sg)=t.TempF32(nm.FfnGate(g));var(wu,su)=t.TempF32(nm.FfnUp(g));var(wd,sd)=t.TempF32(nm.FfnDown(g));var fo=o.FeedForward(x1n,wg,wu,wd,"ff_fb");if(!sg)wg.Dispose();if(!su)wu.Dispose();if(!sd)wd.Dispose();x1n.Dispose();var ou=o.Add(x1,fo,"lo_fb");x1.Dispose();fo.Dispose();x.Dispose();return ou;}
}
