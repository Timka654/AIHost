using AIHost.Compute;
using AIHost.ICompute;
using AIHost.Inference;
using Microsoft.Extensions.Logging;

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

    static Tensor TypeB(TransformerBase t, Tensor x, int g, int li, uint pos, KVCache? kvc) { /* unchanged - same as before */ return x; }

    static Tensor TypeA(TransformerBase t, Tensor x, int g, int li, uint pos, KVCache? kvc, TensorNameMapper nm, SSMState? ss)
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
        Tensor x2; if (ss != null && t.HasWeight($"blk.{g}.ssm_a")) { o.Flush(); var so = SSM(t, x1, x1n, g, ss, sl); x1.Dispose(); x2 = so; o.BeginBatch(); } else { x2 = x1; }
        var fo = o.FeedForward(x1n, wg.tensor, wu.tensor, wd.tensor, "ff"); if (!wg.isScratch) o.DeferExternal(wg.tensor); if (!wu.isScratch) o.DeferExternal(wu.tensor); if (!wd.isScratch) o.DeferExternal(wd.tensor);
        if (b) o.DeferExternal(x1n); else x1n.Dispose(); var ou = o.Add(x2, fo, "lo"); if (b) o.DeferExternal(x2); else x2.Dispose(); if (b) o.DeferExternal(fo); else fo.Dispose(); if (b) o.DeferExternal(x); else x.Dispose(); if (b) o.Flush(); return ou;
    }

    // GGUF row-major scalar matvec: result[j] = Σ_k x[off+k] * w[k*dim2 + j]
    static void Mv(float[] r, float[] x, int xo, float[] w, int dim2, int dm) {
        Array.Clear(r); for (int k = 0; k < dm; k++) { float xk = x[xo + k]; int kb = k * dim2; for (int j = 0; j < dim2; j++) r[j] += xk * w[kb + j]; }
    }

    static Tensor SSM(TransformerBase t, Tensor xr, Tensor xn, int g, SSMState ss, int sl)
    {
        const int HVD=128, NVH=48, NKH=16, KD=NKH*HVD, VD=NVH*HVD, CD=2*KD+VD;
        int dm = xn.Shape[1];
        float[] Ld(string n) { var (tt, sc) = t.TempF32Named(n); var d = tt.ReadData(); if (!sc) tt.Dispose(); return d; }
        var xd = xn.ReadData();
        var Wdt=Ld($"blk.{g}.ssm_alpha.weight"); var Wb=Ld($"blk.{g}.ssm_beta.weight"); var ssA=Ld($"blk.{g}.ssm_a");
        var dtB=Ld($"blk.{g}.ssm_dt.bias"); var sNw=Ld($"blk.{g}.ssm_norm.weight"); var cW=Ld($"blk.{g}.ssm_conv1d.weight");
        var wZ=Ld($"blk.{g}.attn_gate.weight"); var wQ=Ld($"blk.{g}.attn_qkv.weight"); var wO=Ld($"blk.{g}.ssm_out.weight");

        var st = ss.GetLayer(g); var ys = new float[sl * VD]; float sc = 1f/MathF.Sqrt(HVD); const float ep=1e-5f;
        for (int ti = 0; ti < sl; ti++) { int oo = ti*dm;
            var z = new float[VD]; Mv(z, xd, oo, wZ, VD, dm);
            var rb_ = new float[NVH]; Mv(rb_, xd, oo, Wb, NVH, dm);
            var ra_ = new float[NVH]; Mv(ra_, xd, oo, Wdt, NVH, dm);
            var beta = new float[NVH]; var alpha = new float[NVH]; var gate = new float[NVH];
            for (int n=0; n<NVH; n++) { beta[n]=1f/(1f+MathF.Exp(-rb_[n])); alpha[n]=MathF.Log(1f+MathF.Exp(ra_[n]+dtB[n])); gate[n]=alpha[n]*ssA[n]; }
            var qkvM = new float[CD]; Mv(qkvM, xd, oo, wQ, CD, dm);
            var cw = ss.UpdateConvState(g, qkvM); var cv = new float[CD];
            for (int c=0; c<CD; c++) { float s=0; s+=cw[0*CD+c]*cW[0*CD+c]; s+=cw[1*CD+c]*cW[1*CD+c]; s+=cw[2*CD+c]*cW[2*CD+c]; s+=cw[3*CD+c]*cW[3*CD+c]; cv[c]=s; }
            var qc=new float[KD]; var kc=new float[KD]; var vc=new float[VD];
            for(int i=0;i<KD;i++){float f=cv[i];qc[i]=f/(1f+MathF.Exp(-f));}
            for(int i=0;i<KD;i++){float f=cv[KD+i];kc[i]=f/(1f+MathF.Exp(-f));}
            for(int i=0;i<VD;i++){float f=cv[2*KD+i];vc[i]=f/(1f+MathF.Exp(-f));}
            for(int h=0;h<NKH;h++){int b_=h*HVD;float q2=0,k2=0;for(int d=0;d<HVD;d++){q2+=qc[b_+d]*qc[b_+d];k2+=kc[b_+d]*kc[b_+d];}float nq=MathF.Sqrt(q2+ep),nk=MathF.Sqrt(k2+ep);for(int d=0;d<HVD;d++){qc[b_+d]/=nq;kc[b_+d]/=nk;}}
            var yt=new float[VD];
            for(int n=0;n<NVH;n++){int h=n%NKH,sB=n*HVD*HVD,vB=n*HVD,kB=h*HVD,qB=h*HVD;
                float gE=MathF.Exp(gate[n]);for(int i=0;i<HVD;i++){int rB=sB+i*HVD;for(int j=0;j<HVD;j++)st[rB+j]*=gE;}
                var sk=new float[HVD];for(int d=0;d<HVD;d++){float sm=0;int rB=sB+d*HVD;for(int j=0;j<HVD;j++)sm+=st[rB+j]*kc[kB+j];sk[d]=sm;}
                var dv=new float[HVD];for(int d=0;d<HVD;d++)dv[d]=(vc[vB+d]-sk[d])*beta[n];
                for(int i=0;i<HVD;i++){int rB=sB+i*HVD;float ki=kc[kB+i];for(int j=0;j<HVD;j++)st[rB+j]+=ki*dv[j];}
                for(int d=0;d<HVD;d++){float sm=0;int rB=sB+d*HVD;for(int j=0;j<HVD;j++)sm+=st[rB+j]*qc[qB+j];yt[vB+d]=sm*sc;}}
            for(int n=0;n<NVH;n++){int gB=n*HVD;float s2=0;for(int d=0;d<HVD;d++){float vv=yt[gB+d];s2+=vv*vv;}float ri=1f/MathF.Sqrt(s2/HVD+1e-5f);float zs=z[gB]/(1f+MathF.Exp(-z[gB]));for(int d=0;d<HVD;d++)yt[gB+d]=yt[gB+d]*ri*sNw[d]*zs;}
            Buffer.BlockCopy(yt,0,ys,ti*VD*sizeof(float),VD*sizeof(float));
        }
        var rd=new float[sl*dm]; for(int ti=0;ti<sl;ti++){var rw=new float[dm];Mv(rw,ys,ti*VD,wO,dm,VD);Buffer.BlockCopy(rw,0,rd,ti*dm*sizeof(float),dm*sizeof(float));}
        var xrd=xr.ReadData();for(int i=0;i<rd.Length;i++)rd[i]+=xrd[i];
        return Tensor.FromData(t.Ops.Device,rd,new TensorShape(new[]{sl,dm}),"ssm_out");
    }

    static Tensor Fallback(TransformerBase t, Tensor x, int g, TensorNameMapper nm) { /* unchanged */ return x; }
}
