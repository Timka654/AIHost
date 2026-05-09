using System.Numerics;
using AIHost.Compute;
using AIHost.ICompute;
using AIHost.Inference;
using Microsoft.Extensions.Logging;

namespace AIHost.Compute.Formats;

/// <summary>
/// Qwen3.6 hybrid format: Gated Delta Net (linear attention) + Full Attention layers.
/// </summary>
public class QwenHybridFormat : ITransformerFormat
{
    private readonly ILogger<QwenHybridFormat> _logger = AppLogger.Create<QwenHybridFormat>();

    public Tensor ApplyLayer(TransformerBase transformer, Tensor x, int layerIdx, uint position, KVCache? kvCache, SSMState? ssmState)
    {
        int g = transformer.GlobalLayer(layerIdx);
        var nm = transformer._nameMapper!;
        bool hasCombinedQKV = transformer.HasWeight($"blk.{g}.attn_qkv.weight");
        bool hasSeparateQKV = transformer.HasWeight($"blk.{g}.attn_q.weight");
        if (!hasCombinedQKV && !hasSeparateQKV)
            return ApplyLayerSSMFallback(transformer, x, g, nm);
        if (hasSeparateQKV)
            return ApplyLayerTypeB(transformer, x, g, layerIdx, position, kvCache);
        return ApplyLayerCombinedQKV(transformer, x, g, layerIdx, position, kvCache, nm, ssmState);
    }

    // ─── Type B (attention-only) ───────────────────────────────────────────

    private static Tensor ApplyLayerTypeB(TransformerBase transformer, Tensor x, int g, int layerIdx, uint position, KVCache? kvCache)
    {
        var ops = transformer.Ops;
        int seqLen = x.Shape[0];
        bool useBatch = seqLen > 1;

        var wAN = transformer.TempF32Named($"blk.{g}.attn_norm.weight");
        var wQ = transformer.TempF32Named($"blk.{g}.attn_q.weight");
        var wK = transformer.TempF32Named($"blk.{g}.attn_k.weight");
        var wV = transformer.TempF32Named($"blk.{g}.attn_v.weight");
        var wO = transformer.TempF32Named($"blk.{g}.attn_output.weight");
        var wPN = transformer.TempF32Named($"blk.{g}.post_attention_norm.weight");
        var wG = transformer.TempF32Named($"blk.{g}.ffn_gate.weight");
        var wU = transformer.TempF32Named($"blk.{g}.ffn_up.weight");
        var wD = transformer.TempF32Named($"blk.{g}.ffn_down.weight");
        if (useBatch) ops.BeginBatch();

        var xNorm = ops.Clone(x, "attn_norm_in");
        ops.LayerNorm(xNorm, wAN.tensor);
        if (!wAN.isScratch) ops.DeferExternal(wAN.tensor);

        var rawQ = ops.MatMulWeights(xNorm, wQ.tensor, "rawQ");
        var K = ops.MatMulWeights(xNorm, wK.tensor, "K");
        var V = ops.MatMulWeights(xNorm, wV.tensor, "V");
        if (!wQ.isScratch) ops.DeferExternal(wQ.tensor);
        if (!wK.isScratch) ops.DeferExternal(wK.tensor);
        if (!wV.isScratch) ops.DeferExternal(wV.tensor);
        if (useBatch) ops.DeferExternal(xNorm); else xNorm.Dispose();

        int kvDim = K.Shape[1], qTotalDim = rawQ.Shape[1], headDim = 256;
        bool isGatedQ = transformer.HasWeight($"blk.{g}.attn_gate.weight");
        int qDim = transformer.HasWeight($"blk.{g}.attn_output.weight")
            ? transformer._weightCache[$"blk.{g}.attn_output.weight"].Shape[0] : qTotalDim;
        int nQH, qEffectiveDim;
        Tensor gatedQ;

        if (isGatedQ || qTotalDim > qDim)
        {
            var qProj = ops.SliceCols(rawQ, 0, qDim, "q_proj");
            var qGate = ops.SliceCols(rawQ, qDim, qDim, "q_gate");
            if (useBatch) ops.DeferExternal(rawQ); else rawQ.Dispose();
            ops.SiLU(qGate);
            gatedQ = ops.Multiply(qProj, qGate, "gated_q");
            if (useBatch) ops.DeferExternal(qProj); else qProj.Dispose();
            if (useBatch) ops.DeferExternal(qGate); else qGate.Dispose();
            nQH = qDim / headDim; qEffectiveDim = qDim;
        }
        else { gatedQ = rawQ; nQH = qTotalDim / headDim; qEffectiveDim = qTotalDim; }

        var qNormW = transformer.GetOrBuildTiledNorm($"blk.{g}.attn_q_norm.weight", headDim, qEffectiveDim);
        var kNormW = transformer.GetOrBuildTiledNorm($"blk.{g}.attn_k_norm.weight", headDim, kvDim);
        if (qNormW != null) ops.LayerNorm(gatedQ, qNormW);
        if (kNormW != null) ops.LayerNorm(K, kNormW);

        int nKVH = kvDim / headDim;
        ops.ApplyRoPEFull(gatedQ, position, nQH, headDim, transformer._ropeFreqBase);
        ops.ApplyRoPEFull(K, position, nKVH, headDim, transformer._ropeFreqBase);

        Tensor attnOut;
        if (kvCache != null) { kvCache.Add(layerIdx, K, V); var (cK, cV) = kvCache.Get(layerIdx); attnOut = ops.MultiHeadAttention(gatedQ, cK!, cV!, nQH, position, "attn_out_B"); ops.DeferExternal(gatedQ); }
        else { attnOut = ops.MultiHeadAttention(gatedQ, K, V, nQH, position, "attn_out_B"); ops.DeferExternal(gatedQ); ops.DeferExternal(K); ops.DeferExternal(V); }

        Tensor attnProj = wO.tensor.Shape[0] == attnOut.Shape[1] ? ops.MatMulWeights(attnOut, wO.tensor, "attn_proj_B") : ops.MatMulWeightsT(attnOut, wO.tensor, "attn_proj_B");
        if (!wO.isScratch) ops.DeferExternal(wO.tensor);
        if (useBatch) ops.DeferExternal(attnOut); else attnOut.Dispose();
        var x1 = ops.Add(x, attnProj, "x_after_attn_B");
        if (useBatch) ops.DeferExternal(attnProj); else attnProj.Dispose();

        var x1Norm = ops.Clone(x1, "post_attn_norm_in"); ops.LayerNorm(x1Norm, wPN.tensor);
        if (!wPN.isScratch) ops.DeferExternal(wPN.tensor);
        var ffnOut = ops.FeedForward(x1Norm, wG.tensor, wU.tensor, wD.tensor, "ffn_B");
        if (!wG.isScratch) ops.DeferExternal(wG.tensor); if (!wU.isScratch) ops.DeferExternal(wU.tensor); if (!wD.isScratch) ops.DeferExternal(wD.tensor);
        if (useBatch) ops.DeferExternal(x1Norm); else x1Norm.Dispose();

        var output = ops.Add(x1, ffnOut, "layer_out_B");
        if (useBatch) ops.DeferExternal(x1); else x1.Dispose();
        if (useBatch) ops.DeferExternal(ffnOut); else ffnOut.Dispose();
        if (useBatch) ops.DeferExternal(x); else x.Dispose();
        if (useBatch) ops.Flush();
        return output;
    }

    // ─── Type A (combined QKV + Gated Delta Net) ──────────────────────────

    private static Tensor ApplyLayerCombinedQKV(TransformerBase transformer, Tensor x, int g, int layerIdx, uint position,
                                           KVCache? kvCache, TensorNameMapper nm, SSMState? ssmState)
    {
        var ops = transformer.Ops; int seqLen = x.Shape[0]; bool useBatch = seqLen > 1;
        var wAN = transformer.TempF32(nm.AttnNorm(g));
        var wQKV = transformer.TempF32(nm.AttnQKV(g));
        string wAOkey = transformer.HasWeight($"blk.{g}.ssm_out.weight") ? $"blk.{g}.ssm_out.weight" : nm.AttnOutput(g);
        var wAO = transformer.TempF32Named(wAOkey);
        var wFN = transformer.TempF32(nm.FfnNorm(g)); var wG = transformer.TempF32(nm.FfnGate(g));
        var wU = transformer.TempF32(nm.FfnUp(g)); var wD = transformer.TempF32(nm.FfnDown(g));
        if (useBatch) ops.BeginBatch();

        var xNorm = ops.Clone(x, "attn_norm_in"); ops.LayerNorm(xNorm, wAN.tensor);
        if (!wAN.isScratch) ops.DeferExternal(wAN.tensor);
        var qkv = ops.MatMulWeights(xNorm, wQKV.tensor, "qkv");
        if (!wQKV.isScratch) ops.DeferExternal(wQKV.tensor);

        int headDim = 256, qDim = transformer._numHeads * headDim, kvDim = transformer._numKVHeads * headDim, nKvH = transformer._numKVHeads;
        int totalQKV = qkv.Shape[1], expectedBase = qDim + 2 * kvDim;
        bool hasQGate = totalQKV > expectedBase;
        int qGateDim = hasQGate ? totalQKV - expectedBase : 0;

        var Q = ops.SliceCols(qkv, 0, qDim, "Q");
        var K = ops.SliceCols(qkv, qDim, kvDim, "K");
        var V = ops.SliceCols(qkv, qDim + kvDim, kvDim, "V");
        Tensor gatedQ;
        if (hasQGate) {
            var qGate = ops.SliceCols(qkv, qDim + 2 * kvDim, qGateDim, "q_gate"); ops.SiLU(qGate);
            var qGateTiled = ops.Concat(qGate, qGate, 1, "q_gate_tiled_2x"); qGateTiled = ops.Concat(qGateTiled, qGate, 1, "q_gate_tiled_3x");
            if (useBatch) ops.DeferExternal(qGate); else qGate.Dispose();
            gatedQ = ops.Multiply(Q, qGateTiled, "gated_q");
            if (useBatch) ops.DeferExternal(Q); else Q.Dispose();
            if (useBatch) ops.DeferExternal(qGateTiled); else qGateTiled.Dispose();
        } else { gatedQ = Q; }
        if (useBatch) ops.DeferExternal(qkv); else qkv.Dispose();

        ops.ApplyRoPEFull(gatedQ, position, transformer._numHeads, headDim, transformer._ropeFreqBase);
        ops.ApplyRoPEFull(K, position, nKvH, headDim, transformer._ropeFreqBase);

        Tensor attnOut;
        if (kvCache != null) { kvCache.Add(layerIdx, K, V); var (cK, cV) = kvCache.Get(layerIdx); attnOut = ops.MultiHeadAttention(gatedQ, cK!, cV!, transformer._numHeads, position, "attn_out"); ops.DeferExternal(gatedQ); }
        else { attnOut = ops.MultiHeadAttention(gatedQ, K, V, transformer._numHeads, position, "attn_out"); ops.DeferExternal(gatedQ); ops.DeferExternal(K); ops.DeferExternal(V); }

        Tensor attnProj = wAO.tensor.Shape[0] == attnOut.Shape[1] ? ops.MatMulWeights(attnOut, wAO.tensor, "attn_proj") : ops.MatMulWeightsT(attnOut, wAO.tensor, "attn_proj");
        if (!wAO.isScratch) ops.DeferExternal(wAO.tensor);
        if (useBatch) ops.DeferExternal(attnOut); else attnOut.Dispose();
        if (useBatch) ops.DeferExternal(xNorm); else xNorm.Dispose();
        var x1 = ops.Add(x, attnProj, "x_after_attn");
        if (useBatch) ops.DeferExternal(attnProj); else attnProj.Dispose();

        var x1Norm = ops.Clone(x1, "post_attn_norm_in"); ops.LayerNorm(x1Norm, wFN.tensor);
        if (!wFN.isScratch) ops.DeferExternal(wFN.tensor);

        Tensor x2;
        if (ssmState != null && transformer.HasWeight($"blk.{g}.ssm_a")) {
            ops.Flush();
            var ssmOut = ApplySSMRecurrence(transformer, x1, x1Norm, g, ssmState, seqLen);
            x1.Dispose(); x2 = ssmOut;
            ops.BeginBatch();
        } else { x2 = x1; }

        var ffnOut = ops.FeedForward(x1Norm, wG.tensor, wU.tensor, wD.tensor, "ffn_out");
        if (!wG.isScratch) ops.DeferExternal(wG.tensor); if (!wU.isScratch) ops.DeferExternal(wU.tensor); if (!wD.isScratch) ops.DeferExternal(wD.tensor);
        if (useBatch) ops.DeferExternal(x1Norm); else x1Norm.Dispose();

        var output = ops.Add(x2, ffnOut, "layer_out");
        if (useBatch) ops.DeferExternal(x2); else x2.Dispose();
        if (useBatch) ops.DeferExternal(ffnOut); else ffnOut.Dispose();
        if (useBatch) ops.DeferExternal(x); else x.Dispose();
        if (useBatch) ops.Flush();
        return output;
    }

    // ─── SSM Recurrence ────────────────────────────────────────────────────

    private struct PerTokenState { public float[] z, beta, alpha, gate, qkvMixed; }

    private static Tensor ApplySSMRecurrence(TransformerBase transformer, Tensor xResidual, Tensor xNorm,
                                       int g, SSMState ssmState, int seqLen)
    {
        const int HEAD_V_DIM = 128, N_V_HEADS = 48, N_K_HEADS = 16;
        const int KEY_DIM = N_K_HEADS * HEAD_V_DIM, VALUE_DIM = N_V_HEADS * HEAD_V_DIM;
        const int CONV_DIM = 2 * KEY_DIM + VALUE_DIM, CONV_KERNEL = 4;
        var ops = transformer.Ops; int dModel = xNorm.Shape[1];

        float[] Load(string name) { var (t, s) = transformer.TempF32Named(name); var d = t.ReadData(); if (!s) t.Dispose(); return d; }

        var Wdt = Load($"blk.{g}.ssm_alpha.weight");   // [dModel, N_V_HEADS] row-major
        var Wb  = Load($"blk.{g}.ssm_beta.weight");    // [dModel, N_V_HEADS]
        var ssA = Load($"blk.{g}.ssm_a");
        var dtBias = Load($"blk.{g}.ssm_dt.bias");
        var sNw = Load($"blk.{g}.ssm_norm.weight");
        var convW = Load($"blk.{g}.ssm_conv1d.weight"); // [4, CONV_DIM] row-major
        var wZ = Load($"blk.{g}.attn_gate.weight");    // [dModel, VALUE_DIM]
        var wQKV = Load($"blk.{g}.attn_qkv.weight");   // [dModel, CONV_DIM]
        var wOut = Load($"blk.{g}.ssm_out.weight");    // [VALUE_DIM, dModel]

        var xData = xNorm.ReadData();

        // TRANSPOSE: GGUF row-major [dModel,dim2] → MatVecMulSIMD expects [dim2,dModel]
        var wZT = Transpose(wZ, dModel, VALUE_DIM);
        var WbT = Transpose(Wb, dModel, N_V_HEADS);
        var WdtT = Transpose(Wdt, dModel, N_V_HEADS);
        var wQKVT = Transpose(wQKV, dModel, CONV_DIM);
        var wOutT = Transpose(wOut, VALUE_DIM, dModel);

        // Phase 1: parallel matvecs (stateless)
        var perToken = new PerTokenState[seqLen];
        Parallel.For(0, seqLen, t => {
            int off = t * dModel; var pt = new PerTokenState();
            pt.z = MatVecMulSIMD(xData, off, wZT, VALUE_DIM, dModel);
            var rB = MatVecMulSIMD(xData, off, WbT, N_V_HEADS, dModel);
            pt.beta = new float[N_V_HEADS]; for (int n = 0; n < N_V_HEADS; n++) pt.beta[n] = 1f/(1f+MathF.Exp(-rB[n]));
            var rA = MatVecMulSIMD(xData, off, WdtT, N_V_HEADS, dModel);
            pt.alpha = new float[N_V_HEADS]; for (int n = 0; n < N_V_HEADS; n++) pt.alpha[n] = MathF.Log(1f+MathF.Exp(rA[n]+dtBias[n]));
            pt.gate = new float[N_V_HEADS]; for (int n = 0; n < N_V_HEADS; n++) pt.gate[n] = pt.alpha[n]*ssA[n];
            pt.qkvMixed = MatVecMulSIMD(xData, off, wQKVT, CONV_DIM, dModel);
            perToken[t] = pt;
        });

        // Phase 2: sequential conv1d + DeltaNet
        var st = ssmState.GetLayer(g);
        var ySeq = new float[seqLen * VALUE_DIM];
        float sc = 1f/MathF.Sqrt(HEAD_V_DIM);
        const float eps = 1e-5f;
        for (int t = 0; t < seqLen; t++) {
            var pt = perToken[t];
            // conv1d (both convWindow and convW are row-major)
            var cw = ssmState.UpdateConvState(g, pt.qkvMixed);
            var cv = new float[CONV_DIM];
            for (int c = 0; c < CONV_DIM; c++) {
                float s = 0;
                s += cw[0 * CONV_DIM + c] * convW[0 * CONV_DIM + c];
                s += cw[1 * CONV_DIM + c] * convW[1 * CONV_DIM + c];
                s += cw[2 * CONV_DIM + c] * convW[2 * CONV_DIM + c];
                s += cw[3 * CONV_DIM + c] * convW[3 * CONV_DIM + c];
                cv[c] = s;
            }
            var qc = new float[KEY_DIM]; var kc = new float[KEY_DIM]; var vc = new float[VALUE_DIM];
            for (int i = 0; i < KEY_DIM; i++) { float f = cv[i];       qc[i] = f/(1f+MathF.Exp(-f)); }
            for (int i = 0; i < KEY_DIM; i++) { float f = cv[KEY_DIM+i];  kc[i] = f/(1f+MathF.Exp(-f)); }
            for (int i = 0; i < VALUE_DIM;i++) { float f = cv[2*KEY_DIM+i];vc[i] = f/(1f+MathF.Exp(-f)); }
            for (int h = 0; h < N_K_HEADS; h++) {
                int b = h*HEAD_V_DIM; float q2=0,k2=0;
                for (int d=0;d<HEAD_V_DIM;d++){q2+=qc[b+d]*qc[b+d];k2+=kc[b+d]*kc[b+d];}
                float nq=MathF.Sqrt(q2+eps),nk=MathF.Sqrt(k2+eps);
                for (int d=0;d<HEAD_V_DIM;d++){qc[b+d]/=nq;kc[b+d]/=nk;}
            }
            var yt = new float[VALUE_DIM];
            for (int n = 0; n < N_V_HEADS; n++) {
                int h=n%N_K_HEADS,sB=n*HEAD_V_DIM*HEAD_V_DIM,vB=n*HEAD_V_DIM,kB=h*HEAD_V_DIM,qB=h*HEAD_V_DIM;
                float gE=MathF.Exp(pt.gate[n]);
                for (int i=0;i<HEAD_V_DIM;i++){int rB=sB+i*HEAD_V_DIM;for(int j=0;j<HEAD_V_DIM;j++)st[rB+j]*=gE;}
                var sk=new float[HEAD_V_DIM];
                for (int d=0;d<HEAD_V_DIM;d++){float sm=0;int rB=sB+d*HEAD_V_DIM;for(int j=0;j<HEAD_V_DIM;j++)sm+=st[rB+j]*kc[kB+j];sk[d]=sm;}
                var dv=new float[HEAD_V_DIM];
                for(int d=0;d<HEAD_V_DIM;d++)dv[d]=(vc[vB+d]-sk[d])*pt.beta[n];
                for(int i=0;i<HEAD_V_DIM;i++){int rB=sB+i*HEAD_V_DIM;float ki=kc[kB+i];for(int j=0;j<HEAD_V_DIM;j++)st[rB+j]+=ki*dv[j];}
                for(int d=0;d<HEAD_V_DIM;d++){float sm=0;int rB=sB+d*HEAD_V_DIM;for(int j=0;j<HEAD_V_DIM;j++)sm+=st[rB+j]*qc[qB+j];yt[vB+d]=sm*sc;}
            }
            for (int n = 0; n < N_V_HEADS; n++) {
                int gB=n*HEAD_V_DIM;float s2=0;
                for (int d=0;d<HEAD_V_DIM;d++){float v=yt[gB+d];s2+=v*v;}
                float ri=1f/MathF.Sqrt(s2/HEAD_V_DIM+1e-5f);
                float zs=pt.z[gB]/(1f+MathF.Exp(-pt.z[gB]));
                for(int d=0;d<HEAD_V_DIM;d++)yt[gB+d]=yt[gB+d]*ri*sNw[d]*zs;
            }
            Buffer.BlockCopy(yt,0,ySeq,t*VALUE_DIM*sizeof(float),VALUE_DIM*sizeof(float));
        }

        // Phase 3: parallel output projection
        var rd = new float[seqLen * dModel];
        Parallel.For(0, seqLen, t => {
            var row = MatVecMulSIMD(ySeq, t*VALUE_DIM, wOutT, dModel, VALUE_DIM);
            Buffer.BlockCopy(row,0,rd,t*dModel*sizeof(float),dModel*sizeof(float));
        });
        var xrd = xResidual.ReadData();
        for (int i=0;i<rd.Length;i++)rd[i]+=xrd[i];
        return Tensor.FromData(ops.Device, rd, new TensorShape(new[]{seqLen,dModel}), "ssm_out");
    }

    // ─── Transpose ─────────────────────────────────────────────────────────

    /// GGUF row-major [dModel,dim2]: element(k,n) at k*dim2 + n
    /// MatVecMulSIMD needs: weightT[n*dModel + k] = gguf[k*dim2 + n]
    private static float[] Transpose(float[] data, int dModel, int dim2)
    {
        var r = new float[data.Length];
        for (int k = 0; k < dModel; k++)
        { int sb = k * dim2; for (int n = 0; n < dim2; n++) r[n * dModel + k] = data[sb + n]; }
        return r;
    }

    // ─── MatVecMulSIMD ─────────────────────────────────────────────────────

    private static float[] MatVecMulSIMD(float[] x, int off, float[] w, int rdim, int dm)
    {
        var r = new float[rdim]; int vs = Vector<float>.Count;
        for (int j = 0; j < rdim; j++) {
            int wo = j * dm; float s = 0; int k = 0;
            if (vs > 1) { var vsu = Vector<float>.Zero;
                for (; k <= dm - vs; k += vs) { vsu += new Vector<float>(x, off+k) * new Vector<float>(w, wo+k); }
                for (int z = 0; z < vs; z++) s += vsu[z]; }
            for (; k < dm; k++) s += x[off+k] * w[wo+k];
            r[j] = s;
        } return r;
    }

    // ─── GPU decode (seqLen=1) ─────────────────────────────────────────────

    private static Tensor ApplySSMDecodeGPU(TransformerBase transformer, Tensor xResidual, Tensor xNorm,
        int g, SSMState ssmState, ComputeOps ops, int dModel)
    {
        const int VALUE_DIM = 6144, CONV_DIM = 10240;
        var (cw,s1)=transformer.TempF32Named($"blk.{g}.ssm_conv1d.weight");
        var (wq,s2)=transformer.TempF32Named($"blk.{g}.attn_qkv.weight");
        var (wz,s3)=transformer.TempF32Named($"blk.{g}.attn_gate.weight");
        var (wb,s4)=transformer.TempF32Named($"blk.{g}.ssm_beta.weight");
        var (wa,s5)=transformer.TempF32Named($"blk.{g}.ssm_alpha.weight");
        var (db,s6)=transformer.TempF32Named($"blk.{g}.ssm_dt.bias");
        var (sa,s7)=transformer.TempF32Named($"blk.{g}.ssm_a");
        var (sn,s8)=transformer.TempF32Named($"blk.{g}.ssm_norm.weight");
        var (wo,s9)=transformer.TempF32Named($"blk.{g}.ssm_out.weight");
        var csb=ssmState.GetGpuConvBuffer(g); var ssb=ssmState.GetGpuStateBuffer(g);
        var scr=Tensor.Create(ops.Device,new TensorShape(CONV_DIM+VALUE_DIM+48*3),DataType.F32,"ssm_scratch");
        var out_=Tensor.Create(ops.Device,new TensorShape(VALUE_DIM),DataType.F32,"ssm_output");
        ops.SsmGdnDecode(xNorm,cw,new Tensor(csb,new TensorShape(CONV_DIM*3),DataType.F32,"cs"),
            wq,wz,wb,wa,db,sa,scr,new Tensor(ssb,new TensorShape(SSMState.STATE_DIM),DataType.F32,"ss"),sn,out_);
        var yt=out_.ReadData(); var woD=wo.ReadData();
        var rd=new float[dModel];
        for(int j=0;j<dModel;j++){float s=0;for(int i=0;i<VALUE_DIM;i++)s+=yt[i]*woD[i+j*VALUE_DIM];rd[j]=s;}
        var xrd=xResidual.ReadData(); for(int j=0;j<dModel;j++)rd[j]+=xrd[j];
        scr.Dispose();out_.Dispose();
        if(!s1)cw.Dispose();if(!s2)wq.Dispose();if(!s3)wz.Dispose();if(!s4)wb.Dispose();if(!s5)wa.Dispose();
        if(!s6)db.Dispose();if(!s7)sa.Dispose();if(!s8)sn.Dispose();if(!s9)wo.Dispose();
        ssmState.SyncGpuStateToCpu(g);
        return Tensor.FromData(ops.Device,rd,new TensorShape(new[]{1,dModel}),"ssm_decode_out");
    }

    // ─── Fallback ──────────────────────────────────────────────────────────

    private static Tensor ApplyLayerSSMFallback(TransformerBase transformer, Tensor x, int g, TensorNameMapper nm)
    {
        var ops = transformer.Ops; var xNorm = ops.Clone(x, "attn_norm_in");
        var (wAN,sAN)=transformer.TempF32(nm.AttnNorm(g)); ops.LayerNorm(xNorm,wAN); if(!sAN)wAN.Dispose();
        Tensor ao; if(transformer.HasWeight(nm.AttnQKV(g))){var(w,s)=transformer.TempF32(nm.AttnQKV(g));var q=ops.MatMulWeights(xNorm,w,"qkv_f");if(!s)w.Dispose();
            int qd=transformer._numHeads*256,kd=transformer._numKVHeads*256;var Q=ops.SliceCols(q,0,qd,"Q_f");var K=ops.SliceCols(q,qd,kd,"K_f");var V=ops.SliceCols(q,qd+kd,kd,"V_f");q.Dispose();
            ops.ApplyRoPEFull(Q,0,transformer._numHeads,256,transformer._ropeFreqBase);ops.ApplyRoPEFull(K,0,transformer._numKVHeads,256,transformer._ropeFreqBase);
            ao=ops.MultiHeadAttention(Q,K,V,transformer._numHeads,0,"attn_f");ops.DeferExternal(Q);ops.DeferExternal(K);ops.DeferExternal(V);}
        else{var(wq,sq)=transformer.TempF32(nm.AttnQ(g));var(wk,sk)=transformer.TempF32(nm.AttnK(g));var(wv,sv)=transformer.TempF32(nm.AttnV(g));
            var Q=ops.MatMulWeights(xNorm,wq,"Q_f");var K=ops.MatMulWeights(xNorm,wk,"K_f");var V=ops.MatMulWeights(xNorm,wv,"V_f");
            if(!sq)wq.Dispose();if(!sk)wk.Dispose();if(!sv)wv.Dispose();
            ops.ApplyRoPEFull(Q,0,transformer._numHeads,256,transformer._ropeFreqBase);ops.ApplyRoPEFull(K,0,transformer._numKVHeads,256,transformer._ropeFreqBase);
            ao=ops.MultiHeadAttention(Q,K,V,transformer._numHeads,0,"attn_f");ops.DeferExternal(Q);ops.DeferExternal(K);ops.DeferExternal(V);}
        xNorm.Dispose();var(wo,so)=transformer.TempF32(nm.AttnOutput(g));
        Tensor ap=wo.Shape[0]==ao.Shape[1]?ops.MatMulWeights(ao,wo,"ap_f"):ops.MatMulWeightsT(ao,wo,"ap_f");if(!so)wo.Dispose();ao.Dispose();
        var x1=ops.Add(x,ap,"xa_f");ap.Dispose();var x1n=ops.Clone(x1,"pn_f");var(wp,sp)=transformer.TempF32(nm.FfnNorm(g));ops.LayerNorm(x1n,wp);if(!sp)wp.Dispose();
        var(wg,sg)=transformer.TempF32(nm.FfnGate(g));var(wu,su)=transformer.TempF32(nm.FfnUp(g));var(wd,sd)=transformer.TempF32(nm.FfnDown(g));
        var fo=ops.FeedForward(x1n,wg,wu,wd,"ff_f");if(!sg)wg.Dispose();if(!su)wu.Dispose();if(!sd)wd.Dispose();x1n.Dispose();
        var o=ops.Add(x1,fo,"lo_f");x1.Dispose();fo.Dispose();x.Dispose();return o;
    }
}
