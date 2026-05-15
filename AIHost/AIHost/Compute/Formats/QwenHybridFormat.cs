using AIHost.Compute;
using AIHost.ICompute;
using AIHost.Inference;
using Microsoft.Extensions.Logging;
using System.Buffers;
using System.Numerics;

namespace AIHost.Compute.Formats;

public class QwenHybridFormat : ITransformerFormat
{
    private readonly ILogger<QwenHybridFormat> _logger = AppLogger.Create<QwenHybridFormat>();

    public Tensor ApplyLayer(TransformerBase t, Tensor x, int li, uint pos, KVCache? kvc, SSMState? ss)
    {
        var _ts = GlobalProfiler.Start();
        int g = t.GlobalLayer(li); var nm = t._nameMapper!;
        // Qwen3.5: attn_qkv.weight exists on ALL 64 layers (SSM + attention).
        // Discriminate: SSM layers have ssm_conv1d.weight; attention layers DO NOT.
        bool isSsm  = t.HasWeight($"blk.{g}.ssm_conv1d.weight");
        bool hasQ   = t.HasWeight($"blk.{g}.attn_q.weight");
        bool hasQkv = t.HasWeight($"blk.{g}.attn_qkv.weight");
        if (g <= 7 && QwenDbgTrace.Once("layer_type", g))
            _logger.LogWarning("[ROUTE] L{L} isSsm={IsSsm} hasQ={HasQ} hasQkv={HasQkv} → {Route}",
                g, isSsm, hasQ, hasQkv, isSsm ? "TypeA" : "TypeB");
        Tensor result;
        if (isSsm)
            result = TypeA(t, x, g, li, pos, kvc, nm, ss);
        else
            result = TypeB(t, x, g, li, pos, kvc, nm);
        GlobalProfiler.End(_ts, "Qwen.ApplyLayer");
        return result;
    }


    // ─── Type A ────────────────────────────────────────────────────────
    // Recurrent (Gated Delta Net) layer: NO standard MultiHeadAttention.
    // SSM_Gpu IS the full attention mechanism for this layer type.
    static Tensor TypeA(TransformerBase t, Tensor x, int g, int li, uint pos, KVCache? kvc, TensorNameMapper nm, SSMState? ss)
    {
        // OPTIMIZATION: Always use batch mode — eliminates 15-20 GPU fence waits per layer in decode.
        // Batch accumulates all GPU dispatches into one command buffer, flushed once at layer end.
        var o = t.Ops; int sl = x.Shape[0]; bool b = true;
        t._logger.LogInformation("[TypeA] Layer {Layer} START sl={SeqLen}", g, sl);
        var _ts = GlobalProfiler.Start();

        // Only load norms and FFN weights here; attn_qkv/ssm weights are loaded inside SSM_Gpu.
        var a = t.TempF32(nm.AttnNorm(g));
        var fn = t.TempF32(nm.FfnNorm(g));
        // For MoE models (Qwen3Next): use shared expert weights if available.
        // The full expert matrix (ffn_gate_exps.weight) is a 3D MoE tensor — wrong for dense FFN.
        string wgName = nm.FfnGateShexp(g) ?? nm.FfnGate(g);
        string wuName = nm.FfnUpShexp(g)   ?? nm.FfnUp(g);
        string wdName = nm.FfnDownShexp(g) ?? nm.FfnDown(g);
        var wg = t.TempF32Named(wgName); var wu = t.TempF32Named(wuName);
        var wd = t.TempF32Named(wdName);
        GlobalProfiler.End(_ts, "LayerA.LoadW");

        if (b) o.BeginBatch();

        // Normalize input — this is the GDN block input (pre-GDN norm = attn_norm).
        var _ts2 = GlobalProfiler.Start();
        var xn = o.Clone(x, "an"); o.LayerNorm(xn, a.tensor);
        if (!a.isScratch) o.DeferExternal(a.tensor);
        GlobalProfiler.End(_ts2, "LayerA.AttnNorm");

        // GDN block (Gated Delta Net) — the sole attention mechanism for recurrent layers.
        var _ts6 = GlobalProfiler.Start();
        Tensor x1;
        if (ss != null)
        {
            try
            {
                // xr=x (residual), xn=attn_norm(x) (GDN input)
                x1 = SSM_Gpu(t, o, x, xn, g, ss, sl);
                // x is consumed as residual inside SSM_Gpu.Add — dispose it now.
                if (b) o.DeferExternal(x); else x.Dispose();
            }
            catch (Exception ex)
            {
                t._logger.LogError(ex, "[SSM_GPU] Failed at layer {Layer}", g);
                throw;
            }
        }
        else
        {
            t._logger.LogWarning("[TypeA] No SSM state for recurrent layer {Layer} — identity pass-through", g);
            x1 = x; // no-op; x disposed via x1 below
        }
        if (b) o.DeferExternal(xn); else xn.Dispose();
        GlobalProfiler.End(_ts6, "LayerA.GDN");

        // Post-GDN norm for FFN (post_attention_norm / ffn_norm)
        var _ts5 = GlobalProfiler.Start();
        var x1n = o.Clone(x1, "pn"); o.LayerNorm(x1n, fn.tensor);
        if (!fn.isScratch) o.DeferExternal(fn.tensor);
        GlobalProfiler.End(_ts5, "LayerA.FfnNorm");

        var _ts7 = GlobalProfiler.Start();
        var fo = o.FeedForward(x1n, wg.tensor, wu.tensor, wd.tensor, "ff");
        if (!wg.isScratch) o.DeferExternal(wg.tensor);
        if (!wu.isScratch) o.DeferExternal(wu.tensor);
        if (!wd.isScratch) o.DeferExternal(wd.tensor);
        GlobalProfiler.End(_ts7, "LayerA.FFN");

        if (b) o.DeferExternal(x1n); else x1n.Dispose();
        var ou = o.Add(x1, fo, "lo");
        if (b) o.DeferExternal(x1); else x1.Dispose();
        if (b) o.DeferExternal(fo); else fo.Dispose();

        var _ts8 = GlobalProfiler.Start();
        if (b) o.Flush();
        GlobalProfiler.End(_ts8, "LayerA.Flush");

        return ou;
    }

    // ─── Type B ────────────────────────────────────────────────────────
    static Tensor TypeB(TransformerBase t, Tensor x, int g, int li, uint pos, KVCache? kvc, TensorNameMapper nm)
    {
        // OPTIMIZATION: Always use batch mode (same as TypeA/SSM_Gpu).
        var o = t.Ops; int sl = x.Shape[0]; bool b = true;
        var anW = t.TempF32Named($"blk.{g}.attn_norm.weight"); var qW = t.TempF32Named($"blk.{g}.attn_q.weight"); var kW = t.TempF32Named($"blk.{g}.attn_k.weight"); var vW = t.TempF32Named($"blk.{g}.attn_v.weight");
        var oW = t.TempF32Named($"blk.{g}.attn_output.weight"); var pW = t.TempF32Named($"blk.{g}.post_attention_norm.weight");
        // MoE fallback: prefer shared-expert weights (ffn_*_shexp) if present (Qwen3Next), else dense ffn_*.weight.
        string gwName = nm.FfnGateShexp(g) ?? $"blk.{g}.ffn_gate.weight";
        string uwName = nm.FfnUpShexp(g)   ?? $"blk.{g}.ffn_up.weight";
        string dwName = nm.FfnDownShexp(g) ?? $"blk.{g}.ffn_down.weight";
        var gW = t.TempF32Named(gwName); var uW = t.TempF32Named(uwName); var dW = t.TempF32Named(dwName);
        if (b) o.BeginBatch();
        var xn = o.Clone(x, "anB"); o.LayerNorm(xn, anW.tensor); if (!anW.isScratch) o.DeferExternal(anW.tensor);
        var rq = o.MatMulWeights(xn, qW.tensor, "rqB"); var K = o.MatMulWeights(xn, kW.tensor, "KB"); var V = o.MatMulWeights(xn, vW.tensor, "VB");
        if (!qW.isScratch) o.DeferExternal(qW.tensor); if (!kW.isScratch) o.DeferExternal(kW.tensor); if (!vW.isScratch) o.DeferExternal(vW.tensor);
        if (b) o.DeferExternal(xn); else xn.Dispose();
        int kd = K.Shape[1], qtd = rq.Shape[1], hd = t._headDim;
        // Per-head QK-norm: norm weight dimension = actual head dim (128 for Qwen3.5, 256 for Qwen3.6).
        int qknDim = t.HasWeight($"blk.{g}.attn_q_norm.weight")
            ? t._weightCache[$"blk.{g}.attn_q_norm.weight"].Shape[0] : hd;
        int kknDim = t.HasWeight($"blk.{g}.attn_k_norm.weight")
            ? t._weightCache[$"blk.{g}.attn_k_norm.weight"].Shape[0] : hd;
        int totalHeads = t._numHeads;   // always 48 for Qwen3.5/3.6
        // Gate detection: per-head Q output = qtd / totalHeads.
        // Qwen3.5: per-head = 12288/48 = 256 = qknDim*2 → fused Q(128)+gate(128)
        // Qwen3.6: per-head = 12288/48 = 256 = qknDim    → Q only, no gate
        // Qwen3Next: detachable gate via attn_gate.weight.
        bool fusedGate = (qtd / totalHeads) == qknDim * 2;
        bool detachableGate = t.HasWeight($"blk.{g}.attn_gate.weight");
        bool ig = fusedGate || detachableGate;
        int qd = totalHeads * qknDim;  // pure Q part: 48*128=6144 (Qwen3.5), 48*256=12288 (Qwen3.6, no gate)
        // Only log during decode (sl==1): in non-batch mode every MaybeFlush() does a full GPU sync.
        bool dbg = sl == 1 && QwenDbgTrace.Once("TypeB", g);
        if (dbg) QwenDbgTrace.Row0($"rq_raw_g{g}", rq, 16);
        Tensor gq; Tensor? attnOutGate = null; int nqh;
        if (ig) {
            var rqDi = o.DeinterleaveQGate(rq, (uint)totalHeads, (uint)qknDim, "rqDiB");
            if (dbg) QwenDbgTrace.Row0($"rqDi_g{g}", rqDi, 16);
            if (b) o.DeferExternal(rq); else rq.Dispose();
            gq = o.SliceCols(rqDi, 0, qd, "qpB");
            attnOutGate = o.SliceCols(rqDi, qd, qd, "qgB");
            if (dbg) {
                QwenDbgTrace.Row0($"Q_g{g}", gq, 8);
                QwenDbgTrace.Slice($"Q_g{g}_mid", gq, 128, 8);
                QwenDbgTrace.Row0($"gate_g{g}", attnOutGate, 8);
                QwenDbgTrace.Msg($"[DIAG Deinterleave g={g}] qknDim={qknDim} fusedGate={fusedGate} detachableGate={detachableGate} totalHeads={totalHeads} qd={qd}");
            }
            if (b) o.DeferExternal(rqDi); else rqDi.Dispose();
            nqh = totalHeads;
        } else { gq = rq; nqh = totalHeads; }
        if (t.HasWeight($"blk.{g}.attn_q_norm.weight"))
        {
            var (qnW, qnS) = t.TempF32Named($"blk.{g}.attn_q_norm.weight");
            o.LayerNormVirtual(gq, qnW, sl * (gq.Shape[1] / qknDim), qknDim);
            if (!qnS) o.DeferExternal(qnW);
        }
        if (t.HasWeight($"blk.{g}.attn_k_norm.weight"))
        {
            var (knW, knS) = t.TempF32Named($"blk.{g}.attn_k_norm.weight");
            o.LayerNormVirtual(K, knW, sl * (K.Shape[1] / kknDim), kknDim);
            if (!knS) o.DeferExternal(knW);
        }
        // RoPE: use actual head dim from norm weight (qknDim=128), not t._headDim (256 generic).
        o.ApplyRoPEFull(gq, pos, gq.Shape[1] / qknDim, qknDim, t._ropeFreqBase, t._ropeDimCount);
        o.ApplyRoPEFull(K, pos, K.Shape[1] / kknDim, kknDim, t._ropeFreqBase, t._ropeDimCount);
        // Diagnostic: log Q/K head-0 magnitude after RoPE for first 4 attention layers.
        // ||Q||² >> 0 = healthy; ≈ 0 or NaN = corrupted by RoPE/norm.
        if (sl == 1 && QwenDbgTrace.Once("RoPE_QK_norm", g))
        {
            var qd0 = gq.Buffer.ReadRange<float>(0, Math.Min(hd, 16));
            var kd0 = K.Buffer.ReadRange<float>(0, Math.Min(hd, 16));
            float qNormSq = qd0.Sum(v => v * v);
            float kNormSq = kd0.Sum(v => v * v);
            QwenDbgTrace.Msg($"[DIAG RoPE g={g}] ropeDim={t._ropeDimCount} headDim={hd} pos={pos} ||Q_h0[:16]||²={qNormSq:F4} ||K_h0[:16]||²={kNormSq:F4}  Q[:4]=[{string.Join(",", qd0.Take(4).Select(v => v.ToString("F3")))}]");
        }
        Tensor ao;
        if (kvc != null) {
            kvc.Add(li, K, V);
            // FIX: In batch (prefill) mode, K/V buffers are not yet flushed to GPU
            // when Add() is called. Force a flush so MultiHeadAttention sees valid data.
            if (b) o.Flush();
            var (ck, cv) = kvc.Get(li);
            ao = o.MultiHeadAttention(gq, ck!, cv!, nqh, pos, "aoB");
            o.DeferExternal(gq);
        }
        else { ao = o.MultiHeadAttention(gq, K, V, nqh, pos, "aoB"); o.DeferExternal(gq); o.DeferExternal(K); o.DeferExternal(V); }
        if (dbg) QwenDbgTrace.Row0($"ao_pregate_g{g}", ao, 8);
        // Apply sigmoid gate to attention output (qwen3next: sigmoid, not SiLU; after attn, not before).
        if (attnOutGate != null) {
            o.Sigmoid(attnOutGate);
            var aoGated = o.Multiply(ao, attnOutGate, "aoGatedB");
            if (dbg) QwenDbgTrace.Row0($"ao_gated_g{g}", aoGated, 8);
            if (b) o.DeferExternal(ao); else ao.Dispose();
            if (b) o.DeferExternal(attnOutGate); else attnOutGate.Dispose();
            ao = aoGated;
        }
        Tensor ap = oW.tensor.Shape[0] == ao.Shape[1] ? o.MatMulWeights(ao, oW.tensor, "apB") : o.MatMulWeightsT(ao, oW.tensor, "apB");
        if (!oW.isScratch) o.DeferExternal(oW.tensor); if (b) o.DeferExternal(ao); else ao.Dispose();
        var x1 = o.Add(x, ap, "xaB"); if (b) o.DeferExternal(ap); else ap.Dispose();
        var x1n = o.Clone(x1, "pnB"); o.LayerNorm(x1n, pW.tensor); if (!pW.isScratch) o.DeferExternal(pW.tensor);
        var fo = o.FeedForward(x1n, gW.tensor, uW.tensor, dW.tensor, "ffB"); if (!gW.isScratch) o.DeferExternal(gW.tensor); if (!uW.isScratch) o.DeferExternal(uW.tensor); if (!dW.isScratch) o.DeferExternal(dW.tensor);
        if (b) o.DeferExternal(x1n); else x1n.Dispose(); var ou = o.Add(x1, fo, "loB"); if (b) o.DeferExternal(x1); else x1.Dispose(); if (b) o.DeferExternal(fo); else fo.Dispose(); if (b) o.DeferExternal(x); else x.Dispose();
        if (b) o.Flush(); return ou;
    }

    // ─── SSM recurrence on GPU (sub-batch + ArrayPool for CPU padding) ─
    static Tensor SSM_Gpu(TransformerBase t, ComputeOps o, Tensor xr, Tensor xn, int g, SSMState ss, int sl)
    {
        int HVD = t._ssmHVD, NVH = t._ssmVH, NKH = t._ssmKH;
        int KD = t._ssmKD, VD = t._ssmVD, CD = t._ssmCD;
        const int SUB_BATCH = 8; // tokens per GPU submit
        int dm = xn.Shape[1];
        // OPTIMIZATION: Always use batch mode (same as TypeA).
        bool b = true;

        // ── Debug: log key params for first few layers ────────────────
        if (g <= 2 && QwenDbgTrace.Once("ssm_params", g))
            t._logger.LogWarning("[SSM_DBG] L{L} HVD={HVD} NVH={NVH} NKH={NKH} KD={KD} VD={VD} CD={CD} dm={dm}",
                g, HVD, NVH, NKH, KD, VD, CD, dm);

        var (convW, _) = t.TempF32Named($"blk.{g}.ssm_conv1d.weight");
        var (wQKV, _) = t.TempF32Named($"blk.{g}.attn_qkv.weight");

        // ── z-projection ──────────────────────────────────────────────
        // Qwen3.5: ssm_gate.weight   Qwen3.6: attn_gate.weight (renamed, same dims)
        // Both are [5120, 6144] = d_model × (NVH*HVD).  Dimensionally verified.
        string wzName = "";
        foreach (var c in new[] { "ssm_gate.weight","ssm_z.weight","ssm_in.weight",
            "ssm_in_proj.weight","ssm_a_proj.weight","attn_gate.weight" })
        { if (t.HasWeight($"blk.{g}.{c}")) { wzName = $"blk.{g}.{c}"; break; } }
        if (wzName == "")
        {
            var known = new[] { "ssm_gate.weight","ssm_z.weight","ssm_in.weight",
                "ssm_in_proj.weight","ssm_a_proj.weight","attn_gate.weight" };
            var found = known.Where(k => t.HasWeight($"blk.{g}.{k}")).ToList();
            throw new InvalidOperationException(
                $"Layer {g}: no SSM z-projection weight. Available: [{string.Join(", ", found)}]");
        }
        var (wZRaw, wZScratch) = t.TempF32Named(wzName);
        Tensor wZ;
        if (wZRaw.Shape[1] == VD)
        {
            wZ = wZRaw;
            t._logger.LogInformation("[SSM_GPU] wZ='{Name}' shape=[{D0},{D1}] OK", wzName, wZRaw.Shape[0], wZRaw.Shape[1]);
        }
        else
        {
            // CPU pad wZ to [dModel, VD]. Use ArrayPool<float> to avoid LOH allocation.
            t._logger.LogWarning("[SSM_GPU] Padding wZ='{Name}' {D0}x{D1} -> {D0}x{VD}",
                wzName, wZRaw.Shape[0], wZRaw.Shape[1], wZRaw.Shape[0], VD);
            var cpuData = wZRaw.Buffer.Read<float>();
            int paddedLen = wZRaw.Shape[0] * VD;
            float[] padded = ArrayPool<float>.Shared.Rent(paddedLen);
            var paddedSpan = padded.AsSpan(0, paddedLen);
            paddedSpan.Clear(); // zero-fill
            for (int r = 0; r < wZRaw.Shape[0]; r++)
                cpuData.AsSpan(r * wZRaw.Shape[1], wZRaw.Shape[1])
                    .CopyTo(paddedSpan.Slice(r * VD, wZRaw.Shape[1]));
            wZ = Tensor.FromData(o.Device, padded, TensorShape.Matrix(wZRaw.Shape[0], VD), "ssm_wz_padded");
            ArrayPool<float>.Shared.Return(padded);
            if (!wZScratch) wZRaw.Dispose();
        }

        var (wBeta, _) = t.TempF32Named($"blk.{g}.ssm_beta.weight");
        var (wAlpha, _) = t.TempF32Named($"blk.{g}.ssm_alpha.weight");
        var (dtBias, _) = t.TempF32Named($"blk.{g}.ssm_dt.bias");
        var (ssA, _) = t.TempF32Named($"blk.{g}.ssm_a");

        var normF32 = t.GetOrBuildTiledNorm($"blk.{g}.ssm_norm.weight", HVD, VD);
        if (normF32 == null) { normF32 = t.TempF32Named($"blk.{g}.ssm_norm.weight").tensor; }

        var (wOut, _) = t.TempF32Named($"blk.{g}.ssm_out.weight");

        var convStateBuf = ss.GetGpuConvBuffer(g);
        var convStateTensor = new Tensor(convStateBuf, new TensorShape(new[] { SSMState.CONV_STATE_DIM }), DataType.F32);
        var ssmStateBuf = ss.GetGpuStateBuffer(g);
        var ssmStateTensor = new Tensor(ssmStateBuf, new TensorShape(new[] { SSMState.STATE_DIM }), DataType.F32);

        var _tsP = GlobalProfiler.Start();
        t._logger.LogInformation("[SSM_GPU] Layer {Layer} sl={SeqLen} dm={DModel} subBatch={Sub}", g, sl, dm, SUB_BATCH);

        Tensor? result = null;

        for (int ti = 0; ti < sl; ti += SUB_BATCH)
        {
            int end = Math.Min(ti + SUB_BATCH, sl);
            Tensor? subResult = null;

            // CRITICAL: scratch must be allocated AFTER arena Reset (from previous Flush).
            // Allocating inside the sub-batch loop guarantees it won't overlap with
            // tokenOut buffers from the previous sub-batch.
            Tensor scratch = o.AllocTempTensor(new TensorShape(new[] { CD + VD + NVH * 3 }), "ssm_scratch")
                ?? Tensor.Create(o.Device, new TensorShape(new[] { CD + VD + NVH * 3 }), DataType.F32, "ssm_scratch");

            // Dispatch SSM for each token in the sub-batch
            for (int i = 0; i < end - ti; i++)
            {
                var _tsTi = GlobalProfiler.Start();
                int tokenIdx = ti + i;
                // Pass full xn tensor with rowIndex instead of cloning per token.
                // SSM shader reads xNorm.data[rowIndex * DMODEL + k].
                // CRITICAL: AllocTempTensor allocates from arena. In non-batch (decode) mode,
                // SsmGdnDecode calls MaybeFlush → Flush → _arena.Reset(), freeing the buffer
                // before the SSM output can be used. Use Tensor.Create for decode;
                // arena is fine for batch (prefill) since Flush happens only at sub-batch end.
                var tokenOut = b
                    ? (o.AllocTempTensor(TensorShape.Matrix(1, VD), $"ssm_tok{tokenIdx}")
                       ?? Tensor.Create(o.Device, TensorShape.Matrix(1, VD), DataType.F32, $"ssm_tok{tokenIdx}"))
                    : Tensor.Create(o.Device, TensorShape.Matrix(1, VD), DataType.F32, $"ssm_tok{tokenIdx}");

                o.SsmGdnDecode(
                    xn,
                    convW, convStateTensor, wQKV, wZ, wBeta, wAlpha, dtBias, ssA,
                    scratch,
                    ssmStateTensor,
                    normF32,
                    tokenOut,
                    (uint)tokenIdx,
                    (uint)t._dModel,
                    (uint)t._ssmHVD,
                    (uint)t._ssmVH,
                    (uint)t._ssmKH,
                    (uint)t._ssmKD,
                    (uint)t._ssmVD,
                    (uint)t._ssmCD,
                    (uint)g);

                // Log during decode only (sl==1): non-batch mode flushes after each op → correct read.
                // In decode, AllocTempTensor returns null (no arena) → fallback Tensor.Create → readable.
                if (tokenIdx == 0 && sl == 1 && QwenDbgTrace.Once("SSM_tok0", g))
                    QwenDbgTrace.Row0($"ssm_out_g{g}_tok0", tokenOut, 8);

                GlobalProfiler.End(_tsTi, "SSM_GPU.Token");

                if (subResult == null)
                    subResult = tokenOut;
                else
                {
                    subResult = o.Concat(subResult, tokenOut, 0, "ssm_subcat");
                    if (b) o.DeferExternal(tokenOut); else tokenOut.Dispose();
                }
            }

            subResult ??= Tensor.Create(o.Device, TensorShape.Matrix(1, VD), DataType.F32, "ssm_empty_sub");

            // Merge with global result
            if (result == null)
                result = subResult;
            else
            {
                result = o.Concat(result, subResult, 0, "ssm_cat");
                if (b) o.DeferExternal(subResult); else subResult.Dispose();
            }

            // Flush the sub-batch so GPU finishes this group before next
            // Note: Flush() also calls _arena.Reset() which reclaims all temp arena memory
            var _tsFlush = GlobalProfiler.Start();
            if (b) o.Flush();
            GlobalProfiler.End(_tsFlush, "SSM_GPU.FlushSub");
            // Trace ssmState after first sub-batch flush (GPU has executed).
            if (g <= 2 && ti == 0 && QwenDbgTrace.Once("ssm_state_post", g))
            {
                QwenDbgTrace.Slice($"L{g}_state_h0", ssmStateTensor, 0, 16);
                QwenDbgTrace.Slice($"L{g}_state_mid", ssmStateTensor,
                    SSMState.STATE_DIM / 2, 8);
            }
        }
        GlobalProfiler.End(_tsP, "SSM_GPU.Total");
        result ??= Tensor.Create(o.Device, TensorShape.Matrix(1, VD), DataType.F32, "ssm_empty");

        var _tsP2 = GlobalProfiler.Start();
        var projected = o.MatMulWeights(result, wOut, "ssm_proj");
        var residual = o.Add(xr, projected, "ssm_out");
        // Trace residual for first 3 layers (decode mode: GPU completed, values readable)
        if (g <= 2 && !b && QwenDbgTrace.Once("ssm_residual", g))
            QwenDbgTrace.Row0($"ssm_res_g{g}", residual, 16);
        if (b) { o.DeferExternal(result); o.DeferExternal(projected); }
        else { result.Dispose(); projected.Dispose(); }
        // Arena scratch is auto-reclaimed by Flush → Reset — no explicit Dispose needed
        GlobalProfiler.End(_tsP2, "SSM_GPU.OutProj");
        t._logger.LogInformation("[SSM_GPU] Layer {Layer} done", g);
        return residual;
    }


    // ─── Fallback ──────────────────────────────────────────────────────
    static Tensor Fallback(TransformerBase t, Tensor x, int g, TensorNameMapper nm)
    {
        var o = t.Ops; var xn = o.Clone(x, "an_fb");
        var (wa, sa) = t.TempF32(nm.AttnNorm(g));
        o.LayerNorm(xn, wa);
        if (!sa) wa.Dispose();
        Tensor ao; int hd = t._headDim; if (t.HasWeight(nm.AttnQKV(g)))
        {
            var (w, s_) = t.TempF32(nm.AttnQKV(g));
            var q = o.MatMulWeights(xn, w, "q_fb");
            if (!s_) w.Dispose();
            int qd = t._numHeads * hd, kd = t._numKVHeads * hd; var Q = o.SliceCols(q, 0, qd, "Q_fb");
            var K = o.SliceCols(q, qd, kd, "K_fb");
            var V = o.SliceCols(q, qd + kd, kd, "V_fb");
            q.Dispose();
            o.ApplyRoPEFull(Q, 0, t._numHeads, hd, t._ropeFreqBase, t._ropeDimCount);
            o.ApplyRoPEFull(K, 0, t._numKVHeads, hd, t._ropeFreqBase, t._ropeDimCount);
            ao = o.MultiHeadAttention(Q, K, V, t._numHeads, 0, "a_fb");
            o.DeferExternal(Q);
            o.DeferExternal(K);
            o.DeferExternal(V);
        }
        else
        {
            var (wq, sq) = t.TempF32(nm.AttnQ(g));
            var (wk, sk) = t.TempF32(nm.AttnK(g));
            var (wv, sv) = t.TempF32(nm.AttnV(g));
            var Q = o.MatMulWeights(xn, wq, "Q_fb");
            var K = o.MatMulWeights(xn, wk, "K_fb");
            var V = o.MatMulWeights(xn, wv, "V_fb");
            if (!sq) wq.Dispose();
            if (!sk) wk.Dispose();
            if (!sv) wv.Dispose();
            o.ApplyRoPEFull(Q, 0, t._numHeads, hd, t._ropeFreqBase, t._ropeDimCount);
            o.ApplyRoPEFull(K, 0, t._numKVHeads, hd, t._ropeFreqBase, t._ropeDimCount);
            ao = o.MultiHeadAttention(Q, K, V, t._numHeads, 0, "a_fb");
            o.DeferExternal(Q);
            o.DeferExternal(K);
            o.DeferExternal(V);
        }
        xn.Dispose();
        var (wo, so) = t.TempF32(nm.AttnOutput(g));
        Tensor ap = wo.Shape[0] == ao.Shape[1] ? o.MatMulWeights(ao, wo, "ap_fb") : o.MatMulWeightsT(ao, wo, "ap_fb");
        if (!so) wo.Dispose();
        ao.Dispose();
        var x1 = o.Add(x, ap, "xa_fb");
        ap.Dispose();
        var x1n = o.Clone(x1, "pn_fb");
        var (wp, sp) = t.TempF32(nm.FfnNorm(g));
        o.LayerNorm(x1n, wp);
        if (!sp) wp.Dispose();
        var (wg, sg) = t.TempF32(nm.FfnGate(g));
        var (wu, su) = t.TempF32(nm.FfnUp(g));
        var (wd, sd) = t.TempF32(nm.FfnDown(g));
        var fo = o.FeedForward(x1n, wg, wu, wd, "ff_fb");
        if (!sg) wg.Dispose();
        if (!su) wu.Dispose();
        if (!sd) wd.Dispose();
        x1n.Dispose();
        var ou = o.Add(x1, fo, "lo_fb");
        x1.Dispose();
        fo.Dispose();
        x.Dispose();
        return ou;
    }
}
