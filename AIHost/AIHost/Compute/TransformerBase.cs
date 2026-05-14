using AIHost.GGUF;
using AIHost.ICompute;
using AIHost.Inference;
using Microsoft.Extensions.Logging;

namespace AIHost.Compute;

/// <summary>
/// Base class for transformer model inference.
/// Provides shared infrastructure: weight cache, scratch buffers, embedding, head projection,
/// multi-GPU support, and CPU reference checks.
/// Subclasses implement per-layer forward pass via ITransformerFormat.
/// </summary>
public class TransformerBase : IDisposable
{
    protected internal readonly ComputeOps _ops;
    protected internal readonly IGGUFModel _model;
    protected internal readonly int _numLayers;
    protected internal readonly int _dModel;
    protected internal readonly int _numHeads;
    protected internal readonly int _numKVHeads;
    protected internal readonly int _headDim;
    protected internal readonly float _ropeFreqBase;
    /// <summary>
    /// Number of head dimensions to apply RoPE to (from rope.dimension_count).
    /// For models with partial RoPE (e.g. Qwen3.5/3.6: 64 of 256), only these
    /// dimensions are rotated; the rest are left unchanged.
    /// Defaults to _headDim (full rotation) when key is absent.
    /// </summary>
    protected internal readonly int _ropeDimCount;
    protected bool _disposed;
    private bool _hiddenDiagFired;
    protected internal readonly ILogger<TransformerBase> _logger = AppLogger.Create<TransformerBase>();

    /// <summary>Max context length from GGUF metadata (0 = unknown).</summary>
    public int ContextLength { get; private set; }

    // Stores QUANTIZED tensors (or F32 for weights already stored as F32 in GGUF).
    protected internal readonly Dictionary<string, Tensor> _weightCache = [];

    // Pre-allocated F32 scratch tensors keyed by weight name (e.g. "attn_q.weight").
    // Reused across all layers and tokens to eliminate per-inference vkAllocateMemory calls.
    protected internal readonly Dictionary<string, Tensor> _scratchF32 = [];

    // For multi-GPU: offset added to local layer index when looking up weight names.
    protected internal int _layerOffset = 0;
    protected internal int _localLayerCount;
    protected internal TensorNameMapper? _nameMapper;

    private ITransformerFormat _format;

    public int LayerCount => _numLayers;
    public int LocalLayerCount => _localLayerCount;
    public ComputeOps Ops => _ops;

    public TransformerBase(IComputeDevice device, IGGUFModel model, ITransformerFormat format)
    {
        _ops = new ComputeOps(device);
        _model = model;
        _format = format;

        var metadata = model.Metadata;

        int ArchInt(string suffix, int def)
        {
            var v = metadata.GetArchValue<object>(suffix);
            if (v is int i) return i;
            if (v is uint u) return (int)u;
            if (v is long l) return (int)l;
            if (v is ulong ul) return (int)ul;
            return def;
        }
        float ArchFlt(string suffix, float def)
        {
            var v = metadata.GetArchValue<object>(suffix);
            if (v is float f) return f;
            if (v is double d) return (float)d;
            return def;
        }

        _numLayers = ArchInt("block_count", 22);
        _dModel = ArchInt("embedding_length", 2048);
        _numHeads = ArchInt("attention.head_count", 32);
        ContextLength = ArchInt("context_length", 0);

        // Read head_dim from GGUF metadata; fall back to d_model/n_heads for legacy models.
        _headDim = ArchInt("attention.key_length", _dModel / _numHeads);

        _ropeFreqBase = ArchFlt("rope.freq_base", 10000.0f);
        // rope.dimension_count: how many dimensions per head get RoPE applied.
        // For Qwen3.5/3.6: 64 (only 32 pairs rotated out of 128 in a 256-dim head).
        // Absent on most models → default to headDim (rotate all pairs).
        _ropeDimCount = ArchInt("rope.dimension_count", _headDim);
        var kvHeads = ArchInt("attention.head_count_kv", 4);
        _numKVHeads = kvHeads;
        _localLayerCount = _numLayers;

        _logger.LogInformation("Transformer: layers={Layers} d_model={DModel} heads={Heads} kv_heads={KvHeads} head_dim={HeadDim} ctx={Ctx} rope={Rope} rope_dim={RopeDim}",
            _numLayers, _dModel, _numHeads, kvHeads, _headDim, ContextLength, _ropeFreqBase, _ropeDimCount);
    }

    // ── Weight loading ────────────────────────────────────────────────────────

    /// <summary>
    /// Upload all model weights to GPU (quantized). Fast: no dequantization here.
    /// </summary>
    public void LoadWeights()
    {
        _logger.LogInformation("[LOAD] Loading model weights...");
        _nameMapper = new TensorNameMapper(_model);

        _logger.LogInformation("[LOAD] Caching embed/norm/output weights");
        CacheWeight(_nameMapper.TokenEmbd);
        CacheWeight(_nameMapper.OutputNorm);
        CacheWeight(_nameMapper.OutputWeight);

        for (int i = 0; i < _numLayers; i++)
        {
            CacheLayerWeights(i);
            if ((i + 1) % 5 == 0 || i == _numLayers - 1)
                _logger.LogInformation("[LOAD] Uploaded layer {Current}/{Total}", i + 1, _numLayers);
        }

        _logger.LogInformation("[LOAD] Weights loaded: {Count} tensors, quantized in VRAM", _weightCache.Count);

        _logger.LogInformation("[LOAD] Allocating scratch buffers...");
        AllocateScratchWeights(_numLayers > 0 ? _nameMapper.AttnNorm(0) : null);
        TryAllocateScratchHead();
        _logger.LogInformation("[LOAD] F32 scratch buffers allocated: {Count} tensors", _scratchF32.Count);
        _logger.LogInformation("[LOAD] Running CPU reference layer 0...");
        RunCpuReferenceLayer0();
        _logger.LogInformation("[LOAD] LoadWeights COMPLETE");
    }

    /// <summary>
    /// Load only a subset of layers plus optionally the embedding table and lm-head.
    /// Used by MultiGPUTransformer to split a model across devices.
    /// </summary>
    public void LoadWeightsPartial(int globalFirstLayer, int globalLastLayer,
                                    bool withEmbedding, bool withHead)
    {
        _layerOffset = globalFirstLayer;
        _localLayerCount = globalLastLayer - globalFirstLayer;

        _logger.LogInformation("[MultiGPU] Loading layers {First}..{Last}{Embed}{Head}",
            globalFirstLayer, globalLastLayer - 1,
            withEmbedding ? " + embedding" : "", withHead ? " + head" : "");

        _nameMapper ??= new TensorNameMapper(_model);

        if (withEmbedding) CacheWeight(_nameMapper.TokenEmbd);
        if (withHead) { CacheWeight(_nameMapper.OutputNorm); CacheWeight(_nameMapper.OutputWeight); }

        for (int i = globalFirstLayer; i < globalLastLayer; i++)
            CacheLayerWeights(i);

        _logger.LogInformation("[MultiGPU] Loaded {Count} tensors for device", _weightCache.Count);
        if (_localLayerCount > 0)
            TryAllocateScratch(_nameMapper!.AttnNorm(globalFirstLayer));
        if (withHead)
            TryAllocateScratchHead();
    }

    // ── Forward pass ──────────────────────────────────────────────────────────

    /// <summary>Token embedding lookup. Returns [seqLen, dModel] on this device's GPU.</summary>
    public Tensor ForwardEmbedding(int[] tokenIds) => EmbeddingLookupOptimized(tokenIds);

    /// <summary>
    /// Run the locally-loaded layers on activation tensor x.
    /// KV-cache uses LOCAL layer indices (0-based for this device).
    /// </summary>
    public Tensor ForwardLayers(Tensor x, uint startPos, KVCache? cache = null,
                                 SSMState? ssmState = null)
    {
        for (int i = 0; i < _localLayerCount; i++)
            x = _format.ApplyLayer(this, x, i, startPos, cache, ssmState);
        return x;
    }

    /// <summary>
    /// Apply output_norm + lm-head projection. Returns logits [seqLen, vocabSize].
    /// </summary>
    public Tensor ForwardHead(Tensor x)
    {
        var _ts = GlobalProfiler.Start();
        var (normF32, normScratch) = TempF32(_nameMapper!.OutputNorm);
        _ops.LayerNorm(x, normF32);
        if (!normScratch) normF32.Dispose();

        // Use chunked matmul if output.weight would need >1 GB F32 (large-vocab models).
        var outWeightCached = _weightCache[_nameMapper.OutputWeight];
        long outF32Bytes = (long)outWeightCached.Shape.TotalElements * sizeof(float);
        if (outF32Bytes > 1L * 1024 * 1024 * 1024)
        {
            var chunkedLogits = _ops.MatMulWeightsLarge(x, outWeightCached, "logits");
            x.Dispose();
            return chunkedLogits;
        }

        var (outF32, outScratch) = TempF32(_nameMapper.OutputWeight);
        var logits = _ops.MatMulWeights(x, outF32, "logits");
        if (!outScratch) outF32.Dispose();
        x.Dispose();
        GlobalProfiler.End(_ts, "Fwd.Head.Detail");
        return logits;
    }

    /// <summary>
    /// Full forward pass through transformer.
    /// </summary>
    public Tensor Forward(int[] tokenIds, uint startPosition = 0, KVCache? kvCache = null,
                           SSMState? ssmState = null)
    {
        _logger.LogInformation("[FWD] Forward BEGIN tokens={Count} pos={Pos} kvCache={HasKV} ssm={HasSSM}",
            tokenIds.Length, startPosition, kvCache != null, ssmState != null);

        if (_nameMapper == null)
            throw new InvalidOperationException("Weights not loaded. Call LoadWeights() first.");

        // 1. Token embedding
        _logger.LogInformation("[FWD] EmbeddingLookup...");
        Tensor x = EmbeddingLookupOptimized(tokenIds);
        _logger.LogInformation("[FWD] EmbeddingLookup OK shape=[{D0},{D1}]", x.Shape[0], x.Shape[1]);

        // Diagnostic: log last-token hidden state at key checkpoints (decode only, first call)
        bool diagEnabled = tokenIds.Length == 1 && !_hiddenDiagFired;
        if (diagEnabled) _hiddenDiagFired = true;

        DiagHidden("embed", x, diagEnabled);

        // 2. All transformer layers
        _logger.LogInformation("[FWD] Entering layer loop ({Count} layers)", _numLayers);
        for (int i = 0; i < _numLayers; i++)
        {
            _logger.LogInformation("[FWD] Layer {Layer} start", i);
            try
            {
                x = _format.ApplyLayer(this, x, i, startPosition, kvCache, ssmState);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "[FWD] Layer {Layer} FAILED", i);
                throw;
            }
            _logger.LogInformation("[FWD] Layer {Layer} OK", i);
            if (i == 0) DiagHidden("after_layer0", x, diagEnabled);
            if (i == 3) DiagHidden("after_layer3", x, diagEnabled);
            if (i == 15) DiagHidden("after_layer15", x, diagEnabled);
            if (i == 31) DiagHidden("after_layer31", x, diagEnabled);
        }
        _logger.LogInformation("[FWD] Layer loop done");
        DiagHidden("before_head", x, diagEnabled);

        // 3. Final layer norm + vocab projection
        //    Delegates to ForwardHead which has chunked MatMulWeightsLarge path
        //    for large-vocab models (Qwen 248K, etc.).
        return ForwardHead(x);
    }

    private void DiagHidden(string tag, Tensor t, bool enabled)
    {
        if (!enabled) return;
        try
        {
            int cols = t.Shape.Rank >= 2 ? t.Shape[1] : t.Shape[0];
            var row = t.Buffer.ReadRange<float>(0, Math.Min(cols, 16));
            float rms = MathF.Sqrt(row.Sum(v => v * v) / row.Length);
            _logger.LogWarning("[DIAG_HIDDEN {Tag}] rms={Rms:F4} first8=[{V}]",
                tag, rms, string.Join(",", row.Take(8).Select(v => v.ToString("F3"))));
        }
        catch { /* ignore */ }
    }

    // ── Protected helpers for formats ─────────────────────────────────────────

    /// <summary>
    /// Returns an F32 tensor from the named cached weight.
    /// Uses a pre-allocated scratch buffer if available (no vkAllocateMemory); otherwise
    /// allocates a new tensor. Caller must NOT dispose scratch tensors (owned by TransformerBase).
    /// </summary>
    protected internal (Tensor tensor, bool isScratch) TempF32(string name)
    {
        var cached = _weightCache[name];

        if (cached.DataType == DataType.F32)
            return (_ops.Clone(cached), false);

        // Use pre-allocated scratch buffer to avoid vkAllocateMemory per inference
        var dot2 = name.IndexOf('.', name.IndexOf('.') + 1);
        var key = dot2 >= 0 ? name[(dot2 + 1)..] : name;
        if (_scratchF32.TryGetValue(key, out var scratch))
        {
            _ops.DequantizeInto(cached, scratch);
            return (scratch, true);
        }

        return (_ops.Dequantize(cached), false);
    }

    /// <summary>
    /// TempF32 by exact name (bypasses the layer-prefix stripping logic).
    /// </summary>
    protected internal (Tensor tensor, bool isScratch) TempF32Named(string name)
    {
        if (!_weightCache.TryGetValue(name, out var cached))
            throw new KeyNotFoundException($"Weight '{name}' not in cache");
        if (cached.DataType == DataType.F32)
            return (_ops.Clone(cached), false);
        int dot2 = name.IndexOf('.', name.IndexOf('.') + 1);
        var key = dot2 >= 0 ? name[(dot2 + 1)..] : name;
        if (_scratchF32.TryGetValue(key, out var scratch))
        {
            _ops.DequantizeInto(cached, scratch);
            return (scratch, true);
        }
        return (_ops.Dequantize(cached), false);
    }

    /// <summary>
    /// Creates a tiled version of a [head_dim] norm weight to cover [total_dim].
    /// Caches result in _scratchF32 to avoid re-allocation every token.
    /// </summary>
    /// <summary>
    /// Creates a tiled version of a [head_dim] norm weight to cover [total_dim].
    /// If the weight is not found, returns null — caller should skip normalization.
    /// Caches result in _scratchF32 to avoid re-allocation every token.
    /// </summary>
    protected internal Tensor? GetOrBuildTiledNorm(string weightName, int headDim, int totalDim)
    {
        string key = $"_tiled_{weightName}";
        if (_scratchF32.TryGetValue(key, out var cached)) return cached;

        if (!_weightCache.TryGetValue(weightName, out var srcW))
            return null; // weight not present — skip normalization

        var srcF32 = _ops.Dequantize(srcW);
        var srcData = srcF32.ReadData();
        srcF32.Dispose();

        int tiles = totalDim / headDim;
        var tiledData = new float[totalDim];
        for (int t = 0; t < tiles; t++)
            Array.Copy(srcData, 0, tiledData, t * headDim, headDim);

        var tiled = Tensor.FromData(_ops.Device, tiledData, new TensorShape(totalDim), key);
        _scratchF32[key] = tiled;
        return tiled;
    }

    /// <summary>Check if a weight exists in the cache.</summary>
    protected internal bool HasWeight(string name) => _weightCache.ContainsKey(name);

    /// <summary>Get global layer index from local index.</summary>
    protected internal int GlobalLayer(int localIdx) => localIdx + _layerOffset;

    // ── Private helpers ───────────────────────────────────────────────────────

    private Tensor EmbeddingLookupOptimized(int[] tokenIds)
    {
        var embCached = _weightCache[_nameMapper!.TokenEmbd];
        if (embCached.DataType == DataType.F32)
        {
            var f32 = _ops.Clone(embCached);
            var result = _ops.EmbeddingLookup(tokenIds, f32, "embeddings");
            f32.Dispose();
            return result;
        }
        return _ops.EmbeddingLookupFromQuantized(tokenIds, embCached, "embeddings");
    }

    private void TryAllocateScratch(string layerPrefix)
    {
        try { AllocateScratchWeights(layerPrefix); }
        catch (Exception ex)
        {
            _logger.LogWarning("[Scratch] Allocation failed for {Layer}: {Error} — will allocate per inference", layerPrefix, ex.Message);
        }
    }

    private void TryAllocateScratchHead()
    {
        foreach (var name in new[] { "output.weight", "token_embd.weight" })
        {
            if (!_weightCache.TryGetValue(name, out var cached)) continue;
            if (cached.DataType == DataType.F32) continue;
            if (_scratchF32.ContainsKey(name)) continue;
            try
            {
                var scratch = Tensor.Create(_ops.Device, cached.Shape, DataType.F32, name + "_scratch");
                _scratchF32[name] = scratch;
                _logger.LogDebug("[MultiGPU] Scratch {Name}: {SizeMB} MB F32", name, scratch.Shape.TotalElements * 4L / 1024 / 1024);
            }
            catch (Exception ex)
            {
                _logger.LogWarning("[Scratch] {Name} failed: {Error} — will allocate per inference", name, ex.Message);
            }
        }
    }

    private void AllocateScratchWeights(string? firstLayerAttnNormName)
    {
        if (firstLayerAttnNormName == null || _nameMapper == null) return;

        int refLayer = _layerOffset;

        // Build list of weight names to allocate scratch for.
        // For Qwen3.6 hybrid models, Type A layers use ssm_out.weight instead of attn_output.weight.
        // We include both to cover both layer types.
        var names = new List<string>();

        if (_nameMapper.HasCombinedQKV)
        {
            names.Add(_nameMapper.AttnNorm(refLayer));
            names.Add(_nameMapper.AttnQKV(refLayer));
            // attn_output.weight may not exist for Type A layers (Qwen3.6 uses ssm_out.weight)
            names.Add(_nameMapper.AttnOutput(refLayer));
            // ssm_out.weight is used as output projection for Type A layers
            names.Add($"blk.{refLayer}.ssm_out.weight");
            // SSM/GDN weights (Qwen3.6 Type A layers)
            names.Add($"blk.{refLayer}.ssm_alpha.weight");
            names.Add($"blk.{refLayer}.ssm_beta.weight");
            names.Add($"blk.{refLayer}.ssm_conv1d.weight");
            names.Add($"blk.{refLayer}.attn_gate.weight");
            names.Add(_nameMapper.FfnNorm(refLayer));
            names.Add(_nameMapper.FfnGate(refLayer));
            names.Add(_nameMapper.FfnUp(refLayer));
            names.Add(_nameMapper.FfnDown(refLayer));
        }
        else
        {
            names.Add(_nameMapper.AttnNorm(refLayer));
            names.Add(_nameMapper.AttnQ(refLayer));
            names.Add(_nameMapper.AttnK(refLayer));
            names.Add(_nameMapper.AttnV(refLayer));
            names.Add(_nameMapper.AttnOutput(refLayer));
            names.Add(_nameMapper.FfnNorm(refLayer));
            names.Add(_nameMapper.FfnGate(refLayer));
            names.Add(_nameMapper.FfnUp(refLayer));
            names.Add(_nameMapper.FfnDown(refLayer));

            // ── Gemma 4 extras ────────────────────────────────────────────────
            names.Add($"blk.{refLayer}.attn_post_norm.weight");
            names.Add($"blk.{refLayer}.ffn_post_norm.weight");
            names.Add($"blk.{refLayer}.layer_out_scale.weight");

            // ── DeepSeek V2/V3/V4 extras ───────────────────────────────────────
            // MLA weights (compressed attention)
            names.Add($"blk.{refLayer}.attn_q_a.weight");
            names.Add($"blk.{refLayer}.attn_q_b.weight");
            names.Add($"blk.{refLayer}.attn_kv_a_mqa.weight");
            names.Add($"blk.{refLayer}.attn_k_b.weight");
            names.Add($"blk.{refLayer}.attn_v_b.weight");
            // MoE shared expert (shexp)
            names.Add($"blk.{refLayer}.ffn_gate_shexp.weight");
            names.Add($"blk.{refLayer}.ffn_up_shexp.weight");
            names.Add($"blk.{refLayer}.ffn_down_shexp.weight");
        }

        string sample = names[0];
        int secondDot = sample.IndexOf('.', sample.IndexOf('.') + 1);
        int prefixLen = secondDot >= 0 ? secondDot + 1 : 0;

        foreach (var name in names)
        {
            if (!_weightCache.TryGetValue(name, out var cached)) continue;
            if (cached.DataType == DataType.F32) continue;

            var scratch = Tensor.Create(_ops.Device, cached.Shape, DataType.F32, name + "_scratch");
            var key = prefixLen > 0 ? name[prefixLen..] : name;
            _scratchF32[key] = scratch;
        }
    }

    private void RunCpuReferenceLayer0()
    {
        try
        {
            // BUGFIX: TempF32 returns scratch buffer (owned by _scratchF32).
            // Disposing it destroys the scratch, causing Use-After-Free on next inference.
            var (embF32, isScratch) = TempF32("token_embd.weight");
            float[] emb = embF32.Buffer.ReadRange<float>(1 * 2048 * 4UL, 2048);
            if (!isScratch) embF32.Dispose();

            // BUGFIX: norm weight is quantized in _weightCache — must dequantize first, not read raw bytes as floats.
            var (normF32, normIsScratch) = TempF32("blk.0.attn_norm.weight");
            float[] normW = normF32.Buffer.Read<float>();
            if (!normIsScratch) normF32.Dispose();
            float sumSq = 0f; foreach (var v in emb) sumSq += v * v;
            float rms = MathF.Sqrt(sumSq / 2048 + 1e-5f);
            float[] xn = new float[2048];
            for (int i = 0; i < 2048; i++) xn[i] = emb[i] / rms * normW[i];

            _logger.LogTrace("[CPURef] BOS emb[:3]=[{E0:F5},{E1:F5},{E2:F5}] rms={Rms:F6}", emb[0], emb[1], emb[2], rms);
            _logger.LogTrace("[CPURef] xnorm[:3]=[{X0:F5},{X1:F5},{X2:F5}] maxAbs={MaxAbs:F4}", xn[0], xn[1], xn[2], xn.Max(MathF.Abs));

            var wqF32 = _ops.Dequantize(_weightCache["blk.0.attn_q.weight"]);
            float[] wqData = wqF32.Buffer.ReadRange<float>(0, 2048 * 4); wqF32.Dispose();
            float[] q4 = new float[4];
            for (int n = 0; n < 4; n++)
                for (int k = 0; k < 2048; k++) q4[n] += xn[k] * wqData[k + n * 2048];
            _logger.LogTrace("[CPURef] Q[:4]=[{Q0:F5},{Q1:F5},{Q2:F5},{Q3:F5}]", q4[0], q4[1], q4[2], q4[3]);

            var wvData = _ops.Dequantize(_weightCache["blk.0.attn_v.weight"]);
            float[] wvArr = wvData.Buffer.ReadRange<float>(0, 2048 * 4); wvData.Dispose();
            float[] bosV4 = new float[4];
            for (int n = 0; n < 4; n++) for (int k = 0; k < 2048; k++) bosV4[n] += xn[k] * wvArr[k + n * 2048];
            _logger.LogTrace("[CPURef] BOS V[:4]=[{V0:F5},{V1:F5},{V2:F5},{V3:F5}]", bosV4[0], bosV4[1], bosV4[2], bosV4[3]);
            var wvGpu = _ops.Dequantize(_weightCache["blk.0.attn_v.weight"]);
            var xnT = Tensor.FromData(_ops.Device, xn, new TensorShape(new[] { 1, 2048 }), "xn_bos_v");
            var bosVgpu = _ops.MatMulWeights(xnT, wvGpu, "V_bos");
            float[] gv = bosVgpu.Buffer.ReadRange<float>(0, 4); bosVgpu.Dispose(); xnT.Dispose(); wvGpu.Dispose();
            _logger.LogTrace("[CPURef] BOS GPU V[:4]=[{V0:F5},{V1:F5},{V2:F5},{V3:F5}] match={Match}", gv[0], gv[1], gv[2], gv[3], Math.Abs(bosV4[0] - gv[0]) < 1e-3f);

            var wqF32Tensor = _ops.Dequantize(_weightCache["blk.0.attn_q.weight"]);
            var xnTensor = Tensor.FromData(_ops.Device, xn, new TensorShape(new[] { 1, 2048 }), "xn_bos");
            var gpuQ = _ops.MatMulWeights(xnTensor, wqF32Tensor, "Q_bos");
            float[] gq = gpuQ.Buffer.ReadRange<float>(0, 4);
            gpuQ.Dispose(); xnTensor.Dispose(); wqF32Tensor.Dispose();
            _logger.LogTrace("[CPURef] GPU Q[:4]=[{Q0:F5},{Q1:F5},{Q2:F5},{Q3:F5}]", gq[0], gq[1], gq[2], gq[3]);
            _logger.LogTrace("[CPURef] Q match={Match} diff={Diff:F6}", Math.Abs(q4[0] - gq[0]) < 1e-3f, Math.Abs(q4[0] - gq[0]));
        }
        catch (Exception ex) { _logger.LogTrace(ex, "[CPURef] FAILED"); }
    }

    // ── Weight cache helpers ──────────────────────────────────────────────────

    /// <summary>
    /// Cache layer weights. Override in subclass to add format-specific weights.
    /// </summary>
    protected virtual void CacheLayerWeights(int globalLayer)
    {
        var nm = _nameMapper!;

        TryCacheWeight(nm.AttnNorm(globalLayer));

        bool layerHasCombined = _model.Tensors.Any(t => t.Name == $"blk.{globalLayer}.attn_qkv.weight");
        bool layerHasSeparate = _model.Tensors.Any(t => t.Name == $"blk.{globalLayer}.attn_q.weight");

        if (layerHasCombined)
        {
            TryCacheWeight($"blk.{globalLayer}.attn_qkv.weight");
            // Qwen3.6 Type A: attn_gate.weight [dModel, qDim] for gated Q
            TryCacheWeight($"blk.{globalLayer}.attn_gate.weight");
        }
        else if (layerHasSeparate)
        {
            TryCacheWeight($"blk.{globalLayer}.attn_q.weight");
            TryCacheWeight($"blk.{globalLayer}.attn_k.weight");
            TryCacheWeight($"blk.{globalLayer}.attn_v.weight");
            TryCacheWeight($"blk.{globalLayer}.attn_q_norm.weight");
            TryCacheWeight($"blk.{globalLayer}.attn_k_norm.weight");
            TryCacheWeight($"blk.{globalLayer}.attn_output.weight");
            // Qwen3.6 Type B: attn_gate.weight may be present for gated Q
            TryCacheWeight($"blk.{globalLayer}.attn_gate.weight");
        }

        if (layerHasCombined)
        {
            // For Qwen3.6 Type A, attn_output.weight does NOT exist.
            // ssm_out.weight [qDim, dModel] serves as attention output projection.
            // Try attn_output.weight first (for models that have it), then fall through.
            TryCacheWeight(nm.AttnOutput(globalLayer));
        }

        TryCacheWeight(nm.FfnNorm(globalLayer));
        TryCacheWeight($"blk.{globalLayer}.post_attention_norm.weight");
        TryCacheWeight(nm.FfnGate(globalLayer));
        TryCacheWeight(nm.FfnUp(globalLayer));
        TryCacheWeight(nm.FfnDown(globalLayer));

        // ── Gemma 4 extras ────────────────────────────────────────────────
        TryCacheWeight($"blk.{globalLayer}.attn_post_norm.weight");
        TryCacheWeight($"blk.{globalLayer}.ffn_post_norm.weight");
        TryCacheWeight($"blk.{globalLayer}.layer_out_scale.weight");

        // ── DeepSeek V2/V3/V4 extras ───────────────────────────────────────
        // MLA weights (compressed attention)
        TryCacheWeight($"blk.{globalLayer}.attn_q_a.weight");
        TryCacheWeight($"blk.{globalLayer}.attn_q_b.weight");
        TryCacheWeight($"blk.{globalLayer}.attn_q_a_norm.weight");
        TryCacheWeight($"blk.{globalLayer}.attn_kv_a_mqa.weight");
        TryCacheWeight($"blk.{globalLayer}.attn_kv_a_norm.weight");
        TryCacheWeight($"blk.{globalLayer}.attn_k_b.weight");
        TryCacheWeight($"blk.{globalLayer}.attn_v_b.weight");
        // MoE router
        TryCacheWeight($"blk.{globalLayer}.ffn_gate_inp.weight");
        TryCacheWeight($"blk.{globalLayer}.ffn_exp_probs_b.bias");
        // MoE shared expert (shexp)
        TryCacheWeight($"blk.{globalLayer}.ffn_gate_shexp.weight");
        TryCacheWeight($"blk.{globalLayer}.ffn_up_shexp.weight");
        TryCacheWeight($"blk.{globalLayer}.ffn_down_shexp.weight");
        // MoE routed experts (load only for diagnostic purposes)
        TryCacheWeight($"blk.{globalLayer}.ffn_gate_exps.weight");
        TryCacheWeight($"blk.{globalLayer}.ffn_up_exps.weight");
        TryCacheWeight($"blk.{globalLayer}.ffn_down_exps.weight");

        // SSM weights (optional)
        TryCacheWeight($"blk.{globalLayer}.ssm_a");
        TryCacheWeight($"blk.{globalLayer}.ssm_alpha.weight");
        TryCacheWeight($"blk.{globalLayer}.ssm_beta.weight");
        TryCacheWeight($"blk.{globalLayer}.ssm_dt.bias");
        TryCacheWeight($"blk.{globalLayer}.ssm_norm.weight");
        TryCacheWeight($"blk.{globalLayer}.ssm_out.weight");
        TryCacheWeight($"blk.{globalLayer}.ssm_conv1d.weight");
        TryCacheWeight($"blk.{globalLayer}.ssm_gate.weight");
    }

    protected void TryCacheWeight(string name)
    {
        if (_model.Tensors.Any(t => t.Name == name))
            CacheWeight(name);
    }

    protected void CacheWeight(string name)
    {
        if (_weightCache.ContainsKey(name)) return;

        var info = _model.Tensors.FirstOrDefault(t => t.Name == name)
            ?? throw new ArgumentException($"Tensor '{name}' not found in GGUF");

        var buffer = _model.LoadTensor(name);
        var dtype = MapType(info.Type);
        var dims = info.Shape.Select(s => (int)s).ToArray();

        _weightCache[name] = new Tensor(buffer, new TensorShape(dims), dtype, name);
    }

    private static DataType MapType(GGUFTensorType t) => t switch
    {
        GGUFTensorType.F32 => DataType.F32,
        GGUFTensorType.F16 => DataType.F16,
        GGUFTensorType.Q4_0 => DataType.Q4_0,
        GGUFTensorType.Q4_1 => DataType.Q4_1,
        GGUFTensorType.Q5_0 => DataType.Q5_0,
        GGUFTensorType.Q5_1 => DataType.Q5_1,
        GGUFTensorType.Q8_0 => DataType.Q8_0,
        GGUFTensorType.Q8_1 => DataType.Q8_1,
        GGUFTensorType.Q2_K => DataType.Q2_K,
        GGUFTensorType.Q3_K => DataType.Q3_K,
        GGUFTensorType.Q4_K => DataType.Q4_K,
        GGUFTensorType.Q5_K => DataType.Q5_K,
        GGUFTensorType.Q6_K => DataType.Q6_K,
        GGUFTensorType.Q8_K => DataType.Q8_K,
        _ => throw new NotSupportedException($"Unsupported tensor type: {t}")
    };

    // ── Disposal ──────────────────────────────────────────────────────────────

    public void Dispose()
    {
        if (_disposed) return;

        foreach (var t in _scratchF32.Values)
            t.Dispose();
        _scratchF32.Clear();

        foreach (var t in _weightCache.Values)
            t.Dispose();
        _weightCache.Clear();

        _ops.Dispose();

        _disposed = true;
        GC.SuppressFinalize(this);
    }
}
