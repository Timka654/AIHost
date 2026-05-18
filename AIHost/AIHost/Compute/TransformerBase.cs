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
    protected internal readonly int _ssmHVD;
    protected internal readonly int _ssmVH;
    protected internal readonly int _ssmKH;
    protected internal readonly int _ssmKD;
    protected internal readonly int _ssmVD;
    protected internal readonly int _ssmCD;
    protected internal readonly float _ropeFreqBase;
    protected internal readonly int _ropeDimCount;
    protected bool _disposed;
    private bool _hiddenDiagFired;
    protected internal readonly ILogger<TransformerBase> _logger = AppLogger.Create<TransformerBase>();

    public int ContextLength { get; private set; }

    protected internal readonly Dictionary<string, Tensor> _weightCache = [];

    // FIX: Channel mode — embedding lookup via CPU dequant from GGUF file.
    // Reading full 285MB into a single byte[] may fail on constrained systems,
    // so we store GGUFTensorInfo for row-by-row ReadTensorDataRange.
    private byte[]? _rawEmbeddingCpuBytes;
    private GGUFTensorInfo? _rawEmbeddingTensorInfo;
    private long _rawEmbeddingBytesPerRow;

    protected internal readonly Dictionary<string, Tensor> _scratchF32 = [];
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
        _headDim = ArchInt("attention.key_length", _dModel / _numHeads);
        _ropeFreqBase = ArchFlt("rope.freq_base", 10000.0f);
        _ropeDimCount = ArchInt("rope.dimension_count", _headDim);
        var kvHeads = ArchInt("attention.head_count_kv", 4);
        _numKVHeads = kvHeads;
        _localLayerCount = _numLayers;

        int ssmInner  = ArchInt("ssm.inner_size", 6144);
        int ssmState  = ArchInt("ssm.state_size", 128);
        int ssmGroups = ArchInt("ssm.group_count", 16);
        int ssmDtRank = ArchInt("ssm.time_step_rank", 48);
        _ssmVH  = ssmDtRank;
        _ssmKH  = ssmGroups;
        _ssmHVD = ssmState;
        _ssmKD  = ssmState * ssmGroups;
        _ssmVD  = ssmDtRank * _ssmHVD;
        _ssmCD  = ssmInner + 2 * _ssmKD;

        _logger.LogInformation("Transformer: layers={Layers} d_model={DModel} heads={Heads} kv_heads={KvHeads} head_dim={HeadDim} ctx={Ctx} rope={Rope} rope_dim={RopeDim}",
            _numLayers, _dModel, _numHeads, kvHeads, _headDim, ContextLength, _ropeFreqBase, _ropeDimCount);
        _logger.LogInformation("Transformer SSM: v_heads={VH} k_heads={KH} hvd={HVD} key_dim={KD} value_dim={VD} conv_dim={CD}",
            _ssmVH, _ssmKH, _ssmHVD, _ssmKD, _ssmVD, _ssmCD);
    }

    // ── Weight loading ────────────────────────────────────────────────────────

    public void LoadWeights()
    {
        _logger.LogInformation("[LOAD] Loading model weights...");
        _nameMapper = new TensorNameMapper(_model);

        _logger.LogInformation("[LOAD] Caching embed/norm/output weights");
        // FIX: Read raw quantized bytes BEFORE CacheWeight so GGUFReader stream
        // position is known. CacheWeight calls LoadTensor→ReadTensorData which
        // repositions the stream.
        LoadRawEmbeddingCpuBytes();
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

    public Tensor ForwardEmbedding(int[] tokenIds) => EmbeddingLookupOptimized(tokenIds);

    public Tensor ForwardLayers(Tensor x, uint startPos, KVCache? cache = null,
                                 SSMState? ssmState = null)
    {
        for (int i = 0; i < _localLayerCount; i++)
            x = _format.ApplyLayer(this, x, i, startPos, cache, ssmState);
        return x;
    }

    public Tensor ForwardHead(Tensor x)
    {
        var _ts = GlobalProfiler.Start();
        var (normF32, normScratch) = TempF32(_nameMapper!.OutputNorm);
        _ops.LayerNorm(x, normF32);
        if (!normScratch) normF32.Dispose();

        var outWeightCached = _weightCache[_nameMapper.OutputWeight];
        long outF32Bytes = (long)outWeightCached.Shape.TotalElements * sizeof(float);
        _logger.LogWarning("[DIAG_HEAD] outWeight shape=[{D0},{D1}] dtype={DT} f32Bytes={Bytes}",
            outWeightCached.Shape[0], outWeightCached.Shape[1], outWeightCached.DataType, outF32Bytes);

        if (outF32Bytes > 1L * 1024 * 1024 * 1024)
        {
            _ops.Flush();
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

    public Tensor Forward(int[] tokenIds, uint startPosition = 0, KVCache? kvCache = null,
                           SSMState? ssmState = null)
    {
        _logger.LogInformation("[FWD] Forward BEGIN tokens={Count} pos={Pos} kvCache={HasKV} ssm={HasSSM}",
            tokenIds.Length, startPosition, kvCache != null, ssmState != null);

        if (_nameMapper == null)
            throw new InvalidOperationException("Weights not loaded. Call LoadWeights() first.");

        // 1. Token embedding (CPU dequant from raw GGUF bytes — no 4.85GB GPU allocation)
        _logger.LogInformation("[FWD] EmbeddingLookup...");
        Tensor x = EmbeddingLookupOptimized(tokenIds);
        _logger.LogInformation("[FWD] EmbeddingLookup OK shape=[{D0},{D1}]", x.Shape[0], x.Shape[1]);

        // FIX: Flush IMMEDIATELY after embedding lookup. When _rawEmbeddingCpuBytes is null
        // (GGUF read failed or F32 table), EmbeddingLookupFromQuantized falls back to GPU
        // dequant which DeferExternals the 4.85GB F32 buffer. Without this Flush, VRAM is
        // exhausted before layer processing starts, causing ErrorOutOfDeviceMemory on SSM weights.
        _ops.Flush();

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
                // FIX: Flush after each layer to prevent TDR on AMD iGPUs
                _ops.Flush();
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

        {
            try
            {
                int dModel = x.Shape[1];
                int lastRow = x.Shape[0] - 1;
                ulong byteOff = (ulong)(lastRow * dModel * sizeof(float));
                int take = Math.Min(dModel, 32);
                var v = x.Buffer.ReadRange<float>(byteOff, take);
                float rms = MathF.Sqrt(v.Sum(val => val * val) / take);
                _logger.LogWarning("[DIAG_FINAL_HIDDEN] seqLen={SeqLen} row={Row} byteOff={Off} rms={Rms:F4} vals=[{Vals}]",
                    x.Shape[0], lastRow, byteOff, rms,
                    string.Join(",", v.Select(f => f.ToString("F4"))));
            }
            catch (Exception diagEx) { _logger.LogWarning("[DIAG_FINAL_HIDDEN] err={Err}", diagEx.Message); }
        }

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

    protected internal (Tensor tensor, bool isScratch) TempF32(string name)
    {
        var cached = _weightCache[name];
        if (cached.DataType == DataType.F32)
            return (_ops.Clone(cached), false);

        var dot2 = name.IndexOf('.', name.IndexOf('.') + 1);
        var key = dot2 >= 0 ? name[(dot2 + 1)..] : name;
        if (_scratchF32.TryGetValue(key, out var scratch))
        {
            _ops.DequantizeInto(cached, scratch);
            return (scratch, true);
        }
        return (_ops.Dequantize(cached), false);
    }

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

    protected internal Tensor? GetOrBuildTiledNorm(string weightName, int headDim, int totalDim)
    {
        string key = $"_tiled_{weightName}";
        if (_scratchF32.TryGetValue(key, out var cached)) return cached;
        if (!_weightCache.TryGetValue(weightName, out var srcW))
            return null;
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

    protected internal bool HasWeight(string name) => _weightCache.ContainsKey(name);
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

        // FIX: CPU dequant from raw bytes (full) or row-by-row GGUF reads.
        // Both paths avoid 4.85GB GPU F32 allocation and 44 TDR-triggering Flushes.
        if (_rawEmbeddingCpuBytes != null)
        {
            return _ops.EmbeddingLookupFromQuantized(tokenIds, embCached,
                _rawEmbeddingCpuBytes, "embeddings");
        }

        if (_rawEmbeddingTensorInfo != null)
        {
            return EmbeddingLookupFromGgufRows(tokenIds, embCached);
        }

        // Last resort: GPU dequant with per-chunk Flush (may trigger TDR)
        _logger.LogWarning("[EMB] No CPU path available — falling back to GPU dequant");
        return _ops.EmbeddingLookupFromQuantized(tokenIds, embCached, null, "embeddings");
    }

    // FIX: Channel mode — read individual quantized rows from GGUF file and
    // dequantize on CPU. Avoids both 4.85GB GPU allocation and 285MB CPU allocation.
    private Tensor EmbeddingLookupFromGgufRows(int[] tokenIds, Tensor embCached)
    {
        int seqLen = tokenIds.Length;
        int dModel = embCached.Shape[0];
        var resultData = new float[seqLen * dModel];
        var reader = _model.Reader;

        for (int i = 0; i < seqLen; i++)
        {
            int tokenId = tokenIds[i];
            ulong byteOffset = (ulong)(tokenId * _rawEmbeddingBytesPerRow);
            byte[] rowBytes = reader.ReadTensorDataRange(_rawEmbeddingTensorInfo!,
                byteOffset, (int)_rawEmbeddingBytesPerRow);
            ComputeOps.DequantizeRowCpu(embCached.DataType, rowBytes, 0,
                resultData, i * dModel, dModel);
        }

        return Tensor.FromData(_ops.Device, resultData,
            new TensorShape(new[] { seqLen, dModel }), "embeddings");
    }

    private void TryAllocateScratch(string layerPrefix)
    {
        try { AllocateScratchWeights(layerPrefix); }
        catch (Exception ex)
        {
            _logger.LogWarning("[Scratch] Allocation failed for {Layer}: {Error} — will allocate per inference", layerPrefix, ex.Message);
        }
    }

    // FIX: Channel mode — don't pre-allocate scratch for token_embd.weight (4.85GB F32).
    // EmbeddingLookupFromQuantizedCpu handles this via CPU dequant from raw GGUF bytes.
    private void TryAllocateScratchHead()
    {
        foreach (var name in new[] { "output.weight" })
        {
            if (!_weightCache.TryGetValue(name, out var cached)) continue;
            if (cached.DataType == DataType.F32) continue;
            if (_scratchF32.ContainsKey(name)) continue;
            try
            {
                var scratch = Tensor.Create(_ops.Device, cached.Shape, DataType.F32, name + "_scratch");
                _scratchF32[name] = scratch;
                _logger.LogDebug("[Scratch] {Name}: {SizeMB} MB F32", name, scratch.Shape.TotalElements * 4L / 1024 / 1024);
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
        var names = new List<string>();

        if (_nameMapper.HasCombinedQKV)
        {
            names.Add(_nameMapper.AttnNorm(refLayer));
            names.Add(_nameMapper.AttnQKV(refLayer));
            names.Add(_nameMapper.AttnOutput(refLayer));
            names.Add($"blk.{refLayer}.ssm_out.weight");
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
            names.Add($"blk.{refLayer}.attn_post_norm.weight");
            names.Add($"blk.{refLayer}.ffn_post_norm.weight");
            names.Add($"blk.{refLayer}.layer_out_scale.weight");
            names.Add($"blk.{refLayer}.attn_q_a.weight");
            names.Add($"blk.{refLayer}.attn_q_b.weight");
            names.Add($"blk.{refLayer}.attn_kv_a_mqa.weight");
            names.Add($"blk.{refLayer}.attn_k_b.weight");
            names.Add($"blk.{refLayer}.attn_v_b.weight");
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

    // FIX: Channel mode — CPU reference layer 0 is skipped. TempF32("token_embd.weight")
    // would allocate 4.85GB F32 via Dequantize→Tensor.Create, fragmenting VRAM.
    // The CPU reference is diagnostic-only; disabling preserves VRAM for inference.
    private void RunCpuReferenceLayer0()
    {
        _logger.LogWarning("[CPURef] SKIPPED — disabled in Channel mode to preserve VRAM");
    }

    // ── Weight cache helpers ──────────────────────────────────────────────────

    protected virtual void CacheLayerWeights(int globalLayer)
    {
        var nm = _nameMapper!;
        TryCacheWeight(nm.AttnNorm(globalLayer));

        bool layerHasCombined = _model.Tensors.Any(t => t.Name == $"blk.{globalLayer}.attn_qkv.weight");
        bool layerHasSeparate = _model.Tensors.Any(t => t.Name == $"blk.{globalLayer}.attn_q.weight");

        if (layerHasCombined)
        {
            TryCacheWeight($"blk.{globalLayer}.attn_qkv.weight");
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
            TryCacheWeight($"blk.{globalLayer}.attn_gate.weight");
        }

        if (layerHasCombined)
            TryCacheWeight(nm.AttnOutput(globalLayer));

        TryCacheWeight(nm.FfnNorm(globalLayer));
        TryCacheWeight($"blk.{globalLayer}.post_attention_norm.weight");
        TryCacheWeight(nm.FfnGate(globalLayer));
        TryCacheWeight(nm.FfnUp(globalLayer));
        TryCacheWeight(nm.FfnDown(globalLayer));

        TryCacheWeight($"blk.{globalLayer}.attn_post_norm.weight");
        TryCacheWeight($"blk.{globalLayer}.ffn_post_norm.weight");
        TryCacheWeight($"blk.{globalLayer}.layer_out_scale.weight");

        TryCacheWeight($"blk.{globalLayer}.attn_q_a.weight");
        TryCacheWeight($"blk.{globalLayer}.attn_q_b.weight");
        TryCacheWeight($"blk.{globalLayer}.attn_q_a_norm.weight");
        TryCacheWeight($"blk.{globalLayer}.attn_kv_a_mqa.weight");
        TryCacheWeight($"blk.{globalLayer}.attn_kv_a_norm.weight");
        TryCacheWeight($"blk.{globalLayer}.attn_k_b.weight");
        TryCacheWeight($"blk.{globalLayer}.attn_v_b.weight");
        TryCacheWeight($"blk.{globalLayer}.ffn_gate_inp.weight");
        TryCacheWeight($"blk.{globalLayer}.ffn_exp_probs_b.bias");
        TryCacheWeight($"blk.{globalLayer}.ffn_gate_shexp.weight");
        TryCacheWeight($"blk.{globalLayer}.ffn_up_shexp.weight");
        TryCacheWeight($"blk.{globalLayer}.ffn_down_shexp.weight");
        TryCacheWeight($"blk.{globalLayer}.ffn_gate_exps.weight");
        TryCacheWeight($"blk.{globalLayer}.ffn_up_exps.weight");
        TryCacheWeight($"blk.{globalLayer}.ffn_down_exps.weight");

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

    // FIX: Channel mode — store embedding tensor info for row-by-row CPU dequant.
    // Two-stage: try full ReadTensorData first (~285MB), fall back to row-by-row
    // ReadTensorDataRange from GGUF file if full read fails or OOM.
    private void LoadRawEmbeddingCpuBytes()
    {
        try
        {
            var embName = _nameMapper!.TokenEmbd;
            var embTensor = _model.Tensors.FirstOrDefault(t => t.Name == embName);
            if (embTensor == null || embTensor.Type == GGUFTensorType.F32)
            {
                _logger.LogInformation("[LOAD] token_embd is F32 — CPU raw bytes not needed");
                return;
            }

            // Always store info for row-by-row fallback
            _rawEmbeddingTensorInfo = embTensor;
            int dModel = (int)embTensor.Shape[0];
            int vocabSize = (int)embTensor.Shape[1];
            _rawEmbeddingBytesPerRow = (long)embTensor.SizeInBytes / vocabSize;

            // Try full read first
            try
            {
                _rawEmbeddingCpuBytes = _model.Reader.ReadTensorData(embTensor);
                _logger.LogInformation("[LOAD] Read {Size}MB raw quantized embedding to CPU (full)",
                    _rawEmbeddingCpuBytes.Length / 1024 / 1024);
            }
            catch (OutOfMemoryException)
            {
                _logger.LogWarning("[LOAD] OOM reading full embedding — will use row-by-row GGUF reads");
                _rawEmbeddingCpuBytes = null;
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[LOAD] Failed to init embedding CPU path — GPU fallback");
            _rawEmbeddingCpuBytes = null;
            _rawEmbeddingTensorInfo = null;
        }
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
