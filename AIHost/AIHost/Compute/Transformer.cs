using AIHost.GGUF;
using AIHost.ICompute;
using AIHost.Inference;

namespace AIHost.Compute;

/// <summary>
/// LLM Transformer model for inference.
///
/// Memory strategy: weights are stored QUANTIZED in the cache (~600 MB for Q4_K_M).
/// Each forward pass dequantizes only the weights needed for the current operation into
/// temporary F32 tensors that are freed immediately after use.
/// Peak GPU memory per layer ≈ 160 MB F32 (vs 4.4 GB if all pre-dequantized).
/// </summary>
public class Transformer : IDisposable
{
    private readonly ComputeOps _ops;
    private readonly IGGUFModel _model;
    private readonly int _numLayers;
    private readonly int _dModel;
    private readonly int _numHeads;
    private bool _disposed;

    /// <summary>Max context length from GGUF metadata (0 = unknown).</summary>
    public int ContextLength { get; private set; }

    // Stores QUANTIZED tensors (or F32 for weights already stored as F32 in GGUF).
    private readonly Dictionary<string, Tensor> _weightCache = [];

    // Pre-allocated F32 scratch tensors keyed by weight name (e.g. "blk.0.attn_q.weight").
    // Reused across all layers and tokens to eliminate per-inference vkAllocateMemory calls.
    // Sized to hold F32 data for any layer (all layers share the same weight shapes).
    private readonly Dictionary<string, Tensor> _scratchF32 = [];

    public int LayerCount => _numLayers;
    public ComputeOps Ops => _ops;


    public Transformer(IComputeDevice device, IGGUFModel model)
    {
        _ops = new ComputeOps(device);
        _model = model;

        var metadata = model.Metadata;
        _numLayers = metadata.GetValue<int>(GGUFMetadata.KeyBlockCount, 22);
        _dModel = metadata.GetValue<int>(GGUFMetadata.KeyEmbeddingLength, 2048);
        _numHeads = metadata.GetValue<int>(GGUFMetadata.KeyAttentionHeadCount, 32);
        ContextLength = metadata.GetValue<int>(GGUFMetadata.KeyContextLength, 0);

        int GetInt(string key, int def) {
            if (metadata.TryGetValue<int>(key, out var vi)) return vi;
            if (metadata.TryGetValue<uint>(key, out var vu)) return (int)vu;
            return def;
        }
        float GetFlt(string key, float def) {
            if (metadata.TryGetValue<float>(key, out var vf)) return vf;
            return def;
        }
        var ropeFreqBase = GetFlt(GGUFMetadata.KeyRopeFreqBase, 10000.0f);
        var ffnLength    = GetInt("llama.feed_forward_length", 0);
        var ropeDimCount = GetInt("llama.rope.dimension_count", _dModel / _numHeads);
        var rmsEps       = GetFlt("llama.attention.layer_norm_rms_epsilon", 1e-5f);
        var kvHeads      = GetInt("llama.attention.head_count_kv", 4);
        Console.WriteLine($"Transformer: layers={_numLayers} d_model={_dModel} heads={_numHeads} kv_heads={kvHeads} ctx={ContextLength} ffn={ffnLength} rope={ropeFreqBase}");
    }

    /// <summary>
    /// Upload all model weights to GPU (quantized). Fast: no dequantization here.
    /// </summary>
    public void LoadWeights()
    {
        Console.WriteLine("Loading model weights...");

        CacheWeight("token_embd.weight");
        CacheWeight("output_norm.weight");
        CacheWeight("output.weight");

        for (int i = 0; i < _numLayers; i++)
        {
            string p = $"blk.{i}";
            CacheWeight($"{p}.attn_norm.weight");
            CacheWeight($"{p}.attn_q.weight");
            CacheWeight($"{p}.attn_k.weight");
            CacheWeight($"{p}.attn_v.weight");
            CacheWeight($"{p}.attn_output.weight");
            CacheWeight($"{p}.ffn_norm.weight");
            CacheWeight($"{p}.ffn_gate.weight");
            CacheWeight($"{p}.ffn_up.weight");
            CacheWeight($"{p}.ffn_down.weight");

            if ((i + 1) % 5 == 0 || i == _numLayers - 1)
                Console.WriteLine($"  Uploaded layer {i + 1}/{_numLayers}");
        }

        Console.WriteLine($"✓ Weights loaded ({_weightCache.Count} tensors, quantized in VRAM)");

        // Pre-allocate F32 scratch tensors using layer 0 as the template for all layers
        // (all layers share the same weight shapes in transformer architectures).
        AllocateScratchWeights("blk.0");
        Console.WriteLine($"✓ F32 scratch buffers allocated ({_scratchF32.Count} tensors)\n");
        // CPU reference verification: run single BOS token through layer 0 on CPU, compare with GPU
        RunCpuReferenceLayer0();
    }

    /// Run layer 0 forward pass on CPU for BOS (token ID 1) and compare key values with GPU.
    private void RunCpuReferenceLayer0()
    {
        try
        {
            // Read embedding for token 1 (BOS) on CPU
            var embQ = _weightCache["token_embd.weight"];
            var (embF32, _) = TempF32("token_embd.weight");
            float[] emb = embF32.Buffer.ReadRange<float>(1 * 2048 * 4UL, 2048);
            embF32.Dispose();

            // Compute RMSNorm on CPU
            var normW = _weightCache["blk.0.attn_norm.weight"].Buffer.Read<float>();
            float sumSq = 0f; foreach (var v in emb) sumSq += v * v;
            float rms = MathF.Sqrt(sumSq / 2048 + 1e-5f);
            float[] xn = new float[2048];
            for (int i = 0; i < 2048; i++) xn[i] = emb[i] / rms * normW[i];

            Console.WriteLine($"[CPURef] BOS emb[:3]=[{emb[0]:F5},{emb[1]:F5},{emb[2]:F5}] rms={rms:F6}");
            Console.WriteLine($"[CPURef] xnorm[:3]=[{xn[0]:F5},{xn[1]:F5},{xn[2]:F5}] maxAbs={xn.Max(MathF.Abs):F4}");

            // Compute Q = xnorm @ wQ (column-major) — first 4 output dims
            var wqF32 = _ops.Dequantize(_weightCache["blk.0.attn_q.weight"]);
            float[] wqData = wqF32.Buffer.ReadRange<float>(0, 2048 * 4); wqF32.Dispose(); // first 4 output cols
            float[] q4 = new float[4];
            for (int n = 0; n < 4; n++)
                for (int k = 0; k < 2048; k++) q4[n] += xn[k] * wqData[k + n * 2048];
            Console.WriteLine($"[CPURef] Q[:4]=[{q4[0]:F5},{q4[1]:F5},{q4[2]:F5},{q4[3]:F5}]");

            // Also compute V for BOS and compare
            var wvData = _ops.Dequantize(_weightCache["blk.0.attn_v.weight"]);
            float[] wvArr = wvData.Buffer.ReadRange<float>(0, 2048 * 4); wvData.Dispose(); // first 4 output dims
            float[] bosV4 = new float[4];
            for (int n=0;n<4;n++) for (int k=0;k<2048;k++) bosV4[n]+=xn[k]*wvArr[k+n*2048];
            Console.WriteLine($"[CPURef] BOS V[:4]=[{bosV4[0]:F5},{bosV4[1]:F5},{bosV4[2]:F5},{bosV4[3]:F5}]");
            var wvGpu = _ops.Dequantize(_weightCache["blk.0.attn_v.weight"]);
            var xnT = Tensor.FromData(_ops.Device, xn, new TensorShape(new[]{1,2048}), "xn_bos_v");
            var bosVgpu = _ops.MatMulWeights(xnT, wvGpu, "V_bos");
            float[] gv = bosVgpu.Buffer.ReadRange<float>(0,4); bosVgpu.Dispose(); xnT.Dispose(); wvGpu.Dispose();
            Console.WriteLine($"[CPURef] BOS GPU V[:4]=[{gv[0]:F5},{gv[1]:F5},{gv[2]:F5},{gv[3]:F5}] match={Math.Abs(bosV4[0]-gv[0])<1e-3f}");

            // GPU Q for comparison: run a minimal forward pass
            var wqF32Tensor = _ops.Dequantize(_weightCache["blk.0.attn_q.weight"]);
            var xnTensor = Tensor.FromData(_ops.Device, xn, new TensorShape(new[]{1,2048}), "xn_bos");
            var gpuQ = _ops.MatMulWeights(xnTensor, wqF32Tensor, "Q_bos");
            float[] gq = gpuQ.Buffer.ReadRange<float>(0, 4);
            gpuQ.Dispose(); xnTensor.Dispose(); wqF32Tensor.Dispose();
            Console.WriteLine($"[CPURef] GPU Q[:4]=[{gq[0]:F5},{gq[1]:F5},{gq[2]:F5},{gq[3]:F5}]");
            Console.WriteLine($"[CPURef] Q match={Math.Abs(q4[0]-gq[0])<1e-3f} diff={Math.Abs(q4[0]-gq[0]):F6}");

            // Check token 6324 ('Hi') - actual token used in test
            var embFull = _ops.Dequantize(_weightCache["token_embd.weight"]);
            float[] hiEmb = embFull.Buffer.ReadRange<float>(6324UL * 2048 * 4, 2048); // token 6324 = 'Hi'
            embFull.Dispose();
            float hiSumSq = 0f; foreach (var v in hiEmb) hiSumSq += v * v;
            float hiRms = MathF.Sqrt(hiSumSq / 2048 + 1e-5f);
            float[] hiXn = new float[2048];
            for (int i = 0; i < 2048; i++) hiXn[i] = hiEmb[i] / hiRms * normW[i];
            float[] hiQ4 = new float[4];
            for (int n = 0; n < 4; n++)
                for (int k = 0; k < 2048; k++) hiQ4[n] += hiXn[k] * wqData[k + n * 2048];
            // GPU version
            var wqF32b = _ops.Dequantize(_weightCache["blk.0.attn_q.weight"]);
            var hiTensor = Tensor.FromData(_ops.Device, hiXn, new TensorShape(new[]{1,2048}), "xn_hi");
            var hiGpuQ = _ops.MatMulWeights(hiTensor, wqF32b, "Q_hi");
            float[] hiGq = hiGpuQ.Buffer.ReadRange<float>(0, 4);
            hiGpuQ.Dispose(); hiTensor.Dispose(); wqF32b.Dispose();
            Console.WriteLine($"[CPURef] Hi(6324) rms={hiRms:F6} emb[:3]=[{hiEmb[0]:F6},{hiEmb[1]:F6},{hiEmb[2]:F6}]");
            Console.WriteLine($"[CPURef] Hi xn[:3]=[{hiXn[0]:F5},{hiXn[1]:F5},{hiXn[2]:F5}]");
            Console.WriteLine($"[CPURef] Hi CPU Q[:4]=[{hiQ4[0]:F5},{hiQ4[1]:F5},{hiQ4[2]:F5},{hiQ4[3]:F5}]");
            Console.WriteLine($"[CPURef] Hi GPU Q[:4]=[{hiGq[0]:F5},{hiGq[1]:F5},{hiGq[2]:F5},{hiGq[3]:F5}]");
            Console.WriteLine($"[CPURef] Hi Q match={Math.Abs(hiQ4[0]-hiGq[0])<1e-3f}");
        }
        catch (Exception ex) { Console.WriteLine($"[CPURef] FAILED: {ex.Message}"); }
    }

    private void AllocateScratchWeights(string layerPrefix)
    {
        var names = new[]
        {
            $"{layerPrefix}.attn_norm.weight",
            $"{layerPrefix}.attn_q.weight",
            $"{layerPrefix}.attn_k.weight",
            $"{layerPrefix}.attn_v.weight",
            $"{layerPrefix}.attn_output.weight",
            $"{layerPrefix}.ffn_norm.weight",
            $"{layerPrefix}.ffn_gate.weight",
            $"{layerPrefix}.ffn_up.weight",
            $"{layerPrefix}.ffn_down.weight",
        };

        foreach (var name in names)
        {
            var cached = _weightCache[name];
            if (cached.DataType == DataType.F32) continue; // No scratch needed

            var scratch = Tensor.Create(_ops.Device, cached.Shape, DataType.F32, name + "_scratch");
            // Strip layer prefix so all layers share the same scratch key (e.g. "attn_q.weight")
            var key = name[(layerPrefix.Length + 1)..];
            _scratchF32[key] = scratch;
        }
    }

    /// <summary>
    /// Forward pass through transformer.
    /// </summary>
    public Tensor Forward(int[] tokenIds, uint startPosition = 0, KVCache? kvCache = null)
    {
        if (!_weightCache.ContainsKey("token_embd.weight"))
            throw new InvalidOperationException("Weights not loaded. Call LoadWeights() first.");
#if DEEP_DEBUG
        Console.WriteLine($"Forward pass: {tokenIds.Length} tokens at position {startPosition}");
#endif

        // 1. Token embedding: dequantize table → lookup → free table
        Tensor x;
        {
            var (embF32, embScratch) = TempF32("token_embd.weight");
            x = _ops.EmbeddingLookup(tokenIds, embF32, "embeddings");
            if (!embScratch) embF32.Dispose();
        }

        // 2. All transformer layers
        for (int i = 0; i < _numLayers; i++)
            x = ApplyLayer(x, i, startPosition, kvCache);

        // 3. Final layer norm
        {
            var (normF32, normScratch) = TempF32("output_norm.weight");
            _ops.LayerNorm(x, normF32);
            if (!normScratch) normF32.Dispose();
        }

        // 4. Vocab projection (GGUF column-major weight)
        Tensor logits;
        {
            // Debug: print x_normed for last token (compare with llama-cpp embed output)
            if (tokenIds.Length <= 3)
            {
                var xd = x.ReadData();
                int last = x.Shape[0] - 1;
                Console.WriteLine($"[XNorm] seqLen={x.Shape[0]} last tok x_normed[:3]=[{xd[last*2048]:F5},{xd[last*2048+1]:F5},{xd[last*2048+2]:F5}] maxAbs={xd.Max(Math.Abs):F4}");
            }
            var (outF32, outScratch) = TempF32("output.weight");
            logits = _ops.MatMulWeights(x, outF32, "logits");
            if (!outScratch) outF32.Dispose();
        }

        x.Dispose();
        return logits;
    }

    private Tensor ApplyLayer(Tensor x, int layerIdx, uint position, KVCache? kvCache = null)
    {
        string p = $"blk.{layerIdx}";
        int seqLen = x.Shape[0];

        // Batch mode (one fence wait per layer): fast for generation (seqLen=1, tiny matrices).
        // For prefill (seqLen > 1) the large MatMuls exceed the Windows TDR timeout (~2 s),
        // so we fall back to per-op flush which keeps each submit safely under the limit.
        // Batch mode for generation (seqLen=1): ~22 fence waits instead of ~200+.
        // Prefill (seqLen > 1) stays per-op to avoid TDR timeout on large matrices.
        bool useBatch = seqLen == 1;

        if (useBatch)
        {
            // Open batch BEFORE dequantizing weights so that all dispatches
            // (dequantize + attention + FFN) land in ONE command buffer → one fence wait.
            _ops.BeginBatch();

            (Tensor t, bool scratch) Wb(string name)
            {
                var (t2, s) = TempF32(name);
                if (!s) _ops.DeferExternal(t2);
                return (t2, s);
            }

            var (wAttnNorm, _) = Wb($"{p}.attn_norm.weight");
            var (wQ,        _) = Wb($"{p}.attn_q.weight");
            var (wK,        _) = Wb($"{p}.attn_k.weight");
            var (wV,        _) = Wb($"{p}.attn_v.weight");
            var (wAttnOut,  _) = Wb($"{p}.attn_output.weight");
            var (wFfnNorm,  _) = Wb($"{p}.ffn_norm.weight");
            var (wGate,     _) = Wb($"{p}.ffn_gate.weight");
            var (wUp,       _) = Wb($"{p}.ffn_up.weight");
            var (wDown,     _) = Wb($"{p}.ffn_down.weight");
            // Single barrier after all independent weight dequants — GPU can run them in parallel.
            _ops.InsertBarrier();

            var outputB = _ops.TransformerLayer(
                x, wAttnNorm, wQ, wK, wV, wAttnOut,
                wFfnNorm, wGate, wUp, wDown,
                _numHeads, position, kvCache, layerIdx);

            _ops.DeferExternal(x);
            _ops.Flush();
            return outputB;
        }

        // Prefill: per-op flush (dequantize → immediate flush → use → dispose).
        var (wAN, sAN) = TempF32($"{p}.attn_norm.weight");
        var (wQ2, sQ)  = TempF32($"{p}.attn_q.weight");
        var (wK2, sK)  = TempF32($"{p}.attn_k.weight");
        var (wV2, sV)  = TempF32($"{p}.attn_v.weight");
        var (wAO, sAO) = TempF32($"{p}.attn_output.weight");
        var (wFN, sFN) = TempF32($"{p}.ffn_norm.weight");
        var (wG,  sG)  = TempF32($"{p}.ffn_gate.weight");
        var (wU,  sU)  = TempF32($"{p}.ffn_up.weight");
        var (wD,  sD)  = TempF32($"{p}.ffn_down.weight");

        var output = _ops.TransformerLayer(
            x, wAN, wQ2, wK2, wV2, wAO,
            wFN, wG, wU, wD,
            _numHeads, position, kvCache, layerIdx);

        void Dispose2(Tensor t, bool isScratch) { if (!isScratch) t.Dispose(); }
        Dispose2(wAN, sAN); Dispose2(wQ2, sQ); Dispose2(wK2, sK); Dispose2(wV2, sV);
        Dispose2(wAO, sAO); Dispose2(wFN, sFN); Dispose2(wG, sG); Dispose2(wU, sU);
        Dispose2(wD, sD);
        x.Dispose();
        return output;
    }

    /// <summary>
    /// Returns an F32 tensor from the named cached weight.
    /// Uses a pre-allocated scratch buffer if available (no vkAllocateMemory); otherwise
    /// allocates a new tensor. Caller must NOT dispose scratch tensors (owned by Transformer).
    /// </summary>
    private (Tensor tensor, bool isScratch) TempF32(string name)
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
    /// Uploads a single weight to GPU (quantized, no dequantization).
    /// </summary>
    private void CacheWeight(string name)
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
        GGUFTensorType.F32  => DataType.F32,
        GGUFTensorType.F16  => DataType.F16,
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






