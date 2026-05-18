using System.Collections.Concurrent;
using System.Text.Json;
using AIHost.Compute;
using AIHost.GGUF;
using AIHost.ICompute;
using AIHost.Inference;
using AIHost.ICompute.Vulkan;
using AIHost.Tokenizer;

namespace AIHost.Config;

/// <summary>
/// Manages model loading, caching, and lifecycle.
/// Thread-safe: ConcurrentDictionary + SemaphoreSlim guard against
/// concurrent loads, watcher updates, and auto-unload races.
/// </summary>
public class ModelManager : IDisposable
{
    private readonly string _modelsDirectory;
    private readonly string _cacheDirectory;
    private readonly IComputeDevice _device;
    private readonly ILogger<ModelManager> _logger;

    public string ModelsDirectory => _modelsDirectory;
    public string CacheDirectory => _cacheDirectory;
    private readonly ConcurrentDictionary<string, ModelInstance> _loadedModels = new();
    private readonly ConcurrentDictionary<string, ModelConfig> _modelConfigs = new();
    // Serializes the slow model-load path so the same model is never loaded twice.
    private readonly SemaphoreSlim _loadLock = new(1, 1);
    private bool _disposed;

    public ModelManager(string modelsDirectory, string cacheDirectory, IComputeDevice device, ILogger<ModelManager> logger)
    {
        // Normalize paths (resolve ./ and ../ to absolute paths)
        _modelsDirectory = Path.GetFullPath(modelsDirectory);
        _cacheDirectory = Path.GetFullPath(cacheDirectory);
        _device = device;
        _logger = logger;

        // Ensure directories exist
        Directory.CreateDirectory(_modelsDirectory);
        Directory.CreateDirectory(_cacheDirectory);

        // Load all model configs
        LoadModelConfigs();
    }

    /// <summary>
    /// Load all model.json files from subdirectories
    /// </summary>
    private void LoadModelConfigs()
    {
        if (!Directory.Exists(_modelsDirectory))
        {
            Directory.CreateDirectory(_modelsDirectory);
            _logger.LogInformation("Created models directory: {Dir}", _modelsDirectory);
            return;
        }

        // Load configs from subdirectories (legacy format: subdir/model.json)
        foreach (var modelDir in Directory.GetDirectories(_modelsDirectory))
        {
            var configPath = Path.Combine(modelDir, "model.json");
            if (!File.Exists(configPath))
                continue;

            try
            {
                var json = File.ReadAllText(configPath);
                var config = JsonSerializer.Deserialize<ModelConfig>(json);

                if (config != null && !string.IsNullOrEmpty(config.Name))
                {
                    _modelConfigs[config.Name] = config;
                    _logger.LogInformation("Loaded model config: {Name}", config.Name);
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Failed to load config from {Path}", configPath);
            }
        }

        // Also load configs directly from models directory (new format: name.json)
        foreach (var configFile in Directory.GetFiles(_modelsDirectory, "*.json"))
        {
            try
            {
                var json = File.ReadAllText(configFile);
                var config = JsonSerializer.Deserialize<ModelConfig>(json);
                var configName = Path.GetFileNameWithoutExtension(configFile);

                if (config != null)
                {
                    // Use filename as name if config.name is empty
                    if (string.IsNullOrEmpty(config.Name))
                        config.Name = configName;
                    
                    _modelConfigs[config.Name] = config;
                    _logger.LogInformation("Loaded model config: {Name} from {File}", config.Name, Path.GetFileName(configFile));
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Failed to load config from {Path}", configFile);
            }
        }

        _logger.LogInformation("Loaded {Count} model configuration(s)", _modelConfigs.Count);
    }

    /// <summary>
    /// Reload a single config from disk
    /// </summary>
    public bool ReloadConfig(string configName)
    {
        try
        {
            // Primary format: {name}/model.json (used by web panel save operations)
            var configPath = Path.Combine(_modelsDirectory, configName, "model.json");
            if (!File.Exists(configPath))
            {
                // Flat format fallback: {name}.json
                configPath = Path.Combine(_modelsDirectory, $"{configName}.json");
                if (!File.Exists(configPath))
                {
                    _logger.LogWarning("Config file not found for: {Name}", configName);
                    return false;
                }
            }

            var json = File.ReadAllText(configPath);
            var config = JsonSerializer.Deserialize<ModelConfig>(json);

            if (config != null)
            {
                if (string.IsNullOrEmpty(config.Name))
                    config.Name = configName;

                _modelConfigs[config.Name] = config;
                _logger.LogInformation("Reloaded model config: {Name}", config.Name);
                return true;
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to reload config: {Name}", configName);
        }
        return false;
    }

    /// <summary>
    /// Get or load a model instance
    /// </summary>
    public async Task<ModelInstance> GetModelAsync(string modelName)
    {
        // Fast path: already loaded — no lock needed with ConcurrentDictionary.
        if (_loadedModels.TryGetValue(modelName, out var instance))
            return instance;

        // Slow path: need to load. Serialize to prevent two concurrent requests
        // both deciding the model isn't loaded and starting two parallel loads.
        await _loadLock.WaitAsync();
        try
        {
            // Double-check after acquiring the semaphore.
            if (_loadedModels.TryGetValue(modelName, out instance))
                return instance;

            return await LoadModelInternalAsync(modelName);
        }
        finally
        {
            _loadLock.Release();
        }
    }

    private async Task<ModelInstance> LoadModelInternalAsync(string modelName)
    {
        // Get config
        if (!_modelConfigs.TryGetValue(modelName, out var config))
            throw new ArgumentException($"Model '{modelName}' not found in configuration");

        // Resolve model path
        var modelPath = ResolveModelPath(config);

        // Download if needed (always to cache directory for URLs)
        if (!File.Exists(modelPath) && config.AutoDownload && IsUrl(config.ModelPath))
        {
            var fileName = Path.GetFileName(new Uri(config.ModelPath).LocalPath);
            var cachePath = Path.Combine(_cacheDirectory, fileName);
            await DownloadModelAsync(config.ModelPath, cachePath);
            modelPath = cachePath;
        }

        if (!File.Exists(modelPath))
            throw new FileNotFoundException($"Model file not found: {modelPath}");

        // Determine compute device.
        // devices[] with 1 entry = single-GPU with explicit provider/index (e.g. ROCm).
        // devices[] with 2+ entries = multi-GPU split.
        bool isMultiGPUConfig = config.Devices is { Length: > 1 };
        bool isSingleDeviceExplicit = config.Devices is { Length: 1 };
        IComputeDevice device = _device;
        IComputeDevice? perModelDevice = null;

        if (isSingleDeviceExplicit)
        {
            // devices:[{index, provider}] with one entry — use that device for single-GPU
            var d0       = config.Devices![0];
            var provider = d0.Provider ?? config.ComputeProvider ?? "vulkan";
            _logger.LogInformation("Creating single device from devices[0]: {Provider}[{Index}] for model {Model}",
                provider, d0.Index, modelName);
            perModelDevice = CreateComputeDevice(provider, d0.Index);
            device = perModelDevice;
        }
        else if (!isMultiGPUConfig && (config.ComputeProvider != null || config.DeviceIndex != null))
        {
            var provider    = config.ComputeProvider ?? "vulkan";
            var deviceIndex = config.DeviceIndex ?? 0;
            _logger.LogInformation("Creating dedicated {Provider} device (index {Index}) for model {Model}", provider, deviceIndex, modelName);
            perModelDevice = CreateComputeDevice(provider, deviceIndex);
            device = perModelDevice;
        }

        string activeProvider = isSingleDeviceExplicit
            ? (config.Devices![0].Provider ?? config.ComputeProvider ?? "vulkan")
            : (config.ComputeProvider ?? "(global)");
        string activeDevice = isSingleDeviceExplicit
            ? config.Devices![0].Index.ToString()
            : (config.DeviceIndex?.ToString() ?? "(global)");

        _logger.LogInformation(
            "Loading model {Model} | provider={Provider} device={Device} keep_alive={KeepAlive}m batch={Batch} mmap={Mmap} mlock={Mlock}",
            modelName, activeProvider, activeDevice,
            config.KeepAliveMinutes?.ToString() ?? "(global)",
            config.BatchSize?.ToString() ?? "8",
            config.EnableMmap, config.EnableMlock);

        _logger.LogInformation("[LoadModel] Step 1/4: reading GGUF file at {Path}", modelPath);

        if (config.EnableMlock && !config.EnableMmap)
            _logger.LogWarning("enable_mlock requires enable_mmap to be true");

        int batchSize = config.BatchSize ?? 8;
        IInferenceEngine engine;

        // ── Multi-GPU path ────────────────────────────────────────────────────
        if (isMultiGPUConfig)
        {
            var devCfgs = config.Devices!;
            _logger.LogInformation("Multi-GPU: {Count} devices [{Indices}]",
                devCfgs.Length,
                string.Join(", ", devCfgs.Select(d => $"{d.Provider ?? config.ComputeProvider ?? "vulkan"}[{d.Index}]" +
                                                       (d.Layers.HasValue ? $"×{d.Layers}L" : ""))));

            // Build layer-split boundaries from per-device Layers field.
            // Last device always gets the remainder (its Layers value is ignored).
            int[]? layerSplit = BuildLayerSplit(devCfgs);

            var xfm = new MultiGPUTransformer(
                devices: devCfgs.Select(d =>
                    CreateComputeDevice(d.Provider ?? config.ComputeProvider ?? "vulkan", d.Index)).ToArray(),
                modelFactory: d => new AIHost.GGUF.LazyGGUFModel(
                    modelPath, d, config.EnableMmap, config.EnableMlock,
                    requireDeviceLocal: !config.AllowSharedMemory),
                layerSplit: layerSplit);

            engine = new MultiGPUInferenceEngine(xfm, BPETokenizer.FromGGUF(xfm.PrimaryModel.Reader), batchSize, config.MaxConcurrency);
        }
        else
        {
            // ── Single-GPU path ───────────────────────────────────────────
            IGGUFModel ggufModel = new AIHost.GGUF.LazyGGUFModel(
                modelPath, device,
                config.EnableMmap, config.EnableMlock,
                requireDeviceLocal: !config.AllowSharedMemory);
            var tokenizer = BPETokenizer.FromGGUF(ggufModel.Reader);
            var transformer = TransformerFactory.Create(device, ggufModel);
            transformer.LoadWeights();

            // ── Arena allocator (Vulkan only) ───────────────────────────
            TryAttachArena(device, transformer, config);

            engine = new InferenceEngine(transformer, tokenizer, transformer.Ops, batchSize, config.MaxConcurrency);
        }

        var systemMessages = await LoadSystemMessagesAsync(config);

        var loaded = new ModelInstance
        {
            Name = modelName,
            Config = config,
            Engine = engine,
            Device = perModelDevice,
            SystemMessages = systemMessages,
            IsChatModel = config.ChatTemplate
                ?? config.ModelPath.Contains("-Instruct", StringComparison.OrdinalIgnoreCase)
                || config.ModelPath.Contains("-Chat", StringComparison.OrdinalIgnoreCase)
                || config.ModelPath.Contains("-IT", StringComparison.OrdinalIgnoreCase),
            LoadedAt = DateTime.UtcNow
        };

        _loadedModels[modelName] = loaded;

        _logger.LogInformation("Model '{Model}' loaded successfully", modelName);

        return loaded;
    }

    /// <summary>
    /// Resolve model path (searches in models dir, then cache)
    /// </summary>
    private string ResolveModelPath(ModelConfig config)
    {
        var path = config.ModelPath;

        // If URL, check cache first, then model dir
        if (IsUrl(path))
        {
            var fileName = Path.GetFileName(new Uri(path).LocalPath);
            
            // Check cache directory first
            var cachePath = Path.Combine(_cacheDirectory, fileName);
            if (File.Exists(cachePath))
                return cachePath;
            
            // Fall back to model directory
            return Path.Combine(_modelsDirectory, config.Name, fileName);
        }

        // If absolute path, use as-is
        if (Path.IsPathRooted(path))
            return path;

        // Relative path: check model directory first
        var modelDirPath = Path.Combine(_modelsDirectory, config.Name, path);
        if (File.Exists(modelDirPath))
            return modelDirPath;

        // Check cache directory as fallback
        var cacheDirPath = Path.Combine(_cacheDirectory, path);
        if (File.Exists(cacheDirPath))
            return cacheDirPath;

        // Return model directory path (will be used for download)
        return modelDirPath;
    }

    /// <summary>
    /// Check if string is a URL
    /// </summary>
    private static bool IsUrl(string path)
        => path.StartsWith("http://") || path.StartsWith("https://");

    /// <summary>
    /// Download model from URL
    /// </summary>
    private async Task DownloadModelAsync(string url, string destinationPath)
    {
        _logger.LogInformation("Downloading model from {Url}", url);
        
        var directory = Path.GetDirectoryName(destinationPath);
        if (!string.IsNullOrEmpty(directory))
            Directory.CreateDirectory(directory);

        using var client = new HttpClient();
        client.Timeout = TimeSpan.FromHours(1);

        var response = await client.GetAsync(url, HttpCompletionOption.ResponseHeadersRead);
        response.EnsureSuccessStatusCode();

        var totalBytes = response.Content.Headers.ContentLength ?? 0;
        var downloadedBytes = 0L;

        await using var contentStream = await response.Content.ReadAsStreamAsync();
        await using var fileStream = new FileStream(destinationPath, FileMode.Create, FileAccess.Write, FileShare.None);

        var buffer = new byte[8192];
        int bytesRead;

        while ((bytesRead = await contentStream.ReadAsync(buffer)) > 0)
        {
            await fileStream.WriteAsync(buffer.AsMemory(0, bytesRead));
            downloadedBytes += bytesRead;

            if (totalBytes > 0 && downloadedBytes % (10 * 1024 * 1024) == 0) // Log every 10MB
            {
                var progress = (double)downloadedBytes / totalBytes * 100;
                _logger.LogInformation("Download progress: {Pct:F1}% ({MB}MB / {TotalMB}MB)", progress, downloadedBytes / (1024 * 1024), totalBytes / (1024 * 1024));
            }
        }

        _logger.LogInformation("Model downloaded: {Path}", destinationPath);
    }

    /// <summary>
    /// Load system messages from config
    /// </summary>
    private async Task<List<string>> LoadSystemMessagesAsync(ModelConfig config)
    {
        var messages = new List<string>(config.SystemMessages);

        // Load from external files
        foreach (var filePath in config.SystemMessageFiles)
        {
            var resolvedPath = Path.IsPathRooted(filePath) 
                ? filePath 
                : Path.Combine(_modelsDirectory, config.Name, filePath);

            if (File.Exists(resolvedPath))
            {
                var content = await File.ReadAllTextAsync(resolvedPath);
                messages.Add(content);
            }
            else
            {
                _logger.LogWarning("System message file not found: {Path}", resolvedPath);
            }
        }

        return messages;
    }

    /// <summary>
    /// List all available models
    /// </summary>
    public IEnumerable<string> ListModels()
    {
        return _modelConfigs.Keys;
    }

    /// <summary>
    /// Get model configuration
    /// </summary>
    public ModelConfig? GetModelConfig(string modelName)
    {
        _modelConfigs.TryGetValue(modelName, out var config);
        return config;
    }

    /// <summary>
    /// Unload a model from memory
    /// </summary>
    public bool UnloadModel(string modelName, bool force = false)
    {
        if (!_loadedModels.TryGetValue(modelName, out var instance)) return false;

        if (!force && instance.IsActive)
        {
            _logger.LogWarning("Skipping unload of {Model} — {N} active request(s) in progress", modelName, instance.ActiveRequests);
            return false;
        }

        if (_loadedModels.TryRemove(modelName, out instance))
        {
            instance.Dispose();
            _logger.LogInformation("Unloaded model: {Model}", modelName);
            return true;
        }
        return false;
    }

    /// <summary>
    /// Get all loaded models with statistics
    /// </summary>
    public Dictionary<string, ModelInstance> GetLoadedModels()
    {
        return new Dictionary<string, ModelInstance>(_loadedModels);
    }

    /// <summary>
    /// Reload a model (unload + load)
    /// </summary>
    public async Task ReloadModelAsync(string modelName)
    {
        UnloadModel(modelName);
        await GetModelAsync(modelName);
    }

    /// <summary>
    /// Update model statistics after request
    /// </summary>
    public void UpdateModelStats(string modelName, string prompt, double tps)
    {
        if (_loadedModels.TryGetValue(modelName, out var model))
            model.UpdateStats(prompt, tps);
    }

    /// <summary>
    /// Get all available model configurations
    /// </summary>
    public Dictionary<string, ModelConfig> GetAllConfigs()
    {
        return new Dictionary<string, ModelConfig>(_modelConfigs);
    }

    /// <summary>
    /// Register or update a model config at runtime (called by config watcher).
    /// If the model is currently loaded, it is unloaded so it reloads on next request.
    /// </summary>
    public void RegisterOrUpdateConfig(ModelConfig config)
    {
        bool isUpdate = _modelConfigs.ContainsKey(config.Name);
        _modelConfigs[config.Name] = config;

        if (isUpdate && _loadedModels.ContainsKey(config.Name))
        {
            _logger.LogInformation("Model config changed, unloading for hot-reload: {Model}", config.Name);
            UnloadModel(config.Name);
        }
        else
        {
            if (isUpdate)
                _logger.LogInformation("Model config updated (not loaded): {Model}", config.Name);
            else
                _logger.LogInformation("New model available: {Model}", config.Name);
        }
    }

    /// <summary>
    /// Remove a model config at runtime (called by config watcher on file deletion).
    /// Unloads the model if currently running.
    /// </summary>
    public void RemoveConfig(string modelName)
    {
        _modelConfigs.TryRemove(modelName, out _);
        if (_loadedModels.ContainsKey(modelName))
        {
            _logger.LogInformation("Model config removed, unloading: {Model}", modelName);
            UnloadModel(modelName);
        }
        else
        {
            _logger.LogInformation("Model config removed: {Model}", modelName);
        }
    }

    /// <summary>
    /// Converts per-device Layers fields into the int[] layerSplit expected by MultiGPUTransformer.
    /// Returns null if all devices have Layers = null (auto even split).
    /// The last device's Layers value is always ignored — it gets the remainder.
    /// </summary>
    private static int[]? BuildLayerSplit(MultiGpuDeviceConfig[] devices)
    {
        int n = devices.Length;
        // If no device specifies Layers, use auto-split
        if (devices.Take(n - 1).All(d => d.Layers == null)) return null;

        // Build cumulative boundaries: layerSplit[i] = first layer of device i+1
        var splits = new int[n - 1];
        int consumed = 0;
        for (int i = 0; i < n - 1; i++)
        {
            if (devices[i].Layers is int l)
                consumed += l;
            // If Layers is null for a non-last device: leave as-is (will be filled by MultiGPUTransformer auto logic)
            // but we still need a value — use 0 as sentinel meaning "auto from here"
            splits[i] = consumed;
        }
        return splits;
    }

    /// <summary>
    /// Create compute device by provider name
    /// </summary>
    private static IComputeDevice CreateComputeDevice(string provider, int deviceIndex)
    {
        return provider.ToLower() switch
        {
            "vulkan" => new AIHost.ICompute.Vulkan.VulkanComputeDevice(deviceIndex),
            "cuda" => new AIHost.ICompute.CUDA.CudaComputeDevice(deviceIndex),
            "rocm" => new AIHost.ICompute.ROCm.ROCmComputeDevice(deviceIndex),
            _ => throw new ArgumentException($"Unknown compute provider: {provider}")
        };
    }

    /// <summary>
    /// Create and attach a VulkanArenaAllocator to ComputeOps if the device is Vulkan.
    /// Arena size: auto-calculated from model params (context_size+max_tokens+dModel) or taken from config.
    /// Fail-fast: if arena doesn't fit in VRAM, throws InsufficientVramException at init time.
    /// </summary>
    private void TryAttachArena(IComputeDevice device, TransformerBase transformer, ModelConfig config)
    {
        if (device is not AIHost.ICompute.Vulkan.VulkanComputeDevice vkDev) return;

        int arenaMb = config.ArenaSizeMb ?? CalculateArenaSizeMb(transformer, config);
        _logger.LogInformation("[Arena] Creating VulkanArenaAllocator ({Size} MB)", arenaMb);

        try
        {
            var arena = new VulkanArenaAllocator(vkDev.DeviceContext, (ulong)arenaMb * 1024 * 1024);
            transformer.Ops.AttachArena(arena);
            _logger.LogInformation("[Arena] Attached {Size} MB arena to ComputeOps", arenaMb);
        }
        catch (Exception ex)
        {
            throw new AIHost.ICompute.InsufficientVramException(
                (ulong)arenaMb * 1024 * 1024,
                $"Arena allocator: {ex.Message}. Reduce context_size, max_tokens, or arena_size_mb.");
        }
    }

    /// <summary>
    /// Auto-calculate arena size from model parameters.
    /// Formula: per_frame = wZ_padded + tempPool + scratch + concatBuffer; kvCache = attentionLayers × 2 × (ctx+maxT) × kvHeads × headDim × 4.
    /// Rounded up to nearest 64 MB with 10% safety margin.
    /// </summary>
    private static int CalculateArenaSizeMb(TransformerBase t, ModelConfig config)
    {
        const int HVD = 128, NVH = 48, NKH = 16, KD = NKH * HVD, VD = NVH * HVD, CD = 2 * KD + VD;
        const int SUB_BATCH = 8;
        int dm = t._dModel;
        int ctx = config.Parameters.ContextSize;
        int maxT = config.Parameters.MaxTokens;
        int kvHeads = t._numKVHeads;
        int headDim = t._headDim;

        long perFrame = (long)dm * VD * 4L                     // wZ padded
                      + SUB_BATCH * (VD + dm) * 4L            // temp pool
                      + (CD + VD + NVH * 3) * 4L              // scratch
                      + (long)ctx * VD * 4L;                  // concat buffer

        // Approximate attention layers: assume half of all layers are Type B (attention)
        // This is a heuristic for Qwen hybrid models.
        int attnLayers = t._numLayers / 2;
        long kvCache = (long)attnLayers * 2L
                     * (ctx + maxT)
                     * kvHeads * headDim * 4L;

        long total = perFrame + kvCache;
        // Round to 64 MB with 10% safety margin (rounded is already in MB)
        long rounded = (long)((total * 1.1 + 64L * 1024 * 1024 - 1) / (64L * 1024 * 1024)) * 64;
        return (int)Math.Max(64, rounded); // minimum 64 MB
    }

    public void Dispose()
    {
        if (_disposed) return;

        foreach (var instance in _loadedModels.Values)
            instance.Dispose();

        _loadedModels.Clear();
        _loadLock.Dispose();
        _disposed = true;
        GC.SuppressFinalize(this);
    }
}

/// <summary>
/// Instance of a loaded model
/// </summary>
public class ModelInstance : IDisposable
{
    public string Name { get; set; } = "";
    public ModelConfig Config { get; set; } = null!;
    public IInferenceEngine Engine { get; set; } = null!;
    public IComputeDevice? Device { get; set; }
    public List<string> SystemMessages { get; set; } = new();
    public bool IsChatModel { get; set; }
    public DateTime LoadedAt { get; set; }

    // Statistics — all writes go through UpdateStats() which holds _statsLock.
    private readonly Lock _statsLock = new();
    public long TotalRequests { get; private set; }
    public double AverageTPS { get; private set; }
    public DateTime? LastRequestAt { get; private set; }
    public string? LastPrompt { get; private set; }

    // Active request tracking — prevents auto-unload while inference is running.
    private int _activeRequests;
    public int ActiveRequests => _activeRequests;
    public bool IsActive => _activeRequests > 0;

    /// <summary>
    /// Call at request start; returns an IDisposable that decrements on Dispose.
    /// Also bumps LastRequestAt so the auto-unload idle timer resets from
    /// when the request STARTS — not from when it finishes (which could be
    /// 9+ minutes later for long generations on slow hardware).
    /// Usage: using var _ = model.TrackRequest();
    /// </summary>
    public IDisposable TrackRequest()
    {
        Interlocked.Increment(ref _activeRequests);
        // Touch LastRequestAt at request START so the idle clock resets now.
        lock (_statsLock) { LastRequestAt = DateTime.UtcNow; }
        return new RequestScope(this);
    }

    private void ReleaseRequest()
    {
        Interlocked.Decrement(ref _activeRequests);
        // Touch LastRequestAt at request END too so the full idle window is
        // measured from when the client last received a response.
        lock (_statsLock) { LastRequestAt = DateTime.UtcNow; }
    }

    private sealed class RequestScope(ModelInstance owner) : IDisposable
    {
        public void Dispose() => owner.ReleaseRequest();
    }

    /// <summary>
    /// Thread-safe stats update called after each completed request.
    /// </summary>
    public void UpdateStats(string prompt, double tps)
    {
        lock (_statsLock)
        {
            TotalRequests++;
            LastRequestAt = DateTime.UtcNow;
            LastPrompt = prompt.Length > 100 ? prompt[..100] + "..." : prompt;
            AverageTPS = (AverageTPS * (TotalRequests - 1) + tps) / TotalRequests;
        }
    }

    public void Dispose()
    {
        Engine?.Dispose();
        Device?.Dispose();
        GC.SuppressFinalize(this);
    }
}
