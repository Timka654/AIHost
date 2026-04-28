using System.Collections.Concurrent;
using System.Text.Json;
using AIHost.Compute;
using AIHost.GGUF;
using AIHost.ICompute;
using AIHost.Inference;
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
    private readonly IComputeDevice _device;

    public string ModelsDirectory => _modelsDirectory;
    private readonly ConcurrentDictionary<string, ModelInstance> _loadedModels = new();
    private readonly ConcurrentDictionary<string, ModelConfig> _modelConfigs = new();
    // Serializes the slow model-load path so the same model is never loaded twice.
    private readonly SemaphoreSlim _loadLock = new(1, 1);
    private bool _disposed;

    public ModelManager(string modelsDirectory, IComputeDevice device)
    {
        _modelsDirectory = modelsDirectory;
        _device = device;
        
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
            Console.WriteLine($"Created models directory: {_modelsDirectory}");
            return;
        }

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
                    Console.WriteLine($"Loaded model config: {config.Name}");
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Failed to load config from {configPath}: {ex.Message}");
            }
        }

        Console.WriteLine($"Loaded {_modelConfigs.Count} model configuration(s)");
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

        // Download if needed
        if (!File.Exists(modelPath) && config.AutoDownload && IsUrl(config.ModelPath))
        {
            await DownloadModelAsync(config.ModelPath, modelPath);
        }

        if (!File.Exists(modelPath))
            throw new FileNotFoundException($"Model file not found: {modelPath}");

        // Determine which device to use
        IComputeDevice device = _device;
        IComputeDevice? perModelDevice = null;
        
        // Create per-model device if specific settings are provided
        if (config.ComputeProvider != null || config.DeviceIndex != null)
        {
            var provider = config.ComputeProvider ?? "vulkan";
            var deviceIndex = config.DeviceIndex ?? 0;
            
            Console.WriteLine($"Creating dedicated {provider} device (index {deviceIndex}) for model {modelName}");
            perModelDevice = CreateComputeDevice(provider, deviceIndex);
            device = perModelDevice;
        }

        // Load model
        Console.WriteLine($"Loading model: {modelName} from {modelPath}");
        Console.WriteLine($"  Provider: {config.ComputeProvider ?? "(global)"}");
        Console.WriteLine($"  Device Index: {config.DeviceIndex?.ToString() ?? "(global)"}");
        Console.WriteLine($"  Keep Alive: {config.KeepAliveMinutes?.ToString() ?? "(global)"} minutes");
        Console.WriteLine($"  GPU Layers: {config.NumGpuLayers?.ToString() ?? "all"}");
        Console.WriteLine($"  Batch Size: {config.BatchSize?.ToString() ?? "8 (default)"}");
        Console.WriteLine($"  Memory Mapping: {config.EnableMmap}");
        Console.WriteLine($"  Memory Lock: {config.EnableMlock}");
        
        // Warnings for features not yet fully integrated
        if (config.NumGpuLayers.HasValue && config.NumGpuLayers.Value >= 0)
        {
            Console.WriteLine($"  ⚠ Warning: num_gpu_layers is configured but requires Transformer changes for hybrid CPU/GPU");
        }
        if (config.EnableMlock && !config.EnableMmap)
        {
            Console.WriteLine($"  ⚠ Warning: enable_mlock requires enable_mmap to be true");
        }
        
        // LazyGGUFModel owns its GGUFReader — reuse it for the tokenizer
        // to avoid opening and parsing the same file twice.
        IGGUFModel ggufModel = new AIHost.GGUF.LazyGGUFModel(modelPath, device, config.EnableMmap, config.EnableMlock);
        var tokenizer = BPETokenizer.FromGGUF(ggufModel.Reader);
        // Transformer owns its ComputeOps; share it with InferenceEngine
        var transformer = new Transformer(device, ggufModel);

        // Use configured batch size or default to 8
        int batchSize = config.BatchSize ?? 8;
        var engine = new InferenceEngine(transformer, tokenizer, transformer.Ops, batchSize);

        // Load system messages
        var systemMessages = await LoadSystemMessagesAsync(config);

        var loaded = new ModelInstance
        {
            Name = modelName,
            Config = config,
            Engine = engine,
            Device = perModelDevice,
            SystemMessages = systemMessages,
            LoadedAt = DateTime.UtcNow
        };

        _loadedModels[modelName] = loaded;

        Console.WriteLine($"✓ Model '{modelName}' loaded successfully");

        return loaded;
    }

    /// <summary>
    /// Resolve model path (absolute or relative to model directory)
    /// </summary>
    private string ResolveModelPath(ModelConfig config)
    {
        var path = config.ModelPath;

        // If URL, use cache path
        if (IsUrl(path))
        {
            var fileName = Path.GetFileName(new Uri(path).LocalPath);
            return Path.Combine(_modelsDirectory, config.Name, fileName);
        }

        // If absolute path, use as-is
        if (Path.IsPathRooted(path))
            return path;

        // Relative to model directory
        return Path.Combine(_modelsDirectory, config.Name, path);
    }

    /// <summary>
    /// Check if string is a URL
    /// </summary>
    private bool IsUrl(string path)
    {
        return path.StartsWith("http://") || path.StartsWith("https://");
    }

    /// <summary>
    /// Download model from URL
    /// </summary>
    private async Task DownloadModelAsync(string url, string destinationPath)
    {
        Console.WriteLine($"Downloading model from {url}...");
        
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
                Console.WriteLine($"Download progress: {progress:F1}% ({downloadedBytes / (1024 * 1024)}MB / {totalBytes / (1024 * 1024)}MB)");
            }
        }

        Console.WriteLine($"✓ Model downloaded: {destinationPath}");
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
                Console.WriteLine($"Warning: System message file not found: {resolvedPath}");
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
    public void UnloadModel(string modelName)
    {
        if (_loadedModels.TryRemove(modelName, out var instance))
        {
            instance.Dispose();
            Console.WriteLine($"Unloaded model: {modelName}");
        }
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
            Console.WriteLine($"Model config changed, unloading for hot-reload: {config.Name}");
            UnloadModel(config.Name);
        }
        else
        {
            Console.WriteLine(isUpdate
                ? $"Model config updated (not loaded): {config.Name}"
                : $"New model available: {config.Name}");
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
            Console.WriteLine($"Model config removed, unloading: {modelName}");
            UnloadModel(modelName);
        }
        else
        {
            Console.WriteLine($"Model config removed: {modelName}");
        }
    }

    /// <summary>
    /// Create compute device by provider name
    /// </summary>
    private IComputeDevice CreateComputeDevice(string provider, int deviceIndex)
    {
        return provider.ToLower() switch
        {
            "vulkan" => new AIHost.ICompute.Vulkan.VulkanComputeDevice(deviceIndex),
            "cuda" => new AIHost.ICompute.CUDA.CudaComputeDevice(deviceIndex),
            "rocm" => new AIHost.ICompute.ROCm.ROCmComputeDevice(deviceIndex),
            _ => throw new ArgumentException($"Unknown compute provider: {provider}")
        };
    }

    public void Dispose()
    {
        if (_disposed) return;

        foreach (var instance in _loadedModels.Values)
            instance.Dispose();

        _loadedModels.Clear();
        _loadLock.Dispose();
        _disposed = true;
    }
}

/// <summary>
/// Instance of a loaded model
/// </summary>
public class ModelInstance : IDisposable
{
    public string Name { get; set; } = "";
    public ModelConfig Config { get; set; } = null!;
    public InferenceEngine Engine { get; set; } = null!;
    public IComputeDevice? Device { get; set; }
    public List<string> SystemMessages { get; set; } = new();
    public DateTime LoadedAt { get; set; }

    // Statistics — all writes go through UpdateStats() which holds _statsLock.
    private readonly Lock _statsLock = new();
    public long TotalRequests { get; private set; }
    public double AverageTPS { get; private set; }
    public DateTime? LastRequestAt { get; private set; }
    public string? LastPrompt { get; private set; }

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
    }
}
