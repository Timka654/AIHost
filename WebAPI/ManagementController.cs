using Microsoft.AspNetCore.Mvc;
using AIHost.Config;
using AIHost.Logging;
using AIHost.Services;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace AIHost.WebAPI;

/// <summary>
/// Model status response
/// </summary>
public class ModelStatusResponse
{
    [JsonPropertyName("name")]
    public string Name { get; set; } = "";

    [JsonPropertyName("loaded_at")]
    public DateTime LoadedAt { get; set; }

    [JsonPropertyName("total_requests")]
    public long TotalRequests { get; set; }

    [JsonPropertyName("average_tps")]
    public double AverageTPS { get; set; }

    [JsonPropertyName("last_request_at")]
    public DateTime? LastRequestAt { get; set; }

    [JsonPropertyName("last_prompt")]
    public string? LastPrompt { get; set; }

    [JsonPropertyName("config")]
    public ModelConfig? Config { get; set; }
}

/// <summary>
/// Download model request
/// </summary>
public class DownloadModelRequest
{
    [JsonPropertyName("name")]
    public string Name { get; set; } = "";

    [JsonPropertyName("url")]
    public string Url { get; set; } = "";

    [JsonPropertyName("format")]
    public string Format { get; set; } = "gguf";
}

/// <summary>
/// Chat request
/// </summary>
public class ChatRequest
{
    [JsonPropertyName("model_name")]
    public string ModelName { get; set; } = "";

    [JsonPropertyName("message")]
    public string Message { get; set; } = "";

    [JsonPropertyName("system_message")]
    public string? SystemMessage { get; set; }

    [JsonPropertyName("temperature")]
    public float? Temperature { get; set; }

    [JsonPropertyName("top_k")]
    public int? TopK { get; set; }

    [JsonPropertyName("top_p")]
    public float? TopP { get; set; }

    [JsonPropertyName("max_tokens")]
    public int? MaxTokens { get; set; }

    /// <summary>Stream tokens as they are generated (SSE / NDJSON format).</summary>
    [JsonPropertyName("stream")]
    public bool Stream { get; set; } = false;
}

/// <summary>
/// Chat message
/// </summary>
public class ChatMessage
{
    [JsonPropertyName("role")]
    public string Role { get; set; } = "";

    [JsonPropertyName("content")]
    public string Content { get; set; } = "";
}

/// <summary>
/// System resource info
/// </summary>
public class SystemResourceResponse
{
    [JsonPropertyName("total_memory_bytes")]
    public long TotalMemoryBytes { get; set; }

    [JsonPropertyName("available_memory_bytes")]
    public long AvailableMemoryBytes { get; set; }

    [JsonPropertyName("used_memory_bytes")]
    public long UsedMemoryBytes { get; set; }

    [JsonPropertyName("cpu_usage_percent")]
    public double CpuUsagePercent { get; set; }

    [JsonPropertyName("thread_count")]
    public int ThreadCount { get; set; }

    [JsonPropertyName("process_memory_bytes")]
    public long ProcessMemoryBytes { get; set; }

    [JsonPropertyName("gc_total_memory_bytes")]
    public long GcTotalMemoryBytes { get; set; }
}

/// <summary>
/// Buffer pool statistics
/// </summary>
public class BufferPoolStatsResponse
{
    [JsonPropertyName("pooled_count")]
    public int PooledCount { get; set; }

    [JsonPropertyName("active_count")]
    public int ActiveCount { get; set; }

    [JsonPropertyName("total_allocations")]
    public long TotalAllocations { get; set; }

    [JsonPropertyName("pool_hits")]
    public long PoolHits { get; set; }

    [JsonPropertyName("pool_misses")]
    public long PoolMisses { get; set; }

    [JsonPropertyName("hit_rate_percent")]
    public double HitRatePercent { get; set; }

    [JsonPropertyName("pooled_memory_bytes")]
    public long PooledMemoryBytes { get; set; }

    [JsonPropertyName("active_memory_bytes")]
    public long ActiveMemoryBytes { get; set; }
}

/// <summary>
/// Management API controller for admin operations
/// </summary>
[ApiController]
[Route("manage")]
public class ManagementController : ControllerBase
{
    private readonly ModelManager _modelManager;
    private readonly RequestLogger _requestLogger;
    private readonly DownloadManager _downloadManager;
    private readonly InMemoryLoggerProvider _inMemoryLoggerProvider;
    private readonly ILogger<ManagementController> _logger;

    public ManagementController(ModelManager modelManager, RequestLogger requestLogger,
                                 DownloadManager downloadManager,
                                 InMemoryLoggerProvider inMemoryLoggerProvider,
                                 ILogger<ManagementController> logger)
    {
        _modelManager = modelManager;
        _requestLogger = requestLogger;
        _downloadManager = downloadManager;
        _inMemoryLoggerProvider = inMemoryLoggerProvider;
        _logger = logger;
    }

    /// <summary>
    /// Get build version
    /// </summary>
    [HttpGet("build_version")]
    public IActionResult GetBuildVersion()
    {
        return Ok(BuildInfo.Date);
    }

    /// <summary>
    /// Get all loaded models with statistics
    /// </summary>
    [HttpGet("models")]
    public IActionResult GetLoadedModels()
    {
        var models = _modelManager.GetLoadedModels();
        var response = models.Select(kvp => new ModelStatusResponse
        {
            Name = kvp.Key,
            LoadedAt = kvp.Value.LoadedAt,
            TotalRequests = kvp.Value.TotalRequests,
            AverageTPS = kvp.Value.AverageTPS,
            LastRequestAt = kvp.Value.LastRequestAt,
            LastPrompt = kvp.Value.LastPrompt,
            Config = kvp.Value.Config
        }).ToList();

        return Ok(response);
    }

    /// <summary>
    /// Get specific model status
    /// </summary>
    [HttpGet("models/{name}")]
    public IActionResult GetModelStatus(string name)
    {
        var models = _modelManager.GetLoadedModels();
        if (!models.TryGetValue(name, out var model))
        {
            return NotFound(new { error = $"Model '{name}' not loaded" });
        }

        var response = new ModelStatusResponse
        {
            Name = name,
            LoadedAt = model.LoadedAt,
            TotalRequests = model.TotalRequests,
            AverageTPS = model.AverageTPS,
            LastRequestAt = model.LastRequestAt,
            LastPrompt = model.LastPrompt,
            Config = model.Config
        };

        return Ok(response);
    }

    /// <summary>
    /// Unload a model from memory
    /// </summary>
    [HttpDelete("models/{name}")]
    public IActionResult UnloadModel(string name)
    {
        _modelManager.UnloadModel(name);
        return Ok(new { message = $"Model '{name}' unloaded" });
    }

    /// <summary>
    /// Reload a model
    /// </summary>
    [HttpPost("models/{name}/reload")]
    public async Task<IActionResult> ReloadModel(string name)
    {
        try
        {
            await _modelManager.ReloadModelAsync(name);
            return Ok(new { message = $"Model '{name}' reloaded successfully" });
        }
        catch (Exception ex)
        {
            return StatusCode(500, new { error = $"Failed to reload model: {ex.Message}" });
        }
    }

    /// <summary>
    /// Get recent request logs
    /// </summary>
    [HttpGet("logs")]
    public IActionResult GetLogs([FromQuery] int count = 100)
    {
        var logs = _requestLogger.GetRecentLogs(count);
        return Ok(logs);
    }

    /// <summary>
    /// Get logs for specific model
    /// </summary>
    [HttpGet("logs/{modelName}")]
    public IActionResult GetModelLogs(string modelName, [FromQuery] int count = 100)
    {
        var logs = _requestLogger.GetModelLogs(modelName, count);
        return Ok(logs);
    }

    /// <summary>
    /// Clear all logs
    /// </summary>
    [HttpDelete("logs")]
    public IActionResult ClearLogs()
    {
        _requestLogger.Clear();
        return Ok(new { message = "Logs cleared" });
    }

    /// <summary>
    /// Get in-memory debug logs (from ILogger infrastructure).
    /// Optional ?category= substring filter (e.g. "Transformer" or "Inference").
    /// </summary>
    [HttpGet("debug-logs")]
    public IActionResult GetInMemoryLogs([FromQuery] string? category = null)
    {
        try
        {
            LogEntry[] entries;
            if (!string.IsNullOrEmpty(category))
                entries = _inMemoryLoggerProvider.GetEntries(category);
            else
                entries = _inMemoryLoggerProvider.GetAllEntries();

            return Ok(new
            {
                total = entries.Length,
                entries = entries.Select(e => new
                {
                    timestamp = e.Timestamp,
                    level = e.Level.ToString(),
                    category = e.Category,
                    message = e.Message,
                    exception = e.Exception
                })
            });
        }
        catch (Exception ex)
        {
            return StatusCode(500, new { error = $"Failed to get debug logs: {ex.Message}" });
        }
    }

    /// <summary>
    /// Clear in-memory debug logs.
    /// </summary>
    [HttpDelete("debug-logs")]
    public IActionResult ClearInMemoryLogs()
    {
        try
        {
            _inMemoryLoggerProvider.ClearAll();
            return Ok(new { message = "Debug logs cleared" });
        }
        catch (Exception ex)
        {
            return StatusCode(500, new { error = $"Failed to clear debug logs: {ex.Message}" });
        }
    }

    /// <summary>
    /// List all available model configs
    /// </summary>
    [HttpGet("configs")]
    public IActionResult GetAllConfigs()
    {
        var configs = _modelManager.GetAllConfigs();
        return Ok(configs);
    }

    /// <summary>
    /// Create or update a model config
    /// </summary>
    [HttpPost("configs")]
    public async Task<IActionResult> CreateOrUpdateConfig([FromBody] ModelConfig config)
    {
        if (string.IsNullOrEmpty(config.Name))
            return BadRequest(new { error = "Model name is required" });

        try
        {
            var modelDir = Path.Combine(_modelManager.ModelsDirectory, config.Name);
            Directory.CreateDirectory(modelDir);

            var configPath = Path.Combine(modelDir, "model.json");
            var json = JsonSerializer.Serialize(config, new JsonSerializerOptions { WriteIndented = true });
            await System.IO.File.WriteAllTextAsync(configPath, json);

            _modelManager.RegisterOrUpdateConfig(config);

            return Ok(new { message = $"Model config '{config.Name}' saved successfully" });
        }
        catch (Exception ex)
        {
            return StatusCode(500, new { error = $"Failed to save config: {ex.Message}" });
        }
    }

    /// <summary>
    /// Delete a model config
    /// </summary>
    [HttpDelete("configs/{name}")]
    public IActionResult DeleteConfig(string name)
    {
        try
        {
            _modelManager.RemoveConfig(name);
            return Ok(new { message = $"Model config '{name}' deleted" });
        }
        catch (Exception ex)
        {
            return StatusCode(500, new { error = $"Failed to delete config: {ex.Message}" });
        }
    }

    /// <summary>
    /// Clear model cache directory
    /// </summary>
    [HttpDelete("cache")]
    public IActionResult ClearCache()
    {
        try
        {
            if (!Directory.Exists(_modelManager.CacheDirectory))
                return Ok(new { message = "Cache directory does not exist" });

            var cacheDir = _modelManager.CacheDirectory;
            var files = Directory.GetFiles(cacheDir, "*", SearchOption.AllDirectories);
            var dirs = Directory.GetDirectories(cacheDir, "*", SearchOption.AllDirectories);

            foreach (var file in files)
                System.IO.File.Delete(file);
            foreach (var dir in dirs)
                System.IO.Directory.Delete(dir, recursive: true);

            return Ok(new { message = $"Cache cleared ({files.Length} files, {dirs.Length} directories)" });
        }
        catch (Exception ex)
        {
            return StatusCode(500, new { error = $"Failed to clear cache: {ex.Message}" });
        }
    }

    /// <summary>
    /// Get cache directory info
    /// </summary>
    [HttpGet("cache")]
    public IActionResult GetCacheInfo()
    {
        try
        {
            if (!Directory.Exists(_modelManager.CacheDirectory))
                return Ok(new { path = _modelManager.CacheDirectory, file_count = 0, total_size = 0, dir_count = 0 });

            var files = Directory.GetFiles(_modelManager.CacheDirectory, "*", SearchOption.AllDirectories);
            var dirs = Directory.GetDirectories(_modelManager.CacheDirectory, "*", SearchOption.AllDirectories);
            var totalSize = files.Sum(f => new FileInfo(f).Length);

            return Ok(new
            {
                path = _modelManager.CacheDirectory,
                file_count = files.Length,
                total_size = totalSize,
                dir_count = dirs.Length
            });
        }
        catch (Exception ex)
        {
            return StatusCode(500, new { error = $"Failed to get cache info: {ex.Message}" });
        }
    }

    /// <summary>
    /// Direct chat endpoint (auto-loads model if not loaded)
    /// </summary>
    [HttpPost("chat")]
    public async Task<IActionResult> DirectChat([FromBody] ChatRequest request)
    {
        if (string.IsNullOrEmpty(request.ModelName))
            return BadRequest(new { error = "Model name is required" });

        if (string.IsNullOrEmpty(request.Message))
            return BadRequest(new { error = "Message is required" });

        try
        {
            // Try to reload config if not found in memory
            _modelManager.ReloadConfig(request.ModelName);
            
            // Auto-load model if not already loaded
            var model = await _modelManager.GetModelAsync(request.ModelName);

            var modelConfig = _modelManager.GetModelConfig(request.ModelName);
            var config = new AIHost.Inference.GenerationConfig
            {
                MaxNewTokens        = request.MaxTokens ?? modelConfig?.Parameters.MaxTokens ?? 512,
                Temperature         = request.Temperature ?? modelConfig?.Parameters.Temperature ?? 0.7f,
                TopK                = request.TopK ?? modelConfig?.Parameters.TopK ?? 40,
                TopP                = request.TopP ?? modelConfig?.Parameters.TopP ?? 0.9f,
                RepetitionPenalty   = modelConfig?.Parameters.RepetitionPenalty ?? 1.1f,
                FrequencyPenalty    = modelConfig?.Parameters.FrequencyPenalty ?? 0.0f,
                PresencePenalty     = modelConfig?.Parameters.PresencePenalty ?? 0.0f,
                Seed                = modelConfig?.Parameters.Seed ?? -1,
                UseKVCache          = modelConfig?.Parameters.UseKVCache ?? true,
                StopSequences       = modelConfig?.Parameters.Stop.ToList() ?? []
            };

            // Apply chat template so instruction-tuned models receive properly formatted input.
            var formattedPrompt = BuildDirectChatPrompt(model, request);

            if (request.Stream)
            {
                using var _ = model.TrackRequest();
                return await DirectChatStream(request, model, config, formattedPrompt);
            }

            using var __ = model.TrackRequest();
            var ct = HttpContext.RequestAborted;
            var sw = System.Diagnostics.Stopwatch.StartNew();
            var response = model.Engine.Generate(formattedPrompt, config, ct);
            sw.Stop();

            // Count actual tokens by encoding the response
            var responseTokens = model.Engine.Tokenizer.Encode(response, addBos: false, addEos: false);
            var tokenCount = responseTokens.Length;
            var tps = sw.Elapsed.TotalSeconds > 0 ? tokenCount / sw.Elapsed.TotalSeconds : 0;
            _modelManager.UpdateModelStats(request.ModelName, request.Message, tps);

            _requestLogger.LogRequest(new AIHost.Logging.RequestLogEntry
            {
                Timestamp = DateTime.UtcNow,
                Endpoint = "/manage/chat",
                Method = "POST",
                ModelName = request.ModelName,
                Prompt = request.Message.Length > 200 ? request.Message[..200] : request.Message,
                TokensGenerated = tokenCount,
                DurationMs = sw.Elapsed.TotalMilliseconds,
                TPS = tps,
                Success = true
            });

            return Ok(new
            {
                model = request.ModelName,
                response = response,
                tokens = tokenCount,
                finish_reason = "stop"
            });
        }
        catch (ArgumentException ex) when (ex.Message.Contains("not found in configuration") ||
                                            ex.Message.Contains("not found in GGUF"))
        {
            _logger.LogError(ex, "Model/tensor not found for DirectChat: {Model}", request.ModelName);
            return NotFound(new { error = $"Model or tensor not found: {ex.Message}", stack = ex.ToString() });
        }
        catch (ArgumentException ex)
        {
            // Shape mismatches, invalid arguments, etc. — these are inference bugs, not 404
            _logger.LogError(ex, "Inference argument error for model {Model}: {Msg}", request.ModelName, ex.Message);
            return StatusCode(500, new { error = ex.Message, type = ex.GetType().FullName, stack = ex.ToString() });
        }
        catch (FileNotFoundException ex)
        {
            _logger.LogError(ex, "File/library not found during DirectChat for model {Model}", request.ModelName);
            return NotFound(new { error = "Model file or required library not found.", detail = ex.Message, stack = ex.ToString() });
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "DirectChat failed for model {Model}: {Type}: {Message}", request.ModelName, ex.GetType().FullName, ex.Message);
            return StatusCode(500, new { error = ex.Message, type = ex.GetType().FullName, detail = ex.InnerException?.Message, stack = ex.ToString() });
        }
    }

    /// <summary>Applies the model's chat template to the DirectChat message.</summary>
    private static string BuildDirectChatPrompt(AIHost.Config.ModelInstance model, ChatRequest request)
    {
        var tokenizer = model.Engine.Tokenizer;
        bool isQwen   = tokenizer.GetTokenId("<|im_start|>") >= 0;
        var sb = new System.Text.StringBuilder();

        var systemText = request.SystemMessage
                      ?? (model.SystemMessages.Count > 0 ? string.Join("\n", model.SystemMessages) : null);

        if (isQwen)
        {
            if (!string.IsNullOrEmpty(systemText))
                sb.Append($"<|im_start|>system\n{systemText}<|im_end|>\n");
            sb.Append($"<|im_start|>user\n{request.Message}<|im_end|>\n");
            sb.Append("<|im_start|>assistant\n");
        }
        else
        {
            if (!string.IsNullOrEmpty(systemText))
                sb.Append($"<|system|>\n{systemText}\n</s>\n");
            sb.Append($"<|user|>\n{request.Message}\n</s>\n");
            sb.Append("<|assistant|>\n");
        }
        return sb.ToString();
    }

    /// <summary>
    /// Streaming variant of DirectChat — sends NDJSON chunks as each token arrives.
    /// Each chunk: {"token":"...", "done":false}
    /// Final chunk: {"tokens":N, "tps":X, "done":true}  (no "response" field to avoid duplication)
    /// </summary>
    private async Task<IActionResult> DirectChatStream(ChatRequest request,
        AIHost.Config.ModelInstance model, AIHost.Inference.GenerationConfig config,
        string? formattedPrompt = null)
    {
        Response.ContentType = "application/x-ndjson";
        Response.Headers["Cache-Control"] = "no-cache";
        Response.Headers["X-Accel-Buffering"] = "no"; // disable nginx buffering

        var prompt = formattedPrompt ?? request.Message;
        var sw = System.Diagnostics.Stopwatch.StartNew();
        int tokenCount = 0;

        var ct = HttpContext.RequestAborted;
        try
        {
            model.Engine.GenerateStreaming(prompt, config, async token =>
            {
                if (ct.IsCancellationRequested) return;
                tokenCount++;
                var chunk = System.Text.Json.JsonSerializer.Serialize(new { token, done = false });
                await Response.WriteAsync(chunk + "\n", ct);
                await Response.Body.FlushAsync(ct);
            }, ct);
        }
        catch (OperationCanceledException)
        {
            _logger.LogDebug("Streaming DirectChat cancelled for model {Model}", request.ModelName);
            return new EmptyResult();
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Streaming DirectChat failed for model {Model}", request.ModelName);
            var errChunk = System.Text.Json.JsonSerializer.Serialize(new { error = ex.Message, done = true });
            await Response.WriteAsync(errChunk + "\n");
            return new EmptyResult();
        }

        sw.Stop();
        var tps = sw.Elapsed.TotalSeconds > 0 ? tokenCount / sw.Elapsed.TotalSeconds : 0;
        _modelManager.UpdateModelStats(request.ModelName, request.Message, tps);

        var final = System.Text.Json.JsonSerializer.Serialize(new
        {
            tokens = tokenCount,
            tps = Math.Round(tps, 2),
            done = true
        });
        await Response.WriteAsync(final + "\n");
        return new EmptyResult();
    }

    /// <summary>
    /// Download model from URL
    /// </summary>
    [HttpPost("download")]
    public async Task<IActionResult> DownloadModel([FromBody] DownloadModelRequest request)
    {
        if (string.IsNullOrEmpty(request.Name) || string.IsNullOrEmpty(request.Url))
        {
            return BadRequest(new { error = "Name and URL are required" });
        }

        try
        {
            // Download to cache directory
            var fileName = Path.GetFileName(new Uri(request.Url).LocalPath);
            var cachePath = Path.Combine(_modelManager.CacheDirectory, fileName);
            
            // Create cache directory if needed
            Directory.CreateDirectory(_modelManager.CacheDirectory);

            // Download
            Console.WriteLine($"Downloading {request.Name} from {request.Url} to cache...");
            using var client = new HttpClient();
            client.Timeout = TimeSpan.FromHours(2);

            using var response = await client.GetAsync(request.Url, HttpCompletionOption.ResponseHeadersRead);
            response.EnsureSuccessStatusCode();

            using var fileStream = System.IO.File.Create(cachePath);
            await response.Content.CopyToAsync(fileStream);

            Console.WriteLine($"Downloaded {request.Name} to {cachePath}");

            // Create model config in models directory
            var modelDir = Path.Combine(_modelManager.ModelsDirectory, request.Name);
            Directory.CreateDirectory(modelDir);
            
            var config = new ModelConfig
            {
                Name = request.Name,
                ModelPath = fileName, // Relative path - will be found in cache
                Format = request.Format,
                Description = $"Downloaded from {request.Url}"
            };

            var configPath = Path.Combine(modelDir, "model.json");
            var json = JsonSerializer.Serialize(config, new JsonSerializerOptions { WriteIndented = true });
            await System.IO.File.WriteAllTextAsync(configPath, json);

            return Ok(new { 
                message = $"Model '{request.Name}' downloaded successfully to cache",
                cache_path = cachePath,
                config_path = configPath
            });
        }
        catch (Exception ex)
        {
            return StatusCode(500, new { error = $"Download failed: {ex.Message}" });
        }
    }

    /// <summary>
    /// Get system resource information
    /// </summary>
    [HttpGet("system/resources")]
    public IActionResult GetSystemResources()
    {
        try
        {
            var process = System.Diagnostics.Process.GetCurrentProcess();
            var gcInfo = GC.GetGCMemoryInfo();

            // Get available physical memory (platform-specific)
            long availableMemory = 0;
            long totalMemory = 0;
            
            if (OperatingSystem.IsWindows())
            {
                // Windows: Use WMI if available
                try
                {
#pragma warning disable CA1416 // Platform-specific
                    var searcher = new System.Management.ManagementObjectSearcher("SELECT * FROM Win32_OperatingSystem");
                    foreach (System.Management.ManagementObject obj in searcher.Get())
                    {
                        availableMemory = Convert.ToInt64(obj["FreePhysicalMemory"]) * 1024;
                        totalMemory = Convert.ToInt64(obj["TotalVisibleMemorySize"]) * 1024;
                    }
#pragma warning restore CA1416
                }
                catch
                {
                    // Fallback if WMI is not available
                    totalMemory = gcInfo.TotalAvailableMemoryBytes;
                    availableMemory = totalMemory - gcInfo.MemoryLoadBytes;
                }
            }
            else if (OperatingSystem.IsLinux())
            {
                // Linux: Parse /proc/meminfo
                try
                {
                    var memInfo = System.IO.File.ReadAllLines("/proc/meminfo");
                    foreach (var line in memInfo)
                    {
                        if (line.StartsWith("MemTotal:"))
                        {
                            var parts = line.Split(new[] { ' ' }, StringSplitOptions.RemoveEmptyEntries);
                            totalMemory = long.Parse(parts[1]) * 1024; // Convert from KB to bytes
                        }
                        else if (line.StartsWith("MemAvailable:"))
                        {
                            var parts = line.Split(new[] { ' ' }, StringSplitOptions.RemoveEmptyEntries);
                            availableMemory = long.Parse(parts[1]) * 1024; // Convert from KB to bytes
                        }
                    }
                }
                catch
                {
                    // Fallback if /proc/meminfo is not readable
                    totalMemory = gcInfo.TotalAvailableMemoryBytes;
                    availableMemory = totalMemory - gcInfo.MemoryLoadBytes;
                }
            }
            else
            {
                // Generic fallback for other platforms (macOS, etc.)
                totalMemory = gcInfo.TotalAvailableMemoryBytes;
                availableMemory = totalMemory - gcInfo.MemoryLoadBytes;
            }

            var response = new SystemResourceResponse
            {
                TotalMemoryBytes = totalMemory,
                AvailableMemoryBytes = availableMemory,
                UsedMemoryBytes = totalMemory - availableMemory,
                CpuUsagePercent = process.TotalProcessorTime.TotalMilliseconds / Environment.ProcessorCount / Environment.TickCount * 100,
                ThreadCount = process.Threads.Count,
                ProcessMemoryBytes = process.WorkingSet64,
                GcTotalMemoryBytes = GC.GetTotalMemory(false)
            };

            return Ok(response);
        }
        catch (Exception ex)
        {
            return StatusCode(500, new { error = $"Failed to get system resources: {ex.Message}" });
        }
    }

    /// <summary>
    /// Get compute device information
    /// </summary>
    [HttpGet("system/compute")]
    public IActionResult GetComputeInfo()
    {
        try
        {
            var models = _modelManager.GetLoadedModels();
            var computeDevices = models
                .Where(m => m.Value.Device != null)
                .Select(m => new
                {
                    model = m.Key,
                    device_name = m.Value.Device?.ProviderName ?? "Unknown",
                    device_type = m.Value.Device?.GetType().Name ?? "Unknown"
                })
                .ToList();

            return Ok(new
            {
                devices = computeDevices,
                total_models = models.Count,
                has_gpu = computeDevices.Any()
            });
        }
        catch (Exception ex)
        {
            return StatusCode(500, new { error = $"Failed to get compute info: {ex.Message}" });
        }
    }

    /// <summary>
    /// Get VRAM information for all loaded models
    /// </summary>
    [HttpGet("system/vram")]
    public IActionResult GetVramInfo()
    {
        try
        {
            var models = _modelManager.GetLoadedModels();
            var devices = new List<object>();

            foreach (var kvp in models)
            {
                var model = kvp.Value;
                if (model.Device == null) continue;

                var memInfo = model.Device.GetMemoryInfo();
                devices.Add(new
                {
                    model = kvp.Key,
                    device_name = model.Device.ProviderName,
                    total_bytes = memInfo.TotalBytes,
                    available_bytes = memInfo.AvailableBytes,
                    used_bytes = memInfo.UsedBytes,
                    tracked_allocated_bytes = memInfo.TrackedAllocatedBytes,
                    supports_native_query = memInfo.SupportsNativeQuery
                });
            }

            // Глобальная статистика трекера
            var trackerStats = new
            {
                total_allocated_bytes = ComputeBufferBase.TotalAllocatedBytes,
                peak_allocated_bytes = ComputeBufferBase.PeakAllocatedBytes,
                active_buffer_count = ComputeBufferBase.ActiveBufferCount,
                total_buffer_count = ComputeBufferBase.TotalBufferCount,
                allocation_count = ComputeBufferBase.AllocationCount,
                free_count = ComputeBufferBase.FreeCount
            };

            return Ok(new
            {
                devices,
                tracker = trackerStats
            });
        }
        catch (Exception ex)
        {
            return StatusCode(500, new { error = $"Failed to get VRAM info: {ex.Message}" });
        }
    }

    /// <summary>
    /// Get buffer pool statistics (if available)
    /// </summary>
    [HttpGet("system/buffers")]
    public IActionResult GetBufferPoolStats()
    {
        try
        {
            // This would need access to the buffer pool instance
            // For now, return a placeholder
            return Ok(new
            {
                message = "Buffer pool stats not yet implemented",
                available = false
            });
        }
        catch (Exception ex)
        {
            return StatusCode(500, new { error = $"Failed to get buffer stats: {ex.Message}" });
        }
    }

    /// <summary>
    /// Get performance metrics for all models
    /// </summary>
    [HttpGet("performance")]
    public IActionResult GetPerformanceMetrics()
    {
        try
        {
            var models = _modelManager.GetLoadedModels();
            var metrics = models.Select(kvp => new
            {
                model = kvp.Key,
                total_requests = kvp.Value.TotalRequests,
                average_tps = kvp.Value.AverageTPS,
                uptime_seconds = (DateTime.UtcNow - kvp.Value.LoadedAt).TotalSeconds,
                last_request_seconds_ago = kvp.Value.LastRequestAt.HasValue
                    ? (DateTime.UtcNow - kvp.Value.LastRequestAt.Value).TotalSeconds
                    : (double?)null
            }).ToList();

            var totalRequests = metrics.Sum(m => m.total_requests);
            var averageTPS = metrics.Any() ? metrics.Average(m => m.average_tps) : 0;

            return Ok(new
            {
                models = metrics,
                summary = new
                {
                    total_models = models.Count,
                    total_requests = totalRequests,
                    average_tps = averageTPS
                }
            });
        }
        catch (Exception ex)
        {
            return StatusCode(500, new { error = $"Failed to get performance metrics: {ex.Message}" });
        }
    }

    /// <summary>
    /// Get current server status
    /// </summary>
    [HttpGet("status")]
    public IActionResult GetServerStatus()
    {
        try
        {
            var process = System.Diagnostics.Process.GetCurrentProcess();
            var models = _modelManager.GetLoadedModels();

            return Ok(new
            {
                status = "running",
                uptime_seconds = (DateTime.UtcNow - process.StartTime.ToUniversalTime()).TotalSeconds,
                loaded_models = models.Count,
                total_requests = models.Sum(m => m.Value.TotalRequests),
                memory_mb = process.WorkingSet64 / (1024.0 * 1024.0),
                thread_count = process.Threads.Count,
                version = "1.0.0"
            });
        }
        catch (Exception ex)
        {
            return StatusCode(500, new { error = $"Failed to get server status: {ex.Message}" });
        }
    }

    /// <summary>
    /// Get model configuration schema
    /// </summary>
    [HttpGet("configs/schema")]
    public IActionResult GetConfigSchema()
    {
        try
        {
            var schemaPath = Path.Combine("data", "model.schema.json");
            if (!System.IO.File.Exists(schemaPath))
                return NotFound(new { error = "Schema file not found" });

            var schema = System.IO.File.ReadAllText(schemaPath);
            return Content(schema, "application/json");
        }
        catch (Exception ex)
        {
            return StatusCode(500, new { error = $"Failed to get schema: {ex.Message}" });
        }
    }

    /// <summary>
    /// Get model configuration example
    /// </summary>
    [HttpGet("configs/example")]
    public IActionResult GetConfigExample()
    {
        try
        {
            var examplePath = Path.Combine("data", "model.example.json");
            if (!System.IO.File.Exists(examplePath))
                return NotFound(new { error = "Example file not found" });

            var example = System.IO.File.ReadAllText(examplePath);
            return Content(example, "application/json");
        }
        catch (Exception ex)
        {
            return StatusCode(500, new { error = $"Failed to get example: {ex.Message}" });
        }
    }

    /// <summary>
    /// Get specific model config
    /// </summary>
    [HttpGet("configs/{name}")]
    public IActionResult GetConfig(string name)
    {
        try
        {
            var config = _modelManager.GetModelConfig(name);
            if (config == null)
                return NotFound(new { error = $"Config '{name}' not found" });

            return Ok(config);
        }
        catch (Exception ex)
        {
            return StatusCode(500, new { error = $"Failed to get config: {ex.Message}" });
        }
    }

    /// <summary>
    /// Update existing model config
    /// </summary>
    [HttpPut("configs/{name}")]
    public async Task<IActionResult> UpdateConfig(string name, [FromBody] ModelConfig config)
    {
        if (string.IsNullOrEmpty(config.Name) || config.Name != name)
            return BadRequest(new { error = "Config name mismatch" });

        try
        {
            var modelDir = Path.Combine(_modelManager.ModelsDirectory, config.Name);
            Directory.CreateDirectory(modelDir);

            var configPath = Path.Combine(modelDir, "model.json");
            var json = JsonSerializer.Serialize(config, new JsonSerializerOptions { WriteIndented = true });
            await System.IO.File.WriteAllTextAsync(configPath, json);

            _modelManager.RegisterOrUpdateConfig(config);

            return Ok(new { message = $"Model config '{config.Name}' updated successfully" });
        }
        catch (Exception ex)
        {
            return StatusCode(500, new { error = $"Failed to update config: {ex.Message}" });
        }
    }

    /// <summary>
    /// Start downloading a model
    /// </summary>
    [HttpPost("downloads")]
    public IActionResult StartDownload([FromBody] StartDownloadRequest request)
    {
        try
        {
            if (string.IsNullOrWhiteSpace(request.Url))
                return BadRequest(new { error = "URL is required" });

            // Resolve filename the same way DownloadManager would
            var filename = string.IsNullOrWhiteSpace(request.Filename)
                ? Path.GetFileName(new Uri(request.Url).LocalPath)
                : request.Filename;

            if (!request.Force)
            {
                var existingPath = Path.Combine(_downloadManager.CacheDirectory, filename);
                if (System.IO.File.Exists(existingPath))
                    return Conflict(new
                    {
                        error = "file_exists",
                        message = $"File '{filename}' already exists in cache.",
                        filename,
                        path = existingPath
                    });
            }

            var downloadId = _downloadManager.StartDownload(request.Url, filename);
            return Ok(new { download_id = downloadId, message = "Download started" });
        }
        catch (Exception ex)
        {
            return StatusCode(500, new { error = $"Failed to start download: {ex.Message}" });
        }
    }

    /// <summary>
    /// Get all downloads
    /// </summary>
    [HttpGet("downloads")]
    public IActionResult GetDownloads()
    {
        try
        {
            var downloads = _downloadManager.GetAllDownloads();
            var response = downloads.Select(d => new
            {
                id