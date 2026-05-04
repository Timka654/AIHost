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

    public ManagementController(ModelManager modelManager, RequestLogger requestLogger, DownloadManager downloadManager)
    {
        _modelManager = modelManager;
        _requestLogger = requestLogger;
        _downloadManager = downloadManager;
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

            var config = new AIHost.Inference.GenerationConfig
            {
                MaxNewTokens = request.MaxTokens ?? 512,
                Temperature = request.Temperature ?? 0.7f,
                TopK = request.TopK ?? 40,
                TopP = request.TopP ?? 0.9f,
                RepetitionPenalty = 1.1f,
                Seed = -1,
                UseKVCache = true
            };

            var sw = System.Diagnostics.Stopwatch.StartNew();
            var response = model.Engine.Generate(request.Message, config);
            sw.Stop();

            var tokenCount = response.Length;
            var tps = sw.Elapsed.TotalSeconds > 0 ? tokenCount / sw.Elapsed.TotalSeconds : 0;
            _modelManager.UpdateModelStats(request.ModelName, request.Message, tps);

            return Ok(new
            {
                model = request.ModelName,
                response = response,
                tokens = tokenCount,
                finish_reason = "stop"
            });
        }
        catch (ArgumentException ex)
        {
            return NotFound(new { error = $"Model config '{request.ModelName}' not found. Please check the config exists in the Configs section. Details: {ex.Message}" });
        }
        catch (FileNotFoundException ex)
        {
            return NotFound(new { error = $"Model file not found. Please initialize/download the model first. Details: {ex.Message}" });
        }
        catch (Exception ex)
        {
            return StatusCode(500, new { error = $"Chat failed: {ex.Message}" });
        }
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
                id = d.Id,
                url = d.Url,
                filename = d.Filename,
                status = d.Status.ToString().ToLower(),
                total_bytes = d.TotalBytes,
                downloaded_bytes = d.DownloadedBytes,
                progress = Math.Round(d.Progress, 2),
                error = d.Error,
                start_time = d.StartTime,
                end_time = d.EndTime
            });

            return Ok(response);
        }
        catch (Exception ex)
        {
            return StatusCode(500, new { error = $"Failed to get downloads: {ex.Message}" });
        }
    }

    /// <summary>
    /// Get specific download status
    /// </summary>
    [HttpGet("downloads/{downloadId}")]
    public IActionResult GetDownload(string downloadId)
    {
        try
        {
            var download = _downloadManager.GetDownload(downloadId);
            if (download == null)
                return NotFound(new { error = "Download not found" });

            return Ok(new
            {
                id = download.Id,
                url = download.Url,
                filename = download.Filename,
                status = download.Status.ToString().ToLower(),
                total_bytes = download.TotalBytes,
                downloaded_bytes = download.DownloadedBytes,
                progress = Math.Round(download.Progress, 2),
                error = download.Error,
                start_time = download.StartTime,
                end_time = download.EndTime
            });
        }
        catch (Exception ex)
        {
            return StatusCode(500, new { error = $"Failed to get download: {ex.Message}" });
        }
    }

    /// <summary>
    /// Cancel a download
    /// </summary>
    [HttpDelete("downloads/{downloadId}")]
    public IActionResult CancelDownload(string downloadId)
    {
        try
        {
            var success = _downloadManager.CancelDownload(downloadId);
            if (!success)
                return NotFound(new { error = "Download not found" });

            return Ok(new { message = "Download cancelled" });
        }
        catch (Exception ex)
        {
            return StatusCode(500, new { error = $"Failed to cancel download: {ex.Message}" });
        }
    }

    /// <summary>
    /// Clean up completed/failed downloads
    /// </summary>
    [HttpPost("downloads/cleanup")]
    public IActionResult CleanupDownloads()
    {
        try
        {
            _downloadManager.CleanupCompleted();
            return Ok(new { message = "Cleanup completed" });
        }
        catch (Exception ex)
        {
            return StatusCode(500, new { error = $"Failed to cleanup downloads: {ex.Message}" });
        }
    }

    /// <summary>
    /// Get all authentication tokens
    /// </summary>
    [HttpGet("tokens")]
    public IActionResult GetTokens()
    {
        try
        {
            var tokensPath = Path.Combine("data", "config", "tokens.txt");
            if (!System.IO.File.Exists(tokensPath))
                return Ok(new { tokens = Array.Empty<object>() });

            var lines = System.IO.File.ReadAllLines(tokensPath);
            var tokens = new List<object>();

            foreach (var line in lines)
            {
                var trimmed = line.Trim();
                if (string.IsNullOrEmpty(trimmed) || trimmed.StartsWith("#"))
                    continue;

                try
                {
                    var entry = Middleware.TokenEntry.Parse(trimmed);
                    tokens.Add(new
                    {
                        token = entry.Token,
                        access_level = entry.AccessLevel.ToString().ToLower()
                    });
                }
                catch
                {
                    // Skip invalid lines
                }
            }

            return Ok(new { tokens });
        }
        catch (Exception ex)
        {
            return StatusCode(500, new { error = $"Failed to get tokens: {ex.Message}" });
        }
    }

    /// <summary>
    /// Add a new token
    /// </summary>
    [HttpPost("tokens")]
    public async Task<IActionResult> AddToken([FromBody] AddTokenRequest request)
    {
        try
        {
            if (string.IsNullOrWhiteSpace(request.Token))
                return BadRequest(new { error = "Token is required" });

            var tokensPath = Path.Combine("data", "config", "tokens.txt");
            Directory.CreateDirectory(Path.GetDirectoryName(tokensPath)!);

            // Parse access level
            var accessLevel = request.AccessLevel?.ToUpperInvariant() switch
            {
                "A" or "ALL" => Middleware.TokenAccessLevel.All,
                "M" or "MANAGE" => Middleware.TokenAccessLevel.Manage,
                "U" or "USER" => Middleware.TokenAccessLevel.User,
                _ => Middleware.TokenAccessLevel.User
            };

            var entry = new Middleware.TokenEntry
            {
                Token = request.Token.Trim(),
                AccessLevel = accessLevel
            };

            // Check if token already exists
            if (System.IO.File.Exists(tokensPath))
            {
                var existingLines = await System.IO.File.ReadAllLinesAsync(tokensPath);
                foreach (var line in existingLines)
                {
                    if (line.Contains(entry.Token))
                        return BadRequest(new { error = "Token already exists" });
                }
            }

            // Append token
            await System.IO.File.AppendAllTextAsync(tokensPath, $"{entry}\n");

            return Ok(new { message = "Token added successfully", token = entry.Token, access_level = entry.AccessLevel.ToString().ToLower() });
        }
        catch (Exception ex)
        {
            return StatusCode(500, new { error = $"Failed to add token: {ex.Message}" });
        }
    }

    /// <summary>
    /// Delete a token
    /// </summary>
    [HttpDelete("tokens/{token}")]
    public async Task<IActionResult> DeleteToken(string token)
    {
        try
        {
            var tokensPath = Path.Combine("data", "config", "tokens.txt");
            if (!System.IO.File.Exists(tokensPath))
                return NotFound(new { error = "Tokens file not found" });

            var lines = await System.IO.File.ReadAllLinesAsync(tokensPath);
            var newLines = new List<string>();
            var found = false;

            foreach (var line in lines)
            {
                if (line.Contains(token))
                {
                    found = true;
                    continue; // Skip this line
                }
                newLines.Add(line);
            }

            if (!found)
                return NotFound(new { error = "Token not found" });

            await System.IO.File.WriteAllLinesAsync(tokensPath, newLines);

            return Ok(new { message = "Token deleted successfully" });
        }
        catch (Exception ex)
        {
            return StatusCode(500, new { error = $"Failed to delete token: {ex.Message}" });
        }
    }

    /// <summary>
    /// Update token access level
    /// </summary>
    [HttpPut("tokens/{token}")]
    public async Task<IActionResult> UpdateToken(string token, [FromBody] UpdateTokenRequest request)
    {
        try
        {
            var tokensPath = Path.Combine("data", "config", "tokens.txt");
            if (!System.IO.File.Exists(tokensPath))
                return NotFound(new { error = "Tokens file not found" });

            var accessLevel = request.AccessLevel?.ToUpperInvariant() switch
            {
                "A" or "ALL" => Middleware.TokenAccessLevel.All,
                "M" or "MANAGE" => Middleware.TokenAccessLevel.Manage,
                "U" or "USER" => Middleware.TokenAccessLevel.User,
                _ => Middleware.TokenAccessLevel.User
            };

            var lines = await System.IO.File.ReadAllLinesAsync(tokensPath);
            var newLines = new List<string>();
            var found = false;

            foreach (var line in lines)
            {
                if (line.Contains(token))
                {
                    found = true;
                    var entry = new Middleware.TokenEntry { Token = token, AccessLevel = accessLevel };
                    newLines.Add(entry.ToString());
                }
                else
                {
                    newLines.Add(line);
                }
            }

            if (!found)
                return NotFound(new { error = "Token not found" });

            await System.IO.File.WriteAllLinesAsync(tokensPath, newLines);

            return Ok(new { message = "Token updated successfully" });
        }
        catch (Exception ex)
        {
            return StatusCode(500, new { error = $"Failed to update token: {ex.Message}" });
        }
    }

    /// <summary>
    /// Get server configuration
    /// </summary>
    [HttpGet("server-config")]
    public IActionResult GetServerConfig()
    {
        try
        {
            var configPath = Path.Combine(AppContext.BaseDirectory, "data", "config", "server.config.json");
            if (!System.IO.File.Exists(configPath))
            {
                // Create default config if it doesn't exist
                Directory.CreateDirectory(Path.GetDirectoryName(configPath)!);
                var defaultConfig = new
                {
                    models_directory = "./data/models",
                    cache_directory = "./data/cache",
                    host = "localhost",
                    port = 11434,
                    compute_provider = "vulkan",
                    auto_unload_minutes = 30,
                    manage_token = "",
                    tokens_file = "./data/config/tokens.txt"
                };
                var configJson = JsonSerializer.Serialize(defaultConfig, new JsonSerializerOptions { WriteIndented = true });
                System.IO.File.WriteAllText(configPath, configJson);
                return Content(configJson, "application/json");
            }

            var existingConfig = System.IO.File.ReadAllText(configPath);
            return Content(existingConfig, "application/json");
        }
        catch (Exception ex)
        {
            return StatusCode(500, new { error = $"Failed to get server config: {ex.Message}" });
        }
    }

    /// <summary>
    /// Update server configuration
    /// </summary>
    [HttpPut("server-config")]
    public async Task<IActionResult> UpdateServerConfig([FromBody] JsonElement config)
    {
        try
        {
            var configPath = Path.Combine(AppContext.BaseDirectory, "data", "config", "server.config.json");
            Directory.CreateDirectory(Path.GetDirectoryName(configPath)!);

            // Validate JSON structure
            var configJson = JsonSerializer.Serialize(config, new JsonSerializerOptions { WriteIndented = true });
            await System.IO.File.WriteAllTextAsync(configPath, configJson);

            return Ok(new { message = "Server config updated successfully. Restart server to apply changes." });
        }
        catch (Exception ex)
        {
            return StatusCode(500, new { error = $"Failed to update server config: {ex.Message}" });
        }
    }

    /// <summary>
    /// Load a model from config (auto-load if not already loaded)
    /// </summary>
    [HttpPost("models/{name}/load")]
    public async Task<IActionResult> LoadModel(string name)
    {
        try
        {
            // Check if model is already loaded
            var loadedModels = _modelManager.GetLoadedModels();
            if (loadedModels.ContainsKey(name))
            {
                return Ok(new { message = "Model already loaded", name });
            }

            // Try to load the model
            var model = await _modelManager.GetModelAsync(name);
            
            return Ok(new { message = "Model loaded successfully", name });
        }
        catch (FileNotFoundException ex)
        {
            return NotFound(new { error = $"Model config not found: {ex.Message}" });
        }
        catch (Exception ex)
        {
            return StatusCode(500, new { error = $"Failed to load model: {ex.Message}" });
        }
    }

    /// <summary>
    /// Initialize model from config (download if needed)
    /// </summary>
    [HttpPost("configs/{name}/initialize")]
    public async Task<IActionResult> InitializeModel(string name)
    {
        try
        {
            // Configs are saved as {ModelsDirectory}/{name}/model.json
            var configPath = Path.Combine(_modelManager.ModelsDirectory, name, "model.json");
            // Fall back to flat format
            if (!System.IO.File.Exists(configPath))
                configPath = Path.Combine(_modelManager.ModelsDirectory, $"{name}.json");

            if (!System.IO.File.Exists(configPath))
                return NotFound(new { error = $"Config not found. Searched:\n  {Path.Combine(_modelManager.ModelsDirectory, name, "model.json")}\n  {Path.Combine(_modelManager.ModelsDirectory, $"{name}.json")}" });

            var configJson = await System.IO.File.ReadAllTextAsync(configPath);
            var config = JsonSerializer.Deserialize<ModelConfig>(configJson);

            if (config == null)
                return BadRequest(new { error = "Invalid config format" });

            if (string.IsNullOrEmpty(config.ModelPath))
                return BadRequest(new { error = "Config has no model path (\"model\" field) specified" });

            // Check if model file exists in cache directory
            var modelPath = Path.IsPathRooted(config.ModelPath)
                ? config.ModelPath
                : Path.Combine(_modelManager.CacheDirectory, config.ModelPath);

            if (System.IO.File.Exists(modelPath))
                return Ok(new { message = "Model file already exists", path = modelPath });

            // If model field is a URL, start download
            if (config.ModelPath.StartsWith("http://") || config.ModelPath.StartsWith("https://"))
            {
                var filename = Path.GetFileName(new Uri(config.ModelPath).LocalPath);
                var downloadId = _downloadManager.StartDownload(config.ModelPath, filename);

                return Ok(new
                {
                    message = "Download started",
                    download_id = downloadId,
                    filename = filename
                });
            }

            return BadRequest(new { error = $"Model path is not a URL and file not found at: {modelPath}" });
        }
        catch (Exception ex)
        {
            return StatusCode(500, new { error = $"Failed to initialize model: {ex.Message}" });
        }
    }
}

public class AddTokenRequest
{
    [JsonPropertyName("token")]
    public string Token { get; set; } = "";

    [JsonPropertyName("access_level")]
    public string AccessLevel { get; set; } = "U";
}

public class UpdateTokenRequest
{
    [JsonPropertyName("access_level")]
    public string AccessLevel { get; set; } = "U";
}

public class StartDownloadRequest
{
    [JsonPropertyName("url")]
    public string Url { get; set; } = "";

    [JsonPropertyName("filename")]
    public string? Filename { get; set; }

    [JsonPropertyName("force")]
    public bool Force { get; set; } = false;
}