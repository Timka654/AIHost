using Microsoft.AspNetCore.Mvc;
using AIHost.Config;
using AIHost.Logging;
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
/// Management API controller for admin operations
/// </summary>
[ApiController]
[Route("manage")]
public class ManagementController : ControllerBase
{
    private readonly ModelManager _modelManager;
    private readonly RequestLogger _requestLogger;

    public ManagementController(ModelManager modelManager, RequestLogger requestLogger)
    {
        _modelManager = modelManager;
        _requestLogger = requestLogger;
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
            if (!Directory.Exists(_modelManager.ModelsDirectory))
                return Ok(new { message = "Cache directory does not exist" });

            var cacheDir = _modelManager.ModelsDirectory;
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
            if (!Directory.Exists(_modelManager.ModelsDirectory))
                return Ok(new { path = _modelManager.ModelsDirectory, file_count = 0, total_size = 0, dir_count = 0 });

            var files = Directory.GetFiles(_modelManager.ModelsDirectory, "*", SearchOption.AllDirectories);
            var dirs = Directory.GetDirectories(_modelManager.ModelsDirectory, "*", SearchOption.AllDirectories);
            var totalSize = files.Sum(f => new FileInfo(f).Length);

            return Ok(new
            {
                path = _modelManager.ModelsDirectory,
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
    /// Direct chat endpoint
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

            var response = model.Engine.Generate(request.Message, config);

            return Ok(new
            {
                model = request.ModelName,
                response = response,
                tokens = response.Length,
                finish_reason = "stop"
            });
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
            // Create model config
            var modelDir = Path.Combine(_modelManager.ModelsDirectory, request.Name);

            Directory.CreateDirectory(modelDir);

            var fileName = Path.GetFileName(new Uri(request.Url).LocalPath);
            var modelPath = Path.Combine(modelDir, fileName);

            // Download
            Console.WriteLine($"Downloading {request.Name} from {request.Url}...");
            using var client = new HttpClient();
            client.Timeout = TimeSpan.FromHours(2);

            using var response = await client.GetAsync(request.Url, HttpCompletionOption.ResponseHeadersRead);
            response.EnsureSuccessStatusCode();

            using var fileStream = System.IO.File.Create(modelPath);
            await response.Content.CopyToAsync(fileStream);

            Console.WriteLine($"Downloaded {request.Name} to {modelPath}");

            // Create model.json
            var config = new ModelConfig
            {
                Name = request.Name,
                ModelPath = modelPath,
                Format = request.Format,
                Description = $"Downloaded from {request.Url}"
            };

            var configPath = Path.Combine(modelDir, "model.json");
            var json = JsonSerializer.Serialize(config, new JsonSerializerOptions { WriteIndented = true });
            await System.IO.File.WriteAllTextAsync(configPath, json);

            return Ok(new { 
                message = $"Model '{request.Name}' downloaded successfully",
                path = modelPath
            });
        }
        catch (Exception ex)
        {
            return StatusCode(500, new { error = $"Download failed: {ex.Message}" });
        }
    }
}