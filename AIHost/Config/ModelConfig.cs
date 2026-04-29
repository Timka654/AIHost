using System.Text.Json.Serialization;

namespace AIHost.Config;

/// <summary>
/// Configuration for a single model
/// </summary>
public class ModelConfig
{
    /// <summary>
    /// Unique model identifier (e.g., "tinyllama-chat")
    /// </summary>
    [JsonPropertyName("name")]
    public string Name { get; set; } = "";

    /// <summary>
    /// Model file path or URL
    /// </summary>
    [JsonPropertyName("model")]
    public string ModelPath { get; set; } = "";

    /// <summary>
    /// Model format: "gguf" (default), "safetensors", etc.
    /// </summary>
    [JsonPropertyName("format")]
    public string Format { get; set; } = "gguf";

    /// <summary>
    /// Auto-download if model file doesn't exist
    /// </summary>
    [JsonPropertyName("auto_download")]
    public bool AutoDownload { get; set; } = false;

    /// <summary>
    /// Generation parameters
    /// </summary>
    [JsonPropertyName("parameters")]
    public GenerationParameters Parameters { get; set; } = new();

    /// <summary>
    /// System messages (Ollama Modelfile format)
    /// </summary>
    [JsonPropertyName("system_messages")]
    public List<string> SystemMessages { get; set; } = new();

    /// <summary>
    /// Paths to external system message files
    /// </summary>
    [JsonPropertyName("system_message_files")]
    public List<string> SystemMessageFiles { get; set; } = new();

    /// <summary>
    /// Model description
    /// </summary>
    [JsonPropertyName("description")]
    public string? Description { get; set; }

    /// <summary>
    /// Model tags (e.g., ["chat", "instruct"])
    /// </summary>
    [JsonPropertyName("tags")]
    public List<string> Tags { get; set; } = new();

    /// <summary>
    /// Compute provider for this model: "vulkan", "cuda", "rocm" (null = use global)
    /// </summary>
    [JsonPropertyName("compute_provider")]
    public string? ComputeProvider { get; set; }

    /// <summary>
    /// GPU device index (null = use global device_index)
    /// </summary>
    [JsonPropertyName("device_index")]
    public int? DeviceIndex { get; set; }

    /// <summary>
    /// Keep-alive time in minutes before auto-unload (null = use global auto_unload_minutes, 0 = never unload)
    /// Ollama-style: how long to keep model in memory after last use
    /// </summary>
    [JsonPropertyName("keep_alive")]
    public int? KeepAliveMinutes { get; set; }

    /// <summary>
    /// Number of layers to offload to GPU (-1 = all, 0 = CPU only, N = specific count)
    /// Useful for hybrid CPU/GPU inference when VRAM is limited
    /// </summary>
    [JsonPropertyName("num_gpu_layers")]
    public int? NumGpuLayers { get; set; }

    /// <summary>
    /// Preferred batch size for inference (null = use default)
    /// </summary>
    [JsonPropertyName("batch_size")]
    public int? BatchSize { get; set; }

    /// <summary>
    /// Use memory mapping for model loading (default true)
    /// Reduces memory usage but may be slower on some systems
    /// </summary>
    [JsonPropertyName("enable_mmap")]
    public bool EnableMmap { get; set; } = true;

    /// <summary>
    /// Lock model in RAM to prevent swapping (default false)
    /// Only enable if you have sufficient RAM
    /// </summary>
    [JsonPropertyName("enable_mlock")]
    public bool EnableMlock { get; set; } = false;

    /// <summary>
    /// Allow the model to fall back to shared GPU memory (system RAM) when dedicated VRAM
    /// is insufficient. Default false — loading fails fast with a clear error rather than
    /// silently running at PCIe speeds (10–20x slower than VRAM).
    /// Set to true only if you intentionally want CPU-side inference.
    /// </summary>
    [JsonPropertyName("allow_shared_memory")]
    public bool AllowSharedMemory { get; set; } = false;
}

/// <summary>
/// Generation parameters for the model
/// </summary>
public class GenerationParameters
{
    /// <summary>
    /// Temperature (0.0 - 2.0, default 0.7)
    /// </summary>
    [JsonPropertyName("temperature")]
    public float Temperature { get; set; } = 0.7f;

    /// <summary>
    /// Top-K sampling (default 40)
    /// </summary>
    [JsonPropertyName("top_k")]
    public int TopK { get; set; } = 40;

    /// <summary>
    /// Top-P sampling (default 0.9)
    /// </summary>
    [JsonPropertyName("top_p")]
    public float TopP { get; set; } = 0.9f;

    /// <summary>
    /// Repetition penalty (default 1.1)
    /// </summary>
    [JsonPropertyName("repetition_penalty")]
    public float RepetitionPenalty { get; set; } = 1.1f;

    /// <summary>
    /// Context size in tokens (default 2048)
    /// </summary>
    [JsonPropertyName("context_size")]
    public int ContextSize { get; set; } = 2048;

    /// <summary>
    /// Maximum number of tokens to generate (default 512)
    /// </summary>
    [JsonPropertyName("max_tokens")]
    public int MaxTokens { get; set; } = 512;

    /// <summary>
    /// Random seed (-1 for random, default -1)
    /// </summary>
    [JsonPropertyName("seed")]
    public int Seed { get; set; } = -1;

    /// <summary>
    /// Use KV cache for efficiency (default true)
    /// </summary>
    [JsonPropertyName("use_kv_cache")]
    public bool UseKVCache { get; set; } = true;

    /// <summary>
    /// KV cache quantization: "none", "int8", "int4" (default "none")
    /// </summary>
    [JsonPropertyName("kv_cache_quantization")]
    public string KVCacheQuantization { get; set; } = "none";

    /// <summary>
    /// Stop sequences (e.g., ["\n\n", "User:"])
    /// </summary>
    [JsonPropertyName("stop")]
    public List<string> Stop { get; set; } = new();
}

/// <summary>
/// Global server configuration
/// </summary>
public class ServerConfig
{
    /// <summary>
    /// Models directory path (default "./data/models")
    /// </summary>
    [JsonPropertyName("models_directory")]
    public string ModelsDirectory { get; set; } = "./data/models";

    /// <summary>
    /// Server host (default "localhost")
    /// </summary>
    [JsonPropertyName("host")]
    public string Host { get; set; } = "localhost";

    /// <summary>
    /// Server port (default 11434 - Ollama compatible)
    /// </summary>
    [JsonPropertyName("port")]
    public int Port { get; set; } = 11434;

    /// <summary>
    /// Enable OpenAI API endpoints (default true)
    /// </summary>
    [JsonPropertyName("enable_openai_api")]
    public bool EnableOpenAIAPI { get; set; } = true;

    /// <summary>
    /// Enable Ollama API endpoints (default true)
    /// </summary>
    [JsonPropertyName("enable_ollama_api")]
    public bool EnableOllamaAPI { get; set; } = true;

    /// <summary>
    /// Compute provider: "vulkan", "cuda", "rocm" (default "vulkan")
    /// </summary>
    [JsonPropertyName("compute_provider")]
    public string ComputeProvider { get; set; } = "vulkan";

    /// <summary>
    /// GPU device index (default 0)
    /// </summary>
    [JsonPropertyName("device_index")]
    public int DeviceIndex { get; set; } = 0;

    /// <summary>
    /// Enable CORS (default true)
    /// </summary>
    [JsonPropertyName("enable_cors")]
    public bool EnableCORS { get; set; } = true;

    /// <summary>
    /// Log level: "debug", "info", "warning", "error" (default "info")
    /// </summary>
    [JsonPropertyName("log_level")]
    public string LogLevel { get; set; } = "info";

    /// <summary>
    /// Management token for admin API endpoints (null = no auth required)
    /// </summary>
    [JsonPropertyName("manage_token")]
    public string? ManageToken { get; set; } = null;

    /// <summary>
    /// Path to tokens file for request authentication (null = no auth)
    /// </summary>
    [JsonPropertyName("tokens_file")]
    public string? TokensFile { get; set; } = null;

    /// <summary>
    /// Directory for persistent logs (default "./data/logs")
    /// </summary>
    [JsonPropertyName("logs_directory")]
    public string LogsDirectory { get; set; } = "./data/logs";

    /// <summary>
    /// Directory for cache files (default "./data/cache")
    /// </summary>
    [JsonPropertyName("cache_directory")]
    public string CacheDirectory { get; set; } = "./data/cache";

    /// <summary>
    /// Enable persistent logs to disk (default true)
    /// </summary>
    [JsonPropertyName("persistent_logs")]
    public bool PersistentLogs { get; set; } = true;

    /// <summary>
    /// Maximum number of log files to keep (default 10)
    /// </summary>
    [JsonPropertyName("max_log_files")]
    public int MaxLogFiles { get; set; } = 10;

    /// <summary>
    /// Auto-unload models after N minutes of inactivity (0 = disabled)
    /// </summary>
    [JsonPropertyName("auto_unload_minutes")]
    public int AutoUnloadMinutes { get; set; } = 30;

    /// <summary>
    /// Rate limit: requests per minute (0 = unlimited)
    /// </summary>
    [JsonPropertyName("rate_limit_requests_per_minute")]
    public int RateLimitRequestsPerMinute { get; set; } = 60;
}
