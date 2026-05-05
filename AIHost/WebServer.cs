using AIHost;
using AIHost.Config;
using AIHost.ICompute;
using AIHost.ICompute.Vulkan;
using AIHost.ICompute.CUDA;
using AIHost.ICompute.ROCm;
using AIHost.Logging;
using AIHost.Middleware;
using AIHost.Services;
using System.Text.Json;

// Manual test mode: bypass web server entirely
if (args.Contains("--manual"))
{
    AIHost.ManualTests.TestRunner.RunInteractiveTests(args.Where(a => a != "--manual").ToArray());
    return;
}

// Initialize data directory from templates if needed
{
    var dataDir = Path.Combine(AppContext.BaseDirectory, "data");
    var dataInitDir = Path.Combine(AppContext.BaseDirectory, "data_init");

    // If data_init doesn't exist, skip initialization
    if (Directory.Exists(dataInitDir))
    {
        // Create data directory if it doesn't exist
        Directory.CreateDirectory(dataDir);

        // Check if data directory is empty or has no essential files
        var needsInitialization = !File.Exists(Path.Combine(dataDir, "model.schema.json")) ||
                                  !File.Exists(Path.Combine(dataDir, "model.example.json")) ||
                                  !Directory.Exists(Path.Combine(dataDir, "config"));

        if (needsInitialization)
        {
            Console.WriteLine("📦 Initializing data directory from templates...");

            // Copy schema and example
            var schemaSource = Path.Combine(dataInitDir, "model.schema.json");
            var schemaTarget = Path.Combine(dataDir, "model.schema.json");
            if (File.Exists(schemaSource) && !File.Exists(schemaTarget))
            {
                File.Copy(schemaSource, schemaTarget);
                Console.WriteLine($"  ✓ Copied model.schema.json");
            }

            var exampleSource = Path.Combine(dataInitDir, "model.example.json");
            var exampleTarget = Path.Combine(dataDir, "model.example.json");
            if (File.Exists(exampleSource) && !File.Exists(exampleTarget))
            {
                File.Copy(exampleSource, exampleTarget);
                Console.WriteLine($"  ✓ Copied model.example.json");
            }

            // Copy config directory
            var configSourceDir = Path.Combine(dataInitDir, "config");
            var configTargetDir = Path.Combine(dataDir, "config");
            if (Directory.Exists(configSourceDir))
            {
                Directory.CreateDirectory(configTargetDir);
                
                foreach (var file in Directory.GetFiles(configSourceDir))
                {
                    var filename = Path.GetFileName(file);
                    var targetFile = Path.Combine(configTargetDir, filename);
                    if (!File.Exists(targetFile))
                    {
                        File.Copy(file, targetFile);
                        Console.WriteLine($"  ✓ Copied config/{filename}");
                    }
                }
            }

            // Create other necessary subdirectories
            Directory.CreateDirectory(Path.Combine(dataDir, "models"));
            Directory.CreateDirectory(Path.Combine(dataDir, "cache"));
            Directory.CreateDirectory(Path.Combine(dataDir, "logs"));

            Console.WriteLine("✓ Data directory initialized successfully");
        }
    }
}

var builder = WebApplication.CreateBuilder(args);

// Locate server.json — prefer data/config, fall back to config/ for backward compat.
var configPath = Path.Combine(AppContext.BaseDirectory, "data", "config", "server.json");
if (!File.Exists(configPath))
    configPath = Path.Combine(AppContext.BaseDirectory, "config", "server.json");

// Load config for early startup decisions (port, device, etc.).
// System.Text.Json is used so [JsonPropertyName] attributes on ServerConfig are respected.
var serverConfig = new ServerConfig();

if (File.Exists(configPath))
{
    var json = await File.ReadAllTextAsync(configPath);
    serverConfig = JsonSerializer.Deserialize<ServerConfig>(json) ?? new ServerConfig();
    Console.WriteLine($"✓ Loaded server configuration from {configPath}");
}
else
{
    Console.WriteLine($"⚠ Server config not found, using defaults");
    Directory.CreateDirectory(Path.GetDirectoryName(configPath)!);
    var defaultJson = JsonSerializer.Serialize(serverConfig, new JsonSerializerOptions { WriteIndented = true });
    await File.WriteAllTextAsync(configPath, defaultJson);
    Console.WriteLine($"✓ Created default config at {configPath}");
}

// Also wire server.json into the IConfiguration pipeline so that services can
// access it via IConfiguration and it participates in the standard config chain.
builder.Configuration.AddJsonFile(configPath, optional: true, reloadOnChange: false);

// Ensure directory structure and template files exist
EnsureDirectoryStructure(serverConfig);

// Configure Kestrel
builder.WebHost.ConfigureKestrel(options =>
{
    options.ListenAnyIP(serverConfig.Port);
});

// Configure logging level from server config
builder.Logging.ClearProviders();
builder.Logging.AddConsole();
var minLogLevel = serverConfig.LogLevel.ToLowerInvariant() switch
{
    "trace"   => Microsoft.Extensions.Logging.LogLevel.Trace,
    "debug"   => Microsoft.Extensions.Logging.LogLevel.Debug,
    "warning" => Microsoft.Extensions.Logging.LogLevel.Warning,
    "error"   => Microsoft.Extensions.Logging.LogLevel.Error,
    _         => Microsoft.Extensions.Logging.LogLevel.Information
};
builder.Logging.SetMinimumLevel(minLogLevel);

// Add services
builder.Services.AddControllers()
    .AddJsonOptions(options =>
    {
        options.JsonSerializerOptions.PropertyNamingPolicy = System.Text.Json.JsonNamingPolicy.SnakeCaseLower;
        options.JsonSerializerOptions.DefaultIgnoreCondition = System.Text.Json.Serialization.JsonIgnoreCondition.WhenWritingNull;
    });

builder.Services.AddEndpointsApiExplorer();
builder.Services.AddSwaggerGen();

// Configure CORS
if (serverConfig.EnableCORS)
{
    builder.Services.AddCors(options =>
    {
        options.AddDefaultPolicy(policy =>
        {
            policy.AllowAnyOrigin()
                  .AllowAnyMethod()
                  .AllowAnyHeader();
        });
    });
}

// Initialize compute device
IComputeDevice computeDevice = serverConfig.ComputeProvider.ToLower() switch
{
    "cuda" => new CudaComputeDevice(serverConfig.DeviceIndex),
    "rocm" => new ROCmComputeDevice(serverConfig.DeviceIndex),
    "vulkan" => new VulkanComputeDevice(serverConfig.DeviceIndex),
    _ => new VulkanComputeDevice(serverConfig.DeviceIndex)
};

var deviceName = computeDevice switch
{
    VulkanComputeDevice vk => vk.DeviceName,
    CudaComputeDevice cuda => cuda.DeviceName,
    _ => "Unknown"
};
Console.WriteLine($"✓ Initialized compute device: {deviceName} ({computeDevice.ProviderName})");

// Initialize directories
Directory.CreateDirectory(serverConfig.LogsDirectory);
Directory.CreateDirectory(serverConfig.CacheDirectory);
Directory.CreateDirectory(serverConfig.ModelsDirectory);

// Initialize request logger
var requestLogger = new RequestLogger(
    maxLogs: 1000,
    logsDirectory: serverConfig.LogsDirectory,
    persistentLogs: serverConfig.PersistentLogs,
    maxLogFiles: serverConfig.MaxLogFiles
);
builder.Services.AddSingleton(requestLogger);
Console.WriteLine($"✓ Request logger initialized (persistent: {serverConfig.PersistentLogs})");

// Initialize model manager via factory so ILogger<ModelManager> is injected from DI.
builder.Services.AddSingleton(sp =>
    new ModelManager(
        serverConfig.ModelsDirectory,
        serverConfig.CacheDirectory,
        computeDevice,
        sp.GetRequiredService<ILogger<ModelManager>>()));
builder.Services.AddSingleton(serverConfig);

// Initialize download manager
builder.Services.AddSingleton(sp =>
    new DownloadManager(serverConfig.CacheDirectory));

// Register the global compute device so future services can receive it via DI.
// Note: disposed manually at shutdown — DI doesn't own singleton instances registered this way.
builder.Services.AddSingleton<IComputeDevice>(computeDevice);

// Background services registered through DI — participate in graceful shutdown.
builder.Services.AddHostedService<ModelAutoUnloadService>();
builder.Services.AddHostedService<ModelConfigWatcherService>();

var app = builder.Build();

// Initialize AppLogger so non-DI classes (Transformer, InferenceEngine, etc.) get typed loggers
AppLogger.Initialize(app.Services.GetRequiredService<Microsoft.Extensions.Logging.ILoggerFactory>());

var modelManager = app.Services.GetRequiredService<ModelManager>();

// Configure middleware
if (app.Environment.IsDevelopment())
{
    app.UseSwagger();
    app.UseSwaggerUI();
}

if (serverConfig.EnableCORS)
{
    app.UseCors();
}

// Rate limiting middleware
if (serverConfig.RateLimitRequestsPerMinute > 0)
{
    app.UseRateLimit(serverConfig.RateLimitRequestsPerMinute);
    Console.WriteLine($"✓ Rate limiting: {serverConfig.RateLimitRequestsPerMinute} requests/minute");
}

// Token authentication middleware
app.UseTokenAuth();

var tokensFile = serverConfig.TokensFile != null ? Path.Combine(AppContext.BaseDirectory, serverConfig.TokensFile) : null;
if (tokensFile != null && File.Exists(tokensFile))
{
    Console.WriteLine($"✓ Token authentication enabled: {tokensFile}");
}
else
{
    Console.WriteLine("⚠ Token authentication disabled (no tokens file)");
}

// (Auto-unload is now managed by DI as ModelAutoUnloadService IHostedService)

if (serverConfig.ManageToken != null)
{
    Console.WriteLine("✓ Management token configured");
}

// Static files for web UI
app.UseDefaultFiles();
app.UseStaticFiles();

app.UseRouting();

// Map controllers conditionally
if (serverConfig.EnableOllamaAPI)
{
    app.MapControllers();
    Console.WriteLine("✓ Ollama API enabled: /api/*");
}

if (serverConfig.EnableOpenAIAPI)
{
    app.MapControllers();
    Console.WriteLine("✓ OpenAI API enabled: /v1/*");
}

// Always enable management API
app.MapControllers();
Console.WriteLine("✓ Management API enabled: /manage/*");

// Health check endpoint
app.MapGet("/health", () => new
{
    status = "ok",
    version = "1.0.0",
    provider = computeDevice.ProviderName,
    device = deviceName,
    apis = new
    {
        ollama = serverConfig.EnableOllamaAPI,
        openai = serverConfig.EnableOpenAIAPI
    }
});

Console.WriteLine($"\n🚀 AIHost Server started on http://{serverConfig.Host}:{serverConfig.Port}");
Console.WriteLine($"   Models directory: {Path.GetFullPath(serverConfig.ModelsDirectory)}");
Console.WriteLine($"   Available models: {modelManager.ListModels().Count()}");
Console.WriteLine($"\n📚 API Documentation:");
Console.WriteLine($"   Ollama: http://{serverConfig.Host}:{serverConfig.Port}/api/generate");
Console.WriteLine($"   OpenAI: http://{serverConfig.Host}:{serverConfig.Port}/v1/chat/completions");
Console.WriteLine($"   Swagger: http://{serverConfig.Host}:{serverConfig.Port}/swagger\n");

await app.RunAsync();

// Cleanup
modelManager.Dispose();
computeDevice.Dispose();

/// <summary>
/// Ensure all required directories and template files exist
/// </summary>
static void EnsureDirectoryStructure(ServerConfig config)
{
    // Ensure all directories exist
    var directories = new[]
    {
        config.ModelsDirectory,
        config.LogsDirectory,
        config.CacheDirectory,
        Path.GetDirectoryName(config.TokensFile) ?? "data/config"
    };

    foreach (var dir in directories.Where(d => !string.IsNullOrEmpty(d)))
    {
        if (!Directory.Exists(dir))
        {
            Directory.CreateDirectory(dir);
            Console.WriteLine($"✓ Created directory: {dir}");
        }
    }

    // Create template tokens.txt if it doesn't exist
    if (!string.IsNullOrEmpty(config.TokensFile))
    {
        var tokensPath = Path.IsPathRooted(config.TokensFile)
            ? config.TokensFile
            : Path.Combine(AppContext.BaseDirectory, config.TokensFile);

        if (!File.Exists(tokensPath))
    {
        var tokensTemplate = @"# Authentication Tokens Configuration
# Format: <Modifier>:<Token>
# Modifiers:
#   A - All access (full access to everything)
#   M - Manage access (can use /manage endpoints)
#   U - User access (can use API endpoints, not /manage)
#
# Example tokens (uncomment to enable):
# A:admin-full-access-token-123456
# M:manager-token-789012
# U:user-api-token-345678
";
        File.WriteAllText(tokensPath, tokensTemplate);
        Console.WriteLine($"✓ Created template tokens file: {tokensPath}");
        Console.WriteLine($"  ⚠ Edit {Path.GetFileName(tokensPath)} to add your API tokens");
    }
    }

    // Check if any models exist
    var modelDirs = Directory.Exists(config.ModelsDirectory)
        ? Directory.GetDirectories(config.ModelsDirectory)
        : Array.Empty<string>();

    if (modelDirs.Length == 0)
    {
        // Create example model directory with template
        var exampleModelDir = Path.Combine(config.ModelsDirectory, "example-model");
        Directory.CreateDirectory(exampleModelDir);

        var exampleConfig = new
        {
            name = "example-model",
            model = "C:\\path\\to\\your\\model.gguf",
            format = "gguf",
            description = "Example model configuration - edit this file to configure your model",
            tags = new[] { "example", "template" },
            compute_provider = "vulkan",
            device_index = 0,
            keep_alive = 30,
            enable_mmap = true,
            enable_mlock = false,
            num_gpu_layers = -1,
            parameters = new
            {
                temperature = 0.7,
                top_k = 40,
                top_p = 0.9,
                repetition_penalty = 1.1,
                context_size = 2048,
                max_tokens = 512
            }
        };

        var exampleJson = JsonSerializer.Serialize(exampleConfig, new JsonSerializerOptions { WriteIndented = true });
        var exampleConfigPath = Path.Combine(exampleModelDir, "model.json");
        File.WriteAllText(exampleConfigPath, exampleJson);

        Console.WriteLine($"✓ Created example model config: {exampleConfigPath}");
        Console.WriteLine($"  ⚠ Edit model.json to configure your model:");
        Console.WriteLine($"     1. Set 'model' path to your .gguf file");
        Console.WriteLine($"     2. Update 'name' to match your model");
        Console.WriteLine($"     3. Adjust parameters as needed");
    }
}
