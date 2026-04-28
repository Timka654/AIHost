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

// Initialize model manager
var modelManager = new ModelManager(serverConfig.ModelsDirectory, computeDevice);
builder.Services.AddSingleton(modelManager);
builder.Services.AddSingleton(serverConfig);
// Register the global compute device so future services can receive it via DI.
// Note: disposed manually at shutdown — DI doesn't own singleton instances registered this way.
builder.Services.AddSingleton<IComputeDevice>(computeDevice);

// Background services registered through DI — participate in graceful shutdown.
builder.Services.AddHostedService<ModelAutoUnloadService>();
builder.Services.AddHostedService<ModelConfigWatcherService>();

var app = builder.Build();

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
        var tokensTemplate = @"# API Authentication Tokens
# Add one token per line. Lines starting with # are comments.
# Example tokens (replace with your own!):
# your-secret-token-here
# another-token-12345

# Uncomment the line below to enable authentication:
# my-secret-api-token
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
