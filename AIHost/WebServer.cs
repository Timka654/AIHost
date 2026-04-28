using AIHost.Config;
using AIHost.ICompute;
using AIHost.ICompute.Vulkan;
using AIHost.ICompute.CUDA;
using AIHost.ICompute.ROCm;
using AIHost.Logging;
using AIHost.Middleware;
using AIHost.Services;
using System.Text.Json;

var builder = WebApplication.CreateBuilder(args);

// Load server configuration
var configPath = Path.Combine(AppContext.BaseDirectory, "data", "config", "server.json");
if (!File.Exists(configPath))
{
    // Fallback to old location for backward compatibility
    configPath = Path.Combine(AppContext.BaseDirectory, "config", "server.json");
}
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
    
    // Create default config
    Directory.CreateDirectory(Path.GetDirectoryName(configPath)!);
    var defaultJson = JsonSerializer.Serialize(serverConfig, new JsonSerializerOptions { WriteIndented = true });
    await File.WriteAllTextAsync(configPath, defaultJson);
    Console.WriteLine($"✓ Created default config at {configPath}");
}

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
var tokensFile = serverConfig.TokensFile != null ? Path.Combine(AppContext.BaseDirectory, serverConfig.TokensFile) : null;
app.UseTokenAuth(tokensFile, serverConfig.ManageToken);

if (tokensFile != null && File.Exists(tokensFile))
{
    Console.WriteLine($"✓ Token authentication enabled: {tokensFile}");
}
else
{
    Console.WriteLine("⚠ Token authentication disabled (no tokens file)");
}

// Start auto-unload service
var autoUnloadService = new ModelAutoUnloadService(modelManager, serverConfig.AutoUnloadMinutes);

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
