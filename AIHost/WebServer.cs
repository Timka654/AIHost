using AIHost;
using AIHost.Compute;
using AIHost.Config;
using AIHost.ICompute;
using AIHost.ICompute.Vulkan;
using AIHost.ICompute.CUDA;
using AIHost.ICompute.ROCm;
using AIHost.Logging;
using AIHost.Middleware;
using AIHost.Services;
using System.Runtime.InteropServices;
using System.Text.Json;
using Microsoft.Extensions.Logging;

namespace AIHost;

class WebServer
{
    static ILogger logger = null!; // Initialized in Run()
    public static async Task RunAsync(string[] args)
    {
        var builder = WebApplication.CreateBuilder(args);

        builder.Logging.ClearProviders();
        builder.Logging.AddConsole();
        var inMemoryLoggerProvider = new InMemoryLoggerProvider(maxEntries: 900_000);
        builder.Logging.AddProvider(inMemoryLoggerProvider);
        builder.Services.AddSingleton(inMemoryLoggerProvider);

        logger = new ILoggerConsoleWrapper(inMemoryLoggerProvider.CreateLogger(nameof(WebServer)));

        // Locate server.json — prefer data/config, fall back to config/ for backward compat.
        var configPath = Path.Combine(AppContext.BaseDirectory, "data", "config", "server.config.json");
        if (!File.Exists(configPath))
            configPath = Path.Combine(AppContext.BaseDirectory, "config", "server.json");

        var serverConfig = new ServerConfig();

        if (File.Exists(configPath))
        {
            var json = await File.ReadAllTextAsync(configPath);
            serverConfig = JsonSerializer.Deserialize<ServerConfig>(json) ?? new ServerConfig();
            logger.LogInformation("✓ Loaded server configuration from {ConfigPath}", configPath);
        }
        else
        {
            logger.LogWarning("⚠ Server config not found, using defaults");
            Directory.CreateDirectory(Path.GetDirectoryName(configPath)!);
            var defaultJson = JsonSerializer.Serialize(serverConfig, new JsonSerializerOptions { WriteIndented = true });
            await File.WriteAllTextAsync(configPath, defaultJson);
            logger.LogInformation("✓ Created default config at {ConfigPath}", configPath);
        }

        GlobalProfiler.Enabled = serverConfig.ProfilingEnabled;
        logger.LogInformation("  Profiling: {State}", GlobalProfiler.Enabled ? "enabled" : "disabled");


        EnsureDirectoryStructure(serverConfig);

        var minLogLevel = serverConfig.LogLevel.ToLowerInvariant() switch

        {
            "trace" => LogLevel.Trace,
            "debug" => LogLevel.Debug,
            "warning" => LogLevel.Warning,
            "error" => LogLevel.Error,
            _ => LogLevel.Information
        };

        builder.Logging.SetMinimumLevel(minLogLevel);


        builder.Configuration.AddJsonFile(configPath, optional: true, reloadOnChange: false);

        builder.WebHost.ConfigureKestrel(options =>
        {
            options.ListenAnyIP(serverConfig.Port);
        });


        // ── Build info ───────────────────────────────────────────────────────────────
        logger.LogInformation("AIHost build: {Date}", BuildInfo.Date);

        AppDomain.CurrentDomain.UnhandledException += (_, args) =>
        {
            var ex = args.ExceptionObject as Exception;
            logger.LogCritical(ex, "[FATAL] Unhandled: {Message}", ex?.Message ?? "(null)");
        };
        TaskScheduler.UnobservedTaskException += (_, args) =>
        {
            logger.LogCritical(args.Exception!, "[FATAL] Unobserved task: {Message}", args.Exception?.Message ?? "(null)");
            args.SetObserved();
        };

        // ── Silk.NET.Shaderc native library resolver ─────────────────────────────────
        NativeLibrary.SetDllImportResolver(
            typeof(Silk.NET.Shaderc.Shaderc).Assembly,
            (libName, assembly, searchPath) =>
            {
                if (!libName.Contains("shaderc", StringComparison.OrdinalIgnoreCase))
                    return IntPtr.Zero;

                string[] candidates = RuntimeInformation.IsOSPlatform(OSPlatform.Linux)
                    ? ["libshaderc_shared.so.1", "libshaderc_shared.so", "libshaderc.so.1",
               "shaderc_shared", "libshaderc_combined.so.1"]
                    : ["shaderc_shared", "shaderc"];

                foreach (var name in candidates)
                    if (NativeLibrary.TryLoad(name, assembly, searchPath, out var h))
                    {
                        logger.LogInformation("[Shaderc] Loaded native library as '{Name}'", name);
                        return h;
                    }

                logger.LogWarning("[Shaderc] WARNING: could not load shaderc — tried: {Candidates}", string.Join(", ", candidates));
                return IntPtr.Zero;
            });

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

            if (Directory.Exists(dataInitDir))
            {
                Directory.CreateDirectory(dataDir);

                var needsInitialization = !File.Exists(Path.Combine(dataDir, "model.schema.json")) ||
                                          !File.Exists(Path.Combine(dataDir, "model.example.json")) ||
                                          !Directory.Exists(Path.Combine(dataDir, "config"));

                if (needsInitialization)
                {
                    logger.LogInformation("📦 Initializing data directory from templates...");

                    var schemaSource = Path.Combine(dataInitDir, "model.schema.json");
                    var schemaTarget = Path.Combine(dataDir, "model.schema.json");
                    if (File.Exists(schemaSource) && !File.Exists(schemaTarget))
                    {
                        File.Copy(schemaSource, schemaTarget);
                        logger.LogInformation("  ✓ Copied model.schema.json");
                    }

                    var exampleSource = Path.Combine(dataInitDir, "model.example.json");
                    var exampleTarget = Path.Combine(dataDir, "model.example.json");
                    if (File.Exists(exampleSource) && !File.Exists(exampleTarget))
                    {
                        File.Copy(exampleSource, exampleTarget);
                        logger.LogInformation("  ✓ Copied model.example.json");
                    }

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
                                logger.LogInformation("  ✓ Copied config/{Filename}", filename);
                            }
                        }
                    }

                    Directory.CreateDirectory(Path.Combine(dataDir, "models"));
                    Directory.CreateDirectory(Path.Combine(dataDir, "cache"));
                    Directory.CreateDirectory(Path.Combine(dataDir, "logs"));

                    logger.LogInformation("✓ Data directory initialized successfully");
                }
            }
        }

        builder.Services.AddControllers()
            .AddJsonOptions(options =>
            {
                options.JsonSerializerOptions.PropertyNamingPolicy = System.Text.Json.JsonNamingPolicy.SnakeCaseLower;
                options.JsonSerializerOptions.DefaultIgnoreCondition = System.Text.Json.Serialization.JsonIgnoreCondition.WhenWritingNull;
            });

        builder.Services.AddEndpointsApiExplorer();
        builder.Services.AddSwaggerGen();

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
        logger.LogInformation("✓ Initialized compute device: {DeviceName} ({Provider})", deviceName, computeDevice.ProviderName);

        Directory.CreateDirectory(serverConfig.LogsDirectory);
        Directory.CreateDirectory(serverConfig.CacheDirectory);
        Directory.CreateDirectory(serverConfig.ModelsDirectory);

        var requestLogger = new RequestLogger(
            maxLogs: 1000,
            logsDirectory: serverConfig.LogsDirectory,
            persistentLogs: serverConfig.PersistentLogs,
            maxLogFiles: serverConfig.MaxLogFiles
        );
        builder.Services.AddSingleton(requestLogger);
        logger.LogInformation("✓ Request logger initialized (persistent: {Persistent})", serverConfig.PersistentLogs);

        builder.Services.AddSingleton(sp =>
            new ModelManager(
                serverConfig.ModelsDirectory,
                serverConfig.CacheDirectory,
                computeDevice,
                sp.GetRequiredService<ILogger<ModelManager>>()));
        builder.Services.AddSingleton(serverConfig);

        builder.Services.AddSingleton(sp =>
            new DownloadManager(serverConfig.CacheDirectory));

        builder.Services.AddSingleton<IComputeDevice>(computeDevice);

        builder.Services.AddHostedService<ModelAutoUnloadService>();
        builder.Services.AddHostedService<ModelConfigWatcherService>();

        var app = builder.Build();

        AppLogger.Initialize(app.Services.GetRequiredService<ILoggerFactory>());

        logger = AppLogger.Create<WebServer>();

        var modelManager = app.Services.GetRequiredService<ModelManager>();

        if (app.Environment.IsDevelopment())
        {
            app.UseSwagger();
            app.UseSwaggerUI();
        }

        if (serverConfig.EnableCORS)
        {
            app.UseCors();
        }

        if (serverConfig.RateLimitRequestsPerMinute > 0)
        {
            app.UseRateLimit(serverConfig.RateLimitRequestsPerMinute);
            logger.LogInformation("✓ Rate limiting: {Rate} requests/minute", serverConfig.RateLimitRequestsPerMinute);
        }

        app.UseTokenAuth();

        var tokensFile = serverConfig.TokensFile != null ? Path.Combine(AppContext.BaseDirectory, serverConfig.TokensFile) : null;
        if (tokensFile != null && File.Exists(tokensFile))
        {
            logger.LogInformation("✓ Token authentication enabled: {TokensFile}", tokensFile);
        }
        else
        {
            logger.LogWarning("⚠ Token authentication disabled (no tokens file)");
        }

        if (serverConfig.ManageToken != null)
        {
            logger.LogInformation("✓ Management token configured");
        }

        app.UseDefaultFiles();
        app.UseStaticFiles();

        app.UseRouting();

        if (serverConfig.EnableOllamaAPI)
        {
            app.MapControllers();
            logger.LogInformation("✓ Ollama API enabled: /api/*");
        }

        if (serverConfig.EnableOpenAIAPI)
        {
            app.MapControllers();
            logger.LogInformation("✓ OpenAI API enabled: /v1/*");
        }

        app.MapControllers();
        logger.LogInformation("✓ Management API enabled: /manage/*");

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

        logger.LogInformation("🚀 AIHost Server started on http://{Host}:{Port}", serverConfig.Host, serverConfig.Port);
        logger.LogInformation("   Models directory: {Directory}", Path.GetFullPath(serverConfig.ModelsDirectory));
        logger.LogInformation("   Available models: {Count}", modelManager.ListModels().Count());
        logger.LogInformation("📚 API Documentation:");
        logger.LogInformation("   Ollama: http://{Host}:{Port}/api/generate", serverConfig.Host, serverConfig.Port);
        logger.LogInformation("   OpenAI: http://{Host}:{Port}/v1/chat/completions", serverConfig.Host, serverConfig.Port);
        logger.LogInformation("   Swagger: http://{Host}:{Port}/swagger", serverConfig.Host, serverConfig.Port);

        try { await app.RunAsync(); }
        catch (Exception ex)
        {
            logger.LogCritical(ex, "[FATAL] Host crashed: {Message}", ex.Message);
            throw;
        }

        modelManager.Dispose();
        computeDevice.Dispose();

        VulkanGlobalContext.DestroyAll();
    }
    /// <summary>
    /// Ensure all required directories and template files exist
    /// </summary>
    static void EnsureDirectoryStructure(ServerConfig config)
    {
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
                logger.LogInformation("✓ Created directory: {Directory}", dir);
            }
        }

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
                logger.LogInformation("✓ Created template tokens file: {Path}", tokensPath);
                logger.LogWarning("  ⚠ Edit {Filename} to add your API tokens", Path.GetFileName(tokensPath));
            }
        }

        var modelDirs = Directory.Exists(config.ModelsDirectory)
            ? Directory.GetDirectories(config.ModelsDirectory)
            : Array.Empty<string>();

        if (modelDirs.Length == 0)
        {
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

            logger.LogInformation("✓ Created example model config: {Path}", exampleConfigPath);
            logger.LogWarning("  ⚠ Edit model.json to configure your model:");
            logger.LogWarning("     1. Set 'model' path to your .gguf file");
            logger.LogWarning("     2. Update 'name' to match your model");
            logger.LogWarning("     3. Adjust parameters as needed");
        }
    }

}

