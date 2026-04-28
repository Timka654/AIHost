using System.Text.Json;
using AIHost.Config;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;

namespace AIHost.Services;

/// <summary>
/// Watches the models directory for model.json changes and keeps ModelManager in sync.
///
/// Behavior:
///   - model.json created/changed → RegisterOrUpdateConfig → unloads running instance for hot-reload
///   - model.json deleted         → RemoveConfig → unloads running instance
/// </summary>
public sealed class ModelConfigWatcherService : BackgroundService
{
    private readonly ModelManager _modelManager;
    private readonly ServerConfig _serverConfig;
    private readonly ILogger<ModelConfigWatcherService> _logger;
    private FileSystemWatcher? _watcher;

    private static readonly JsonSerializerOptions _jsonOptions = new()
    {
        PropertyNameCaseInsensitive = true
    };

    public ModelConfigWatcherService(
        ModelManager modelManager,
        ServerConfig serverConfig,
        ILogger<ModelConfigWatcherService> logger)
    {
        _modelManager = modelManager;
        _serverConfig = serverConfig;
        _logger = logger;
    }

    protected override Task ExecuteAsync(CancellationToken stoppingToken)
    {
        var modelsDir = _serverConfig.ModelsDirectory;

        if (!Directory.Exists(modelsDir))
        {
            _logger.LogWarning("Models directory not found, config watcher inactive: {Dir}", modelsDir);
            return Task.CompletedTask;
        }

        _watcher = new FileSystemWatcher(modelsDir)
        {
            Filter = "model.json",
            IncludeSubdirectories = true,
            NotifyFilter = NotifyFilters.LastWrite | NotifyFilters.FileName | NotifyFilters.DirectoryName,
            EnableRaisingEvents = true
        };

        _watcher.Changed += OnConfigFileChanged;
        _watcher.Created += OnConfigFileChanged;
        _watcher.Deleted += OnConfigFileDeleted;

        _logger.LogInformation("Model config watcher started: {Dir}", Path.GetFullPath(modelsDir));

        // Block until the host signals cancellation.
        return Task.Delay(Timeout.Infinite, stoppingToken)
            .ContinueWith(_ => { }, TaskScheduler.Default); // disposal handled in StopAsync
    }

    public override Task StopAsync(CancellationToken cancellationToken)
    {
        // Dispose watcher before base stops the background task.
        _watcher?.Dispose();
        _watcher = null;
        return base.StopAsync(cancellationToken);
    }

    private void OnConfigFileChanged(object _, FileSystemEventArgs e)
    {
        var fullPath = e.FullPath;
        var name = e.Name;
        Task.Run(async () =>
        {
            // Small debounce — OS may fire before file write completes.
            await Task.Delay(200);
            try
            {
                var json = await File.ReadAllTextAsync(fullPath);
                var config = JsonSerializer.Deserialize<ModelConfig>(json, _jsonOptions);

                if (config == null || string.IsNullOrWhiteSpace(config.Name))
                {
                    _logger.LogWarning("Skipping invalid model config: {Path}", fullPath);
                    return;
                }

                _modelManager.RegisterOrUpdateConfig(config);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error processing model config {Name}", name);
            }
        });
    }

    private void OnConfigFileDeleted(object _, FileSystemEventArgs e)
    {
        var modelName = Path.GetFileName(Path.GetDirectoryName(e.FullPath));
        if (string.IsNullOrWhiteSpace(modelName)) return;

        _modelManager.RemoveConfig(modelName);
    }
}
