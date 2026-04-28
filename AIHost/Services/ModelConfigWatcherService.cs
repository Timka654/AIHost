using System.Text.Json;
using AIHost.Config;
using Microsoft.Extensions.Hosting;

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

    private static readonly JsonSerializerOptions _jsonOptions = new()
    {
        PropertyNameCaseInsensitive = true
    };

    public ModelConfigWatcherService(ModelManager modelManager, ServerConfig serverConfig)
    {
        _modelManager = modelManager;
        _serverConfig = serverConfig;
    }

    protected override Task ExecuteAsync(CancellationToken stoppingToken)
    {
        var modelsDir = _serverConfig.ModelsDirectory;

        if (!Directory.Exists(modelsDir))
        {
            Console.WriteLine($"⚠ Models directory not found, config watcher inactive: {modelsDir}");
            return Task.CompletedTask;
        }

        var watcher = new FileSystemWatcher(modelsDir)
        {
            Filter = "model.json",
            IncludeSubdirectories = true,
            NotifyFilter = NotifyFilters.LastWrite | NotifyFilters.FileName | NotifyFilters.DirectoryName,
            EnableRaisingEvents = true
        };

        watcher.Changed += OnConfigFileChanged;
        watcher.Created += OnConfigFileChanged;
        watcher.Deleted += OnConfigFileDeleted;

        Console.WriteLine($"✓ Model config watcher started: {Path.GetFullPath(modelsDir)}");

        // Hold the watcher alive until the host stops, then clean up.
        return Task.Delay(Timeout.Infinite, stoppingToken)
            .ContinueWith(_ => watcher.Dispose(), TaskScheduler.Default);
    }

    private void OnConfigFileChanged(object _, FileSystemEventArgs e)
    {
        // Debounce without blocking the threadpool: fire-and-forget async task.
        // The OS may deliver the event before the file write is complete.
        var fullPath = e.FullPath;
        var name = e.Name;
        Task.Run(async () =>
        {
            await Task.Delay(200);
            try
            {
                var json = await File.ReadAllTextAsync(fullPath);
                var config = JsonSerializer.Deserialize<ModelConfig>(json, _jsonOptions);

                if (config == null || string.IsNullOrWhiteSpace(config.Name))
                {
                    Console.WriteLine($"⚠ Skipping invalid model config: {fullPath}");
                    return;
                }

                _modelManager.RegisterOrUpdateConfig(config);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"⚠ Error processing model config {name}: {ex.Message}");
            }
        });
    }

    private void OnConfigFileDeleted(object _, FileSystemEventArgs e)
    {
        // Derive model name from the parent directory name (models/<name>/model.json).
        var modelName = Path.GetFileName(Path.GetDirectoryName(e.FullPath));
        if (string.IsNullOrWhiteSpace(modelName)) return;

        _modelManager.RemoveConfig(modelName);
    }
}
