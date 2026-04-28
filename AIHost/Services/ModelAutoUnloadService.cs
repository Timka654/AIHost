using AIHost.Config;
using Microsoft.Extensions.Hosting;

namespace AIHost.Services;

/// <summary>
/// Auto-unloads models that have been idle longer than their keep-alive threshold.
/// Runs as a hosted background service integrated with the ASP.NET Core lifetime.
/// </summary>
public sealed class ModelAutoUnloadService : BackgroundService
{
    private readonly ModelManager _modelManager;
    private readonly ServerConfig _serverConfig;

    public ModelAutoUnloadService(ModelManager modelManager, ServerConfig serverConfig)
    {
        _modelManager = modelManager;
        _serverConfig = serverConfig;
    }

    protected override async Task ExecuteAsync(CancellationToken stoppingToken)
    {
        if (_serverConfig.AutoUnloadMinutes <= 0)
        {
            Console.WriteLine("⚠ Auto-unload disabled");
            return;
        }

        Console.WriteLine($"✓ Auto-unload enabled: {_serverConfig.AutoUnloadMinutes} minutes of inactivity");

        while (!stoppingToken.IsCancellationRequested)
        {
            await Task.Delay(TimeSpan.FromMinutes(1), stoppingToken);
            CheckAndUnloadInactiveModels();
        }
    }

    private void CheckAndUnloadInactiveModels()
    {
        try
        {
            var models = _modelManager.GetLoadedModels();
            var now = DateTime.UtcNow;

            foreach (var (name, model) in models)
            {
                var keepAliveMinutes = model.Config.KeepAliveMinutes ?? _serverConfig.AutoUnloadMinutes;
                if (keepAliveMinutes == 0) continue;

                var lastActivity = model.LastRequestAt ?? model.LoadedAt;
                var idleMinutes = (now - lastActivity).TotalMinutes;

                if (idleMinutes >= keepAliveMinutes)
                {
                    Console.WriteLine($"Auto-unloading {name} (idle {idleMinutes:F1}m, threshold {keepAliveMinutes}m)");
                    _modelManager.UnloadModel(name);
                }
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error in auto-unload service: {ex.Message}");
        }
    }
}
