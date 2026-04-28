using AIHost.Config;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;

namespace AIHost.Services;

/// <summary>
/// Auto-unloads models that have been idle longer than their keep-alive threshold.
/// Runs as a hosted background service integrated with the ASP.NET Core lifetime.
/// </summary>
public sealed class ModelAutoUnloadService : BackgroundService
{
    private readonly ModelManager _modelManager;
    private readonly ServerConfig _serverConfig;
    private readonly ILogger<ModelAutoUnloadService> _logger;

    public ModelAutoUnloadService(
        ModelManager modelManager,
        ServerConfig serverConfig,
        ILogger<ModelAutoUnloadService> logger)
    {
        _modelManager = modelManager;
        _serverConfig = serverConfig;
        _logger = logger;
    }

    protected override async Task ExecuteAsync(CancellationToken stoppingToken)
    {
        if (_serverConfig.AutoUnloadMinutes <= 0)
        {
            _logger.LogWarning("Auto-unload disabled");
            return;
        }

        _logger.LogInformation("Auto-unload enabled: {Minutes} minutes of inactivity", _serverConfig.AutoUnloadMinutes);

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
                    _logger.LogInformation(
                        "Auto-unloading {Model} (idle {Idle:F1}m, threshold {Threshold}m)",
                        name, idleMinutes, keepAliveMinutes);
                    _modelManager.UnloadModel(name);
                }
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error in auto-unload service");
        }
    }
}
