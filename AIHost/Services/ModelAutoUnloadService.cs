using System.Collections.Concurrent;
using System.Timers;

namespace AIHost.Services;

/// <summary>
/// Auto-unload service for inactive models
/// </summary>
public class ModelAutoUnloadService : IDisposable
{
    private readonly Config.ModelManager _modelManager;
    private readonly System.Timers.Timer _timer;
    private readonly int _inactiveMinutes;
    private bool _disposed;

    public ModelAutoUnloadService(Config.ModelManager modelManager, int inactiveMinutes)
    {
        _modelManager = modelManager;
        _inactiveMinutes = inactiveMinutes;

        if (_inactiveMinutes > 0)
        {
            _timer = new System.Timers.Timer(TimeSpan.FromMinutes(1).TotalMilliseconds);
            _timer.Elapsed += CheckAndUnloadInactiveModels;
            _timer.Start();
            Console.WriteLine($"✓ Auto-unload enabled: {_inactiveMinutes} minutes of inactivity");
        }
        else
        {
            _timer = null!;
            Console.WriteLine("⚠ Auto-unload disabled");
        }
    }

    private void CheckAndUnloadInactiveModels(object? sender, ElapsedEventArgs e)
    {
        try
        {
            var models = _modelManager.GetLoadedModels();
            var now = DateTime.UtcNow;

            foreach (var kvp in models)
            {
                var model = kvp.Value;
                var lastActivity = model.LastRequestAt ?? model.LoadedAt;
                var inactiveDuration = now - lastActivity;

                // Use model-specific keep_alive if set, otherwise use global setting
                var keepAliveMinutes = model.Config.KeepAliveMinutes ?? _inactiveMinutes;
                
                // If keep_alive is 0, never unload this model
                if (keepAliveMinutes == 0)
                    continue;

                if (inactiveDuration.TotalMinutes >= keepAliveMinutes)
                {
                    Console.WriteLine($"Auto-unloading inactive model: {kvp.Key} (inactive for {inactiveDuration.TotalMinutes:F1} minutes, keep_alive: {keepAliveMinutes}m)");
                    _modelManager.UnloadModel(kvp.Key);
                }
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error in auto-unload service: {ex.Message}");
        }
    }

    public void Dispose()
    {
        if (_disposed) return;

        _timer?.Stop();
        _timer?.Dispose();
        _disposed = true;
    }
}
