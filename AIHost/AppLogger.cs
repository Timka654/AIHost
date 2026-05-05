using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;

namespace AIHost;

/// <summary>
/// Static logger factory for non-DI classes (Transformer, InferenceEngine, ComputeOps, etc.).
/// Initialized from WebServer.cs after the DI container is built.
/// Returns NullLogger instances before initialization (no-op, safe for pre-startup code).
/// </summary>
public static class AppLogger
{
    private static ILoggerFactory _factory = NullLoggerFactory.Instance;

    /// <summary>Call once after builder.Build() in WebServer.cs.</summary>
    public static void Initialize(ILoggerFactory factory) => _factory = factory;

    public static ILogger<T> Create<T>() => _factory.CreateLogger<T>();
}
