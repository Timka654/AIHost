using Microsoft.Extensions.Logging;
using System.Collections.Concurrent;

namespace AIHost.Logging;

/// <summary>
/// In-memory logger that stores recent log entries for debugging via API.
/// Thread-safe, bounded ring buffer to avoid memory leaks.
/// </summary>
public class InMemoryLogger : ILogger
{
    private readonly string _categoryName;
    private readonly int _maxEntries;
    private readonly ConcurrentQueue<LogEntry> _entries = new();

    public InMemoryLogger(string categoryName, int maxEntries = 5000)
    {
        _categoryName = categoryName;
        _maxEntries = maxEntries;
    }

    public IDisposable? BeginScope<TState>(TState state) where TState : notnull => null;

    public bool IsEnabled(LogLevel logLevel) => true;

    public void Log<TState>(LogLevel logLevel, EventId eventId, TState state,
                            Exception? exception, Func<TState, Exception?, string> formatter)
    {
        if (!IsEnabled(logLevel)) return;

        var entry = new LogEntry
        {
            Timestamp = DateTime.UtcNow,
            Level = logLevel,
            Category = _categoryName,
            Message = formatter(state, exception),
            Exception = exception?.ToString()
        };

        _entries.Enqueue(entry);

        // Trim if exceeded max
        while (_entries.Count > _maxEntries)
            _entries.TryDequeue(out _);
    }

    /// <summary>Get all stored log entries (newest last).</summary>
    public LogEntry[] GetEntries() => _entries.ToArray();

    /// <summary>Clear all stored entries.</summary>
    public void Clear() { while (_entries.TryDequeue(out _)) { } }
}

public class LogEntry
{
    public DateTime Timestamp { get; set; }
    public LogLevel Level { get; set; }
    public string Category { get; set; } = "";
    public string Message { get; set; } = "";
    public string? Exception { get; set; }
}

/// <summary>
/// ILoggerProvider that creates InMemoryLogger instances.
/// Register in DI to capture all log output.
/// </summary>
public class InMemoryLoggerProvider : ILoggerProvider
{
    private readonly ConcurrentDictionary<string, InMemoryLogger> _loggers = new();
    private readonly int _maxEntries;

    public InMemoryLoggerProvider(int maxEntries = 5000)
    {
        _maxEntries = maxEntries;
    }

    public ILogger CreateLogger(string categoryName)
    {
        return _loggers.GetOrAdd(categoryName, name => new InMemoryLogger(name, _maxEntries));
    }

    /// <summary>Get all log entries from all categories, newest last.</summary>
    public LogEntry[] GetAllEntries()
    {
        return _loggers.Values
            .SelectMany(l => l.GetEntries())
            .OrderBy(e => e.Timestamp)
            .ToArray();
    }

    /// <summary>Get entries filtered by category (substring match).</summary>
    public LogEntry[] GetEntries(string categoryFilter)
    {
        return _loggers
            .Where(kv => kv.Key.Contains(categoryFilter, StringComparison.OrdinalIgnoreCase))
            .SelectMany(kv => kv.Value.GetEntries())
            .OrderBy(e => e.Timestamp)
            .ToArray();
    }

    /// <summary>Clear all loggers.</summary>
    public void ClearAll()
    {
        foreach (var logger in _loggers.Values)
            logger.Clear();
    }

    public void Dispose() { _loggers.Clear(); }
}
