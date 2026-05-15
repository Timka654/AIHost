using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Console;
using Microsoft.Extensions.Options;
using System.Collections.Concurrent;
using System.Text.RegularExpressions;

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

    public InMemoryLogger(string categoryName, int maxEntries = 50000)

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

    public int Count => _entries.Count;

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

    public InMemoryLoggerProvider(int maxEntries = 50000)

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
    public LogEntry[] GetEntries(out int count, string? categoryFilter = null, int? skip = null, int? take = null, string? searchExpression = null)
    {
        var selector = _loggers.AsEnumerable();

        if (!string.IsNullOrEmpty(categoryFilter))
            selector = selector.Where(kv => kv.Key.Contains(categoryFilter, StringComparison.OrdinalIgnoreCase));


        Regex? regex = null;

        IEnumerable<LogEntry> _logsSelector = selector
            .SelectMany(kv => kv.Value.GetEntries())
            .OrderBy(e => e.Timestamp);

        if (!string.IsNullOrEmpty(searchExpression))
        {
            regex = new Regex(searchExpression, RegexOptions.IgnoreCase | RegexOptions.Compiled);

            _logsSelector = _logsSelector.Where(e => regex.IsMatch(e.Message));
        }

        count = _logsSelector.Count();

        return _logsSelector
            .Skip(skip ?? 0)
            .Take(take ?? 2000)
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

internal class ILoggerConsoleWrapper(ILogger logger) : ILogger
{
    public IDisposable? BeginScope<TState>(TState state) where TState : notnull
    {
        return null;
    }

    public bool IsEnabled(LogLevel logLevel)
        => true;

    public void Log<TState>(LogLevel logLevel, EventId eventId, TState state, Exception? exception, Func<TState, Exception?, string> formatter)
    {
        logger.Log<TState>(logLevel, eventId, state, exception, formatter);
        Console.WriteLine(formatter(state, exception));
    }
}