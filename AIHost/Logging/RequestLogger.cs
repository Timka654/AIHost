using System.Collections.Concurrent;
using System.Text.Json;

namespace AIHost.Logging;

/// <summary>
/// Log entry for API requests
/// </summary>
public class RequestLogEntry
{
    public DateTime Timestamp { get; set; }
    public string Endpoint { get; set; } = "";
    public string Method { get; set; } = "";
    public string ModelName { get; set; } = "";
    public string Prompt { get; set; } = "";
    public int TokensGenerated { get; set; }
    public double DurationMs { get; set; }
    public double TPS { get; set; }
    public bool Success { get; set; }
    public string? Error { get; set; }
}

/// <summary>
/// Request logger with in-memory circular buffer and optional disk persistence
/// </summary>
public class RequestLogger
{
    private readonly ConcurrentQueue<RequestLogEntry> _logs = new();
    private readonly int _maxLogs;
    private readonly string? _logsDirectory;
    private readonly bool _persistentLogs;
    private readonly int _maxLogFiles;
    private readonly object _fileLock = new();

    public RequestLogger(int maxLogs = 1000, string? logsDirectory = null, bool persistentLogs = true, int maxLogFiles = 10)
    {
        _maxLogs = maxLogs;
        _logsDirectory = logsDirectory;
        _persistentLogs = persistentLogs;
        _maxLogFiles = maxLogFiles;

        if (_persistentLogs && !string.IsNullOrEmpty(_logsDirectory))
        {
            Directory.CreateDirectory(_logsDirectory);
            Console.WriteLine($"✓ Persistent logs enabled: {_logsDirectory}");
        }
    }

    /// <summary>
    /// Add request log entry
    /// </summary>
    public void LogRequest(RequestLogEntry entry)
    {
        _logs.Enqueue(entry);

        // Keep only last N logs
        while (_logs.Count > _maxLogs)
        {
            _logs.TryDequeue(out _);
        }

        // Write to disk if enabled
        if (_persistentLogs && !string.IsNullOrEmpty(_logsDirectory))
        {
            Task.Run(() => WriteLogToDisk(entry));
        }
    }

    /// <summary>
    /// Write log entry to disk
    /// </summary>
    private void WriteLogToDisk(RequestLogEntry entry)
    {
        try
        {
            var dateStr = DateTime.UtcNow.ToString("yyyy-MM-dd");
            var logFile = Path.Combine(_logsDirectory!, $"requests-{dateStr}.jsonl");

            var json = JsonSerializer.Serialize(entry);
            lock (_fileLock)
            {
                File.AppendAllText(logFile, json + Environment.NewLine);
            }

            // Cleanup old log files
            CleanupOldLogs();
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Failed to write log to disk: {ex.Message}");
        }
    }

    /// <summary>
    /// Remove old log files beyond max limit
    /// </summary>
    private void CleanupOldLogs()
    {
        try
        {
            var logFiles = Directory.GetFiles(_logsDirectory!, "requests-*.jsonl")
                .OrderByDescending(f => f)
                .ToList();

            if (logFiles.Count > _maxLogFiles)
            {
                foreach (var oldFile in logFiles.Skip(_maxLogFiles))
                {
                    File.Delete(oldFile);
                }
            }
        }
        catch
        {
            // Ignore cleanup errors
        }
    }

    /// <summary>
    /// Get recent logs (newest first)
    /// </summary>
    public List<RequestLogEntry> GetRecentLogs(int count = 100)
    {
        return _logs
            .Reverse()
            .Take(count)
            .ToList();
    }

    /// <summary>
    /// Get logs for specific model
    /// </summary>
    public List<RequestLogEntry> GetModelLogs(string modelName, int count = 100)
    {
        return _logs
            .Where(l => l.ModelName == modelName)
            .Reverse()
            .Take(count)
            .ToList();
    }

    /// <summary>
    /// Load logs from disk for a specific date
    /// </summary>
    public List<RequestLogEntry> LoadLogsFromDisk(DateTime date, int maxCount = 1000)
    {
        if (string.IsNullOrEmpty(_logsDirectory))
            return new List<RequestLogEntry>();

        try
        {
            var dateStr = date.ToString("yyyy-MM-dd");
            var logFile = Path.Combine(_logsDirectory, $"requests-{dateStr}.jsonl");

            if (!File.Exists(logFile))
                return new List<RequestLogEntry>();

            var logs = new List<RequestLogEntry>();
            var lines = File.ReadLines(logFile).Reverse().Take(maxCount);

            foreach (var line in lines)
            {
                if (string.IsNullOrWhiteSpace(line))
                    continue;

                try
                {
                    var entry = JsonSerializer.Deserialize<RequestLogEntry>(line);
                    if (entry != null)
                        logs.Add(entry);
                }
                catch
                {
                    // Skip invalid lines
                }
            }

            return logs;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Failed to load logs from disk: {ex.Message}");
            return new List<RequestLogEntry>();
        }
    }

    /// <summary>
    /// Clear all logs
    /// </summary>
    public void Clear()
    {
        _logs.Clear();
    }
}
