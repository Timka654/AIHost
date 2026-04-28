using System.Diagnostics;

namespace AIHost.Compute;

/// <summary>
/// Profiler for measuring compute operation performance
/// </summary>
public class ComputeProfiler
{
    private readonly Dictionary<string, ProfileEntry> _entries = new();
    private readonly object _lockObject = new();

    /// <summary>
    /// Start measuring an operation
    /// </summary>
    public ProfileScope Begin(string operationName)
    {
        return new ProfileScope(this, operationName);
    }

    internal void RecordOperation(string name, long elapsedTicks)
    {
        lock (_lockObject)
        {
            if (!_entries.TryGetValue(name, out var entry))
            {
                entry = new ProfileEntry { Name = name };
                _entries[name] = entry;
            }

            entry.CallCount++;
            entry.TotalTicks += elapsedTicks;
            entry.MinTicks = entry.CallCount == 1 ? elapsedTicks : Math.Min(entry.MinTicks, elapsedTicks);
            entry.MaxTicks = Math.Max(entry.MaxTicks, elapsedTicks);
        }
    }

    /// <summary>
    /// Get all profiling results
    /// </summary>
    public ProfileResult[] GetResults()
    {
        lock (_lockObject)
        {
            return _entries.Values
                .Select(e => new ProfileResult
                {
                    Name = e.Name,
                    CallCount = e.CallCount,
                    TotalMilliseconds = e.TotalTicks * 1000.0 / Stopwatch.Frequency,
                    AverageMilliseconds = (e.TotalTicks / (double)e.CallCount) * 1000.0 / Stopwatch.Frequency,
                    MinMilliseconds = e.MinTicks * 1000.0 / Stopwatch.Frequency,
                    MaxMilliseconds = e.MaxTicks * 1000.0 / Stopwatch.Frequency
                })
                .OrderByDescending(r => r.TotalMilliseconds)
                .ToArray();
        }
    }

    /// <summary>
    /// Get a summary report of profiling results
    /// </summary>
    public string GetSummary()
    {
        var results = GetResults();
        if (results.Length == 0)
            return "No profiling data collected";

        var sb = new System.Text.StringBuilder();
        sb.AppendLine("=== Compute Profiling Summary ===");
        sb.AppendLine($"{"Operation",-30} {"Calls",8} {"Total(ms)",12} {"Avg(ms)",10} {"Min(ms)",10} {"Max(ms)",10}");
        sb.AppendLine(new string('-', 90));

        foreach (var result in results)
        {
            sb.AppendLine($"{result.Name,-30} {result.CallCount,8} {result.TotalMilliseconds,12:F3} " +
                         $"{result.AverageMilliseconds,10:F3} {result.MinMilliseconds,10:F3} {result.MaxMilliseconds,10:F3}");
        }

        double totalTime = results.Sum(r => r.TotalMilliseconds);
        sb.AppendLine(new string('-', 90));
        sb.AppendLine($"{"TOTAL",-30} {results.Sum(r => r.CallCount),8} {totalTime,12:F3}");

        return sb.ToString();
    }

    /// <summary>
    /// Clear all profiling data
    /// </summary>
    public void Clear()
    {
        lock (_lockObject)
        {
            _entries.Clear();
        }
    }
}

/// <summary>
/// Scoped profiler that automatically records elapsed time
/// </summary>
public struct ProfileScope : IDisposable
{
    private readonly ComputeProfiler _profiler;
    private readonly string _name;
    private readonly long _startTicks;

    internal ProfileScope(ComputeProfiler profiler, string name)
    {
        _profiler = profiler;
        _name = name;
        _startTicks = Stopwatch.GetTimestamp();
    }

    public void Dispose()
    {
        long elapsed = Stopwatch.GetTimestamp() - _startTicks;
        _profiler.RecordOperation(_name, elapsed);
    }
}

internal class ProfileEntry
{
    public string Name { get; set; } = string.Empty;
    public long CallCount { get; set; }
    public long TotalTicks { get; set; }
    public long MinTicks { get; set; }
    public long MaxTicks { get; set; }
}

/// <summary>
/// Profiling result for a single operation
/// </summary>
public struct ProfileResult
{
    public string Name { get; set; }
    public long CallCount { get; set; }
    public double TotalMilliseconds { get; set; }
    public double AverageMilliseconds { get; set; }
    public double MinMilliseconds { get; set; }
    public double MaxMilliseconds { get; set; }

    public override string ToString()
    {
        return $"{Name}: {CallCount} calls, {AverageMilliseconds:F3}ms avg " +
               $"({MinMilliseconds:F3}-{MaxMilliseconds:F3}ms), {TotalMilliseconds:F3}ms total";
    }
}
