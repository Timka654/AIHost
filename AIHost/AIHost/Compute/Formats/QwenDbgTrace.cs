using AIHost.ICompute;
using Microsoft.Extensions.Logging;

namespace AIHost.Compute.Formats;

/// <summary>
/// One-shot debug tracer for Qwen inference.
/// Logs tensor values at key pipeline stages — fires ONCE per (tag, layer) pair per process lifetime.
/// Category: QwenDbgTrace → query via /manage/debug-logs?category=QwenDbgTrace
/// </summary>
public sealed class QwenDbgTrace
{
    private static readonly ILogger _log = AppLogger.Create<QwenDbgTrace>();
    private static readonly HashSet<string> _logged = new();

    /// <summary>Returns true the FIRST time this (tag, layer) pair is encountered — false on all subsequent calls.</summary>
    public static bool Once(string tag, int layer)
    {
        string key = $"{tag}:{layer}";
        lock (_logged) return _logged.Add(key);
    }

    /// <summary>
    /// Reads first <paramref name="n"/> elements of row 0 of a 2-D F32 tensor and logs them.
    /// Forces GPU→CPU sync — use sparingly (debug only, one call per layer).
    /// </summary>
    public static void Row0(string tag, Tensor t, int n = 12)
    {
        try
        {
            if (t.Shape.Rank != 2)
            {
                _log.LogWarning("[QDBG] {Tag} rank={Rank} shape={Shape} (not 2D, skipping row read)", tag, t.Shape.Rank, t.Shape);
                return;
            }
            var data = t.ReadRow(0);
            int take = Math.Min(n, data.Length);
            var vals = string.Join(", ", data.Take(take).Select(v => v.ToString("G5")));
            _log.LogWarning("[QDBG] {Tag} shape=[{R}×{C}] row0[0..{N}]: [{Vals}]",
                tag, t.Shape[0], t.Shape[1], take, vals);
        }
        catch (Exception ex)
        {
            _log.LogWarning("[QDBG] {Tag} read failed: {Err}", tag, ex.Message);
        }
    }

    /// <summary>Log shape only (no GPU sync).</summary>
    public static void Shape(string tag, Tensor t)
    {
        _log.LogWarning("[QDBG] {Tag} shape={Shape}", tag, t.Shape);
    }

    /// <summary>Log a plain message.</summary>
    public static void Msg(string msg) => _log.LogWarning("[QDBG] {Msg}", msg);
}
