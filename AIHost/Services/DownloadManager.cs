using System.Collections.Concurrent;

namespace AIHost.Services;

public class DownloadManager
{
    private readonly ConcurrentDictionary<string, DownloadTask> _downloads = new();
    private readonly HttpClient _httpClient = new();
    private readonly string _cacheDirectory;

    public DownloadManager(string cacheDirectory)
    {
        // Normalize path (resolve ./ and ../ to absolute path)
        _cacheDirectory = Path.GetFullPath(cacheDirectory);
        Directory.CreateDirectory(_cacheDirectory);
    }

    public string CacheDirectory => _cacheDirectory;

    public string StartDownload(string url, string? filename = null)
    {
        var downloadId = Guid.NewGuid().ToString();
        var targetFilename = filename ?? Path.GetFileName(new Uri(url).LocalPath);
        var targetPath = Path.Combine(_cacheDirectory, targetFilename);

        var downloadTask = new DownloadTask
        {
            Id = downloadId,
            Url = url,
            Filename = targetFilename,
            TargetPath = targetPath,
            Status = DownloadStatus.Pending,
            StartTime = DateTime.UtcNow
        };

        _downloads[downloadId] = downloadTask;

        // Start download in background
        _ = Task.Run(async () => await ExecuteDownload(downloadId));

        return downloadId;
    }

    private async Task ExecuteDownload(string downloadId)
    {
        if (!_downloads.TryGetValue(downloadId, out var task))
            return;

        try
        {
            task.Status = DownloadStatus.Downloading;

            using var response = await _httpClient.GetAsync(task.Url, HttpCompletionOption.ResponseHeadersRead, task.CancellationTokenSource.Token);
            response.EnsureSuccessStatusCode();

            task.TotalBytes = response.Content.Headers.ContentLength ?? 0;

            using var contentStream = await response.Content.ReadAsStreamAsync(task.CancellationTokenSource.Token);
            using var fileStream = new FileStream(task.TargetPath, FileMode.Create, FileAccess.Write, FileShare.None, 8192, true);

            var buffer = new byte[8192];
            int bytesRead;
            long totalRead = 0;

            while ((bytesRead = await contentStream.ReadAsync(buffer, task.CancellationTokenSource.Token)) > 0)
            {
                await fileStream.WriteAsync(buffer.AsMemory(0, bytesRead), task.CancellationTokenSource.Token);
                totalRead += bytesRead;
                task.DownloadedBytes = totalRead;
                task.Progress = task.TotalBytes > 0 ? (double)totalRead / task.TotalBytes * 100 : 0;
            }

            task.Status = DownloadStatus.Completed;
            task.EndTime = DateTime.UtcNow;
        }
        catch (OperationCanceledException)
        {
            task.Status = DownloadStatus.Cancelled;
            task.Error = "Download was cancelled";
            
            // Clean up partial file
            if (File.Exists(task.TargetPath))
            {
                try { File.Delete(task.TargetPath); } catch { }
            }
        }
        catch (Exception ex)
        {
            task.Status = DownloadStatus.Failed;
            task.Error = ex.Message;
            task.EndTime = DateTime.UtcNow;
        }
    }

    public DownloadTask? GetDownload(string downloadId)
    {
        _downloads.TryGetValue(downloadId, out var task);
        return task;
    }

    public List<DownloadTask> GetAllDownloads()
    {
        return _downloads.Values.ToList();
    }

    public bool CancelDownload(string downloadId)
    {
        if (_downloads.TryGetValue(downloadId, out var task))
        {
            task.CancellationTokenSource.Cancel();
            return true;
        }
        return false;
    }

    public void CleanupCompleted()
    {
        var completedIds = _downloads
            .Where(kvp => kvp.Value.Status == DownloadStatus.Completed || 
                         kvp.Value.Status == DownloadStatus.Failed ||
                         kvp.Value.Status == DownloadStatus.Cancelled)
            .Select(kvp => kvp.Key)
            .ToList();

        foreach (var id in completedIds)
        {
            _downloads.TryRemove(id, out _);
        }
    }
}

public class DownloadTask
{
    public string Id { get; set; } = string.Empty;
    public string Url { get; set; } = string.Empty;
    public string Filename { get; set; } = string.Empty;
    public string TargetPath { get; set; } = string.Empty;
    public DownloadStatus Status { get; set; }
    public long TotalBytes { get; set; }
    public long DownloadedBytes { get; set; }
    public double Progress { get; set; }
    public string? Error { get; set; }
    public DateTime StartTime { get; set; }
    public DateTime? EndTime { get; set; }
    public CancellationTokenSource CancellationTokenSource { get; } = new();
}

public enum DownloadStatus
{
    Pending,
    Downloading,
    Completed,
    Failed,
    Cancelled
}
