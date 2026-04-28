using System.Text.Json;
using System.Text.Json.Serialization;

namespace AIHost.External;

/// <summary>
/// HuggingFace model info
/// </summary>
public class HFModelInfo
{
    [JsonPropertyName("id")]
    public string Id { get; set; } = "";

    [JsonPropertyName("author")]
    public string Author { get; set; } = "";

    [JsonPropertyName("downloads")]
    public long Downloads { get; set; }

    [JsonPropertyName("likes")]
    public int Likes { get; set; }

    [JsonPropertyName("tags")]
    public List<string> Tags { get; set; } = new();
}

/// <summary>
/// HuggingFace file info
/// </summary>
public class HFFileInfo
{
    [JsonPropertyName("rfilename")]
    public string Name { get; set; } = "";

    [JsonPropertyName("size")]
    public long Size { get; set; }
}

/// <summary>
/// HuggingFace model downloader
/// </summary>
public class HuggingFaceDownloader : IDisposable
{
    private const string HF_API = "https://huggingface.co/api";
    private readonly HttpClient _client;
    private bool _disposed;

    public HuggingFaceDownloader()
    {
        _client = new HttpClient();
        _client.Timeout = TimeSpan.FromHours(2);
        _client.DefaultRequestHeaders.Add("User-Agent", "AIHost/1.0");
    }

    public void Dispose()
    {
        if (_disposed) return;
        _client.Dispose();
        _disposed = true;
    }

    /// <summary>
    /// Search models on HuggingFace
    /// </summary>
    public async Task<List<HFModelInfo>> SearchModelsAsync(string query, string filter = "gguf", int limit = 20)
    {
        try
        {
            var url = $"{HF_API}/models?search={Uri.EscapeDataString(query)}&filter={filter}&limit={limit}&sort=downloads";
            var response = await _client.GetStringAsync(url);
            var models = JsonSerializer.Deserialize<List<HFModelInfo>>(response) ?? new();
            return models;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"HuggingFace search failed: {ex.Message}");
            return new List<HFModelInfo>();
        }
    }

    /// <summary>
    /// Get model files list
    /// </summary>
    public async Task<List<HFFileInfo>> GetModelFilesAsync(string modelId)
    {
        try
        {
            var url = $"{HF_API}/models/{modelId}/tree/main";
            var response = await _client.GetStringAsync(url);
            var files = JsonSerializer.Deserialize<List<HFFileInfo>>(response) ?? new();
            return files.Where(f => f.Name.EndsWith(".gguf", StringComparison.OrdinalIgnoreCase)).ToList();
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Failed to get model files: {ex.Message}");
            return new List<HFFileInfo>();
        }
    }

    /// <summary>
    /// Get direct download URL for a file
    /// </summary>
    public string GetDownloadUrl(string modelId, string fileName)
    {
        return $"https://huggingface.co/{modelId}/resolve/main/{fileName}";
    }

    /// <summary>
    /// Download model file with progress
    /// </summary>
    public async Task DownloadModelFileAsync(string modelId, string fileName, string destPath, IProgress<double>? progress = null)
    {
        var url = GetDownloadUrl(modelId, fileName);
        var directory = Path.GetDirectoryName(destPath);
        if (!string.IsNullOrEmpty(directory))
        {
            Directory.CreateDirectory(directory);
        }

        Console.WriteLine($"Downloading {modelId}/{fileName}...");

        using var response = await _client.GetAsync(url, HttpCompletionOption.ResponseHeadersRead);
        response.EnsureSuccessStatusCode();

        var totalBytes = response.Content.Headers.ContentLength ?? 0;
        var downloadedBytes = 0L;

        using var contentStream = await response.Content.ReadAsStreamAsync();
        using var fileStream = File.Create(destPath);

        var buffer = new byte[8192];
        int bytesRead;

        while ((bytesRead = await contentStream.ReadAsync(buffer, 0, buffer.Length)) > 0)
        {
            await fileStream.WriteAsync(buffer, 0, bytesRead);
            downloadedBytes += bytesRead;

            if (totalBytes > 0 && progress != null)
            {
                var percent = (double)downloadedBytes / totalBytes * 100;
                progress.Report(percent);
            }

            if (downloadedBytes % (10 * 1024 * 1024) == 0) // Log every 10MB
            {
                Console.WriteLine($"Downloaded {downloadedBytes / 1024 / 1024}MB / {totalBytes / 1024 / 1024}MB");
            }
        }

        Console.WriteLine($"Download complete: {destPath}");
    }
}
