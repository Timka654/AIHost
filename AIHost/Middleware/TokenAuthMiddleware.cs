using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.Logging;
using AIHost.Config;

namespace AIHost.Middleware;

/// <summary>
/// Middleware for token-based authentication with dynamic token reloading
/// </summary>
public class TokenAuthMiddleware : IDisposable
{
    private readonly RequestDelegate _next;
    private readonly ILogger<TokenAuthMiddleware> _logger;
    private readonly HashSet<string> _validTokens = new();
    private readonly string? _manageToken;
    private readonly string? _tokensFilePath;
    private readonly object _lockObj = new();
    private readonly FileSystemWatcher? _fileWatcher;
    private bool _disposed;

    public TokenAuthMiddleware(RequestDelegate next, ServerConfig serverConfig, ILogger<TokenAuthMiddleware> logger)
    {
        _logger = logger;
        _next = next;
        _manageToken = serverConfig.ManageToken;
        
        var tokensFile = serverConfig.TokensFile;
        
        // Setup file path and load initial tokens
        if (!string.IsNullOrEmpty(tokensFile))
        {
            _tokensFilePath = Path.IsPathRooted(tokensFile) 
                ? tokensFile 
                : Path.Combine(AppContext.BaseDirectory, tokensFile);
            
            // Load initial tokens
            LoadTokens(_tokensFilePath);
            
            // Setup file watcher if directory exists
            var directory = Path.GetDirectoryName(_tokensFilePath);
            var fileName = Path.GetFileName(_tokensFilePath);
            
            if (!string.IsNullOrEmpty(directory) && Directory.Exists(directory))
            {
                try
                {
                    _fileWatcher = new FileSystemWatcher(directory, fileName)
                    {
                        NotifyFilter = NotifyFilters.LastWrite | NotifyFilters.Size | NotifyFilters.CreationTime,
                        EnableRaisingEvents = true
                    };
                    
                    _fileWatcher.Changed += OnTokensFileChanged;
                    _fileWatcher.Created += OnTokensFileChanged;
                    _fileWatcher.Deleted += OnTokensFileDeleted;
                    
                    _logger.LogInformation("Token file watcher enabled: {File}", fileName);
                }
                catch (Exception ex)
                {
                    _logger.LogWarning(ex, "Failed to setup token file watcher");
                }
            }
        }
    }
    
    private void OnTokensFileChanged(object sender, FileSystemEventArgs e)
    {
        if (string.IsNullOrEmpty(_tokensFilePath)) return;

        var path = _tokensFilePath;
        Task.Run(async () =>
        {
            // Small debounce — OS may fire before file write completes.
            await Task.Delay(100);
            try
            {
                LoadTokens(path);
                int count;
                lock (_lockObj) { count = _validTokens.Count; }
                _logger.LogInformation("Tokens reloaded: {Count} token(s)", count);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Failed to reload tokens");
            }
        });
    }
    
    private void OnTokensFileDeleted(object sender, FileSystemEventArgs e)
    {
        lock (_lockObj)
        {
            _validTokens.Clear();
        }
        _logger.LogWarning("Tokens file deleted — authentication disabled");
    }

    private void LoadTokens(string filePath)
    {
        try
        {
            if (!File.Exists(filePath))
            {
                lock (_lockObj)
                {
                    _validTokens.Clear();
                }
                _logger.LogWarning("Tokens file not found: {Path}", filePath);
                return;
            }
            
            var lines = File.ReadAllLines(filePath);
            var newTokens = new HashSet<string>();
            
            foreach (var line in lines)
            {
                var token = line.Trim();
                if (!string.IsNullOrEmpty(token) && !token.StartsWith("#"))
                {
                    newTokens.Add(token);
                }
            }
            
            lock (_lockObj)
            {
                _validTokens.Clear();
                foreach (var token in newTokens)
                {
                    _validTokens.Add(token);
                }
            }
            
            var tokenCount = newTokens.Count;
            if (tokenCount == 0)
                _logger.LogWarning("No valid tokens found — authentication disabled");
            else
                _logger.LogInformation("Loaded {Count} valid token(s)", tokenCount);
        }
        catch (Exception ex)
        {
            Console.WriteLine($"❌ Failed to load tokens: {ex.Message}");
        }
    }
    
    /// <summary>
    /// Check if authentication is required (based on whether we have valid tokens)
    /// </summary>
    private bool RequireAuth
    {
        get
        {
            lock (_lockObj)
            {
                return _validTokens.Count > 0;
            }
        }
    }

    public async Task InvokeAsync(HttpContext context)
    {
        var path = context.Request.Path.Value ?? "";

        // Skip auth for health check and Swagger
        if (path == "/" || path.StartsWith("/swagger") || path.StartsWith("/v1/models") || path.StartsWith("/api/tags"))
        {
            await _next(context);
            return;
        }

        // Management endpoints require manage token
        if (path.StartsWith("/manage"))
        {
            if (!string.IsNullOrEmpty(_manageToken))
            {
                var authHeader = context.Request.Headers["Authorization"].FirstOrDefault();
                var providedToken = authHeader?.Replace("Bearer ", "").Trim();

                if (providedToken != _manageToken)
                {
                    context.Response.StatusCode = 401;
                    await context.Response.WriteAsJsonAsync(new { error = "Unauthorized - invalid management token" });
                    return;
                }
            }
            await _next(context);
            return;
        }

        // Regular API endpoints - check tokens if we have any configured
        if (RequireAuth)
        {
            var authHeader = context.Request.Headers["Authorization"].FirstOrDefault();
            var providedToken = authHeader?.Replace("Bearer ", "").Trim();

            bool isValid = false;
            if (!string.IsNullOrEmpty(providedToken))
            {
                lock (_lockObj)
                {
                    isValid = _validTokens.Contains(providedToken);
                }
            }

            if (!isValid)
            {
                context.Response.StatusCode = 401;
                await context.Response.WriteAsJsonAsync(new { error = "Unauthorized - invalid or missing token" });
                return;
            }
        }

        await _next(context);
    }
    
    public void Dispose()
    {
        if (_disposed) return;
        
        if (_fileWatcher != null)
        {
            _fileWatcher.Changed -= OnTokensFileChanged;
            _fileWatcher.Created -= OnTokensFileChanged;
            _fileWatcher.Deleted -= OnTokensFileDeleted;
            _fileWatcher.Dispose();
        }
        
        _disposed = true;
    }
}

public static class TokenAuthMiddlewareExtensions
{
    public static IApplicationBuilder UseTokenAuth(this IApplicationBuilder builder)
    {
        return builder.UseMiddleware<TokenAuthMiddleware>();
    }
}
