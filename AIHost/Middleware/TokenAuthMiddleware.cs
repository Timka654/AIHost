using Microsoft.AspNetCore.Http;
using AIHost.Config;

namespace AIHost.Middleware;

/// <summary>
/// Middleware for token-based authentication with dynamic token reloading
/// </summary>
public class TokenAuthMiddleware : IDisposable
{
    private readonly RequestDelegate _next;
    private readonly HashSet<string> _validTokens = new();
    private readonly string? _manageToken;
    private readonly string? _tokensFilePath;
    private readonly object _lockObj = new();
    private readonly FileSystemWatcher? _fileWatcher;
    private bool _disposed;

    public TokenAuthMiddleware(RequestDelegate next, ServerConfig serverConfig)
    {
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
                    
                    Console.WriteLine($"✓ Token file watcher enabled: {fileName}");
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"⚠ Failed to setup file watcher: {ex.Message}");
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
                Console.WriteLine($"🔄 Tokens reloaded: {_validTokens.Count} token(s)");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"⚠ Failed to reload tokens: {ex.Message}");
            }
        });
    }
    
    private void OnTokensFileDeleted(object sender, FileSystemEventArgs e)
    {
        lock (_lockObj)
        {
            _validTokens.Clear();
        }
        Console.WriteLine($"⚠ Tokens file deleted - authentication disabled");
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
                Console.WriteLine($"⚠ Tokens file not found: {filePath}");
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
            
            if (newTokens.Count == 0)
            {
                Console.WriteLine($"⚠ No valid tokens found - authentication disabled");
            }
            else
            {
                Console.WriteLine($"✓ Loaded {_validTokens.Count} valid token(s)");
            }
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
