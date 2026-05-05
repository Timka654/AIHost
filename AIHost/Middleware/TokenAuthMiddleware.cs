using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.Logging;
using AIHost.Config;

namespace AIHost.Middleware;

/// <summary>
/// Token access level
/// </summary>
public enum TokenAccessLevel
{
    /// <summary>
    /// User access - can use API endpoints (not /manage)
    /// </summary>
    User,
    
    /// <summary>
    /// Management access - can use /manage endpoints
    /// </summary>
    Manage,
    
    /// <summary>
    /// All access - full access to everything
    /// </summary>
    All
}

/// <summary>
/// Token entry with access level
/// </summary>
public class TokenEntry
{
    public string Token { get; set; } = "";
    public TokenAccessLevel AccessLevel { get; set; }
    
    public static TokenEntry Parse(string line)
    {
        var parts = line.Split(':', 2);
        if (parts.Length != 2)
            throw new FormatException($"Invalid token format: {line}");
        
        var modifier = parts[0].Trim().ToUpperInvariant();
        var token = parts[1].Trim();
        
        var accessLevel = modifier switch
        {
            "A" => TokenAccessLevel.All,
            "M" => TokenAccessLevel.Manage,
            "U" => TokenAccessLevel.User,
            _ => throw new FormatException($"Invalid access modifier: {modifier}")
        };
        
        return new TokenEntry
        {
            Token = token,
            AccessLevel = accessLevel
        };
    }
    
    public override string ToString()
    {
        var modifier = AccessLevel switch
        {
            TokenAccessLevel.All => "A",
            TokenAccessLevel.Manage => "M",
            TokenAccessLevel.User => "U",
            _ => "U"
        };
        return $"{modifier}:{Token}";
    }
}

/// <summary>
/// Middleware for token-based authentication with dynamic token reloading
/// </summary>
public class TokenAuthMiddleware : IDisposable
{
    private readonly RequestDelegate _next;
    private readonly ILogger<TokenAuthMiddleware> _logger;
    private readonly Dictionary<string, TokenAccessLevel> _validTokens = new();
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
            var newTokens = new Dictionary<string, TokenAccessLevel>();
            
            foreach (var line in lines)
            {
                var trimmedLine = line.Trim();
                if (string.IsNullOrEmpty(trimmedLine) || trimmedLine.StartsWith("#"))
                    continue;
                
                try
                {
                    var entry = TokenEntry.Parse(trimmedLine);
                    newTokens[entry.Token] = entry.AccessLevel;
                }
                catch (FormatException ex)
                {
                    _logger.LogWarning("Skipping invalid token line: {Error}", ex.Message);
                }
            }
            
            lock (_lockObj)
            {
                _validTokens.Clear();
                foreach (var kvp in newTokens)
                {
                    _validTokens[kvp.Key] = kvp.Value;
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
            _logger.LogError(ex, "Failed to load tokens");
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

        // Management endpoints require manage token or A/M level access
        if (path.StartsWith("/manage"))
        {
            // First check manage token (legacy)
            if (!string.IsNullOrEmpty(_manageToken))
            {
                var authHeader = context.Request.Headers["Authorization"].FirstOrDefault();
                var providedToken = authHeader?.Replace("Bearer ", "").Trim();

                if (providedToken == _manageToken)
                {
                    await _next(context);
                    return;
                }
            }
            
            // Check token-based access with A or M level
            if (RequireAuth)
            {
                var authHeader = context.Request.Headers["Authorization"].FirstOrDefault();
                var providedToken = authHeader?.Replace("Bearer ", "").Trim();

                bool hasAccess = false;
                if (!string.IsNullOrEmpty(providedToken))
                {
                    lock (_lockObj)
                    {
                        if (_validTokens.TryGetValue(providedToken, out var level))
                        {
                            hasAccess = level == TokenAccessLevel.All || level == TokenAccessLevel.Manage;
                        }
                    }
                }

                if (!hasAccess)
                {
                    context.Response.StatusCode = 401;
                    await context.Response.WriteAsJsonAsync(new { error = "Unauthorized - management access required" });
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
                    // Any valid token grants access to regular API endpoints
                    isValid = _validTokens.ContainsKey(providedToken);
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
