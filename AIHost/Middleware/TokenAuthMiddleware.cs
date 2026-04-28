using Microsoft.AspNetCore.Http;
using System.Collections.Concurrent;

namespace AIHost.Middleware;

/// <summary>
/// Middleware for token-based authentication
/// </summary>
public class TokenAuthMiddleware
{
    private readonly RequestDelegate _next;
    private readonly HashSet<string> _validTokens = new();
    private readonly string? _manageToken;
    private readonly bool _requireAuth;
    private readonly object _lockObj = new();

    public TokenAuthMiddleware(RequestDelegate next, string? tokensFile, string? manageToken)
    {
        _next = next;
        _manageToken = manageToken;
        
        // Load tokens from file if provided
        if (!string.IsNullOrEmpty(tokensFile) && File.Exists(tokensFile))
        {
            LoadTokens(tokensFile);
            _requireAuth = true;
        }
        else
        {
            _requireAuth = false;
            Console.WriteLine("No tokens file found - authentication disabled");
        }
    }

    private void LoadTokens(string filePath)
    {
        try
        {
            var lines = File.ReadAllLines(filePath);
            lock (_lockObj)
            {
                _validTokens.Clear();
                foreach (var line in lines)
                {
                    var token = line.Trim();
                    if (!string.IsNullOrEmpty(token) && !token.StartsWith("#"))
                    {
                        _validTokens.Add(token);
                    }
                }
            }
            Console.WriteLine($"Loaded {_validTokens.Count} valid tokens from {filePath}");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Failed to load tokens: {ex.Message}");
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

        // Regular API endpoints - check tokens if auth is enabled
        if (_requireAuth)
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
}

public static class TokenAuthMiddlewareExtensions
{
    public static IApplicationBuilder UseTokenAuth(this IApplicationBuilder builder, string? tokensFile, string? manageToken)
    {
        return builder.UseMiddleware<TokenAuthMiddleware>(tokensFile, manageToken);
    }
}
