using Microsoft.AspNetCore.Http;
using System.Collections.Concurrent;

namespace AIHost.Middleware;

/// <summary>
/// Rate limiting middleware
/// </summary>
public class RateLimitMiddleware
{
    private readonly RequestDelegate _next;
    private readonly int _requestsPerMinute;
    private readonly ConcurrentDictionary<string, Queue<DateTime>> _requests = new();
    private readonly object _lockObj = new();

    public RateLimitMiddleware(RequestDelegate next, int requestsPerMinute)
    {
        _next = next;
        _requestsPerMinute = requestsPerMinute;
    }

    public async Task InvokeAsync(HttpContext context)
    {
        if (_requestsPerMinute <= 0)
        {
            await _next(context);
            return;
        }

        // Get client identifier (IP or token)
        var clientId = GetClientId(context);
        var now = DateTime.UtcNow;
        
        bool rateLimitExceeded = false;

        lock (_lockObj)
        {
            if (!_requests.TryGetValue(clientId, out var requestQueue))
            {
                requestQueue = new Queue<DateTime>();
                _requests[clientId] = requestQueue;
            }

            // Remove requests older than 1 minute
            while (requestQueue.Count > 0 && (now - requestQueue.Peek()).TotalMinutes > 1)
            {
                requestQueue.Dequeue();
            }

            // Check rate limit
            if (requestQueue.Count >= _requestsPerMinute)
            {
                rateLimitExceeded = true;
            }
            else
            {
                // Add current request
                requestQueue.Enqueue(now);
            }
        }

        if (rateLimitExceeded)
        {
            context.Response.StatusCode = 429;
            await context.Response.WriteAsJsonAsync(new
            {
                error = "Rate limit exceeded",
                limit = _requestsPerMinute,
                retry_after = 60
            });
            return;
        }

        await _next(context);
    }

    private string GetClientId(HttpContext context)
    {
        // Try to get token from Authorization header
        var authHeader = context.Request.Headers["Authorization"].FirstOrDefault();
        if (!string.IsNullOrEmpty(authHeader))
        {
            var token = authHeader.Replace("Bearer ", "").Trim();
            if (!string.IsNullOrEmpty(token))
                return $"token:{token}";
        }

        // Fall back to IP address
        var ip = context.Connection.RemoteIpAddress?.ToString() ?? "unknown";
        return $"ip:{ip}";
    }
}

public static class RateLimitMiddlewareExtensions
{
    public static IApplicationBuilder UseRateLimit(this IApplicationBuilder builder, int requestsPerMinute)
    {
        return builder.UseMiddleware<RateLimitMiddleware>(requestsPerMinute);
    }
}
