using Silk.NET.Vulkan;
using System.Reflection;

namespace AIHost.Compute;

/// <summary>
/// Lazy loader for shader source code from embedded files
/// </summary>
public class ShaderLoader
{
    private static readonly ILogger _logger = AppLogger.Create<ShaderLoader>();

    private static readonly Dictionary<string, string> _cache = new();
    private static readonly string _shaderBasePath = Path.Combine(
        Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location) ?? "",
        "Shaders"
    );

    static readonly Dictionary<string, string> _extensions = new()
    {
        { "vulkan", ".glsl" },
        { "rocm", ".hip" }
    };

    /// <summary>
    /// Load shader source code for specified provider
    /// </summary>
    /// <param name="provider">Provider name (Vulkan, ROCm, etc.)</param>
    /// <param name="shaderName">Shader file name without extension</param>
    /// <returns>Shader source code</returns>
    public static string Load(string provider, string shaderName)
    {
        if (_cache.TryGetValue(shaderName, out string? cached))
            return cached;

        string[] pathes = CombinePath(provider, shaderName);

        foreach (string path in pathes)
        {
            _logger.LogInformation($"Try loading shader \"{path}\" for {provider}.{shaderName}");

            if (File.Exists(path))
            {
                string source = File.ReadAllText(path);
                _cache[shaderName] = source;
            _logger.LogInformation($"Try loading shader - success for {provider}.{shaderName}");
                return source;
            }
        }

        _logger.LogError($"Try loading shader - failed for {provider}.{shaderName}");

        throw new FileNotFoundException($"Shader not found: {provider}/{shaderName}");
    }

    private static string[] CombinePath(string provider, string shaderName)
    {
        if (!_extensions.TryGetValue(provider.ToLower(), out string? ext))
        {
            _logger.LogError($"Try loading shader for {provider}.{shaderName} - unknown provider \"{provider}\"");

            throw new InvalidOperationException($"Not found extension for provider '{provider}'");
        }

        var p1 = Path.Combine(_shaderBasePath, "override", provider);
        var p2 = Path.Combine(_shaderBasePath, provider);

        return [Path.Combine(p1, shaderName + ext), Path.Combine(p2, shaderName + ext)];
    }

    /// <summary>
    /// Clear shader cache
    /// </summary>
    public static void ClearCache()
    {
        _cache.Clear();
    }
}
