using System.Reflection;

namespace AIHost.Compute;

/// <summary>
/// Lazy loader for shader source code from embedded files
/// </summary>
public static class ShaderLoader
{
    private static readonly Dictionary<string, string> _cache = new();
    private static readonly string _shaderBasePath = Path.Combine(
        Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location) ?? "",
        "Shaders"
    );

    /// <summary>
    /// Load shader source code for specified provider
    /// </summary>
    /// <param name="provider">Provider name (Vulkan, ROCm, etc.)</param>
    /// <param name="shaderName">Shader file name without extension</param>
    /// <returns>Shader source code</returns>
    public static string Load(string provider, string shaderName)
    {
        string key = $"{provider}/{shaderName}";
        
        if (_cache.TryGetValue(key, out string? cached))
            return cached;

        string[] extensions = provider.ToLower() switch
        {
            "vulkan" => new[] { ".glsl", ".comp" },
            "rocm" => new[] { ".hip", ".cu" },
            _ => new[] { ".glsl" }
        };

        foreach (var ext in extensions)
        {
            string path = Path.Combine(_shaderBasePath, provider, shaderName + ext);
            
            if (File.Exists(path))
            {
                string source = File.ReadAllText(path);
                _cache[key] = source;
                return source;
            }
        }

        throw new FileNotFoundException($"Shader not found: {provider}/{shaderName}");
    }

    /// <summary>
    /// Check if shader exists
    /// </summary>
    public static bool Exists(string provider, string shaderName)
    {
        try
        {
            Load(provider, shaderName);
            return true;
        }
        catch (FileNotFoundException)
        {
            return false;
        }
    }

    /// <summary>
    /// Clear shader cache
    /// </summary>
    public static void ClearCache()
    {
        _cache.Clear();
    }

    /// <summary>
    /// Get all available shaders for a provider
    /// </summary>
    public static string[] GetAvailableShaders(string provider)
    {
        string providerPath = Path.Combine(_shaderBasePath, provider);
        
        if (!Directory.Exists(providerPath))
            return Array.Empty<string>();

        return Directory.GetFiles(providerPath)
            .Select(Path.GetFileNameWithoutExtension)
            .Where(name => !string.IsNullOrEmpty(name))
            .ToArray()!;
    }
}
