using System.Text.Json.Serialization;

namespace AIHost.WebAPI;

// === Ollama API Models ===

/// <summary>
/// Ollama /api/generate request
/// </summary>
public class OllamaGenerateRequest
{
    [JsonPropertyName("model")]
    public string Model { get; set; } = "";

    [JsonPropertyName("prompt")]
    public string Prompt { get; set; } = "";

    [JsonPropertyName("system")]
    public string? System { get; set; }

    [JsonPropertyName("template")]
    public string? Template { get; set; }

    [JsonPropertyName("context")]
    public int[]? Context { get; set; }

    [JsonPropertyName("stream")]
    public bool Stream { get; set; } = false;

    [JsonPropertyName("raw")]
    public bool Raw { get; set; } = false;

    [JsonPropertyName("options")]
    public OllamaOptions? Options { get; set; }
}

/// <summary>
/// Ollama /api/chat request
/// </summary>
public class OllamaChatRequest
{
    [JsonPropertyName("model")]
    public string Model { get; set; } = "";

    [JsonPropertyName("messages")]
    public List<OllamaMessage> Messages { get; set; } = new();

    [JsonPropertyName("stream")]
    public bool Stream { get; set; } = false;

    [JsonPropertyName("options")]
    public OllamaOptions? Options { get; set; }
}

/// <summary>
/// Ollama message format
/// </summary>
public class OllamaMessage
{
    [JsonPropertyName("role")]
    public string Role { get; set; } = ""; // "system", "user", "assistant"

    [JsonPropertyName("content")]
    public string Content { get; set; } = "";

    [JsonPropertyName("images")]
    public List<string>? Images { get; set; }
}

/// <summary>
/// Ollama generation options
/// </summary>
public class OllamaOptions
{
    [JsonPropertyName("temperature")]
    public float? Temperature { get; set; }

    [JsonPropertyName("top_k")]
    public int? TopK { get; set; }

    [JsonPropertyName("top_p")]
    public float? TopP { get; set; }

    [JsonPropertyName("repeat_penalty")]
    public float? RepeatPenalty { get; set; }

    [JsonPropertyName("seed")]
    public int? Seed { get; set; }

    [JsonPropertyName("num_predict")]
    public int? NumPredict { get; set; }

    [JsonPropertyName("stop")]
    public List<string>? Stop { get; set; }
}

/// <summary>
/// Ollama generate response
/// </summary>
public class OllamaGenerateResponse
{
    [JsonPropertyName("model")]
    public string Model { get; set; } = "";

    [JsonPropertyName("created_at")]
    public string CreatedAt { get; set; } = DateTime.UtcNow.ToString("o");

    [JsonPropertyName("response")]
    public string Response { get; set; } = "";

    [JsonPropertyName("done")]
    public bool Done { get; set; } = true;

    [JsonPropertyName("context")]
    public int[]? Context { get; set; }

    [JsonPropertyName("total_duration")]
    public long TotalDuration { get; set; }

    [JsonPropertyName("load_duration")]
    public long LoadDuration { get; set; }

    [JsonPropertyName("prompt_eval_count")]
    public int PromptEvalCount { get; set; }

    [JsonPropertyName("eval_count")]
    public int EvalCount { get; set; }

    [JsonPropertyName("eval_duration")]
    public long EvalDuration { get; set; }
}

/// <summary>
/// Ollama chat response
/// </summary>
public class OllamaChatResponse
{
    [JsonPropertyName("model")]
    public string Model { get; set; } = "";

    [JsonPropertyName("created_at")]
    public string CreatedAt { get; set; } = DateTime.UtcNow.ToString("o");

    [JsonPropertyName("message")]
    public OllamaMessage Message { get; set; } = new();

    [JsonPropertyName("done")]
    public bool Done { get; set; } = true;

    [JsonPropertyName("total_duration")]
    public long TotalDuration { get; set; }

    [JsonPropertyName("load_duration")]
    public long LoadDuration { get; set; }

    [JsonPropertyName("prompt_eval_count")]
    public int PromptEvalCount { get; set; }

    [JsonPropertyName("eval_count")]
    public int EvalCount { get; set; }

    [JsonPropertyName("eval_duration")]
    public long EvalDuration { get; set; }
}

/// <summary>
/// Ollama /api/tags response (list models)
/// </summary>
public class OllamaTagsResponse
{
    [JsonPropertyName("models")]
    public List<OllamaModelInfo> Models { get; set; } = new();
}

/// <summary>
/// Ollama model info
/// </summary>
public class OllamaModelInfo
{
    [JsonPropertyName("name")]
    public string Name { get; set; } = "";

    [JsonPropertyName("modified_at")]
    public string ModifiedAt { get; set; } = DateTime.UtcNow.ToString("o");

    [JsonPropertyName("size")]
    public long Size { get; set; }

    [JsonPropertyName("digest")]
    public string Digest { get; set; } = "";

    [JsonPropertyName("details")]
    public OllamaModelDetails? Details { get; set; }
}

/// <summary>
/// Ollama model details
/// </summary>
public class OllamaModelDetails
{
    [JsonPropertyName("format")]
    public string Format { get; set; } = "";

    [JsonPropertyName("family")]
    public string Family { get; set; } = "";

    [JsonPropertyName("families")]
    public List<string>? Families { get; set; }

    [JsonPropertyName("parameter_size")]
    public string ParameterSize { get; set; } = "";

    [JsonPropertyName("quantization_level")]
    public string QuantizationLevel { get; set; } = "";
}
