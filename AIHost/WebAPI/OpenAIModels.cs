using System.Text.Json.Serialization;

namespace AIHost.WebAPI;

// === OpenAI API Models ===

/// <summary>
/// OpenAI /v1/chat/completions request
/// </summary>
public class OpenAIChatRequest
{
    [JsonPropertyName("model")]
    public string Model { get; set; } = "";

    [JsonPropertyName("messages")]
    public List<OpenAIMessage> Messages { get; set; } = new();

    [JsonPropertyName("temperature")]
    public float? Temperature { get; set; }

    [JsonPropertyName("top_p")]
    public float? TopP { get; set; }

    [JsonPropertyName("max_tokens")]
    public int? MaxTokens { get; set; }

    [JsonPropertyName("stream")]
    public bool Stream { get; set; } = false;

    [JsonPropertyName("stop")]
    public object? Stop { get; set; } // string or string[]

    [JsonPropertyName("presence_penalty")]
    public float? PresencePenalty { get; set; }

    [JsonPropertyName("frequency_penalty")]
    public float? FrequencyPenalty { get; set; }

    [JsonPropertyName("seed")]
    public int? Seed { get; set; }
}

/// <summary>
/// OpenAI message format
/// </summary>
public class OpenAIMessage
{
    [JsonPropertyName("role")]
    public string Role { get; set; } = ""; // "system", "user", "assistant"

    [JsonPropertyName("content")]
    public string Content { get; set; } = "";

    [JsonPropertyName("name")]
    public string? Name { get; set; }
}

/// <summary>
/// OpenAI chat completion response
/// </summary>
public class OpenAIChatResponse
{
    [JsonPropertyName("id")]
    public string Id { get; set; } = $"chatcmpl-{Guid.NewGuid():N}";

    [JsonPropertyName("object")]
    public string Object { get; set; } = "chat.completion";

    [JsonPropertyName("created")]
    public long Created { get; set; } = DateTimeOffset.UtcNow.ToUnixTimeSeconds();

    [JsonPropertyName("model")]
    public string Model { get; set; } = "";

    [JsonPropertyName("choices")]
    public List<OpenAIChoice> Choices { get; set; } = new();

    [JsonPropertyName("usage")]
    public OpenAIUsage Usage { get; set; } = new();
}

/// <summary>
/// OpenAI choice
/// </summary>
public class OpenAIChoice
{
    [JsonPropertyName("index")]
    public int Index { get; set; }

    [JsonPropertyName("message")]
    public OpenAIMessage Message { get; set; } = new();

    [JsonPropertyName("finish_reason")]
    public string FinishReason { get; set; } = "stop"; // "stop", "length", "content_filter"
}

/// <summary>
/// OpenAI token usage
/// </summary>
public class OpenAIUsage
{
    [JsonPropertyName("prompt_tokens")]
    public int PromptTokens { get; set; }

    [JsonPropertyName("completion_tokens")]
    public int CompletionTokens { get; set; }

    [JsonPropertyName("total_tokens")]
    public int TotalTokens { get; set; }
}

/// <summary>
/// OpenAI /v1/models response
/// </summary>
public class OpenAIModelsResponse
{
    [JsonPropertyName("object")]
    public string Object { get; set; } = "list";

    [JsonPropertyName("data")]
    public List<OpenAIModelInfo> Data { get; set; } = new();
}

/// <summary>
/// OpenAI model info
/// </summary>
public class OpenAIModelInfo
{
    [JsonPropertyName("id")]
    public string Id { get; set; } = "";

    [JsonPropertyName("object")]
    public string Object { get; set; } = "model";

    [JsonPropertyName("created")]
    public long Created { get; set; } = DateTimeOffset.UtcNow.ToUnixTimeSeconds();

    [JsonPropertyName("owned_by")]
    public string OwnedBy { get; set; } = "aihost";
}

/// <summary>
/// OpenAI error response
/// </summary>
public class OpenAIErrorResponse
{
    [JsonPropertyName("error")]
    public OpenAIError Error { get; set; } = new();
}

/// <summary>
/// OpenAI error details
/// </summary>
public class OpenAIError
{
    [JsonPropertyName("message")]
    public string Message { get; set; } = "";

    [JsonPropertyName("type")]
    public string Type { get; set; } = "invalid_request_error";

    [JsonPropertyName("code")]
    public string? Code { get; set; }
}
