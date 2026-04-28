using Microsoft.AspNetCore.Mvc;
using AIHost.Config;
using AIHost.Inference;
using System.Diagnostics;

namespace AIHost.WebAPI;

[ApiController]
[Route("api")]
public class OllamaController : ControllerBase
{
    private readonly ModelManager _modelManager;

    public OllamaController(ModelManager modelManager)
    {
        _modelManager = modelManager;
    }

    /// <summary>
    /// Ollama /api/generate - Generate text from a prompt
    /// </summary>
    [HttpPost("generate")]
    public async Task<IActionResult> Generate([FromBody] OllamaGenerateRequest request)
    {
        var sw = Stopwatch.StartNew();
        long loadDuration = 0;

        try
        {
            // Load model
            var loadSw = Stopwatch.StartNew();
            var model = await _modelManager.GetModelAsync(request.Model);
            loadDuration = loadSw.ElapsedMilliseconds * 1_000_000; // Convert to nanoseconds

            // Build prompt with system message
            var prompt = BuildPrompt(model, request.System, request.Prompt, request.Raw);

            // Get generation parameters
            var config = BuildGenerationConfig(model.Config, request.Options);

            // Generate
            var evalSw = Stopwatch.StartNew();
            var response = model.Engine.Generate(prompt, config);
            var evalDuration = evalSw.ElapsedMilliseconds * 1_000_000;

            // Count tokens (approximate)
            var tokenizer = model.Engine.Tokenizer;
            var promptTokens = tokenizer.Encode(prompt).Length;
            var responseTokens = tokenizer.Encode(response).Length;

            var result = new OllamaGenerateResponse
            {
                Model = request.Model,
                Response = response,
                Done = true,
                TotalDuration = sw.ElapsedMilliseconds * 1_000_000,
                LoadDuration = loadDuration,
                PromptEvalCount = promptTokens,
                EvalCount = responseTokens,
                EvalDuration = evalDuration
            };

            return Ok(result);
        }
        catch (Exception ex)
        {
            return BadRequest(new { error = ex.Message });
        }
    }

    /// <summary>
    /// Ollama /api/chat - Chat with a model
    /// </summary>
    [HttpPost("chat")]
    public async Task<IActionResult> Chat([FromBody] OllamaChatRequest request)
    {
        var sw = Stopwatch.StartNew();
        long loadDuration = 0;

        try
        {
            // Load model
            var loadSw = Stopwatch.StartNew();
            var model = await _modelManager.GetModelAsync(request.Model);
            loadDuration = loadSw.ElapsedMilliseconds * 1_000_000;

            // Build prompt from messages
            var prompt = BuildChatPrompt(model, request.Messages);

            // Get generation parameters
            var config = BuildGenerationConfig(model.Config, request.Options);

            // Generate
            var evalSw = Stopwatch.StartNew();
            var response = model.Engine.Generate(prompt, config);
            var evalDuration = evalSw.ElapsedMilliseconds * 1_000_000;

            // Count tokens
            var tokenizer = model.Engine.Tokenizer;
            var promptTokens = tokenizer.Encode(prompt).Length;
            var responseTokens = tokenizer.Encode(response).Length;

            var result = new OllamaChatResponse
            {
                Model = request.Model,
                Message = new OllamaMessage
                {
                    Role = "assistant",
                    Content = response
                },
                Done = true,
                TotalDuration = sw.ElapsedMilliseconds * 1_000_000,
                LoadDuration = loadDuration,
                PromptEvalCount = promptTokens,
                EvalCount = responseTokens,
                EvalDuration = evalDuration
            };

            return Ok(result);
        }
        catch (Exception ex)
        {
            return BadRequest(new { error = ex.Message });
        }
    }

    /// <summary>
    /// Ollama /api/tags - List available models
    /// </summary>
    [HttpGet("tags")]
    public IActionResult Tags()
    {
        var models = _modelManager.ListModels()
            .Select(name => {
                var config = _modelManager.GetModelConfig(name);
                return new OllamaModelInfo
                {
                    Name = name,
                    Size = GetModelSize(config),
                    Details = new OllamaModelDetails
                    {
                        Format = config?.Format ?? "gguf",
                        ParameterSize = ExtractParameterSize(name),
                        QuantizationLevel = ExtractQuantization(name)
                    }
                };
            })
            .ToList();

        return Ok(new OllamaTagsResponse { Models = models });
    }

    /// <summary>
    /// Ollama /api/show - Show model information
    /// </summary>
    [HttpPost("show")]
    public IActionResult Show([FromBody] Dictionary<string, string> request)
    {
        if (!request.TryGetValue("name", out var modelName))
            return BadRequest(new { error = "Model name required" });

        var config = _modelManager.GetModelConfig(modelName);
        if (config == null)
            return NotFound(new { error = $"Model '{modelName}' not found" });

        return Ok(new
        {
            modelfile = BuildModelfile(config),
            parameters = config.Parameters,
            template = BuildTemplate(config),
            details = new
            {
                format = config.Format,
                parameter_size = ExtractParameterSize(modelName),
                quantization_level = ExtractQuantization(modelName)
            }
        });
    }

    // === Helper Methods ===

    private string BuildPrompt(ModelInstance model, string? system, string prompt, bool raw)
    {
        if (raw)
            return prompt;

        var parts = new List<string>();

        // Add system messages
        if (!string.IsNullOrEmpty(system))
            parts.Add($"System: {system}");
        
        foreach (var sysMsg in model.SystemMessages)
            parts.Add($"System: {sysMsg}");

        // Add user prompt
        parts.Add($"User: {prompt}");
        parts.Add("Assistant:");

        return string.Join("\n\n", parts);
    }

    private string BuildChatPrompt(ModelInstance model, List<OllamaMessage> messages)
    {
        var parts = new List<string>();

        // Add model system messages first
        foreach (var sysMsg in model.SystemMessages)
            parts.Add($"System: {sysMsg}");

        // Add conversation messages
        foreach (var msg in messages)
        {
            var role = msg.Role switch
            {
                "system" => "System",
                "user" => "User",
                "assistant" => "Assistant",
                _ => msg.Role
            };
            parts.Add($"{role}: {msg.Content}");
        }

        // Add assistant prefix
        parts.Add("Assistant:");

        return string.Join("\n\n", parts);
    }

    private GenerationConfig BuildGenerationConfig(ModelConfig modelConfig, OllamaOptions? options)
    {
        var config = new GenerationConfig
        {
            Temperature = modelConfig.Parameters.Temperature,
            TopK = modelConfig.Parameters.TopK,
            TopP = modelConfig.Parameters.TopP,
            RepetitionPenalty = modelConfig.Parameters.RepetitionPenalty,
            MaxNewTokens = modelConfig.Parameters.MaxTokens,
            Seed = modelConfig.Parameters.Seed,
            UseKVCache = modelConfig.Parameters.UseKVCache,
            KVCacheQuantization = ParseKVCacheQuantization(modelConfig.Parameters.KVCacheQuantization)
        };

        // Override with request options
        if (options != null)
        {
            if (options.Temperature.HasValue) config.Temperature = options.Temperature.Value;
            if (options.TopK.HasValue) config.TopK = options.TopK.Value;
            if (options.TopP.HasValue) config.TopP = options.TopP.Value;
            if (options.RepeatPenalty.HasValue) config.RepetitionPenalty = options.RepeatPenalty.Value;
            if (options.NumPredict.HasValue) config.MaxNewTokens = options.NumPredict.Value;
            if (options.Seed.HasValue) config.Seed = options.Seed.Value;
        }

        return config;
    }

    private KVCacheQuantization ParseKVCacheQuantization(string value)
    {
        return value.ToLower() switch
        {
            "int8" => KVCacheQuantization.Int8,
            "int4" => KVCacheQuantization.Int4,
            _ => KVCacheQuantization.None
        };
    }

    private long GetModelSize(ModelConfig? config)
    {
        if (config == null) return 0;
        
        var path = config.ModelPath;
        if (System.IO.File.Exists(path))
            return new System.IO.FileInfo(path).Length;

        return 0;
    }

    private string ExtractParameterSize(string modelName)
    {
        // Try to extract parameter size from model name (e.g., "tinyllama-1.1b" -> "1.1B")
        var match = System.Text.RegularExpressions.Regex.Match(modelName, @"(\d+\.?\d*)[bB]");
        return match.Success ? match.Groups[1].Value + "B" : "unknown";
    }

    private string ExtractQuantization(string modelName)
    {
        // Try to extract quantization from model name (e.g., "Q2_K" -> "Q2_K")
        var match = System.Text.RegularExpressions.Regex.Match(modelName, @"(Q\d+_[KMS]|F16|F32)");
        return match.Success ? match.Value : "unknown";
    }

    private string BuildModelfile(ModelConfig config)
    {
        var lines = new List<string>();
        
        lines.Add($"# Modelfile for {config.Name}");
        lines.Add($"FROM {config.ModelPath}");
        
        foreach (var sysMsg in config.SystemMessages)
            lines.Add($"SYSTEM {sysMsg}");

        lines.Add($"PARAMETER temperature {config.Parameters.Temperature}");
        lines.Add($"PARAMETER top_k {config.Parameters.TopK}");
        lines.Add($"PARAMETER top_p {config.Parameters.TopP}");
        lines.Add($"PARAMETER repeat_penalty {config.Parameters.RepetitionPenalty}");

        return string.Join("\n", lines);
    }

    private string BuildTemplate(ModelConfig config)
    {
        return "{{ .System }}\n\nUser: {{ .Prompt }}\n\nAssistant:";
    }
}
