using Microsoft.AspNetCore.Mvc;
using AIHost.Config;
using AIHost.Inference;
using AIHost.Logging;
using System.Diagnostics;

namespace AIHost.WebAPI;

[ApiController]
[Route("v1")]
public class OpenAIController : ControllerBase
{
    private readonly ModelManager _modelManager;
    private readonly RequestLogger _requestLogger;
    private readonly ILogger<OpenAIController> _logger;

    public OpenAIController(ModelManager modelManager, RequestLogger requestLogger, ILogger<OpenAIController> logger)
    {
        _modelManager = modelManager;
        _requestLogger = requestLogger;
        _logger = logger;
    }

    /// <summary>
    /// OpenAI /v1/chat/completions - Create chat completion
    /// </summary>
    [HttpPost("chat/completions")]
    public async Task<IActionResult> ChatCompletions([FromBody] OpenAIChatRequest request)
    {
        try
        {
            // Load model
            var model = await _modelManager.GetModelAsync(request.Model);

            // Build prompt from messages
            var prompt = BuildPrompt(model, request.Messages);

            // Get generation parameters
            var config = BuildGenerationConfig(model.Config, request);

            // Generate
            using var __ = model.TrackRequest();
            var sw = Stopwatch.StartNew();
            var response = model.Engine.Generate(prompt, config, HttpContext.RequestAborted);
            sw.Stop();

            // Count tokens
            var tokenizer = model.Engine.Tokenizer;
            var promptTokens = tokenizer.Encode(prompt).Length;
            var completionTokens = tokenizer.Encode(response).Length;

            var tps = sw.Elapsed.TotalSeconds > 0 ? completionTokens / sw.Elapsed.TotalSeconds : 0;
            _modelManager.UpdateModelStats(request.Model, prompt, tps);

            _requestLogger.LogRequest(new RequestLogEntry
            {
                Timestamp = DateTime.UtcNow,
                Endpoint = "/v1/chat/completions",
                Method = "POST",
                ModelName = request.Model,
                Prompt = prompt.Length > 200 ? prompt[..200] : prompt,
                TokensGenerated = completionTokens,
                DurationMs = sw.Elapsed.TotalMilliseconds,
                TPS = tps,
                Success = true
            });

            var result = new OpenAIChatResponse
            {
                Model = request.Model,
                Choices = new List<OpenAIChoice>
                {
                    new OpenAIChoice
                    {
                        Index = 0,
                        Message = new OpenAIMessage
                        {
                            Role = "assistant",
                            Content = response
                        },
                        FinishReason = "stop"
                    }
                },
                Usage = new OpenAIUsage
                {
                    PromptTokens = promptTokens,
                    CompletionTokens = completionTokens,
                    TotalTokens = promptTokens + completionTokens
                }
            };

            return Ok(result);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "OpenAI request failed [{Type}]: {Message}", ex.GetType().FullName, ex.Message);
            return BadRequest(new OpenAIErrorResponse
            {
                Error = new OpenAIError
                {
                    Message = $"{ex.Message} | type={ex.GetType().FullName} | inner={ex.InnerException?.Message} | stack={ex.StackTrace}",
                    Type = "invalid_request_error"
                }
            });
        }
    }

    /// <summary>
    /// OpenAI /v1/models - List available models
    /// </summary>
    [HttpGet("models")]
    public IActionResult Models()
    {
        var models = _modelManager.ListModels()
            .Select(name => new OpenAIModelInfo
            {
                Id = name,
                Object = "model",
                OwnedBy = "aihost"
            })
            .ToList();

        return Ok(new OpenAIModelsResponse { Data = models });
    }

    /// <summary>
    /// OpenAI /v1/models/{model} - Retrieve model information
    /// </summary>
    [HttpGet("models/{model}")]
    public IActionResult GetModel(string model)
    {
        var config = _modelManager.GetModelConfig(model);
        if (config == null)
            return NotFound(new OpenAIErrorResponse
            {
                Error = new OpenAIError
                {
                    Message = $"Model '{model}' not found",
                    Type = "invalid_request_error"
                }
            });

        return Ok(new OpenAIModelInfo
        {
            Id = model,
            Object = "model",
            OwnedBy = "aihost"
        });
    }

    // === Helper Methods ===

    private string BuildPrompt(ModelInstance model, List<OpenAIMessage> messages)
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

        // Add assistant prefix for generation
        if (parts.Count > 0 && !parts[^1].StartsWith("Assistant:"))
            parts.Add("Assistant:");

        return string.Join("\n\n", parts);
    }

    private GenerationConfig BuildGenerationConfig(ModelConfig modelConfig, OpenAIChatRequest request)
    {
        var config = new GenerationConfig
        {
            Temperature = request.Temperature ?? modelConfig.Parameters.Temperature,
            TopP = request.TopP ?? modelConfig.Parameters.TopP,
            TopK = modelConfig.Parameters.TopK,
            RepetitionPenalty = modelConfig.Parameters.RepetitionPenalty,
            MaxNewTokens = request.MaxTokens ?? modelConfig.Parameters.MaxTokens,
            Seed = request.Seed ?? modelConfig.Parameters.Seed,
            UseKVCache = modelConfig.Parameters.UseKVCache,
            KVCacheQuantization = ParseKVCacheQuantization(modelConfig.Parameters.KVCacheQuantization)
        };

        // Handle stop sequences (not yet implemented in GenerationConfig)
        // if (request.Stop != null)
        // {
        //     if (request.Stop is string stopStr)
        //         config.StopSequences = new List<string> { stopStr };
        //     else if (request.Stop is string[] stopArr)
        //         config.StopSequences = stopArr.ToList();
        // }

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
}
