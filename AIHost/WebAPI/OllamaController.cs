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

    [HttpGet("version")]
    public async Task<IActionResult> Version()
    {
        return Ok(new
        {
            version = "0.21.1"
        });
    }

    /// <summary>
    /// Ollama /api/generate - Generate text from a prompt
    /// </summary>
    [HttpPost("generate")]
    public async Task<IActionResult> Generate([FromBody] OllamaGenerateRequest request)
    {
        var sw = Stopwatch.StartNew();

        try
        {
            var loadSw = Stopwatch.StartNew();
            var model = await _modelManager.GetModelAsync(request.Model);
            var loadDuration = loadSw.ElapsedMilliseconds * 1_000_000L;

            var prompt = BuildPrompt(model, request.System, request.Prompt, request.Raw);
            var config = BuildGenerationConfig(model.Config, request.Options, model.Engine.ContextLength);
            var tokenizer = model.Engine.Tokenizer;
            var promptTokens = tokenizer.Encode(prompt).Length;

            if (request.Stream)
                return await StreamGenerate(model, prompt, config, request.Model, loadDuration, promptTokens, sw);

            var evalSw = Stopwatch.StartNew();
            var fullOutput = model.Engine.Generate(prompt, config);
            var evalDuration = evalSw.ElapsedMilliseconds * 1_000_000L;

            // Return only the generated part, not the full prompt+output
            var generated = StripPrompt(fullOutput, prompt);
            var evalCount = tokenizer.Encode(generated).Length;

            _modelManager.UpdateModelStats(request.Model, request.Prompt, evalCount / (evalDuration / 1e9));

            return Ok(new OllamaGenerateResponse
            {
                Model = request.Model,
                Response = generated,
                Done = true,
                TotalDuration = sw.ElapsedMilliseconds * 1_000_000L,
                LoadDuration = loadDuration,
                PromptEvalCount = promptTokens,
                EvalCount = evalCount,
                EvalDuration = evalDuration
            });
        }
        catch (Exception ex)
        {
            return BadRequest(new { error = ex.Message });
        }
    }

    private async Task<IActionResult> StreamGenerate(
        Config.ModelInstance model, string prompt, GenerationConfig config,
        string modelName, long loadDuration, int promptTokens, Stopwatch sw)
    {
        Response.ContentType = "application/x-ndjson";
        var evalSw = Stopwatch.StartNew();
        var evalCount = 0;

        model.Engine.GenerateStreaming(prompt, config, token =>
        {
            evalCount++;
            var chunk = System.Text.Json.JsonSerializer.Serialize(new
            {
                model = modelName,
                response = token,
                done = false
            });
            Response.WriteAsync(chunk + "\n").GetAwaiter().GetResult();
            Response.Body.FlushAsync().GetAwaiter().GetResult();
        });

        var finalChunk = System.Text.Json.JsonSerializer.Serialize(new
        {
            model = modelName,
            response = "",
            done = true,
            total_duration = sw.ElapsedMilliseconds * 1_000_000L,
            load_duration = loadDuration,
            prompt_eval_count = promptTokens,
            eval_count = evalCount,
            eval_duration = evalSw.ElapsedMilliseconds * 1_000_000L
        });
        await Response.WriteAsync(finalChunk + "\n");

        var streamTps = evalCount / (evalSw.Elapsed.TotalSeconds > 0 ? evalSw.Elapsed.TotalSeconds : 1);
        _modelManager.UpdateModelStats(modelName, prompt, streamTps);

        return new EmptyResult();
    }

    /// <summary>
    /// Ollama /api/chat - Chat with a model
    /// </summary>
    [HttpPost("chat")]
    public async Task<IActionResult> Chat([FromBody] OllamaChatRequest request)
    {
        var sw = Stopwatch.StartNew();

        try
        {
            var loadSw = Stopwatch.StartNew();
            var model = await _modelManager.GetModelAsync(request.Model);
            var loadDuration = loadSw.ElapsedMilliseconds * 1_000_000L;

            var prompt = BuildChatPrompt(model, request.Messages);
            var config = BuildGenerationConfig(model.Config, request.Options, model.Engine.ContextLength);
            var tokenizer = model.Engine.Tokenizer;
            var promptTokens = tokenizer.Encode(prompt).Length;

            var evalSw = Stopwatch.StartNew();
            var fullOutput = model.Engine.Generate(prompt, config);
            var evalDuration = evalSw.ElapsedMilliseconds * 1_000_000L;

            var generated = StripPrompt(fullOutput, prompt);
            var evalCount = tokenizer.Encode(generated).Length;

            _modelManager.UpdateModelStats(request.Model, prompt, evalCount / (evalDuration / 1e9));

            return Ok(new OllamaChatResponse
            {
                Model = request.Model,
                Message = new OllamaMessage { Role = "assistant", Content = generated },
                Done = true,
                TotalDuration = sw.ElapsedMilliseconds * 1_000_000L,
                LoadDuration = loadDuration,
                PromptEvalCount = promptTokens,
                EvalCount = evalCount,
                EvalDuration = evalDuration
            });
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

    /// <summary>
    /// Strips the prompt prefix from the full decoded output so only generated text is returned.
    /// </summary>
    private static string StripPrompt(string fullOutput, string prompt)
    {
        if (fullOutput.StartsWith(prompt, StringComparison.Ordinal))
            return fullOutput[prompt.Length..].TrimStart('\n', '\r', ' ');

        // Look for the last assistant tag (TinyLlama template or classic format)
        foreach (var marker in new[] { "<|assistant|>", "Assistant:" })
        {
            var idx = fullOutput.LastIndexOf(marker, StringComparison.Ordinal);
            if (idx >= 0)
                return fullOutput[(idx + marker.Length)..].TrimStart('\n', '\r', ' ', '\0');
        }

        return fullOutput;
    }

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
        // TinyLlama / LLaMA chat template used by most instruct models:
        // <|system|>\n{system}\n</s>\n<|user|>\n{user}\n</s>\n<|assistant|>\n
        var sb = new System.Text.StringBuilder();

        // Collect system text: model-level messages + any system role in conversation
        var systemParts = new List<string>(model.SystemMessages);
        foreach (var msg in messages)
            if (msg.Role == "system") systemParts.Add(msg.Content);

        if (systemParts.Count > 0)
        {
            sb.Append("<|system|>\n");
            sb.Append(string.Join("\n", systemParts));
            sb.Append("\n</s>\n");
        }

        foreach (var msg in messages)
        {
            if (msg.Role == "system") continue; // already handled above
            var tag = msg.Role == "assistant" ? "<|assistant|>" : "<|user|>";
            sb.Append($"{tag}\n{msg.Content}\n</s>\n");
        }

        sb.Append("<|assistant|>\n");
        return sb.ToString();
    }

    private GenerationConfig BuildGenerationConfig(ModelConfig modelConfig, OllamaOptions? options,
        int modelContextLength = 0)
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
            KVCacheQuantization = ParseKVCacheQuantization(modelConfig.Parameters.KVCacheQuantization),
            MaxPromptTokens = ComputeMaxPromptTokens(
                modelContextLength, modelConfig.Parameters.ContextSize, modelConfig.Parameters.MaxTokens)
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

    private static int ComputeMaxPromptTokens(int modelCtx, int configCtx, int maxNewTokens)
    {
        int ctx = modelCtx > 0 ? modelCtx : configCtx;
        int limit = ctx - maxNewTokens;
        return limit > 0 ? limit : 0;
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

        lines.Add(FormattableString.Invariant($"PARAMETER temperature {config.Parameters.Temperature}"));
        lines.Add($"PARAMETER top_k {config.Parameters.TopK}");
        lines.Add(FormattableString.Invariant($"PARAMETER top_p {config.Parameters.TopP}"));
        lines.Add(FormattableString.Invariant($"PARAMETER repeat_penalty {config.Parameters.RepetitionPenalty}"));

        return string.Join("\n", lines);
    }

    private string BuildTemplate(ModelConfig config)
    {
        return "{{ .System }}\n\nUser: {{ .Prompt }}\n\nAssistant:";
    }
}
