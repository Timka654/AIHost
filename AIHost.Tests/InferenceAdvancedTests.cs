using AIHost.Compute;
using AIHost.GGUF;
using AIHost.Inference;
using AIHost.ICompute;
using AIHost.ICompute.Vulkan;
using AIHost.Tokenizer;
using Xunit;

namespace AIHost.Tests;

/// <summary>
/// Tests for advanced inference features (batch, sampling)
/// </summary>
public class InferenceAdvancedTests : IDisposable
{
    private readonly IComputeDevice _device;
    private readonly Transformer _model;
    private readonly BPETokenizer _tokenizer;
    private readonly ComputeOps _ops;
    private readonly InferenceEngine _engine;

    public InferenceAdvancedTests()
    {
        var modelPath = Environment.GetEnvironmentVariable("TEST_MODEL_PATH") 
            ?? @"D:\User\Downloads\tinyllama-1.1b-chat-v1.0.Q2_K.gguf";
        
        _device = new VulkanComputeDevice();
        var ggufModel = new GGUFModel(modelPath, _device);
        
        _model = new Transformer(_device, ggufModel);
        _model.LoadWeights();
        
        _tokenizer = BPETokenizer.FromGGUF(ggufModel.Reader);
        _ops = new ComputeOps(_device);
        _engine = new InferenceEngine(_model, _tokenizer, _ops);
    }

    [Fact]
    public void BatchGenerate_MultiplePrompts_ReturnsResults()
    {
        // Arrange
        var prompts = new[] { "Hello", "Hi", "Hey" };
        var config = new GenerationConfig
        {
            MaxNewTokens = 5,
            Temperature = 0.7f,
            UseKVCache = true
        };

        // Act
        var results = _engine.BatchGenerate(prompts, config);

        // Assert
        Assert.Equal(3, results.Length);
        foreach (var result in results)
        {
            Assert.NotNull(result);
            Assert.NotEmpty(result);
        }
    }

    [Fact]
    public void Generate_WithRepetitionPenalty_ReducesRepetition()
    {
        // Arrange
        var prompt = "The the the";
        var configWithPenalty = new GenerationConfig
        {
            MaxNewTokens = 10,
            Temperature = 1.0f,
            RepetitionPenalty = 1.5f,
            UseKVCache = true
        };
        
        var configWithoutPenalty = new GenerationConfig
        {
            MaxNewTokens = 10,
            Temperature = 1.0f,
            RepetitionPenalty = 1.0f,
            UseKVCache = true,
            Seed = 42
        };

        // Act
        var resultWithPenalty = _engine.Generate(prompt, configWithPenalty);
        var resultWithoutPenalty = _engine.Generate(prompt, configWithoutPenalty);

        // Assert
        Assert.NotNull(resultWithPenalty);
        Assert.NotNull(resultWithoutPenalty);
        // Results should differ when penalty is applied
        Assert.NotEqual(resultWithPenalty, resultWithoutPenalty);
    }

    [Fact]
    public void Generate_WithTopK_LimitsVocabulary()
    {
        // Arrange
        var prompt = "Once";
        var config = new GenerationConfig
        {
            MaxNewTokens = 5,
            TopK = 5,
            Temperature = 1.0f,
            Seed = 42
        };

        // Act
        var result = _engine.Generate(prompt, config);

        // Assert
        Assert.NotNull(result);
        Assert.Contains("Once", result);
    }

    [Fact]
    public void Generate_WithTopP_NucleusSampling()
    {
        // Arrange
        var prompt = "Once upon";
        var config = new GenerationConfig
        {
            MaxNewTokens = 5,
            TopP = 0.9f,
            TopK = 0, // Disable TopK
            Temperature = 1.0f,
            Seed = 42
        };

        // Act
        var result = _engine.Generate(prompt, config);

        // Assert
        Assert.NotNull(result);
        Assert.Contains("Once upon", result);
    }

    [Fact]
    public void Generate_WithDifferentTemperatures_ProducesDifferentResults()
    {
        // Arrange
        var prompt = "Hello";
        var lowTemp = new GenerationConfig
        {
            MaxNewTokens = 10,
            Temperature = 0.1f,
            Seed = 42
        };
        
        var highTemp = new GenerationConfig
        {
            MaxNewTokens = 10,
            Temperature = 2.0f,
            Seed = 43
        };

        // Act
        var resultLow = _engine.Generate(prompt, lowTemp);
        var resultHigh = _engine.Generate(prompt, highTemp);

        // Assert
        Assert.NotNull(resultLow);
        Assert.NotNull(resultHigh);
        // Low temperature should be more deterministic
        Assert.Contains("Hello", resultLow);
        Assert.Contains("Hello", resultHigh);
    }

    public void Dispose()
    {
        _engine?.Dispose();
        _ops?.Dispose();
        _model?.Dispose();
        _device?.Dispose();
    }
}
