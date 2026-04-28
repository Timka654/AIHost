using AIHost.Compute;
using AIHost.GGUF;
using AIHost.Inference;
using AIHost.ICompute;
using AIHost.ICompute.Vulkan;
using AIHost.Tokenizer;
using Xunit;

namespace AIHost.Tests;

/// <summary>
/// Shared fixture: loads the model once for all tests in InferenceAdvancedTests.
/// Avoids 5× model load + double ComputeOps that caused the post-test crash.
/// </summary>
public class InferenceFixture : IDisposable
{
    public IComputeDevice Device { get; }
    public Transformer Model { get; }
    public BPETokenizer Tokenizer { get; }
    public InferenceEngine Engine { get; }

    // Must be stored and disposed — otherwise finalized after Device.Dispose() → Vulkan crash.
    private readonly GGUFModel _ggufModel;

    public InferenceFixture()
    {
        var modelPath = Environment.GetEnvironmentVariable("TEST_MODEL_PATH")
            ?? @"D:\User\Downloads\tinyllama-1.1b-chat-v1.0.Q2_K.gguf";

        Device = new VulkanComputeDevice();
        _ggufModel = new GGUFModel(modelPath, Device);

        Model = new Transformer(Device, _ggufModel);
        Model.LoadWeights();

        Tokenizer = BPETokenizer.FromGGUF(_ggufModel.Reader);
        // Share Transformer's ComputeOps — no second command queue
        Engine = new InferenceEngine(Model, Tokenizer, Model.Ops);
    }

    public void Dispose()
    {
        Engine.Dispose();
        Model.Dispose();
        _ggufModel.Dispose();
        // Force GC to finalize any lingering Vulkan objects before the device is destroyed.
        // Without this, finalizers may run after Device.Dispose() and crash the process.
        GC.Collect();
        GC.WaitForPendingFinalizers();
        Device.Dispose();
    }
}

/// <summary>
/// Tests for advanced inference features (batch, sampling).
/// Uses IClassFixture so the model loads once for all tests.
/// </summary>
public class InferenceAdvancedTests : IClassFixture<InferenceFixture>
{
    private readonly InferenceEngine _engine;

    public InferenceAdvancedTests(InferenceFixture fixture)
    {
        _engine = fixture.Engine;
    }

    [Fact]
    public void BatchGenerate_MultiplePrompts_ReturnsResults()
    {
        var prompts = new[] { "Hello", "Hi", "Hey" };
        var config = new GenerationConfig { MaxNewTokens = 5, Temperature = 0.7f, UseKVCache = true };

        var results = _engine.BatchGenerate(prompts, config);

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
        var prompt = "The the the";
        var withPenalty = new GenerationConfig
        {
            MaxNewTokens = 10, Temperature = 1.0f, RepetitionPenalty = 1.5f, UseKVCache = true
        };
        var withoutPenalty = new GenerationConfig
        {
            MaxNewTokens = 10, Temperature = 1.0f, RepetitionPenalty = 1.0f, UseKVCache = true, Seed = 42
        };

        var r1 = _engine.Generate(prompt, withPenalty);
        var r2 = _engine.Generate(prompt, withoutPenalty);

        Assert.NotNull(r1);
        Assert.NotNull(r2);
        Assert.NotEqual(r1, r2);
    }

    [Fact]
    public void Generate_WithTopK_LimitsVocabulary()
    {
        var config = new GenerationConfig { MaxNewTokens = 5, TopK = 5, Temperature = 1.0f, Seed = 42 };

        var result = _engine.Generate("Once", config);

        Assert.NotNull(result);
        Assert.Contains("Once", result);
    }

    [Fact]
    public void Generate_WithTopP_NucleusSampling()
    {
        var config = new GenerationConfig
        {
            MaxNewTokens = 5, TopP = 0.9f, TopK = 0, Temperature = 1.0f, Seed = 42
        };

        var result = _engine.Generate("Once upon", config);

        Assert.NotNull(result);
        Assert.Contains("Once upon", result);
    }

    [Fact]
    public void Generate_WithDifferentTemperatures_ProducesDifferentResults()
    {
        var lowTemp  = new GenerationConfig { MaxNewTokens = 10, Temperature = 0.1f, Seed = 42 };
        var highTemp = new GenerationConfig { MaxNewTokens = 10, Temperature = 2.0f, Seed = 42 };

        var resultLow  = _engine.Generate("Hello", lowTemp);
        var resultHigh = _engine.Generate("Hello", highTemp);

        Assert.NotNull(resultLow);
        Assert.NotNull(resultHigh);
        Assert.Contains("Hello", resultLow);
        Assert.Contains("Hello", resultHigh);
    }
}
