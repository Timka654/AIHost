using AIHost.Tokenizer;

namespace AIHost.Inference;

/// <summary>
/// Common API for single-GPU and multi-GPU inference engines.
/// Controllers and ModelManager use this interface so they work with both.
/// </summary>
public interface IInferenceEngine : IDisposable
{
    BPETokenizer Tokenizer    { get; }
    int          ContextLength { get; }

    string Generate(string prompt, GenerationConfig config);
    void   GenerateStreaming(string prompt, GenerationConfig config, Action<string> onTokenGenerated);
}
