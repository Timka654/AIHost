using AIHost.ICompute;

namespace AIHost.GGUF;

/// <summary>
/// Interface for GGUF model implementations
/// </summary>
public interface IGGUFModel : IDisposable
{
    /// <summary>
    /// GGUF header information
    /// </summary>
    GGUFHeader Header { get; }
    
    /// <summary>
    /// Model metadata
    /// </summary>
    GGUFMetadata Metadata { get; }
    
    /// <summary>
    /// Tensor information list
    /// </summary>
    IReadOnlyList<GGUFTensorInfo> Tensors { get; }
    
    /// <summary>
    /// GGUF reader
    /// </summary>
    GGUFReader Reader { get; }
    
    /// <summary>
    /// Load tensor into GPU memory
    /// </summary>
    IComputeBuffer LoadTensor(string tensorName);
}
