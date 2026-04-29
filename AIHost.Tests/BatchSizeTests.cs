using Xunit;
using AIHost.Inference;
using AIHost.Compute;
using AIHost.ICompute;
using AIHost.GGUF;
using AIHost.Tokenizer;

namespace AIHost.Tests;

public class BatchSizeTests
{
    [Fact]
    public void InferenceEngine_ShouldAcceptBatchSizeParameter()
    {
        var mockDevice = new MockComputeDevice();
        var mockModel = new MockGGUFModel();
        var mockTransformer = new MockTransformer(mockDevice, mockModel);
        var mockTokenizer = new MockBPETokenizer();
        var computeOps = new ComputeOps(mockDevice);
        
        var engine = new InferenceEngine(mockTransformer, mockTokenizer, computeOps, batchSize: 4);
        
        Assert.Equal(4, engine.BatchSize);
    }
    
    [Fact]
    public void InferenceEngine_BatchSizeShouldDefaultTo8()
    {
        var mockDevice = new MockComputeDevice();
        var mockModel = new MockGGUFModel();
        var mockTransformer = new MockTransformer(mockDevice, mockModel);
        var mockTokenizer = new MockBPETokenizer();
        var computeOps = new ComputeOps(mockDevice);
        
        var engine = new InferenceEngine(mockTransformer, mockTokenizer, computeOps);
        
        Assert.Equal(8, engine.BatchSize);
    }
    
    [Fact]
    public void InferenceEngine_BatchSizeShouldBeAtLeastOne()
    {
        var mockDevice = new MockComputeDevice();
        var mockModel = new MockGGUFModel();
        var mockTransformer = new MockTransformer(mockDevice, mockModel);
        var mockTokenizer = new MockBPETokenizer();
        var computeOps = new ComputeOps(mockDevice);
        
        var engine = new InferenceEngine(mockTransformer, mockTokenizer, computeOps, batchSize: 0);
        
        Assert.Equal(1, engine.BatchSize);
    }
    
    [Fact]
    public void ModelConfig_BatchSizeShouldBeNullableInt()
    {
        var config = new Config.ModelConfig();
        Assert.Null(config.BatchSize);
        config.BatchSize = 4;
        Assert.Equal(4, config.BatchSize);
    }
}

internal class MockComputeDevice : IComputeDevice
{
    public string ProviderName => "Mock";
    public string ApiVersion => "1.0.0";
    
    public IComputeBuffer CreateBuffer(ulong size, BufferType type, DataType elementType = DataType.F32, bool requireDeviceLocal = false)
        => new MockComputeBuffer(size);
    
    public IComputeKernel CreateKernel(string source, string entryPoint)
        => throw new NotImplementedException();
    
    public IComputeKernel CreateKernelFromFile(string filePath, string entryPoint)
        => throw new NotImplementedException();
    
    public IComputeCommandQueue CreateCommandQueue()
        => new MockComputeCommandQueue();
    
    public void Synchronize() { }
    public void Dispose() { }
}

internal class MockComputeBuffer : IComputeBuffer
{
    public ulong Size { get; }
    public BufferType Type => BufferType.Storage;
    public DataType ElementType => DataType.F32;
    
    public MockComputeBuffer(ulong size) => Size = size;
    
    public IntPtr GetPointer() => IntPtr.Zero;
    public void Write<T>(T[] data) where T : unmanaged { }
    public T[] Read<T>() where T : unmanaged => Array.Empty<T>();
    public void Dispose() { }
}

internal class MockComputeCommandQueue : IComputeCommandQueue
{
    public void WriteBuffer(IComputeBuffer buffer, ulong offset, byte[] data) { }
    public void ReadBuffer(IComputeBuffer buffer, ulong offset, byte[] data) { }
    public void Dispatch(IComputeKernel kernel, uint[] globalWorkSize, uint[]? localWorkSize = null) { }
    public void Flush() { }
    public void InsertMemoryBarrier() { }
    public void Dispose() { }
}

internal class MockGGUFModel : IGGUFModel
{
    public GGUFHeader Header => new GGUFHeader { Version = 3, TensorCount = 0 };
    public GGUFMetadata Metadata { get; }
    public IReadOnlyList<GGUFTensorInfo> Tensors => Array.Empty<GGUFTensorInfo>();
    public GGUFReader Reader => null!;
    
    public MockGGUFModel()
    {
        Metadata = new GGUFMetadata();
        Metadata.Add(GGUFMetadata.KeyBlockCount, 1);
        Metadata.Add(GGUFMetadata.KeyEmbeddingLength, 128);
        Metadata.Add(GGUFMetadata.KeyAttentionHeadCount, 4);
    }
    
    public IComputeBuffer LoadTensor(string tensorName) => new MockComputeBuffer(1024);
    public void Dispose() { }
}

internal class MockTransformer : Transformer
{
    public MockTransformer(IComputeDevice device, IGGUFModel model) : base(device, model) { }
}

internal class MockBPETokenizer : BPETokenizer
{
    public MockBPETokenizer() : base(new[] { "<unk>", "<bos>", "<eos>", "test" }, 1, 2, 0) { }
}
