using AIHost.Compute;
using AIHost.ICompute;
using AIHost.ICompute.Vulkan;

namespace AIHost.Tests;

public class ComputeOpsTests : IDisposable
{
    private readonly IComputeDevice _device;
    private readonly ComputeOps _ops;

    public ComputeOpsTests()
    {
        _device = new VulkanComputeDevice();
        _ops = new ComputeOps(_device);
    }

    public void Dispose()
    {
        _ops.Dispose();
        GC.Collect();
        GC.WaitForPendingFinalizers();
        _device.Dispose();
    }

    [Fact]
    public void MatMul_LargeMatrices_ProducesValidResult()
    {
        // Arrange
        int M = 64, K = 128, N = 32;
        var random = new Random(42);
        
        float[] aData = Enumerable.Range(0, M * K).Select(_ => (float)(random.NextDouble() * 2 - 1)).ToArray();
        float[] bData = Enumerable.Range(0, K * N).Select(_ => (float)(random.NextDouble() * 2 - 1)).ToArray();

        var A = Tensor.FromData(_device, aData, TensorShape.Matrix(M, K), "A");
        var B = Tensor.FromData(_device, bData, TensorShape.Matrix(K, N), "B");

        // Act
        var C = _ops.MatMul(A, B);
        var result = C.ReadData();

        // Assert
        Assert.Equal(M * N, result.Length);
        Assert.DoesNotContain(float.NaN, result);
        Assert.DoesNotContain(float.PositiveInfinity, result);
        Assert.DoesNotContain(float.NegativeInfinity, result);
        Assert.True(result.Average(Math.Abs) > 0.01, "Result should have non-zero values");

        // Cleanup
        A.Dispose();
        B.Dispose();
        C.Dispose();
    }

    [Fact]
    public void Softmax_SmallVector_SumsToOne()
    {
        // Arrange
        float[] data = { 1.0f, 2.0f, 3.0f, 4.0f, 1.0f };
        var tensor = Tensor.FromData(_device, data, TensorShape.Vector(5), "softmax_input");

        // Act
        _ops.Softmax(tensor);
        var result = tensor.ReadData();

        // Assert
        Assert.Equal(5, result.Length);
        Assert.All(result, x => Assert.InRange(x, 0.0f, 1.0f));
        Assert.InRange(result.Sum(), 0.99f, 1.01f); // Sum should be ~1.0

        // Cleanup
        tensor.Dispose();
    }

    [Fact]
    public void SiLU_ValidInput_ProducesExpectedOutput()
    {
        // Arrange
        float[] data = { -2.0f, -1.0f, 0.0f, 1.0f, 2.0f };
        var tensor = Tensor.FromData(_device, data, TensorShape.Vector(5), "silu_input");

        // Act
        _ops.SiLU(tensor);
        var result = tensor.ReadData();

        // Assert
        Assert.Equal(5, result.Length);
        Assert.InRange(result[2], -0.01f, 0.01f); // SiLU(0) ≈ 0
        Assert.True(result[3] > 0.5f, "SiLU(1) should be > 0.5");
        Assert.True(result[4] > 1.5f, "SiLU(2) should be > 1.5");

        // Cleanup
        tensor.Dispose();
    }

    [Fact]
    public void Add_TwoVectors_ProducesSum()
    {
        // Arrange
        float[] dataA = { 1.0f, 2.0f, 3.0f, 4.0f };
        float[] dataB = { 5.0f, 6.0f, 7.0f, 8.0f };
        
        var A = Tensor.FromData(_device, dataA, TensorShape.Vector(4), "A");
        var B = Tensor.FromData(_device, dataB, TensorShape.Vector(4), "B");

        // Act
        var C = _ops.Add(A, B);
        var result = C.ReadData();

        // Assert
        Assert.Equal(new[] { 6.0f, 8.0f, 10.0f, 12.0f }, result);

        // Cleanup
        A.Dispose();
        B.Dispose();
        C.Dispose();
    }

    [Fact]
    public void LayerNorm_ValidInput_NormalizesCorrectly()
    {
        // Arrange
        float[] data = Enumerable.Range(0, 64).Select(i => (float)i).ToArray();
        float[] weights = Enumerable.Range(0, 64).Select(_ => 1.0f).ToArray();
        
        var tensor = Tensor.FromData(_device, data, TensorShape.Vector(64), "input");
        var weight = Tensor.FromData(_device, weights, TensorShape.Vector(64), "weight");

        // Act
        _ops.LayerNorm(tensor, weight);
        var result = tensor.ReadData();

        // Assert
        Assert.Equal(64, result.Length);
        Assert.DoesNotContain(float.NaN, result);

        // Cleanup
        tensor.Dispose();
        weight.Dispose();
    }
}
