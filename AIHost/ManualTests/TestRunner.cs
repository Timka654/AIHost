using AIHost.ICompute;

namespace AIHost.ManualTests;

/// <summary>
/// Manual test runner for compute operations
/// Extracted from Program.cs for cleaner separation
/// </summary>
public static partial class TestRunner
{
    public static unsafe void RunInteractiveTests(string[] args)
    {
        Console.WriteLine("=== AIHost Compute Provider - Manual Tests ===\n");
        Console.WriteLine("Select test mode:");
        Console.WriteLine("  1. Compute shader test (multiply by 2)");
        Console.WriteLine("  2. Load GGUF model");
        Console.WriteLine("  3. Test dequantization");
        Console.WriteLine("  4. Test matrix multiplication");
        Console.WriteLine("  5. Test tensor operations (softmax, layernorm, etc)");
        Console.WriteLine("  6. Test RoPE and attention");
        Console.WriteLine("  7. Test tokenizer");
        Console.WriteLine("  8. Test multi-head attention");
        Console.WriteLine("  9. Test feed-forward network (FFN)");
        Console.WriteLine("  10. Test transformer forward pass");
        Console.WriteLine("  11. Test autoregressive inference");
        Console.WriteLine("  12. Test InferenceEngine (with real model)");
        
        string? choice;
        string? argModelPath = null;

        if (args.Length >= 1)
        {
            choice = args[0];
            argModelPath = args.Length >= 2 ? args[1] : null;
            Console.WriteLine($"Choice (1-12): {choice}");
        }
        else
        {
            Console.Write("\nChoice (1-12): ");
            choice = Console.ReadLine();
        }
        Console.WriteLine();

        try
        {
            using var provider = new ICompute.Vulkan.VulkanComputeDevice();
            
            Console.WriteLine($"Provider: {provider.ProviderName}");
            Console.WriteLine($"API Version: {provider.ApiVersion}\n");

            switch (choice)
            {
                case "2": TestGGUFLoader(provider); break;
                case "3": TestDequantization(provider); break;
                case "4": TestMatMul(provider); break;
                case "5": TestTensorOps(provider); break;
                case "6": TestRopeAndAttention(provider); break;
                case "7": TestTokenizer(provider); break;
                case "8": TestMultiHeadAttention(provider); break;
                case "9": TestFFN(provider); break;
                case "10": TestTransformer(provider); break;
                case "11": TestInference(provider); break;
                case "12": TestInferenceEngine(provider, argModelPath); break;
                default: TestComputeShader(provider); break;
            }

            provider.Synchronize();
            Console.WriteLine("\nVulkan device synchronized successfully");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error: {ex.Message}");
            Console.WriteLine($"Stack trace:\n{ex.StackTrace}");
        }

        Console.WriteLine("\nPress any key to exit...");
        try
        {
            Console.ReadKey();
        }
        catch (InvalidOperationException)
        {
            // Input was redirected, ignore
        }
    }
}
