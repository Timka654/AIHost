namespace AIHost.ICompute;

/// <summary>
/// Thrown when a model requires dedicated VRAM but the GPU cannot satisfy the allocation.
/// This happens when the model weights exceed available VRAM and the driver would fall back
/// to shared GPU memory (system RAM via PCIe), which is 10–20x slower.
///
/// To suppress this error and allow the fallback, set "allow_shared_memory": true in model.json.
/// </summary>
public sealed class InsufficientVramException : Exception
{
    public ulong RequestedBytes { get; }
    public string ProviderName { get; }

    public InsufficientVramException(ulong requestedBytes, string providerName)
        : base(BuildMessage(requestedBytes, providerName))
    {
        RequestedBytes = requestedBytes;
        ProviderName = providerName;
    }

    private static string BuildMessage(ulong bytes, string provider)
    {
        var mb = bytes / (1024.0 * 1024.0);
        return $"Insufficient dedicated VRAM: {provider} could not allocate {mb:F1} MB in device-local memory. " +
               $"The model would run from system RAM at PCIe speeds (10–20x slower). " +
               $"Options: use a smaller/more quantized model, free VRAM, or set \"allow_shared_memory\": true in model.json to allow the fallback.";
    }
}
