namespace AIHost.ICompute.ROCm;

/// <summary>
/// ROCm/HIP compute device implementation for AMD GPUs
/// </summary>
public class ROCmComputeDevice : ComputeProviderBase
{
    private readonly int _deviceId;
    private readonly string? _gcnArch; // e.g. "gfx1030" — passed to kernels at compile time
    private bool _disposed;

    public override string ProviderName => "ROCm/HIP";
    public override string ApiVersion { get; }

    public ROCmComputeDevice(int deviceId = 0)
    {
        _deviceId = deviceId;

        // Initialize HIP runtime
        HipApi.CheckError(HipApi.hipInit(0), "hipInit");

        // Get device count
        HipApi.CheckError(HipApi.hipGetDeviceCount(out int deviceCount), "hipGetDeviceCount");
        if (deviceCount == 0)
            throw new InvalidOperationException("No HIP-capable devices found");

        if (deviceId >= deviceCount)
            throw new ArgumentException($"Device ID {deviceId} not available. Found {deviceCount} device(s)");

        // Set device
        HipApi.CheckError(HipApi.hipSetDevice(_deviceId), "hipSetDevice");

        // Get device properties
        HipApi.CheckError(HipApi.hipGetDeviceProperties(out var props, _deviceId), "hipGetDeviceProperties");

        var deviceName = System.Text.Encoding.UTF8.GetString(props.name).TrimEnd('\0');
        ApiVersion = $"HIP {props.major}.{props.minor}";

        // Best-effort GCN architecture string for HIPRTC compilation.
        // AMD maps major.minor → gfxNNN differently from CUDA. Known mappings:
        //   9.0 → gfx900 (Vega10), 9.6 → gfx906 (Vega20), 10.1 → gfx1010, 10.3 → gfx1030
        // We build gfx{major}{minor:D2} and let --offload-arch=native override if available.
        _gcnArch = null; // null → kernel uses --offload-arch=native (auto-detect at compile time)

        Console.WriteLine($"ROCm Device: {deviceName}");
        Console.WriteLine($"API Version: {ApiVersion}");
        Console.WriteLine($"Compute Capability: {props.major}.{props.minor}");
        Console.WriteLine($"Multiprocessors: {props.multiProcessorCount}");
        Console.WriteLine($"Global Memory: {props.totalGlobalMem / (1024 * 1024)} MB\n");
    }

    public override IComputeBuffer CreateBuffer(ulong size, BufferType type, DataType elementType = DataType.F32, bool requireDeviceLocal = false)
    {
        // ROCm always allocates in device memory — requireDeviceLocal is always satisfied.
        return new ROCmComputeBuffer(size, type, elementType);
    }

    public override IComputeKernel CreateKernel(string source, string entryPoint)
        => new ROCmComputeKernel(source, entryPoint, _gcnArch);

    public override IComputeKernel CreateKernelFromFile(string filePath, string entryPoint)
        => new ROCmComputeKernel(File.ReadAllText(filePath), entryPoint, _gcnArch);

    public override IComputeCommandQueue CreateCommandQueue()
    {
        return new ROCmComputeCommandQueue();
    }

    public override void Synchronize()
    {
        HipApi.CheckError(HipApi.hipDeviceSynchronize(), "hipDeviceSynchronize");
    }

    public override void Dispose()
    {
        if (_disposed) return;

        Synchronize();
        _disposed = true;
    }
}
