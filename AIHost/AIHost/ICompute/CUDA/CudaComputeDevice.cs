using System.Runtime.InteropServices;

namespace AIHost.ICompute.CUDA;

/// <summary>
/// CUDA compute device implementation
/// </summary>
public unsafe class CudaComputeDevice : ComputeProviderBase
{
    private readonly int _deviceId;
    private bool _disposed;

    public override string ProviderName => "CUDA";
    public override string ApiVersion { get; }
    public string DeviceName { get; private set; } = string.Empty;
    public int DeviceIndex => _deviceId;
    public int ComputeCapabilityMajor { get; private set; }
    public int ComputeCapabilityMinor { get; private set; }

    /// <summary>
    /// Get information about all available CUDA devices
    /// </summary>
    public static DeviceInfo[] GetAvailableDevices()
    {
        var devices = new List<DeviceInfo>();

        var error = CudaApi.GetDeviceCount(out int count);
        if (error != CudaError.Success || count == 0)
            return Array.Empty<DeviceInfo>();

        for (int i = 0; i < count; i++)
        {
            error = CudaApi.GetDeviceProperties(out CudaDeviceProp prop, i);
            if (error == CudaError.Success)
            {
                string name = Marshal.PtrToStringAnsi((IntPtr)prop.name) ?? "Unknown";
                devices.Add(new DeviceInfo
                {
                    Index = i,
                    Name = name,
                    ApiVersion = $"Compute {prop.major}.{prop.minor}",
                    DeviceType = "CUDA GPU"
                });
            }
        }

        return devices.ToArray();
    }

    public CudaComputeDevice() : this(0)
    {
    }

    public CudaComputeDevice(int deviceId)
    {
        _deviceId = deviceId;

        // Set active device
        var error = CudaApi.SetDevice(deviceId);
        CudaApi.CheckError(error, "cudaSetDevice");

        // Get device properties
        error = CudaApi.GetDeviceProperties(out CudaDeviceProp prop, deviceId);
        CudaApi.CheckError(error, "cudaGetDeviceProperties");

        DeviceName = Marshal.PtrToStringAnsi((IntPtr)prop.name) ?? "Unknown CUDA Device";
        ComputeCapabilityMajor = prop.major;
        ComputeCapabilityMinor = prop.minor;
        ApiVersion = $"{prop.major}.{prop.minor}";

        Console.WriteLine($"CUDA Device [{deviceId}]: {DeviceName}");
        Console.WriteLine($"Compute Capability: {ComputeCapabilityMajor}.{ComputeCapabilityMinor}");
        Console.WriteLine($"Total Memory: {prop.totalGlobalMem / (1024 * 1024)}MB");
        Console.WriteLine($"Multiprocessors: {prop.multiProcessorCount}");
    }

    public override IComputeBuffer CreateBuffer(ulong size, BufferType type, DataType elementType = DataType.F32, bool requireDeviceLocal = false)
    {
        // CUDA always allocates in device memory — requireDeviceLocal is always satisfied.
        return new CudaComputeBuffer(size, type, elementType);
    }

    public override IComputeKernel CreateKernel(string source, string entryPoint)
    {
        return new CudaComputeKernel(source, entryPoint);
    }

    public override IComputeKernel CreateKernelFromFile(string filePath, string entryPoint)
    {
        var source = File.ReadAllText(filePath);
        return new CudaComputeKernel(source, entryPoint);
    }

    public override IComputeCommandQueue CreateCommandQueue()
    {
        return new CudaComputeCommandQueue();
    }

    public override void Synchronize()
    {
        var error = CudaApi.DeviceSynchronize();
        CudaApi.CheckError(error, "cudaDeviceSynchronize");
    }

    public override void Dispose()
    {
        if (_disposed) return;

        CudaApi.DeviceReset();
        _disposed = true;
    }
}

/// <summary>
/// Information about a CUDA compute device
/// </summary>
public class DeviceInfo
{
    public int Index { get; set; }
    public string Name { get; set; } = string.Empty;
    public string ApiVersion { get; set; } = string.Empty;
    public string DeviceType { get; set; } = string.Empty;
}
