namespace AIHost.ICompute.ROCm;

/// <summary>
/// ROCm/HIP compute kernel with HIPRTC runtime compilation
/// </summary>
public class ROCmComputeKernel : IComputeKernel
{
    private readonly string _source;
    private readonly string _entryPoint;
    private readonly List<object> _arguments = new();
    private IntPtr _module;
    private IntPtr _function;
    private IntPtr _stream;       // set by ROCmComputeCommandQueue before each dispatch
    private string? _gcnArch;     // set by ROCmComputeDevice after GPU architecture detection
    private bool _compiled;
    private bool _disposed;

    public string Name { get; }
    public KernelArgumentType[] ArgumentTypes => Array.Empty<KernelArgumentType>();

    public ROCmComputeKernel(string source, string entryPoint, string? gcnArch = null)
    {
        _source = source;
        _entryPoint = entryPoint;
        _gcnArch = gcnArch;
        Name = entryPoint;
    }

    /// <summary>Called by ROCmComputeCommandQueue before each Dispatch to route work to the queue's HIP stream.</summary>
    internal void SetStream(IntPtr stream) => _stream = stream;

    public void SetArgument(int index, IComputeBuffer buffer)
    {
        while (_arguments.Count <= index)
            _arguments.Add(null!);
        _arguments[index] = buffer;
    }

    public void SetArgument(int index, object value)
    {
        while (_arguments.Count <= index)
            _arguments.Add(null!);
        _arguments[index] = value;
    }

    public void Compile()
    {
        if (_compiled) return;

        // 1. Create HIPRTC program
        HipRtcApi.CheckError(
            HipRtcApi.hiprtcCreateProgram(out IntPtr prog, _source, _entryPoint, 0, null, null),
            "hiprtcCreateProgram");

        try
        {
            // 2. Compile: use detected architecture or fall back to offload-arch=native
            string archFlag = _gcnArch != null
                ? $"--offload-arch={_gcnArch}"
                : "--offload-arch=native";
            var options = new[] { archFlag };
            var result = HipRtcApi.hiprtcCompileProgram(prog, options.Length, options);

            if (result != HipRtcApi.HipRtcResult.Success)
            {
                // Get compilation log
                HipRtcApi.hiprtcGetProgramLogSize(prog, out ulong logSize);
                if (logSize > 0)
                {
                    var log = new byte[logSize];
                    HipRtcApi.hiprtcGetProgramLog(prog, log);
                    var logStr = System.Text.Encoding.UTF8.GetString(log);
                    throw new InvalidOperationException($"HIP kernel compilation failed:\n{logStr}");
                }
                throw new InvalidOperationException($"HIP kernel compilation failed with code: {result}");
            }

            // 3. Get compiled code
            HipRtcApi.CheckError(HipRtcApi.hiprtcGetCodeSize(prog, out ulong codeSize), "hiprtcGetCodeSize");
            var code = new byte[codeSize];
            HipRtcApi.CheckError(HipRtcApi.hiprtcGetCode(prog, code), "hiprtcGetCode");

            // 4. Load module
            unsafe
            {
                fixed (byte* pCode = code)
                {
                    HipApi.CheckError(
                        HipApi.hipModuleLoadData(out _module, (IntPtr)pCode),
                        "hipModuleLoadData");
                }
            }

            // 5. Get kernel function
            HipApi.CheckError(
                HipApi.hipModuleGetFunction(out _function, _module, _entryPoint),
                "hipModuleGetFunction");

            _compiled = true;
        }
        finally
        {
            // Cleanup HIPRTC program
            var tempProg = prog;
            HipRtcApi.hiprtcDestroyProgram(ref tempProg);
        }
    }

    public void Dispatch(uint[] globalWorkSize, uint[]? localWorkSize)
    {
        if (!_compiled)
            Compile();

        // Prepare kernel arguments as IntPtr array
        var kernelParams = new IntPtr[_arguments.Count];
        var pinnedHandles = new System.Runtime.InteropServices.GCHandle[_arguments.Count];

        try
        {
            for (int i = 0; i < _arguments.Count; i++)
            {
                var arg = _arguments[i];
                if (arg is IComputeBuffer buffer)
                {
                    var ptr = buffer.GetPointer();
                    pinnedHandles[i] = System.Runtime.InteropServices.GCHandle.Alloc(ptr, System.Runtime.InteropServices.GCHandleType.Pinned);
                    kernelParams[i] = pinnedHandles[i].AddrOfPinnedObject();
                }
                else if (arg is uint uintVal)
                {
                    pinnedHandles[i] = System.Runtime.InteropServices.GCHandle.Alloc(uintVal, System.Runtime.InteropServices.GCHandleType.Pinned);
                    kernelParams[i] = pinnedHandles[i].AddrOfPinnedObject();
                }
                else if (arg is float floatVal)
                {
                    pinnedHandles[i] = System.Runtime.InteropServices.GCHandle.Alloc(floatVal, System.Runtime.InteropServices.GCHandleType.Pinned);
                    kernelParams[i] = pinnedHandles[i].AddrOfPinnedObject();
                }
            }

            // Calculate grid and block dimensions
            uint blockX = localWorkSize != null && localWorkSize.Length > 0 ? localWorkSize[0] : 256;
            uint blockY = localWorkSize != null && localWorkSize.Length > 1 ? localWorkSize[1] : 1;
            uint blockZ = localWorkSize != null && localWorkSize.Length > 2 ? localWorkSize[2] : 1;

            uint gridX = (globalWorkSize[0] + blockX - 1) / blockX;
            uint gridY = globalWorkSize.Length > 1 ? (globalWorkSize[1] + blockY - 1) / blockY : 1;
            uint gridZ = globalWorkSize.Length > 2 ? (globalWorkSize[2] + blockZ - 1) / blockZ : 1;

            // Launch kernel on the stream set by the command queue (IntPtr.Zero = default stream)
            HipApi.CheckError(
                HipApi.hipModuleLaunchKernel(
                    _function,
                    gridX, gridY, gridZ,
                    blockX, blockY, blockZ,
                    0,       // shared memory bytes
                    _stream, // use queue's HIP stream for correct ordering
                    kernelParams,
                    null),   // extra
                "hipModuleLaunchKernel");
        }
        finally
        {
            // Unpin all handles
            foreach (var handle in pinnedHandles)
            {
                if (handle.IsAllocated)
                    handle.Free();
            }
        }
    }

    public void Dispose()
    {
        if (_disposed) return;

        if (_module != IntPtr.Zero)
        {
            HipApi.hipModuleUnload(_module); // Don't throw in Dispose
            _module = IntPtr.Zero;
        }

        _disposed = true;
    }
}
