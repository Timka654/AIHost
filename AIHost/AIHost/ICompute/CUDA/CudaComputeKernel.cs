namespace AIHost.ICompute.CUDA;

/// <summary>
/// CUDA compute kernel implementation with NVRTC runtime compilation
/// </summary>
public unsafe class CudaComputeKernel : IComputeKernel
{
    private IntPtr _module;
    private IntPtr _function;
    private readonly string _source;
    private readonly string _entryPoint;
    private readonly List<object> _arguments = new();
    private bool _disposed;

    public string Name => _entryPoint;
    public KernelArgumentType[] ArgumentTypes => Array.Empty<KernelArgumentType>(); // Simplified

    public CudaComputeKernel(string source, string entryPoint)
    {
        _source = source;
        _entryPoint = entryPoint;
        CompileAndLoad();
    }

    private void CompileAndLoad()
    {
        // Compile CUDA source to PTX using NVRTC
        var srcBytes = CudaApi.StringToNullTerminatedBytes(_source);
        var nameBytes = CudaApi.StringToNullTerminatedBytes("kernel.cu");

        var result = CudaApi.CreateProgram(
            out IntPtr prog,
            srcBytes,
            nameBytes,
            0,
            null!,
            null!);
        CudaApi.CheckNvrtcError(result, "nvrtcCreateProgram");

        try
        {
            // Compile options
            var options = new[]
            {
                CudaApi.StringToNullTerminatedBytes("--gpu-architecture=compute_75"), // SM 7.5 (Turing+)
                CudaApi.StringToNullTerminatedBytes("--std=c++17")
            };

            result = CudaApi.CompileProgram(prog, options.Length, options);
            
            if (result != NvrtcResult.Success)
            {
                // Get compilation log
                CudaApi.GetProgramLogSize(prog, out ulong logSize);
                byte[] log = new byte[logSize];
                CudaApi.GetProgramLog(prog, log);
                string logStr = System.Text.Encoding.UTF8.GetString(log);
                
                throw new CudaException($"NVRTC compilation failed:\n{logStr}");
            }

            // Get PTX
            CudaApi.GetPTXSize(prog, out ulong ptxSize);
            byte[] ptx = new byte[ptxSize];
            CudaApi.GetPTX(prog, ptx);

            // Load module from PTX
            var error = CudaApi.ModuleLoadData(out _module, ptx);
            CudaApi.CheckError(error, "cuModuleLoadData");

            // Get function
            var funcNameBytes = CudaApi.StringToNullTerminatedBytes(_entryPoint);
            error = CudaApi.ModuleGetFunction(out _function, _module, funcNameBytes);
            CudaApi.CheckError(error, "cuModuleGetFunction");
        }
        finally
        {
            CudaApi.DestroyProgram(ref prog);
        }
    }

    public void Execute(
        IComputeCommandQueue queue,
        uint globalWorkSizeX, uint globalWorkSizeY, uint globalWorkSizeZ,
        uint localWorkSizeX, uint localWorkSizeY, uint localWorkSizeZ,
        params IComputeBuffer[] buffers)
    {
        if (_disposed)
            throw new ObjectDisposedException(nameof(CudaComputeKernel));

        // Calculate grid dimensions (CUDA blocks)
        uint gridX = (globalWorkSizeX + localWorkSizeX - 1) / localWorkSizeX;
        uint gridY = (globalWorkSizeY + localWorkSizeY - 1) / localWorkSizeY;
        uint gridZ = (globalWorkSizeZ + localWorkSizeZ - 1) / localWorkSizeZ;

        // Prepare kernel parameters (buffer pointers)
        IntPtr* paramPtrs = stackalloc IntPtr[buffers.Length];
        for (int i = 0; i < buffers.Length; i++)
        {
            paramPtrs[i] = buffers[i].GetPointer();
        }

        // Launch kernel
        var cudaQueue = (CudaComputeCommandQueue)queue;
        var error = CudaApi.LaunchKernel(
            _function,
            gridX, gridY, gridZ,
            localWorkSizeX, localWorkSizeY, localWorkSizeZ,
            0, // shared memory
            cudaQueue.GetStream(),
            paramPtrs,
            null);
        
        CudaApi.CheckError(error, "cuLaunchKernel");
    }

    public void SetArgument(int index, object value)
    {
        if (index >= _arguments.Count)
        {
            _arguments.AddRange(new object[index - _arguments.Count + 1]);
        }
        _arguments[index] = value;
    }

    public void Dispatch(uint[] globalWorkSize, uint[]? localWorkSize = null)
    {
        // This requires a command queue - not directly supported
        throw new NotSupportedException("Use Execute() with IComputeCommandQueue instead");
    }

    public void Compile()
    {
        // Already compiled in constructor
    }

    public void Dispose()
    {
        if (_disposed) return;

        if (_module != IntPtr.Zero)
        {
            CudaApi.ModuleUnload(_module);
            _module = IntPtr.Zero;
        }

        _disposed = true;
    }
}
