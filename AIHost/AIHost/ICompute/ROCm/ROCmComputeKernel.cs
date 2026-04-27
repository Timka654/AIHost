namespace AIHost.ICompute.ROCm;

/// <summary>
/// ROCm/HIP compute kernel
/// TODO: Implement HIP kernel compilation and execution
/// </summary>
public class ROCmComputeKernel : IComputeKernel
{
    private readonly string _source;
    private readonly string _entryPoint;
    private readonly ROCmComputeDevice _device;
    private readonly List<object> _arguments = new();
    private bool _compiled;
    private bool _disposed;

    // TODO: Store hipModule_t and hipFunction_t
    // private IntPtr _module;
    // private IntPtr _function;

    public string Name { get; }
    public KernelArgumentType[] ArgumentTypes => Array.Empty<KernelArgumentType>();

    public ROCmComputeKernel(string source, string entryPoint, ROCmComputeDevice device)
    {
        _source = source;
        _entryPoint = entryPoint;
        _device = device;
        Name = entryPoint;
    }

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

    public void Dispatch(uint[] globalWorkSize, uint[]? localWorkSize)
    {
        throw new NotImplementedException();
        // TODO: Launch kernel
    }

    public void Compile()
    {
        if (_compiled) return;

        // TODO: Compile kernel
        // 1. Create hiprtcProgram with source
        // 2. Compile with hiprtcCompileProgram
        // 3. Get PTX/code with hiprtcGetCode
        // 4. Load module with hipModuleLoadData
        // 5. Get function with hipModuleGetFunction
        
        _compiled = true;
        throw new NotImplementedException("ROCm kernel compilation not implemented");
    }

    public void Dispose()
    {
        if (_disposed) return;
        
        // TODO: Cleanup module and function
        // if (_module != IntPtr.Zero)
        //     hipModuleUnload(_module);
        
        _disposed = true;
    }
}
