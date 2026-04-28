using System.Runtime.InteropServices;

namespace AIHost.ICompute.CUDA;

/// <summary>
/// CUDA Runtime API P/Invoke bindings
/// Minimal set of functions needed for compute operations
/// </summary>
public static unsafe class CudaApi
{
    private const string CudaLibrary = "cudart64_12"; // CUDA 12.x
    private const string NvrtcLibrary = "nvrtc64_120_0"; // NVRTC for runtime compilation

    #region Device Management

    [DllImport(CudaLibrary, EntryPoint = "cudaGetDeviceCount")]
    public static extern CudaError GetDeviceCount(out int count);

    [DllImport(CudaLibrary, EntryPoint = "cudaSetDevice")]
    public static extern CudaError SetDevice(int device);

    [DllImport(CudaLibrary, EntryPoint = "cudaGetDevice")]
    public static extern CudaError GetDevice(out int device);

    [DllImport(CudaLibrary, EntryPoint = "cudaGetDeviceProperties")]
    public static extern CudaError GetDeviceProperties(out CudaDeviceProp prop, int device);

    [DllImport(CudaLibrary, EntryPoint = "cudaDeviceSynchronize")]
    public static extern CudaError DeviceSynchronize();

    [DllImport(CudaLibrary, EntryPoint = "cudaDeviceReset")]
    public static extern CudaError DeviceReset();

    #endregion

    #region Memory Management

    [DllImport(CudaLibrary, EntryPoint = "cudaMalloc")]
    public static extern CudaError Malloc(out IntPtr devPtr, ulong size);

    [DllImport(CudaLibrary, EntryPoint = "cudaFree")]
    public static extern CudaError Free(IntPtr devPtr);

    [DllImport(CudaLibrary, EntryPoint = "cudaMemcpy")]
    public static extern CudaError Memcpy(IntPtr dst, IntPtr src, ulong count, CudaMemcpyKind kind);

    [DllImport(CudaLibrary, EntryPoint = "cudaMemset")]
    public static extern CudaError Memset(IntPtr devPtr, int value, ulong count);

    [DllImport(CudaLibrary, EntryPoint = "cudaMemGetInfo")]
    public static extern CudaError MemGetInfo(out ulong free, out ulong total);

    #endregion

    #region Module and Function Management

    [DllImport(CudaLibrary, EntryPoint = "cuModuleLoadData")]
    public static extern CudaError ModuleLoadData(out IntPtr module, byte[] image);

    [DllImport(CudaLibrary, EntryPoint = "cuModuleGetFunction")]
    public static extern CudaError ModuleGetFunction(out IntPtr hfunc, IntPtr hmod, byte[] name);

    [DllImport(CudaLibrary, EntryPoint = "cuModuleUnload")]
    public static extern CudaError ModuleUnload(IntPtr hmod);

    [DllImport(CudaLibrary, EntryPoint = "cuLaunchKernel")]
    public static extern CudaError LaunchKernel(
        IntPtr f,
        uint gridDimX, uint gridDimY, uint gridDimZ,
        uint blockDimX, uint blockDimY, uint blockDimZ,
        uint sharedMemBytes,
        IntPtr hStream,
        IntPtr* kernelParams,
        IntPtr* extra);

    #endregion

    #region NVRTC - Runtime Compilation

    [DllImport(NvrtcLibrary, EntryPoint = "nvrtcCreateProgram")]
    public static extern NvrtcResult CreateProgram(
        out IntPtr prog,
        byte[] src,
        byte[] name,
        int numHeaders,
        byte[][] headers,
        byte[][] includeNames);

    [DllImport(NvrtcLibrary, EntryPoint = "nvrtcDestroyProgram")]
    public static extern NvrtcResult DestroyProgram(ref IntPtr prog);

    [DllImport(NvrtcLibrary, EntryPoint = "nvrtcCompileProgram")]
    public static extern NvrtcResult CompileProgram(IntPtr prog, int numOptions, byte[][] options);

    [DllImport(NvrtcLibrary, EntryPoint = "nvrtcGetPTXSize")]
    public static extern NvrtcResult GetPTXSize(IntPtr prog, out ulong ptxSizeRet);

    [DllImport(NvrtcLibrary, EntryPoint = "nvrtcGetPTX")]
    public static extern NvrtcResult GetPTX(IntPtr prog, byte[] ptx);

    [DllImport(NvrtcLibrary, EntryPoint = "nvrtcGetProgramLogSize")]
    public static extern NvrtcResult GetProgramLogSize(IntPtr prog, out ulong logSizeRet);

    [DllImport(NvrtcLibrary, EntryPoint = "nvrtcGetProgramLog")]
    public static extern NvrtcResult GetProgramLog(IntPtr prog, byte[] log);

    #endregion

    #region Stream Management

    [DllImport(CudaLibrary, EntryPoint = "cudaStreamCreate")]
    public static extern CudaError StreamCreate(out IntPtr stream);

    [DllImport(CudaLibrary, EntryPoint = "cudaStreamSynchronize")]
    public static extern CudaError StreamSynchronize(IntPtr stream);

    [DllImport(CudaLibrary, EntryPoint = "cudaStreamDestroy")]
    public static extern CudaError StreamDestroy(IntPtr stream);

    #endregion

    #region Helper Methods

    public static void CheckError(CudaError error, string operation)
    {
        if (error != CudaError.Success)
            throw new CudaException($"CUDA error in {operation}: {error}");
    }

    public static void CheckNvrtcError(NvrtcResult result, string operation)
    {
        if (result != NvrtcResult.Success)
            throw new CudaException($"NVRTC error in {operation}: {result}");
    }

    public static byte[] StringToNullTerminatedBytes(string str)
    {
        var bytes = System.Text.Encoding.UTF8.GetBytes(str);
        var result = new byte[bytes.Length + 1];
        Array.Copy(bytes, result, bytes.Length);
        result[bytes.Length] = 0;
        return result;
    }

    #endregion
}

/// <summary>
/// CUDA error codes
/// </summary>
public enum CudaError
{
    Success = 0,
    InvalidValue = 1,
    OutOfMemory = 2,
    NotInitialized = 3,
    Deinitialized = 4,
    NoDevice = 100,
    InvalidDevice = 101,
    InvalidMemcpyDirection = 21,
    Unknown = 999
}

/// <summary>
/// NVRTC compilation result codes
/// </summary>
public enum NvrtcResult
{
    Success = 0,
    OutOfMemory = 1,
    ProgramCreationFailure = 2,
    InvalidInput = 3,
    InvalidProgram = 4,
    InvalidOption = 5,
    Compilation = 6,
    BuiltinOperationFailure = 7,
    NoNameExpression = 8,
    NoLoweredNamesBeforeCompilation = 9,
    NameExpressionNotValid = 10,
    InternalError = 11
}

/// <summary>
/// Memory copy direction
/// </summary>
public enum CudaMemcpyKind
{
    HostToHost = 0,
    HostToDevice = 1,
    DeviceToHost = 2,
    DeviceToDevice = 3,
    Default = 4
}

/// <summary>
/// CUDA device properties
/// </summary>
[StructLayout(LayoutKind.Sequential)]
public unsafe struct CudaDeviceProp
{
    public fixed byte name[256];
    public ulong totalGlobalMem;
    public ulong sharedMemPerBlock;
    public int regsPerBlock;
    public int warpSize;
    public ulong memPitch;
    public int maxThreadsPerBlock;
    public fixed int maxThreadsDim[3];
    public fixed int maxGridSize[3];
    public int clockRate;
    public ulong totalConstMem;
    public int major;
    public int minor;
    public ulong textureAlignment;
    public ulong texturePitchAlignment;
    public int deviceOverlap;
    public int multiProcessorCount;
    public int kernelExecTimeoutEnabled;
    public int integrated;
    public int canMapHostMemory;
    public int computeMode;
    public int maxTexture1D;
    public fixed int maxTexture2D[2];
    public fixed int maxTexture3D[3];
    // Additional fields omitted for brevity
}

/// <summary>
/// CUDA exception
/// </summary>
public class CudaException : Exception
{
    public CudaException(string message) : base(message) { }
    public CudaException(string message, Exception inner) : base(message, inner) { }
}
