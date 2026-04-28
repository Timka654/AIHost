using System.Runtime.InteropServices;

namespace AIHost.ICompute.ROCm;

/// <summary>
/// HIP Runtime API P/Invoke bindings for ROCm
/// Documentation: https://rocm.docs.amd.com/projects/HIP/en/latest/reference/index.html
/// </summary>
internal static class HipApi
{
    private const string HipLibrary = "amdhip64"; // Windows: amdhip64.dll, Linux: libamdhip64.so

    #region Error Handling

    public enum HipError
    {
        Success = 0,
        InvalidValue = 1,
        OutOfMemory = 2,
        NotInitialized = 3,
        Deinitialized = 4,
        NoDevice = 100,
        InvalidDevice = 101,
        InvalidImage = 200,
        InvalidContext = 201,
        ContextAlreadyCurrent = 202,
        MapFailed = 205,
        UnmapFailed = 206,
        NoBinaryForGpu = 209,
        AlreadyAcquired = 210,
        NotMapped = 211,
        InvalidSource = 300,
        FileNotFound = 301,
        InvalidHandle = 400,
        NotFound = 500,
        NotReady = 600,
        LaunchFailure = 700,
        RuntimeMemory = 702,
        RuntimeOther = 799,
        Unknown = 999
    }

    public static void CheckError(HipError error, string operation = "")
    {
        if (error != HipError.Success)
            throw new InvalidOperationException($"HIP error during {operation}: {error}");
    }

    #endregion

    #region Device Management

    [DllImport(HipLibrary, CallingConvention = CallingConvention.Cdecl)]
    public static extern HipError hipInit(uint flags);

    [DllImport(HipLibrary, CallingConvention = CallingConvention.Cdecl)]
    public static extern HipError hipGetDeviceCount(out int count);

    [DllImport(HipLibrary, CallingConvention = CallingConvention.Cdecl)]
    public static extern HipError hipSetDevice(int deviceId);

    [DllImport(HipLibrary, CallingConvention = CallingConvention.Cdecl)]
    public static extern HipError hipGetDevice(out int deviceId);

    [DllImport(HipLibrary, CallingConvention = CallingConvention.Cdecl)]
    public static extern HipError hipDeviceSynchronize();

    [DllImport(HipLibrary, CallingConvention = CallingConvention.Cdecl)]
    public static extern HipError hipDeviceReset();

    [StructLayout(LayoutKind.Sequential)]
    public struct HipDeviceProp
    {
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 256)]
        public byte[] name;
        public ulong totalGlobalMem;
        public ulong sharedMemPerBlock;
        public int regsPerBlock;
        public int warpSize;
        public ulong memPitch;
        public int maxThreadsPerBlock;
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 3)]
        public int[] maxThreadsDim;
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 3)]
        public int[] maxGridSize;
        public int clockRate;
        public ulong totalConstMem;
        public int major;
        public int minor;
        public int multiProcessorCount;
        public int l2CacheSize;
        public int maxThreadsPerMultiProcessor;
        public int computeMode;
        public int clockInstructionRate;
        // ... additional fields as needed
    }

    [DllImport(HipLibrary, CallingConvention = CallingConvention.Cdecl)]
    public static extern HipError hipGetDeviceProperties(out HipDeviceProp prop, int deviceId);

    #endregion

    #region Memory Management

    [DllImport(HipLibrary, CallingConvention = CallingConvention.Cdecl)]
    public static extern HipError hipMalloc(out IntPtr devPtr, ulong size);

    [DllImport(HipLibrary, CallingConvention = CallingConvention.Cdecl)]
    public static extern HipError hipFree(IntPtr devPtr);

    [DllImport(HipLibrary, CallingConvention = CallingConvention.Cdecl)]
    public static extern HipError hipMemcpy(IntPtr dst, IntPtr src, ulong sizeBytes, HipMemcpyKind kind);

    [DllImport(HipLibrary, CallingConvention = CallingConvention.Cdecl)]
    public static extern HipError hipMemcpyAsync(IntPtr dst, IntPtr src, ulong sizeBytes, HipMemcpyKind kind, IntPtr stream);

    [DllImport(HipLibrary, CallingConvention = CallingConvention.Cdecl)]
    public static extern HipError hipMemset(IntPtr devPtr, int value, ulong sizeBytes);

    public enum HipMemcpyKind
    {
        HostToHost = 0,
        HostToDevice = 1,
        DeviceToHost = 2,
        DeviceToDevice = 3,
        Default = 4
    }

    #endregion

    #region Stream Management

    [DllImport(HipLibrary, CallingConvention = CallingConvention.Cdecl)]
    public static extern HipError hipStreamCreate(out IntPtr stream);

    [DllImport(HipLibrary, CallingConvention = CallingConvention.Cdecl)]
    public static extern HipError hipStreamDestroy(IntPtr stream);

    [DllImport(HipLibrary, CallingConvention = CallingConvention.Cdecl)]
    public static extern HipError hipStreamSynchronize(IntPtr stream);

    [DllImport(HipLibrary, CallingConvention = CallingConvention.Cdecl)]
    public static extern HipError hipStreamWaitEvent(IntPtr stream, IntPtr hipEvent, uint flags);

    #endregion

    #region Module & Kernel Management

    [DllImport(HipLibrary, CallingConvention = CallingConvention.Cdecl)]
    public static extern HipError hipModuleLoad(out IntPtr module, string fname);

    [DllImport(HipLibrary, CallingConvention = CallingConvention.Cdecl)]
    public static extern HipError hipModuleLoadData(out IntPtr module, IntPtr image);

    [DllImport(HipLibrary, CallingConvention = CallingConvention.Cdecl)]
    public static extern HipError hipModuleUnload(IntPtr module);

    [DllImport(HipLibrary, CallingConvention = CallingConvention.Cdecl)]
    public static extern HipError hipModuleGetFunction(out IntPtr function, IntPtr module, string name);

    [DllImport(HipLibrary, CallingConvention = CallingConvention.Cdecl)]
    public static extern HipError hipModuleLaunchKernel(
        IntPtr function,
        uint gridDimX, uint gridDimY, uint gridDimZ,
        uint blockDimX, uint blockDimY, uint blockDimZ,
        uint sharedMemBytes,
        IntPtr stream,
        IntPtr[] kernelParams,
        IntPtr[] extra);

    #endregion

    #region Event Management

    [DllImport(HipLibrary, CallingConvention = CallingConvention.Cdecl)]
    public static extern HipError hipEventCreate(out IntPtr hipEvent);

    [DllImport(HipLibrary, CallingConvention = CallingConvention.Cdecl)]
    public static extern HipError hipEventDestroy(IntPtr hipEvent);

    [DllImport(HipLibrary, CallingConvention = CallingConvention.Cdecl)]
    public static extern HipError hipEventRecord(IntPtr hipEvent, IntPtr stream);

    [DllImport(HipLibrary, CallingConvention = CallingConvention.Cdecl)]
    public static extern HipError hipEventSynchronize(IntPtr hipEvent);

    #endregion
}

/// <summary>
/// HIPRTC (HIP Runtime Compilation) API bindings
/// Used for compiling HIP kernels at runtime (like Shaderc for Vulkan)
/// </summary>
internal static class HipRtcApi
{
    private const string HipRtcLibrary = "hiprtc"; // Windows: hiprtc.dll, Linux: libhiprtc.so

    public enum HipRtcResult
    {
        Success = 0,
        OutOfMemory = 1,
        ProgramCreationFailure = 2,
        InvalidInput = 3,
        InvalidProgram = 4,
        InvalidOption = 5,
        Compilation = 6,
        BuildinNameExpressionFailure = 7,
        NameExpressionNotValid = 8,
        InternalError = 9
    }

    public static void CheckError(HipRtcResult result, string operation = "")
    {
        if (result != HipRtcResult.Success)
            throw new InvalidOperationException($"HIPRTC error during {operation}: {result}");
    }

    [DllImport(HipRtcLibrary, CallingConvention = CallingConvention.Cdecl)]
    public static extern HipRtcResult hiprtcCreateProgram(
        out IntPtr prog,
        string src,
        string name,
        int numHeaders,
        string[] headers,
        string[] includeNames);

    [DllImport(HipRtcLibrary, CallingConvention = CallingConvention.Cdecl)]
    public static extern HipRtcResult hiprtcDestroyProgram(ref IntPtr prog);

    [DllImport(HipRtcLibrary, CallingConvention = CallingConvention.Cdecl)]
    public static extern HipRtcResult hiprtcCompileProgram(IntPtr prog, int numOptions, string[] options);

    [DllImport(HipRtcLibrary, CallingConvention = CallingConvention.Cdecl)]
    public static extern HipRtcResult hiprtcGetCodeSize(IntPtr prog, out ulong codeSizeRet);

    [DllImport(HipRtcLibrary, CallingConvention = CallingConvention.Cdecl)]
    public static extern HipRtcResult hiprtcGetCode(IntPtr prog, byte[] code);

    [DllImport(HipRtcLibrary, CallingConvention = CallingConvention.Cdecl)]
    public static extern HipRtcResult hiprtcGetProgramLogSize(IntPtr prog, out ulong logSizeRet);

    [DllImport(HipRtcLibrary, CallingConvention = CallingConvention.Cdecl)]
    public static extern HipRtcResult hiprtcGetProgramLog(IntPtr prog, byte[] log);
}
