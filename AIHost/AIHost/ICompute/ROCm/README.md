# ROCm/HIP Provider

ROCm (Radeon Open Compute) provider для AMD GPU acceleration.

## Требования

### Windows
- AMD Radeon RX 5000+ series (RDNA/RDNA2/RDNA3)
- ROCm 6.0+ for Windows (https://rocm.docs.amd.com/en/latest/deploy/windows/index.html)
- Libraries: `amdhip64.dll`, `hiprtc.dll`

### Linux
- AMD Radeon RX 5000+ series
- ROCm 6.0+ (https://rocm.docs.amd.com/en/latest/deploy/linux/index.html)
- Libraries: `libamdhip64.so`, `libhiprtc.so`

## Архитектура

```
AIHost/ICompute/ROCm/
├── HipApi.cs                    # HIP Runtime API bindings (P/Invoke)
├── ROCmComputeDevice.cs         # Device initialization & management
├── ROCmComputeBuffer.cs         # GPU memory (hipMalloc/hipMemcpy)
├── ROCmComputeKernel.cs         # Kernel compilation (HIPRTC) & launch
└── ROCmComputeCommandQueue.cs   # HIP streams & synchronization

AIHost/Shaders/ROCm/
├── matmul.hip                   # Matrix multiplication
├── softmax.hip                  # Softmax activation
├── silu.hip                     # SiLU/Swish activation
├── add.hip                      # Element-wise addition
└── concat_axis1.hip             # Tensor concatenation
```

## Использование

```csharp
// Инициализация ROCm device
using var device = new ROCmComputeDevice(deviceId: 0);

// Создание buffers
var bufferA = device.CreateBuffer(size, BufferType.Storage, DataType.F32);
var bufferB = device.CreateBuffer(size, BufferType.Storage, DataType.F32);

// Запись данных
bufferA.Write(dataA);
bufferB.Write(dataB);

// Создание kernel
var kernel = device.CreateKernel(hipSource, "kernel_name");
kernel.SetArgument(0, bufferA);
kernel.SetArgument(1, bufferB);

// Dispatch
var queue = device.CreateCommandQueue();
queue.Dispatch(kernel, new uint[] { workSize }, null);
queue.Flush();

// Чтение результатов
var result = bufferB.Read<float>();
```

## HIP API

### Device Management
- `hipInit()` - Initialize HIP runtime
- `hipGetDeviceCount()` - Get available GPU count
- `hipSetDevice()` - Set active device
- `hipDeviceSynchronize()` - Wait for device operations

### Memory Management
- `hipMalloc()` - Allocate device memory
- `hipFree()` - Free device memory
- `hipMemcpy()` - Synchronous memory copy
- `hipMemcpyAsync()` - Asynchronous memory copy with stream

### Kernel Execution
- `hipModuleLoadData()` - Load compiled module
- `hipModuleGetFunction()` - Get kernel function
- `hipModuleLaunchKernel()` - Launch kernel with grid/block dimensions

### Runtime Compilation (HIPRTC)
- `hiprtcCreateProgram()` - Create compilation program
- `hiprtcCompileProgram()` - Compile HIP source to binary
- `hiprtcGetCode()` - Retrieve compiled code
- `hiprtcGetProgramLog()` - Get compilation errors

## Совместимость с Vulkan

ROCm provider использует ту же абстракцию `IComputeDevice`, что и Vulkan:
- Shader source: GLSL → HIP/C++ (автоматическая портация ShaderLoader)
- API calls: Vulkan semantics → HIP native API
- Memory model: Unified через `IComputeBuffer` interface

## Производительность

AMD Radeon RX 6000/7000 series показывают отличную производительность с HIP:
- RX 6600 XT (8GB): ~11 TFLOPS FP32
- RX 6700 XT (12GB): ~13 TFLOPS FP32  
- RX 6800 (16GB): ~16 TFLOPS FP32
- RX 7900 XTX (24GB): ~61 TFLOPS FP32

## Debugging

### Проверка установки
```powershell
# Windows
where amdhip64.dll
where hiprtc.dll

# Linux
ldconfig -p | grep hip
```

### Device info
```csharp
var device = new ROCmComputeDevice();
// Console output:
// ROCm Device: AMD Radeon RX 6600 XT
// API Version: HIP 6.0
// Compute Capability: 10.3
// Multiprocessors: 32
// Global Memory: 8192 MB
```

## Ограничения

1. **Windows Support**: ROCm для Windows в beta-состоянии (ROCm 6.0+)
2. **GPU Compatibility**: Требуется RDNA+ архитектура (RX 5000+)
3. **Compilation Time**: HIPRTC компилирует на лету (первый запуск медленнее)
4. **Library Dependencies**: Требуется установка полного ROCm stack

## Roadmap

- [x] HIP API bindings (hipMalloc, hipMemcpy, hipModuleLaunch)
- [x] HIPRTC runtime compilation
- [x] Basic compute kernels (matmul, softmax, silu, add)
- [ ] Quantization kernels (Q2_K, Q3_K, Q4_K, Q5_K, Q6_K)
- [ ] Optimized attention kernels
- [ ] ROCm-specific optimizations (LDS, async copies)
- [ ] Multi-GPU support (peer-to-peer)
- [ ] rocBLAS integration for GEMM
