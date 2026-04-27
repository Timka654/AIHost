using Silk.NET.Vulkan;
using Silk.NET.Shaderc;

namespace AIHost.ICompute.Vulkan;

/// <summary>
/// Ядро вычислений для Vulkan (compute shader)
/// </summary>
internal unsafe class VulkanComputeKernel : ComputeKernelBase
{
    private readonly Vk _vk;
    private readonly Device _device;
    private readonly string _source;
    private readonly string _entryPoint;
    private ShaderModule _shaderModule;
    private Pipeline _pipeline;
    private PipelineLayout _pipelineLayout;
    private DescriptorSetLayout _descriptorSetLayout;
    private DescriptorPool _descriptorPool;
    private DescriptorSet _descriptorSet;
    private readonly List<IComputeBuffer> _bufferArguments = new();
    private bool _compiled;
    private bool _disposed;

    public override string Name => _entryPoint;

    internal VulkanComputeKernel(Vk vk, Device device, string source, string entryPoint)
    {
        _vk = vk;
        _device = device;
        _source = source;
        _entryPoint = entryPoint;
    }

    public override void SetArgument(int index, object value)
    {
        if (value is IComputeBuffer buffer)
        {
            while (_bufferArguments.Count <= index)
                _bufferArguments.Add(null!);
            _bufferArguments[index] = buffer;
        }
        else
        {
            throw new ArgumentException($"Argument type {value?.GetType().Name ?? "null"} not supported");
        }
    }

    public override void Compile()
    {
        if (_compiled) return;

        // Компиляция GLSL -> SPIR-V через Shaderc
        byte[] spirvBytes = CompileGLSLToSPIRV(_source, _entryPoint);
        
        // Создание shader module из SPIR-V
        fixed (byte* spirvPtr = spirvBytes)
        {
            var createInfo = new ShaderModuleCreateInfo
            {
                SType = StructureType.ShaderModuleCreateInfo,
                CodeSize = (nuint)spirvBytes.Length,
                PCode = (uint*)spirvPtr
            };

            if (_vk.CreateShaderModule(_device, &createInfo, null, out _shaderModule) != Result.Success)
                throw new InvalidOperationException("Failed to create shader module");
        }

        // Определяем количество binding'ов из GLSL source
        int bindingCount = ParseBindingCount(_source);
        
        // Создание descriptor set layout для буферов
        var layoutBindings = stackalloc DescriptorSetLayoutBinding[bindingCount];
        
        for (int i = 0; i < bindingCount; i++)
        {
            layoutBindings[i] = new DescriptorSetLayoutBinding
            {
                Binding = (uint)i,
                DescriptorType = DescriptorType.StorageBuffer,
                DescriptorCount = 1,
                StageFlags = ShaderStageFlags.ComputeBit
            };
        }

        var descriptorLayoutInfo = new DescriptorSetLayoutCreateInfo
        {
            SType = StructureType.DescriptorSetLayoutCreateInfo,
            BindingCount = (uint)bindingCount,
            PBindings = layoutBindings
        };

        if (_vk.CreateDescriptorSetLayout(_device, &descriptorLayoutInfo, null, out _descriptorSetLayout) != Result.Success)
            throw new InvalidOperationException("Failed to create descriptor set layout");

        // Создание pipeline layout
        var descriptorSetLayoutLocal = _descriptorSetLayout;
        var pipelineLayoutInfo = new PipelineLayoutCreateInfo
        {
            SType = StructureType.PipelineLayoutCreateInfo,
            SetLayoutCount = 1,
            PSetLayouts = &descriptorSetLayoutLocal
        };

        if (_vk.CreatePipelineLayout(_device, &pipelineLayoutInfo, null, out _pipelineLayout) != Result.Success)
            throw new InvalidOperationException("Failed to create pipeline layout");

        // Создание compute pipeline
        var entryPointBytes = System.Text.Encoding.UTF8.GetBytes(_entryPoint + "\0");
        fixed (byte* pName = entryPointBytes)
        {
            var stageInfo = new PipelineShaderStageCreateInfo
            {
                SType = StructureType.PipelineShaderStageCreateInfo,
                Stage = ShaderStageFlags.ComputeBit,
                Module = _shaderModule,
                PName = pName
            };

            var pipelineInfo = new ComputePipelineCreateInfo
            {
                SType = StructureType.ComputePipelineCreateInfo,
                Stage = stageInfo,
                Layout = _pipelineLayout
            };

            if (_vk.CreateComputePipelines(_device, default, 1, &pipelineInfo, null, out _pipeline) != Result.Success)
                throw new InvalidOperationException("Failed to create compute pipeline");
        }

        // Создание descriptor pool
        var poolSize = new DescriptorPoolSize
        {
            Type = DescriptorType.StorageBuffer,
            DescriptorCount = (uint)bindingCount
        };

        var poolInfo = new DescriptorPoolCreateInfo
        {
            SType = StructureType.DescriptorPoolCreateInfo,
            MaxSets = 1,
            PoolSizeCount = 1,
            PPoolSizes = &poolSize
        };

        if (_vk.CreateDescriptorPool(_device, &poolInfo, null, out _descriptorPool) != Result.Success)
            throw new InvalidOperationException("Failed to create descriptor pool");

        // Выделение descriptor set
        var descriptorSetLayoutForAlloc = _descriptorSetLayout;
        var allocInfo = new DescriptorSetAllocateInfo
        {
            SType = StructureType.DescriptorSetAllocateInfo,
            DescriptorPool = _descriptorPool,
            DescriptorSetCount = 1,
            PSetLayouts = &descriptorSetLayoutForAlloc
        };

        if (_vk.AllocateDescriptorSets(_device, &allocInfo, out _descriptorSet) != Result.Success)
            throw new InvalidOperationException("Failed to allocate descriptor set");

        _compiled = true;
    }

    private byte[] CompileGLSLToSPIRV(string glslSource, string entryPoint)
    {
        var shaderc = Shaderc.GetApi();
        
        var compiler = shaderc.CompilerInitialize();
        if (compiler == null)
            throw new InvalidOperationException("Failed to initialize shaderc compiler");

        var options = shaderc.CompileOptionsInitialize();
        shaderc.CompileOptionsSetTargetEnv(options, (int)TargetEnv.Vulkan, (uint)EnvVersion.Vulkan13);
        shaderc.CompileOptionsSetOptimizationLevel(options, OptimizationLevel.Performance);

        var result = shaderc.CompileIntoSpv(
            compiler,
            glslSource,
            (nuint)glslSource.Length,
            ShaderKind.ComputeShader,
            "shader.comp",
            entryPoint,
            options
        );

        var status = shaderc.ResultGetCompilationStatus(result);
        if (status != CompilationStatus.Success)
        {
            var errorBytes = shaderc.ResultGetErrorMessage(result);
            string errorMessage = System.Runtime.InteropServices.Marshal.PtrToStringAnsi((nint)errorBytes) ?? "Unknown error";
            shaderc.ResultRelease(result);
            shaderc.CompileOptionsRelease(options);
            shaderc.CompilerRelease(compiler);
            throw new InvalidOperationException($"GLSL compilation failed: {errorMessage}");
        }

        nuint spirvLength = shaderc.ResultGetLength(result);
        byte[] spirvBytes = new byte[spirvLength];
        
        var spirvPtr = shaderc.ResultGetBytes(result);
        fixed (byte* dest = spirvBytes)
        {
            System.Buffer.MemoryCopy(
                spirvPtr, 
                dest, 
                spirvBytes.Length, 
                (long)spirvLength
            );
        }

        shaderc.ResultRelease(result);
        shaderc.CompileOptionsRelease(options);
        shaderc.CompilerRelease(compiler);

        return spirvBytes;
    }

    public void UpdateDescriptorSets()
    {
        if (!_compiled)
            throw new InvalidOperationException("Kernel must be compiled before updating descriptor sets");

        var writeDescriptorSets = stackalloc WriteDescriptorSet[_bufferArguments.Count];
        var bufferInfos = stackalloc DescriptorBufferInfo[_bufferArguments.Count];

        for (int i = 0; i < _bufferArguments.Count; i++)
        {
            if (_bufferArguments[i] is not VulkanComputeBuffer vkBuffer)
                throw new InvalidOperationException($"Buffer at index {i} is not a VulkanComputeBuffer");

            bufferInfos[i] = new DescriptorBufferInfo
            {
                Buffer = vkBuffer.VkBuffer,
                Offset = 0,
                Range = vkBuffer.Size
            };

            writeDescriptorSets[i] = new WriteDescriptorSet
            {
                SType = StructureType.WriteDescriptorSet,
                DstSet = _descriptorSet,
                DstBinding = (uint)i,
                DstArrayElement = 0,
                DescriptorCount = 1,
                DescriptorType = DescriptorType.StorageBuffer,
                PBufferInfo = &bufferInfos[i]
            };
        }

        _vk.UpdateDescriptorSets(_device, (uint)_bufferArguments.Count, writeDescriptorSets, 0, null);
    }

    public override void Dispatch(uint[] globalWorkSize, uint[]? localWorkSize = null)
    {
        if (!_compiled)
            Compile();

        if (globalWorkSize == null || globalWorkSize.Length == 0)
            throw new ArgumentException("Global work size must be specified");
    }

    /// <summary>
    /// Парсит GLSL source и определяет количество binding'ов
    /// </summary>
    private static int ParseBindingCount(string glslSource)
    {
        int maxBinding = -1;
        
        // Ищем все "binding = N"
        var lines = glslSource.Split('\n');
        foreach (var line in lines)
        {
            var trimmed = line.Trim();
            if (!trimmed.Contains("binding")) continue;
            
            // Ищем паттерн "binding = число"
            var bindingIndex = trimmed.IndexOf("binding");
            if (bindingIndex < 0) continue;
            
            var equalsIndex = trimmed.IndexOf('=', bindingIndex);
            if (equalsIndex < 0) continue;
            
            var afterEquals = trimmed.Substring(equalsIndex + 1).Trim();
            var endIndex = afterEquals.IndexOfAny(new[] { ')', ',', ' ', '\t' });
            if (endIndex < 0) endIndex = afterEquals.Length;
            
            var numberStr = afterEquals.Substring(0, endIndex).Trim();
            if (int.TryParse(numberStr, out int bindingNum))
            {
                if (bindingNum > maxBinding)
                    maxBinding = bindingNum;
            }
        }
        
        // Возвращаем количество (maxBinding + 1), минимум 2
        return Math.Max(maxBinding + 1, 2);
    }

    public override void Dispose()
    {
        if (_disposed) return;

        if (_descriptorPool.Handle != 0)
            _vk.DestroyDescriptorPool(_device, _descriptorPool, null);

        if (_pipeline.Handle != 0)
            _vk.DestroyPipeline(_device, _pipeline, null);
        
        if (_pipelineLayout.Handle != 0)
            _vk.DestroyPipelineLayout(_device, _pipelineLayout, null);
        
        if (_descriptorSetLayout.Handle != 0)
            _vk.DestroyDescriptorSetLayout(_device, _descriptorSetLayout, null);
        
        if (_shaderModule.Handle != 0)
            _vk.DestroyShaderModule(_device, _shaderModule, null);

        _disposed = true;
    }

    internal Pipeline Pipeline => _pipeline;
    internal PipelineLayout PipelineLayout => _pipelineLayout;
    internal DescriptorSet DescriptorSet => _descriptorSet;
}
