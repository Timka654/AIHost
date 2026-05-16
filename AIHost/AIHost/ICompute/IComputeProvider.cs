namespace AIHost.ICompute;

/// <summary>
/// Базовый класс для реализации провайдера вычислений
/// </summary>
public abstract class ComputeProviderBase : IComputeDevice
{
    public abstract string ProviderName { get; }
    public abstract string ApiVersion { get; }

    public abstract IComputeBuffer CreateBuffer(ulong size, BufferType type, DataType elementType = DataType.F32, bool requireDeviceLocal = false);
    public abstract IComputeKernel CreateKernel(string source, string entryPoint);
    public abstract IComputeKernel CreateKernelFromFile(string filePath, string entryPoint);
    public abstract IComputeCommandQueue CreateCommandQueue();
    public abstract void Synchronize();
    public abstract void Dispose();

    /// <summary>
    /// Получить карту heap'ов устройства. По умолчанию — пустой массив.
    /// Провайдеры с нативным запросом (Vulkan) переопределяют.
    /// </summary>
    public virtual VramHeapInfo[] GetHeapInfo() => Array.Empty<VramHeapInfo>();

    /// <summary>
    /// По умолчанию возвращает информацию только из трекера аллокаций.
    /// Провайдеры, поддерживающие нативный запрос памяти (Vulkan), переопределяют этот метод.
    /// </summary>
    public virtual DeviceMemoryInfo GetMemoryInfo()
    {
        return new DeviceMemoryInfo
        {
            TotalBytes = 0,
            AvailableBytes = 0,
            UsedBytes = 0,
            TrackedAllocatedBytes = ComputeBufferBase.TotalAllocatedBytes,
            SupportsNativeQuery = false
        };
    }
}

/// <summary>
/// Базовый класс для реализации буфера
/// </summary>
public abstract class ComputeBufferBase : IComputeBuffer
{
    // ── Статические поля для трекинга (backing fields для Interlocked) ──
    private static long _totalAllocatedBytes;
    private static long _peakAllocatedBytes;
    private static long _totalBufferCount;
    private static long _activeBufferCount;
    private static long _allocationCount;
    private static long _freeCount;

    // ── Allocation tracking stack traces (for leak debugging) ────────────
    private static readonly System.Collections.Concurrent.ConcurrentQueue<string> _allocTraces = new();
    private static readonly System.Collections.Concurrent.ConcurrentQueue<string> _freeTraces = new();

    /// <summary>Last N allocation stack traces (most recent first).</summary>
    public static string[] GetAllocTraces(int n = 50)
    {
        var all = _allocTraces.ToArray();
        return all.Reverse().Take(n).ToArray();
    }

    /// <summary>Last N free stack traces (most recent first).</summary>
    public static string[] GetFreeTraces(int n = 50)
    {
        var all = _freeTraces.ToArray();
        return all.Reverse().Take(n).ToArray();
    }

    /// <summary>Get allocation stats as a formatted string.</summary>
    public static string GetStats()
    {
        long alloc = Interlocked.Read(ref _allocationCount);
        long free = Interlocked.Read(ref _freeCount);
        long active = Interlocked.Read(ref _activeBufferCount);
        long total = Interlocked.Read(ref _totalAllocatedBytes);
        long peak = Interlocked.Read(ref _peakAllocatedBytes);
        return $"alloc={alloc} free={free} active={active} totalMB={total / 1024 / 1024} peakMB={peak / 1024 / 1024}";
    }

    /// <summary>Глобальный трекер выделенной памяти через все буферы (в байтах).</summary>
    public static long TotalAllocatedBytes => Interlocked.Read(ref _totalAllocatedBytes);

    /// <summary>Пиковое значение выделенной памяти (для статистики).</summary>
    public static long PeakAllocatedBytes => Interlocked.Read(ref _peakAllocatedBytes);

    /// <summary>Общее количество созданных буферов.</summary>
    public static long TotalBufferCount => Interlocked.Read(ref _totalBufferCount);

    /// <summary>Количество активных (не освобожденных) буферов.</summary>
    public static long ActiveBufferCount => Interlocked.Read(ref _activeBufferCount);

    /// <summary>Счетчик аллокаций (для отладки утечек).</summary>
    public static long AllocationCount => Interlocked.Read(ref _allocationCount);

    /// <summary>Счетчик освобождений.</summary>
    public static long FreeCount => Interlocked.Read(ref _freeCount);

    /// <summary>
    /// Уведомить трекер о выделении памяти.
    /// Вызывается из конструкторов конкретных буферов.
    /// </summary>
    protected static void TrackAllocate(ulong size)
    {
        Interlocked.Add(ref _totalAllocatedBytes, (long)size);
        Interlocked.Increment(ref _totalBufferCount);
        Interlocked.Increment(ref _activeBufferCount);
        Interlocked.Increment(ref _allocationCount);

        // Capture stack trace for leak debugging (truncated to 3 deepest frames)
        var st = Environment.StackTrace;
        var lines = st.Split('\n', StringSplitOptions.RemoveEmptyEntries);
        var brief = lines.Length > 3
            ? string.Join(" <- ", lines.Skip(2).Take(3).Select(l => l.TrimStart()))
            : st.Replace('\n', ' ');
        _allocTraces.Enqueue($"[ALLOC  +{size / 1024}KB] active={_activeBufferCount} totalMB={_totalAllocatedBytes / 1024 / 1024} {brief}");
        // Trim excess entries
        while (_allocTraces.Count > 500) _allocTraces.TryDequeue(out _);

        // Обновляем пик
        long current = Interlocked.Read(ref _totalAllocatedBytes);
        long peak;
        do
        {
            peak = Interlocked.Read(ref _peakAllocatedBytes);
            if (current <= peak) break;
        }
        while (Interlocked.CompareExchange(ref _peakAllocatedBytes, current, peak) != peak);
    }

    /// <summary>
    /// Уведомить трекер об освобождении памяти.
    /// Вызывается из Dispose() конкретных буферов.
    /// </summary>
    protected static void TrackFree(ulong size)
    {
        Interlocked.Add(ref _totalAllocatedBytes, -(long)size);
        Interlocked.Decrement(ref _activeBufferCount);
        Interlocked.Increment(ref _freeCount);

        var st = Environment.StackTrace;
        var lines = st.Split('\n', StringSplitOptions.RemoveEmptyEntries);
        var brief = lines.Length > 3
            ? string.Join(" <- ", lines.Skip(2).Take(3).Select(l => l.TrimStart()))
            : st.Replace('\n', ' ');
        _freeTraces.Enqueue($"[FREE  -{size / 1024}KB] active={_activeBufferCount} totalMB={_totalAllocatedBytes / 1024 / 1024} {brief}");
        while (_freeTraces.Count > 500) _freeTraces.TryDequeue(out _);
    }

    public abstract ulong Size { get; }
    public abstract BufferType Type { get; }
    public abstract DataType ElementType { get; }
    public abstract IntPtr GetPointer();
    public abstract void Write<T>(T[] data) where T : unmanaged;
    public abstract T[] Read<T>() where T : unmanaged;

    public virtual T[] ReadRange<T>(ulong byteOffset, int elementCount) where T : unmanaged
    {
        var all = Read<T>();
        int start = (int)(byteOffset / (ulong)System.Runtime.InteropServices.Marshal.SizeOf<T>());
        return all.Skip(start).Take(elementCount).ToArray();
    }

    public abstract void Dispose();
}

/// <summary>
/// Базовый класс для реализации ядра
/// </summary>
public abstract class ComputeKernelBase : IComputeKernel
{
    public abstract string Name { get; }
    public virtual KernelArgumentType[] ArgumentTypes { get; protected set; } = Array.Empty<KernelArgumentType>();
    public abstract void SetArgument(int index, object value);
    public abstract void Dispatch(uint[] globalWorkSize, uint[]? localWorkSize = null);
    public virtual void ClearArguments() { }
    public abstract void Compile();
    public abstract void Dispose();
}

/// <summary>
/// Базовый класс для реализации очереди команд
/// </summary>
public abstract class ComputeCommandQueueBase : IComputeCommandQueue
{
    public abstract void WriteBuffer(IComputeBuffer buffer, ulong offset, byte[] data);
    public abstract void ReadBuffer(IComputeBuffer buffer, ulong offset, byte[] data);
    public abstract void Dispatch(IComputeKernel kernel, uint[] globalWorkSize, uint[]? localWorkSize = null);
    public abstract void Flush();
    public virtual void InsertMemoryBarrier() { } // no-op for non-Vulkan backends
    public abstract void Dispose();
}
