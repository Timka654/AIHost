namespace AIHost.ICompute;

/// <summary>
/// Информация о heap'е памяти с драйвера (vkGetPhysicalDeviceMemoryProperties, cudaMemGetInfo, etc.)
/// </summary>
public class VramHeapInfo
{
    /// <summary>Индекс heap'а в драйвере.</summary>
    public int Index { get; init; }

    /// <summary>Общий размер heap'а в байтах.</summary>
    public long TotalSize { get; init; }

    /// <summary>Свободно (если драйвер поддерживает запрос). -1 если неизвестно.</summary>
    public long AvailableSize { get; init; } = -1;

    /// <summary>DEVICE_LOCAL или эквивалент (выделенная VRAM).</summary>
    public bool IsDeviceLocal { get; init; }

    /// <summary>HOST_VISIBLE (CPU-mappable).</summary>
    public bool IsHostVisible { get; init; }

    /// <summary>HOST_COHERENT (no flush needed).</summary>
    public bool IsHostCoherent { get; init; }

    public override string ToString()
    {
        long avail = AvailableSize >= 0 ? AvailableSize / (1024 * 1024) : -1;
        return $"Heap[{Index}] total={TotalSize / (1024 * 1024)}MB " +
               $"free={(avail >= 0 ? $"{avail}MB" : "?")} " +
               $"{(IsDeviceLocal ? "DEVICE_LOCAL" : "")} " +
               $"{(IsHostVisible ? "HOST_VISIBLE" : "")} " +
               $"{(IsHostCoherent ? "HOST_COHERENT" : "")}";
    }
}

/// <summary>
/// Лимит VRAM на уровне устройства (драйвер/API + deviceIdx).
/// Содержит карту heap'ов и отслеживает зарезервированные байты.
/// </summary>
public class DeviceVramLimit
{
    private readonly VramHeapInfo[] _heaps;
    private long _reservedBytes;
    private long _reservedDeviceLocalBytes;

    /// <summary>Все heap'ы устройства.</summary>
    public IReadOnlyList<VramHeapInfo> Heaps => _heaps;

    /// <summary>Пользовательский лимит в байтах (0 = авто = сумма всех heap'ов).</summary>
    public long MaxBytes { get; }

    /// <summary>Разрешён ли fallback на HOST_VISIBLE память.</summary>
    public bool AllowSharedMemory { get; }

    /// <summary>Уже зарезервировано байт.</summary>
    public long ReservedBytes => Interlocked.Read(ref _reservedBytes);

    /// <summary>Зарезервировано в DEVICE_LOCAL heap'ах.</summary>
    public long ReservedDeviceLocalBytes => Interlocked.Read(ref _reservedDeviceLocalBytes);

    /// <summary>Эффективный лимит = min(сумма heap'ов, MaxBytes если задан).</summary>
    public long EffectiveLimit => MaxBytes > 0
        ? Math.Min(TotalDeviceLocalSize, MaxBytes)
        : TotalDeviceLocalSize;

    /// <summary>Суммарный размер всех DEVICE_LOCAL heap'ов.</summary>
    public long TotalDeviceLocalSize => _heaps.Where(h => h.IsDeviceLocal).Sum(h => h.TotalSize);

    /// <summary>Свободно в DEVICE_LOCAL (EffectiveLimit - ReservedDeviceLocalBytes).</summary>
    public long FreeDeviceLocalBytes => Math.Max(0, EffectiveLimit - ReservedDeviceLocalBytes);

    /// <summary>Суммарный размер HOST_VISIBLE heap'ов (системная RAM + shared).</summary>
    public long TotalHostVisibleSize => _heaps.Where(h => h.IsHostVisible).Sum(h => h.TotalSize);

    public DeviceVramLimit(VramHeapInfo[] heaps, long maxBytes = 0, bool allowSharedMemory = false)
    {
        _heaps = heaps ?? throw new ArgumentNullException(nameof(heaps));
        MaxBytes = maxBytes;
        AllowSharedMemory = allowSharedMemory;
    }

    /// <summary>
    /// Проверить, можно ли зарезервировать ещё size байт.
    /// </summary>
    public bool CanReserve(long size, bool requireDeviceLocal = false)
    {
        if (requireDeviceLocal)
            return FreeDeviceLocalBytes >= size;
        return FreeDeviceLocalBytes >= size || (AllowSharedMemory && TotalHostVisibleSize >= size);
    }

    /// <summary>
    /// Зарезервировать size байт. Уменьшает свободный бюджет.
    /// Возвращает true если успешно, false если превышен лимит.
    /// </summary>
    public bool TryReserve(long size, bool isDeviceLocal)
    {
        if (isDeviceLocal)
        {
            long current = Interlocked.Read(ref _reservedDeviceLocalBytes);
            while (true)
            {
                long newValue = current + size;
                if (newValue > EffectiveLimit)
                    return false;
                long old = Interlocked.CompareExchange(ref _reservedDeviceLocalBytes, newValue, current);
                if (old == current) break;
                current = old;
            }
        }

        Interlocked.Add(ref _reservedBytes, size);
        return true;
    }

    /// <summary>
    /// Зарезервировать size байт. Бросает исключение если превышен лимит.
    /// </summary>
    public void Reserve(long size, bool isDeviceLocal)
    {
        if (!TryReserve(size, isDeviceLocal))
        {
            throw new InsufficientVramException(
                (ulong)size,
                $"VRAM limit exceeded: {FreeDeviceLocalBytes / (1024 * 1024)}MB free, " +
                $"need {size / (1024 * 1024)}MB. Effective limit: {EffectiveLimit / (1024 * 1024)}MB. " +
                $"Reduce context_size or set allow_shared_memory=true.");
        }
    }

    /// <summary>
    /// Освободить зарезервированные байты (при выгрузке тензора).
    /// </summary>
    public void Release(long size, bool isDeviceLocal)
    {
        Interlocked.Add(ref _reservedBytes, -size);
        if (isDeviceLocal)
            Interlocked.Add(ref _reservedDeviceLocalBytes, -size);
    }

    public override string ToString()
    {
        var sb = new System.Text.StringBuilder();
        sb.AppendLine($"DeviceVramLimit: limit={EffectiveLimit / (1024 * 1024)}MB " +
                       $"reserved={ReservedDeviceLocalBytes / (1024 * 1024)}MB " +
                       $"free={FreeDeviceLocalBytes / (1024 * 1024)}MB " +
                       $"shared={AllowSharedMemory}");
        foreach (var h in _heaps)
            sb.AppendLine($"  {h}");
        return sb.ToString();
    }
}

/// <summary>
/// Назначение слоёв на GPU.
/// </summary>
public class LayerAssignment
{
    /// <summary>Индекс устройства (0-based).</summary>
    public int DeviceIndex { get; init; }

    /// <summary>Первый глобальный слой на этом устройстве (включительно).</summary>
    public int FirstLayer { get; init; }

    /// <summary>Последний глобальный слой на этом устройстве (не включительно).</summary>
    public int LastLayerExclusive { get; init; }

    /// <summary>Количество слоёв.</summary>
    public int LayerCount => LastLayerExclusive - FirstLayer;

    /// <summary>Владеет ли эмбеддингом (всегда device 0).</summary>
    public bool OwnsEmbedding { get; init; }

    /// <summary>Владеет ли головой (output_norm + output.weight).</summary>
    public bool OwnsHead { get; init; }
}

/// <summary>
/// План аллокации памяти для модели — рассчитывается до первой аллокации.
/// </summary>
public class ModelAllocationPlan
{
    /// <summary>Имя модели.</summary>
    public string ModelName { get; init; } = "";

    /// <summary>Суммарный размер весов в байтах (quantized).</summary>
    public long WeightBytes { get; set; }

    /// <summary>Размер арены в байтах (переиспользуемый scratch).</summary>
    public long ArenaBytes { get; set; }

    /// <summary>Размер KV-кэша в байтах.</summary>
    public long KvCacheBytes { get; set; }

    /// <summary>Размер per-layer scratch F32 буферов.</summary>
    public long ScratchF32Bytes { get; set; }

    /// <summary>Размер рантайм-пула (максимальная разовая аллокация).</summary>
    public long RuntimePoolBytes { get; set; }

    /// <summary>Суммарный требуемый бюджет.</summary>
    public long TotalBytes => WeightBytes + ArenaBytes + KvCacheBytes + ScratchF32Bytes + RuntimePoolBytes;

    /// <summary>Назначение слоёв по устройствам (1+ записей для multi-GPU).</summary>
    public LayerAssignment[] LayerMap { get; set; } = Array.Empty<LayerAssignment>();

    /// <summary>Количество тензоров весов.</summary>
    public int TensorCount { get; set; }

    /// <summary>Проверить план против лимита устройства.</summary>
    public PlanValidationResult Validate(DeviceVramLimit limit)
    {
        bool fits = limit.CanReserve(TotalBytes, requireDeviceLocal: !limit.AllowSharedMemory);
        return new PlanValidationResult
        {
            IsValid = fits,
            RequiredBytes = TotalBytes,
            AvailableBytes = limit.FreeDeviceLocalBytes,
            LimitBytes = limit.EffectiveLimit,
            ModelName = ModelName,
            Breakdown = new PlanBreakdown
            {
                Weights = WeightBytes,
                Arena = ArenaBytes,
                KvCache = KvCacheBytes,
                ScratchF32 = ScratchF32Bytes,
                RuntimePool = RuntimePoolBytes,
                Total = TotalBytes,
                TensorCount = TensorCount,
                AllowSharedMemory = limit.AllowSharedMemory,
                HostVisibleAvailable = limit.TotalHostVisibleSize
            }
        };
    }

    public override string ToString()
    {
        var sb = new System.Text.StringBuilder();
        sb.AppendLine($"AllocationPlan for '{ModelName}':");
        sb.AppendLine($"  Weights ({TensorCount} tensors):  {WeightBytes / (1024 * 1024),6} MB");
        sb.AppendLine($"  Arena (scratch reuse):           {ArenaBytes / (1024 * 1024),6} MB");
        sb.AppendLine($"  KV Cache:                        {KvCacheBytes / (1024 * 1024),6} MB");
        sb.AppendLine($"  Scratch F32:                     {ScratchF32Bytes / (1024 * 1024),6} MB");
        sb.AppendLine($"  Runtime pool:                    {RuntimePoolBytes / (1024 * 1024),6} MB");
        sb.AppendLine($"  ─────────────────────────────────────────");
        sb.AppendLine($"  TOTAL REQUIRED:                  {TotalBytes / (1024 * 1024),6} MB");
        if (LayerMap.Length > 1)
        {
            sb.AppendLine("  Layer assignments:");
            foreach (var a in LayerMap)
                sb.AppendLine($"    Device {a.DeviceIndex}: layers {a.FirstLayer}..{a.LastLayerExclusive - 1}" +
                               $"{(a.OwnsEmbedding ? " + embedding" : "")}" +
                               $"{(a.OwnsHead ? " + head" : "")}");
        }
        return sb.ToString();
    }
}

/// <summary>
/// Результат валидации плана аллокации против лимита устройства.
/// </summary>
public class PlanValidationResult
{
    public bool IsValid { get; init; }
    public long RequiredBytes { get; init; }
    public long AvailableBytes { get; init; }
    public long LimitBytes { get; init; }
    public string ModelName { get; init; } = "";
    public PlanBreakdown Breakdown { get; init; } = new();

    public string GetDetailedReport()
    {
        var sb = new System.Text.StringBuilder();
        sb.AppendLine("═══════════════════════════════════════════════════════════");
        sb.AppendLine($"VRAM BUDGET EXCEEDED — model: {ModelName}");
        sb.AppendLine("───────────────────────────────────────────────────────────");
        sb.AppendLine("Allocation plan:");
        sb.AppendLine($"  Weights ({Breakdown.TensorCount} tensors):   {Breakdown.Weights / (1024 * 1024),6} MB");
        sb.AppendLine($"  Arena (scratch reuse):                       {Breakdown.Arena / (1024 * 1024),6} MB");
        sb.AppendLine($"  KV Cache:                                    {Breakdown.KvCache / (1024 * 1024),6} MB");
        sb.AppendLine($"  Scratch F32:                                 {Breakdown.ScratchF32 / (1024 * 1024),6} MB");
        sb.AppendLine($"  Runtime pool:                                {Breakdown.RuntimePool / (1024 * 1024),6} MB");
        sb.AppendLine($"  ─────────────────────────────────────");
        sb.AppendLine($"  TOTAL REQUIRED:                              {Breakdown.Total / (1024 * 1024),6} MB");
        sb.AppendLine();
        sb.AppendLine($"Device limit:  {LimitBytes / (1024 * 1024)} MB");
        sb.AppendLine($"Available:     {AvailableBytes / (1024 * 1024)} MB");
        sb.AppendLine($"GAP:           {(Breakdown.Total - AvailableBytes) / (1024 * 1024)} MB");
        sb.AppendLine();
        if (Breakdown.HostVisibleAvailable > 0 && !Breakdown.AllowSharedMemory)
        {
            sb.AppendLine($"HOST_VISIBLE free: {Breakdown.HostVisibleAvailable / (1024 * 1024)} MB (not used — set allow_shared_memory=true)");
            sb.AppendLine();
        }
        sb.AppendLine("Suggestions:");
        sb.AppendLine("  1. Set allow_shared_memory=true for HOST_VISIBLE fallback (PCIe speeds)");
        sb.AppendLine("  2. Reduce context_size to lower KV cache");
        sb.AppendLine("  3. Set arena_size_mb to a smaller value");
        sb.AppendLine("  4. Use multi-GPU (devices[]) to split layers");
        sb.AppendLine("═══════════════════════════════════════════════════════════");
        return sb.ToString();
    }
}

/// <summary>
/// Детализация бюджета для отчёта.
/// </summary>
public class PlanBreakdown
{
    public long Weights { get; init; }
    public long Arena { get; init; }
    public long KvCache { get; init; }
    public long ScratchF32 { get; init; }
    public long RuntimePool { get; init; }
    public long Total { get; init; }
    public int TensorCount { get; init; }
    public bool AllowSharedMemory { get; init; }
    public long HostVisibleAvailable { get; init; }
}
