namespace AIHost.ICompute;

/// <summary>
/// Абстракция над функцией ядра. Принимает аргументы (буферы, числа) и знает, как себя «запустить».
/// </summary>
public interface IComputeKernel : IDisposable
{
    /// <summary>
    /// Имя ядра (для компиляции шейдера)
    /// </summary>
    string Name { get; }

    /// <summary>
    /// Типы аргументов ядра
    /// </summary>
    KernelArgumentType[] ArgumentTypes { get; }

    /// <summary>
    /// Установить аргумент ядра
    /// </summary>
    void SetArgument(int index, object value);

    /// <summary>
    /// Запустить ядро с указанными размерами
    /// </summary>
    /// <param name="globalWorkSize">Глобальные размеры работы</param>
    /// <param name="localWorkSize">Локальные размеры работы (опционально)</param>
    void Dispatch(uint[] globalWorkSize, uint[]? localWorkSize = null);

    /// <summary>
    /// Компилировать ядро
    /// </summary>
    void Compile();
}

/// <summary>
/// Типы аргументов для ядра
/// </summary>
public enum KernelArgumentType
{
    Buffer,
    Uniform,
    Image,
    Sampler,
    Other
}