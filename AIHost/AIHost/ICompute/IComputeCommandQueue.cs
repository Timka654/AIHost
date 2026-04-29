namespace AIHost.ICompute;

/// <summary>
/// Очередь команд для выполнения вычислений
/// </summary>
public interface IComputeCommandQueue : IDisposable
{
    /// <summary>
    /// Добавить команду записи буфера
    /// </summary>
    void WriteBuffer(IComputeBuffer buffer, ulong offset, byte[] data);

    /// <summary>
    /// Добавить команду чтения буфера
    /// </summary>
    void ReadBuffer(IComputeBuffer buffer, ulong offset, byte[] data);

    /// <summary>
    /// Добавить команду выполнения ядра
    /// </summary>
    void Dispatch(IComputeKernel kernel, uint[] globalWorkSize, uint[]? localWorkSize = null);

    /// <summary>
    /// Выполнить все команды в очереди
    /// </summary>
    void Flush();

    /// <summary>
    /// Insert a compute-to-compute memory barrier so the next dispatch sees writes
    /// from all previous dispatches. Does NOT submit or wait — use inside a batch.
    /// </summary>
    void InsertMemoryBarrier();
}
