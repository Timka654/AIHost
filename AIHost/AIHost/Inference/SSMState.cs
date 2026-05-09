using AIHost.ICompute;

namespace AIHost.Inference;

/// <summary>
/// Per-layer recurrent state for Gated Delta Net (linear attention) blocks.
///
/// Two types of state per layer:
///   1. SSM state: matrix [head_v_dim=128, head_v_dim=128, n_v_heads=48] = 786432 float
///      This is the recurrent state S for the Gated Delta Net.
///      Initialized to zeros (correct initial state).
///
///   2. Conv state: sliding window [conv_kernel_size-1=3, conv_dim=10240] = 30720 float
///      This holds the last (kernel_size-1) input values for conv1d.
///      Initialized to zeros.
///
/// Dimensions (from GGUF metadata):
///   ssm_d_state = 128 (head_v_dim)
///   ssm_n_group = 48  (n_v_heads)
///   ssm_d_inner = 6144 (= n_v_heads * head_v_dim)
///   n_k_heads = 16, key_dim = 2048
///   conv_dim = 2*key_dim + value_dim = 4096 + 6144 = 10240
///   ssm_d_conv = 4 (kernel_size)
/// </summary>
public sealed class SSMState : IDisposable
{
    private const int HEAD_V_DIM = 128;     // ssm_d_state
    private const int N_V_HEADS = 48;       // ssm_n_group
    private const int CONV_DIM = 10240;     // conv_dim = 2*key_dim + value_dim
    private const int CONV_KERNEL = 4;      // ssm_d_conv
    private const int CONV_STATE_LEN = CONV_KERNEL - 1; // 3

    /// <summary>SSM state size per layer: [128, 128, 48] = 786432 floats</summary>
    public const int STATE_DIM = HEAD_V_DIM * HEAD_V_DIM * N_V_HEADS;

    /// <summary>Conv state size per layer: [3, 10240] = 30720 floats</summary>
    public const int CONV_STATE_DIM = CONV_STATE_LEN * CONV_DIM;

    private readonly Dictionary<int, float[]> _ssmStates = new();
    private readonly Dictionary<int, float[]> _convStates = new();
    private bool _disposed;

    // GPU buffers for SSM state (one per layer, created on demand)
    private readonly IComputeDevice? _device;
    private readonly Dictionary<int, IComputeBuffer> _ssmGpuBuffers = new();
    private readonly Dictionary<int, IComputeBuffer> _convGpuBuffers = new();

    public SSMState(IComputeDevice? device = null)
    {
        _device = device;
    }

    /// <summary>
    /// Get (or lazily create) the SSM recurrent state matrix for one layer.
    /// Returns float[HEAD_V_DIM * HEAD_V_DIM * N_V_HEADS] = float[786432].
    /// Layout: [head_v_dim, head_v_dim, n_v_heads] — column-major per group.
    /// </summary>
    public float[] GetLayer(int layerIndex)
    {
        if (!_ssmStates.TryGetValue(layerIndex, out var h))
        {
            h = new float[STATE_DIM]; // zero-initialised = correct initial state
            _ssmStates[layerIndex] = h;
        }
        return h;
    }

    /// <summary>
    /// Get (or lazily create) the conv1d sliding window state for one layer.
    /// Returns float[CONV_STATE_LEN * CONV_DIM] = float[30720].
    /// Layout: [conv_state_len, conv_dim] — last (kernel_size-1) inputs.
    /// </summary>
    public float[] GetConvState(int layerIndex)
    {
        if (!_convStates.TryGetValue(layerIndex, out var c))
        {
            c = new float[CONV_STATE_DIM]; // zero-initialised
            _convStates[layerIndex] = c;
        }
        return c;
    }

    /// <summary>
    /// Get GPU buffer for SSM state. Creates and uploads if needed.
    /// </summary>
    public IComputeBuffer GetGpuStateBuffer(int layerIndex)
    {
        if (_device == null)
            throw new InvalidOperationException("SSMState created without GPU device");

        if (!_ssmGpuBuffers.TryGetValue(layerIndex, out var buf))
        {
            buf = _device.CreateBuffer((ulong)(STATE_DIM * sizeof(float)), BufferType.Storage, DataType.F32);
            var cpuState = GetLayer(layerIndex);
            buf.Write(cpuState);
            _ssmGpuBuffers[layerIndex] = buf;
        }
        return buf;
    }

    /// <summary>
    /// Get GPU buffer for conv state. Creates and uploads if needed.
    /// </summary>
    public IComputeBuffer GetGpuConvBuffer(int layerIndex)
    {
        if (_device == null)
            throw new InvalidOperationException("SSMState created without GPU device");

        if (!_convGpuBuffers.TryGetValue(layerIndex, out var buf))
        {
            buf = _device.CreateBuffer((ulong)(CONV_STATE_DIM * sizeof(float)), BufferType.Storage, DataType.F32);
            var cpuState = GetConvState(layerIndex);
            buf.Write(cpuState);
            _convGpuBuffers[layerIndex] = buf;
        }
        return buf;
    }

    /// <summary>
    /// Read GPU state back to CPU after decode step.
    /// </summary>
    public void SyncGpuStateToCpu(int layerIndex)
    {
        if (_ssmGpuBuffers.TryGetValue(layerIndex, out var buf))
        {
            var data = buf.Read<float>();
            var state = GetLayer(layerIndex);
            Buffer.BlockCopy(data, 0, state, 0, data.Length * sizeof(float));
        }
        if (_convGpuBuffers.TryGetValue(layerIndex, out var convBuf))
        {
            var data = convBuf.Read<float>();
            var state = GetConvState(layerIndex);
            Buffer.BlockCopy(data, 0, state, 0, data.Length * sizeof(float));
        }
    }

    /// <summary>
    /// Update conv state with new input and return the full window for conv1d.
    /// conv_state holds last (kernel_size-1) = 3 values of dimension conv_dim.
    /// New input is [conv_dim] floats.
    /// Returns the concatenated [conv_state (3, conv_dim), new_input (1, conv_dim)]
    /// as a flat float[4 * conv_dim] ready for conv1d with kernel_size=4.
    /// </summary>
    public float[] UpdateConvState(int layerIndex, float[] newInput)
    {
        var convState = GetConvState(layerIndex);
        int convDim = CONV_DIM;
        int stateLen = CONV_STATE_LEN;

        // Shift conv state: remove oldest, keep last (stateLen-1)
        Buffer.BlockCopy(convState, convDim * sizeof(float),
                         convState, 0,
                         (stateLen - 1) * convDim * sizeof(float));

        // Copy new input to the last slot
        Buffer.BlockCopy(newInput, 0,
                         convState, (stateLen - 1) * convDim * sizeof(float),
                         convDim * sizeof(float));

        // Build full window: [convState (3, convDim), newInput (1, convDim)]
        var fullWindow = new float[CONV_KERNEL * convDim];
        Buffer.BlockCopy(convState, 0, fullWindow, 0, stateLen * convDim * sizeof(float));
        Buffer.BlockCopy(newInput, 0, fullWindow, stateLen * convDim * sizeof(float), convDim * sizeof(float));

        return fullWindow;
    }

    public void Clear()
    {
        foreach (var h in _ssmStates.Values)
            Array.Clear(h, 0, h.Length);
        foreach (var c in _convStates.Values)
            Array.Clear(c, 0, c.Length);
    }

    public void Dispose()
    {
        if (_disposed) return;
        _ssmStates.Clear();
        _convStates.Clear();
        foreach (var buf in _ssmGpuBuffers.Values) buf.Dispose();
        _ssmGpuBuffers.Clear();
        foreach (var buf in _convGpuBuffers.Values) buf.Dispose();
        _convGpuBuffers.Clear();
        _disposed = true;
    }
}
