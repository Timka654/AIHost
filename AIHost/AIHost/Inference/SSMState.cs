namespace AIHost.Inference;

/// <summary>
/// Per-layer recurrent state for SSM (State Space Model) blocks.
/// Holds h [n_groups=48, group_dim=128] = [6144] float32 per layer.
/// Maintained on CPU between tokens (24 KB per model = negligible overhead).
/// </summary>
public sealed class SSMState : IDisposable
{
    private const int N_GROUPS    = 48;
    private const int GROUP_DIM   = 128;
    public  const int STATE_DIM   = N_GROUPS * GROUP_DIM; // 6144

    private readonly Dictionary<int, float[]> _layers = new();
    private bool _disposed;

    /// <summary>Get (or lazily create) the state vector for one layer.</summary>
    public float[] GetLayer(int layerIndex)
    {
        if (!_layers.TryGetValue(layerIndex, out var h))
        {
            h = new float[STATE_DIM]; // zero-initialised = correct initial state
            _layers[layerIndex] = h;
        }
        return h;
    }

    public void Clear()
    {
        foreach (var h in _layers.Values)
            Array.Clear(h, 0, h.Length);
    }

    public void Dispose()
    {
        if (_disposed) return;
        _layers.Clear();
        _disposed = true;
    }
}
