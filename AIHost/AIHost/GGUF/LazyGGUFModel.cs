using AIHost.ICompute;
using System.IO.MemoryMappedFiles;

namespace AIHost.GGUF;

/// <summary>
/// Lazy-loading GGUF model with memory mapping support
/// Loads tensors on-demand and supports split GGUF files
/// </summary>
public class LazyGGUFModel : IDisposable
{
    private readonly GGUFReader _reader;
    private readonly IComputeDevice _device;
    private readonly Dictionary<string, IComputeBuffer> _tensorBuffers = new();
    private readonly MemoryMappedFile? _memoryMappedFile;
    private readonly string _filePath;
    private readonly List<string> _splitFiles = new();
    private bool _disposed;
    private bool _useMemoryMapping;

    public GGUFHeader Header => _reader.Header;
    public GGUFMetadata Metadata => _reader.Metadata;
    public IReadOnlyList<GGUFTensorInfo> Tensors => _reader.Tensors;
    public GGUFReader Reader => _reader;
    public bool UseMemoryMapping => _useMemoryMapping;
    public int LoadedTensorCount => _tensorBuffers.Count;
    public int TotalTensorCount => Tensors.Count;

    public LazyGGUFModel(string filePath, IComputeDevice device, bool useMemoryMapping = true)
    {
        _filePath = filePath;
        _device = device;
        _useMemoryMapping = useMemoryMapping;

        // Check for split files
        _splitFiles = DiscoverSplitFiles(filePath);
        
        if (_splitFiles.Count > 1)
        {
            Console.WriteLine($"Discovered {_splitFiles.Count} split GGUF files");
            // For now, load from first file
            // TODO: Implement proper multi-file support
            filePath = _splitFiles[0];
            _useMemoryMapping = false; // Disable mmap for split files for now
        }

        // Initialize reader
        _reader = new GGUFReader(filePath);
        _reader.Load();

        // Create memory-mapped file if requested
        if (_useMemoryMapping && _splitFiles.Count == 1)
        {
            try
            {
                _memoryMappedFile = MemoryMappedFile.CreateFromFile(
                    filePath, 
                    FileMode.Open, 
                    null, 
                    0, 
                    MemoryMappedFileAccess.Read);
                
                Console.WriteLine($"Memory-mapped file created for {Path.GetFileName(filePath)}");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Warning: Failed to create memory-mapped file: {ex.Message}");
                Console.WriteLine("Falling back to standard file I/O");
                _memoryMappedFile = null;
                _useMemoryMapping = false;
            }
        }
    }

    /// <summary>
    /// Load a tensor on-demand (lazy loading)
    /// </summary>
    public unsafe IComputeBuffer LoadTensor(string tensorName)
    {
        // Check cache
        if (_tensorBuffers.TryGetValue(tensorName, out var cachedBuffer))
            return cachedBuffer;

        // Find tensor
        var tensor = Tensors.FirstOrDefault(t => t.Name == tensorName);
        if (tensor == null)
            throw new ArgumentException($"Tensor '{tensorName}' not found in model");

        // Read data (with memory mapping if available)
        byte[] data;
        
        if (_useMemoryMapping && _memoryMappedFile != null)
        {
            data = ReadTensorWithMemoryMapping(tensor);
        }
        else
        {
            data = _reader.ReadTensorData(tensor);
        }

        // Create buffer
        var dataType = MapTensorTypeToDataType(tensor.Type);
        var buffer = _device.CreateBuffer(tensor.SizeInBytes, BufferType.Storage, dataType);

        // Upload data
        fixed (byte* src = data)
        {
            var dest = buffer.GetPointer();
            System.Buffer.MemoryCopy(src, dest.ToPointer(), (long)buffer.Size, data.Length);
        }

        // Cache
        _tensorBuffers[tensorName] = buffer;
        return buffer;
    }

    /// <summary>
    /// Load multiple tensors matching a predicate
    /// </summary>
    public Dictionary<string, IComputeBuffer> LoadTensors(Func<GGUFTensorInfo, bool> predicate)
    {
        var result = new Dictionary<string, IComputeBuffer>();
        foreach (var tensor in Tensors.Where(predicate))
        {
            result[tensor.Name] = LoadTensor(tensor.Name);
        }
        return result;
    }

    /// <summary>
    /// Unload a tensor to free GPU memory
    /// </summary>
    public void UnloadTensor(string tensorName)
    {
        if (_tensorBuffers.TryGetValue(tensorName, out var buffer))
        {
            buffer.Dispose();
            _tensorBuffers.Remove(tensorName);
            Console.WriteLine($"Unloaded tensor: {tensorName}");
        }
    }

    /// <summary>
    /// Unload all tensors to free GPU memory
    /// </summary>
    public void UnloadAllTensors()
    {
        foreach (var buffer in _tensorBuffers.Values)
        {
            buffer.Dispose();
        }
        _tensorBuffers.Clear();
        Console.WriteLine("Unloaded all tensors");
    }

    /// <summary>
    /// Get memory usage statistics
    /// </summary>
    public long GetGPUMemoryUsageBytes()
    {
        return _tensorBuffers.Values.Sum(b => (long)b.Size);
    }

    private unsafe byte[] ReadTensorWithMemoryMapping(GGUFTensorInfo tensor)
    {
        if (_memoryMappedFile == null)
            throw new InvalidOperationException("Memory-mapped file not available");

        using var accessor = _memoryMappedFile.CreateViewAccessor(
            (long)tensor.Offset, 
            (long)tensor.SizeInBytes, 
            MemoryMappedFileAccess.Read);

        byte[] data = new byte[tensor.SizeInBytes];
        accessor.ReadArray(0, data, 0, data.Length);
        
        return data;
    }

    private DataType MapTensorTypeToDataType(GGUFTensorType tensorType)
    {
        // GGUF tensor types
        return tensorType switch
        {
            GGUFTensorType.F32 => DataType.F32,
            GGUFTensorType.F16 => DataType.F16,
            GGUFTensorType.Q4_0 => DataType.Q4_0,
            GGUFTensorType.Q4_1 => DataType.Q4_1,
            GGUFTensorType.Q5_0 => DataType.Q5_0,
            GGUFTensorType.Q5_1 => DataType.Q5_1,
            GGUFTensorType.Q8_0 => DataType.Q8_0,
            GGUFTensorType.Q8_1 => DataType.Q8_1,
            GGUFTensorType.Q2_K => DataType.Q2_K,
            GGUFTensorType.Q3_K => DataType.Q3_K,
            GGUFTensorType.Q4_K => DataType.Q4_K,
            GGUFTensorType.Q5_K => DataType.Q5_K,
            GGUFTensorType.Q6_K => DataType.Q6_K,
            _ => DataType.F32
        };
    }

    /// <summary>
    /// Discover split GGUF files (e.g., model-00001-of-00004.gguf)
    /// </summary>
    private List<string> DiscoverSplitFiles(string filePath)
    {
        var files = new List<string> { filePath };
        
        var directory = Path.GetDirectoryName(filePath) ?? ".";
        var fileName = Path.GetFileNameWithoutExtension(filePath);
        
        // Pattern: model-00001-of-00004 or model.gguf.00001.of.00004
        var splitPattern1 = System.Text.RegularExpressions.Regex.Match(
            fileName, @"^(.+)-\d{5}-of-(\d{5})$");
        
        if (splitPattern1.Success)
        {
            string baseName = splitPattern1.Groups[1].Value;
            int totalParts = int.Parse(splitPattern1.Groups[2].Value);
            
            files.Clear();
            for (int i = 1; i <= totalParts; i++)
            {
                string splitFile = Path.Combine(directory, 
                    $"{baseName}-{i:D5}-of-{totalParts:D5}.gguf");
                
                if (File.Exists(splitFile))
                    files.Add(splitFile);
            }
        }
        
        return files;
    }

    public void Dispose()
    {
        if (_disposed) return;

        UnloadAllTensors();
        _memoryMappedFile?.Dispose();
        _reader?.Dispose();
        
        _disposed = true;
    }
}
