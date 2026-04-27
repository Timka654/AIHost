namespace AIHost.Compute;

/// <summary>
/// Описание формы тензора
/// </summary>
public readonly struct TensorShape
{
    public int[] Dimensions { get; }
    public int Rank => Dimensions.Length;
    public int TotalElements { get; }

    public TensorShape(params int[] dimensions)
    {
        Dimensions = dimensions;
        TotalElements = 1;
        foreach (var dim in dimensions)
            TotalElements *= dim;
    }

    public int this[int index] => Dimensions[index];

    public override string ToString()
    {
        return $"[{string.Join(" × ", Dimensions)}]";
    }

    // Фабричные методы для типичных форм
    public static TensorShape Vector(int size) => new(size);
    public static TensorShape Matrix(int rows, int cols) => new(rows, cols);
    public static TensorShape Tensor3D(int dim0, int dim1, int dim2) => new(dim0, dim1, dim2);
    public static TensorShape Tensor4D(int dim0, int dim1, int dim2, int dim3) => new(dim0, dim1, dim2, dim3);
}
