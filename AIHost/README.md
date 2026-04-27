# AIHost - GPU-Accelerated LLM Inference Engine

GPU-based LLM inference engine with Vulkan compute support and planned ROCm integration.

## ✅ Реализовано

### Compute Infrastructure
- **Vulkan Compute Provider** - GPU вычисления через Vulkan API
- **Tensor Operations** - MatMul, Softmax, LayerNorm, SiLU, RoPE, Attention, FFN
- **Quantization** - Q2_K, Q3_K, Q4_K, Q5_K, Q6_K деквантизация
- **KV-Cache** - с tensor concatenation для эффективной генерации

### Model Support
- **GGUF Loader** - загрузка моделей в GGUF формате
- **Transformer** - полный forward pass через все слои
- **BPE Tokenizer** - базовая поддержка токенизации

### Inference Engine
- **Sampling Strategies**:
  - Temperature scaling
  - Top-K filtering
  - Top-P (nucleus) sampling
- **Generation Modes**:
  - Standard generation
  - Streaming generation with callbacks
- **KV-cache integration** для ускорения autoregressive generation

### Testing
- **8 unit тестов** через xUnit:
  - MatMul (64×128×32 матрицы)
  - Softmax
  - SiLU activation
  - Element-wise Add
  - LayerNorm
  - Tokenizer encode/decode

## 📁 Структура проекта

```
AIHost/
├── ICompute/               # Compute provider interfaces
│   ├── Vulkan/            # Vulkan implementation
│   └── ROCm/              # ROCm stub (TODO)
├── Compute/               # High-level tensor operations
│   ├── ComputeOps.cs     # Tensor operations API
│   ├── Tensor.cs         # GPU tensor wrapper
│   ├── Transformer.cs    # LLM transformer model
│   └── ShaderLoader.cs   # Lazy shader loading
├── Shaders/               # External shader files
│   ├── Vulkan/           # GLSL compute shaders
│   │   ├── matmul.glsl
│   │   ├── softmax.glsl
│   │   ├── silu.glsl
│   │   ├── add.glsl
│   │   └── concat_axis1.glsl
│   └── ROCm/             # HIP kernels (TODO)
├── Inference/            # Inference engine
│   └── InferenceEngine.cs
├── GGUF/                 # GGUF format support
├── Tokenizer/            # Tokenization
└── Tests/                # xUnit tests

AIHost.Tests/
├── ComputeOpsTests.cs    # Compute operations tests
└── TokenizerTests.cs     # Tokenizer tests
```

## 🚀 Использование

### Запуск тестов
```bash
dotnet test
```

### Inference с реальной моделью
```bash
dotnet run
# Выбрать опцию 12: Test InferenceEngine
# Указать путь к .gguf файлу (например TinyLlama Q2_K)
```

### Пример кода
```csharp
using var provider = new VulkanComputeDevice();
using var ops = new ComputeOps(provider);

// Load model
var model = new GGUFModel("tinyllama-1.1b.Q2_K.gguf", provider);
var transformer = new Transformer(provider, model);
transformer.LoadWeights();

// Create inference engine
var tokenizer = new BPETokenizer(/* ... */);
using var engine = new InferenceEngine(transformer, tokenizer, ops);

// Generate text
var config = new GenerationConfig
{
    MaxNewTokens = 50,
    Temperature = 0.8f,
    TopK = 40,
    TopP = 0.9f
};

string output = engine.Generate("Hello", config);
```

## 📋 Roadmap

### ✅ Completed
- [x] Vulkan compute provider
- [x] Основные tensor операции
- [x] Q2_K/Q3_K/Q4_K/Q5_K/Q6_K деквантизация
- [x] KV-cache с concatenation
- [x] Sampling strategies (temp, top-k, top-p)
- [x] Streaming generation
- [x] Unit тесты
- [x] ROCm provider structure (stub)
- [x] Shader externalization

### 🔄 In Progress
- [ ] Полная интеграция с реальной моделью
- [ ] ROCm/HIP implementation

### 📝 TODO
- [ ] ROCm provider implementation (HIP API bindings)
- [ ] Beam search
- [ ] Batch inference
- [ ] Профилирование и оптимизация
- [ ] REST API (ASP.NET Core)
- [ ] Загрузка токенизатора из tokenizer.json
- [ ] Поддержка других моделей (Llama 2/3, Mistral)

## 🔧 Технологии

- **.NET 10.0** - C# unsafe code для GPU interop
- **Vulkan Compute** - через Silk.NET.Vulkan 2.21.0
- **xUnit** - для unit тестов
- **GGUF** - формат квантизованных моделей

## ⚡ Performance

Текущие результаты (AMD Radeon RX 6600 XT):
- Dequantization Q2_K: ~189ms для больших тензоров
- MatMul 64×128×32: ~0.82ms
- Softmax: sum=1.0 ✓
- Unit тесты: 8/8 pass за ~800ms

## 📄 License

MIT (или ваша лицензия)
