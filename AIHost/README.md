# 🤖 AIHost - Production-Ready LLM Inference Server

High-performance, multi-backend LLM inference server with Ollama and OpenAI API compatibility. Built with .NET 10.0, supporting Vulkan, CUDA, and ROCm compute backends.

## ✨ Features

- 🚀 **Multi-Backend GPU Support** - Vulkan, CUDA, ROCm
- 🔌 **API Compatible** - Ollama `/api/*` and OpenAI `/v1/*` endpoints
- 🌐 **Web Management Console** - Modern UI for model management
- 🐳 **Docker Ready** - Complete containerization with GPU support
- 🔐 **Enterprise Security** - Token auth, rate limiting, management tokens
- 📊 **Advanced Monitoring** - Request logs, model statistics, TPS tracking
- ⚡ **Optimized Performance** - KV cache, batch inference, multi-GPU
- 📦 **HuggingFace Integration** - Direct model downloads
- 🔄 **Auto-Unload** - Automatic cleanup of inactive models
- 💾 **Persistent Logs** - JSON Lines format with rotation

## 🎯 Quick Start

### Local Deployment

```bash
# Build and run
dotnet build
dotnet run -- --web

# Open browser
http://localhost:11434
```

### Docker Deployment

```bash
# Build and start
docker-compose up -d

# Open browser
http://localhost:11434
```

## 📁 Data Structure

All data is organized in the `data/` directory for easy Docker volume mounting:

```
data/
├── config/
│   ├── server.json         # Server configuration
│   └── tokens.txt          # Access tokens
├── models/
│   └── {model-name}/
│       ├── model.json      # Model config
│       └── *.gguf          # Model file
├── logs/
│   └── requests-*.jsonl    # Request logs by date
└── cache/                  # Temporary cache
```

## 🔧 Configuration

### Server Configuration (`data/config/server.json`)

```json
{
  "models_directory": "./data/models",
  "port": 11434,
  "compute_provider": "vulkan",
  "manage_token": "your-secret-token",
  "tokens_file": "./data/config/tokens.txt",
  "persistent_logs": true,
  "auto_unload_minutes": 30,
  "rate_limit_requests_per_minute": 60
}
```

### Model Configuration (`data/models/{model}/model.json`)

Each model can override global settings:

```json
{
  "name": "tinyllama-chat",
  "model": "path/to/model.gguf",
  
  "compute_provider": "vulkan",
  "device_index": 0,
  "keep_alive": 30,
  "enable_mmap": true,
  
  "parameters": {
    "temperature": 0.7,
    "context_size": 2048
  }
}
```

**Per-model settings:**
- `compute_provider` - Override GPU backend (vulkan/cuda/rocm)
- `device_index` - Select specific GPU device
- `keep_alive` - Minutes before auto-unload (0 = never)
- `enable_mmap` - Use memory mapping to reduce RAM

See [MANAGEMENT.md](MANAGEMENT.md) for complete configuration reference.

## 🌐 API Examples

### Ollama API

```bash
curl http://localhost:11434/api/generate \
  -H "Authorization: Bearer your-token" \
  -d '{"model": "tinyllama-chat", "prompt": "Hello"}'
```

### OpenAI API

```bash
curl http://localhost:11434/v1/chat/completions \
  -H "Authorization: Bearer your-token" \
  -d '{"model": "tinyllama-chat", "messages": [{"role": "user", "content": "Hello"}]}'
```

### Python (OpenAI SDK)

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:11434/v1", api_key="your-token")
response = client.chat.completions.create(
    model="tinyllama-chat",
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.choices[0].message.content)
```

## 🐳 Docker

### Quick Start

```bash
docker-compose up -d
```

### With NVIDIA GPU

Ensure `nvidia-docker2` is installed, then:

```bash
docker-compose up -d
```

GPU is automatically detected via docker-compose.yml configuration.

### With AMD GPU (ROCm)

Update `data/config/server.json`:

```json
{
  "compute_provider": "rocm"
}
```

Then start with ROCm device mappings:

```bash
docker-compose up -d
```

## 🔒 Security

### Token Authentication

Create `data/config/tokens.txt`:

```
sk-user1-token-123
sk-user2-token-456
```

Each line is a valid access token. Lines starting with `#` are comments.

### Management Token

Set in `data/config/server.json`:

```json
{
  "manage_token": "your-secret-admin-token"
}
```

Required for management endpoints (`/manage/*`).

### Rate Limiting

Configured per client (IP or token):

```json
{
  "rate_limit_requests_per_minute": 60
}
```

Returns `429 Too Many Requests` when limit exceeded.

## 📊 Management API

### Model Statistics

```bash
curl -H "Authorization: Bearer admin-token" \
  http://localhost:11434/manage/models
```

### Reload Model

```bash
curl -X POST \
  -H "Authorization: Bearer admin-token" \
  http://localhost:11434/manage/models/tinyllama-chat/reload
```

### View Request Logs

```bash
curl -H "Authorization: Bearer admin-token" \
  http://localhost:11434/manage/logs?count=50
```

### Download Model

```bash
curl -X POST \
  -H "Authorization: Bearer admin-token" \
  -H "Content-Type: application/json" \
  http://localhost:11434/manage/download \
  -d '{
    "name": "llama2-7b",
    "url": "https://huggingface.co/.../model.gguf"
  }'
```

## 📚 Documentation

- [**FEATURES.md**](FEATURES.md) - Complete feature list
- [**WEBAPI.md**](WEBAPI.md) - Full API reference
- [**MANAGEMENT.md**](MANAGEMENT.md) - Management console guide
- [**QUICKSTART.md**](QUICKSTART.md) - Quick start guide
- [**DOCKER.md**](DOCKER.md) - Docker deployment guide

## 🧪 Testing

```bash
dotnet test
```

**54/54 tests passing** ✅

## 🔥 Performance

### TinyLlama 1.1B Q2_K on AMD RX 6600 XT

- **TPS**: ~45 tokens/second
- **Context**: 2048 tokens
- **KV Cache**: Enabled with INT8 quantization

### Optimization Tips

1. Use quantized models (Q4_K_M, Q5_K_M)
2. Enable KV cache quantization
3. Set appropriate `auto_unload_minutes`
4. Use batch inference for multiple prompts
5. Configure rate limiting

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details.

---

**Built with ❤️ using .NET 10.0, Vulkan, CUDA, and ROCm**

**⭐ Star us on GitHub if you find this useful!**
