# AIHost WebAPI - Quick Start Guide

## 🚀 Запуск веб-сервера

### 1. Запуск в режиме веб-сервера

```bash
cd E:\my_dev\AIHost\AIHost
dotnet run -- --web
```

Сервер запустится на http://localhost:11434

### 2. Проверка работы

```bash
# Health check
curl http://localhost:11434/

# Список моделей (Ollama)
curl http://localhost:11434/api/tags

# Список моделей (OpenAI)
curl http://localhost:11434/v1/models
```

## 📁 Структура конфигурации

```
AIHost/
├── config/
│   └── server.json                    # Настройки сервера
├── models/
│   └── tinyllama-chat/
│       ├── model.json                 # Конфиг модели
│       ├── system.txt                 # Системное сообщение (опционально)
│       └── tinyllama-1.1b-chat.gguf   # Файл модели
```

### Пример server.json

```json
{
  "models_directory": "./models",
  "host": "localhost",
  "port": 11434,
  "enable_openai_api": true,
  "enable_ollama_api": true,
  "compute_provider": "vulkan",
  "device_index": 0,
  "enable_cors": true,
  "log_level": "info"
}
```

### Пример model.json

```json
{
  "name": "tinyllama-chat",
  "model": "D:\\path\\to\\model.gguf",
  "format": "gguf",
  "auto_download": false,
  "parameters": {
    "temperature": 0.7,
    "top_k": 40,
    "top_p": 0.9,
    "repetition_penalty": 1.1,
    "context_size": 2048,
    "max_tokens": 512,
    "seed": -1,
    "use_kv_cache": true,
    "kv_cache_quantization": "none",
    "stop": []
  },
  "system_messages": [
    "You are a helpful AI assistant."
  ],
  "system_message_files": [
    "system.txt"
  ],
  "description": "TinyLlama chat model",
  "tags": ["chat", "1.1b"]
}
```

## 🔌 API примеры

### Ollama API - Generate

```bash
curl http://localhost:11434/api/generate -d '{
  "model": "tinyllama-chat",
  "prompt": "What is the capital of France?",
  "stream": false
}'
```

### Ollama API - Chat

```bash
curl http://localhost:11434/api/chat -d '{
  "model": "tinyllama-chat",
  "messages": [
    {"role": "user", "content": "Hello!"}
  ]
}'
```

### OpenAI API - Chat Completions

```bash
curl http://localhost:11434/v1/chat/completions -d '{
  "model": "tinyllama-chat",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"}
  ]
}'
```

## 🐍 Python примеры

### С OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="dummy"  # Не требуется
)

response = client.chat.completions.create(
    model="tinyllama-chat",
    messages=[
        {"role": "user", "content": "Hello!"}
    ]
)

print(response.choices[0].message.content)
```

### С requests

```python
import requests

response = requests.post(
    "http://localhost:11434/api/chat",
    json={
        "model": "tinyllama-chat",
        "messages": [
            {"role": "user", "content": "Hello!"}
        ]
    }
)

print(response.json()["message"]["content"])
```

## 🌐 JavaScript примеры

```javascript
// Fetch API
const response = await fetch('http://localhost:11434/v1/chat/completions', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    model: 'tinyllama-chat',
    messages: [
      { role: 'user', content: 'Hello!' }
    ]
  })
});

const data = await response.json();
console.log(data.choices[0].message.content);
```

## ⚙️ Compute Provider

Доступные провайдеры:
- **vulkan** (по умолчанию) - AMD/NVIDIA через Vulkan
- **cuda** - NVIDIA CUDA
- **rocm** - AMD ROCm/HIP

Установка в config/server.json:
```json
{
  "compute_provider": "vulkan",
  "device_index": 0
}
```

## 📊 Параметры генерации

| Параметр | Описание | По умолчанию |
|----------|----------|--------------|
| temperature | Креативность (0.0-2.0) | 0.7 |
| top_k | Top-K сэмплирование | 40 |
| top_p | Top-P (nucleus) | 0.9 |
| repetition_penalty | Штраф за повторы | 1.1 |
| max_tokens | Максимум токенов | 512 |
| context_size | Размер контекста | 2048 |
| seed | Random seed (-1 = random) | -1 |
| use_kv_cache | Использовать KV кэш | true |

## 🔐 Автозагрузка моделей

Если модель не найдена локально:

```json
{
  "model": "https://huggingface.co/user/model.gguf",
  "auto_download": true
}
```

При первом запросе модель будет скачана автоматически.

## 📝 Системные сообщения

### Вариант 1: В конфиге

```json
{
  "system_messages": [
    "You are a helpful assistant.",
    "Always be professional."
  ]
}
```

### Вариант 2: Внешние файлы

```json
{
  "system_message_files": [
    "system.txt",
    "guidelines.txt"
  ]
}
```

Файлы создаются в папке модели (`models/tinyllama-chat/system.txt`).

## 🐛 Troubleshooting

### Порт занят

Измените порт в `config/server.json`:
```json
{
  "port": 8080
}
```

### Модель не загружается

1. Проверьте путь в `model.json`
2. Убедитесь что файл существует
3. Проверьте формат (должен быть GGUF)

### GPU не обнаружен

1. Установите драйвера Vulkan/CUDA/ROCm
2. Проверьте `compute_provider` в `config/server.json`
3. Попробуйте другой `device_index`

## 📚 Полная документация

См. [WEBAPI.md](WEBAPI.md) для детальной документации API.

## 🎯 Быстрый тест

```bash
# 1. Запустить сервер
dotnet run -- --web

# 2. В другом терминале
curl http://localhost:11434/api/generate -d '{"model":"tinyllama-chat","prompt":"2+2=?"}'
```

Готово! 🎉
