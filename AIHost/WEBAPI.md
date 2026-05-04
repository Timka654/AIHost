# AIHost Web API

Веб-сервер для запуска LLM моделей с поддержкой Ollama и OpenAI API.

## 🚀 Быстрый старт

### 1. Конфигурация сервера

Создайте `config/server.json`:

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

### 2. Конфигурация модели

Создайте папку модели: `data/models/tinyllama-chat/model.json`

```json
{
  "name": "tinyllama-chat",
  "model": "D:\\path\\to\\model.gguf",
  "format": "gguf",
  "auto_download": false,
  
  "compute_provider": "vulkan",
  "device_index": 0,
  "keep_alive": 30,
  "enable_mmap": true,
  
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

### 3. Системные сообщения

**Вариант 1: В конфиге**
```json
{
  "system_messages": [
    "You are a helpful assistant.",
    "Always be polite and professional."
  ]
}
```

**Вариант 2: Внешние файлы**
```json
{
  "system_message_files": [
    "system.txt",
    "guidelines.txt"
  ]
}
```

Создайте `models/tinyllama-chat/system.txt`:
```
You are a helpful, respectful and honest assistant.
Always answer as helpfully as possible.
```

### 4. Запуск сервера

```bash
dotnet run --project AIHost -- --web
```

Сервер запустится на http://localhost:11434

## 📡 API Endpoints

### Ollama API (совместимость)

#### POST /api/generate
Генерация текста из промпта

**Запрос:**
```json
{
  "model": "tinyllama-chat",
  "prompt": "What is the capital of France?",
  "stream": false,
  "options": {
    "temperature": 0.7,
    "top_k": 40,
    "num_predict": 100
  }
}
```

**Ответ:**
```json
{
  "model": "tinyllama-chat",
  "created_at": "2026-04-28T10:00:00Z",
  "response": "The capital of France is Paris.",
  "done": true,
  "total_duration": 1234567890,
  "prompt_eval_count": 10,
  "eval_count": 8
}
```

#### POST /api/chat
Чат с моделью

**Запрос:**
```json
{
  "model": "tinyllama-chat",
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful assistant."
    },
    {
      "role": "user",
      "content": "Hello!"
    }
  ],
  "stream": false
}
```

**Ответ:**
```json
{
  "model": "tinyllama-chat",
  "created_at": "2026-04-28T10:00:00Z",
  "message": {
    "role": "assistant",
    "content": "Hello! How can I help you today?"
  },
  "done": true
}
```

#### GET /api/tags
Список доступных моделей

**Ответ:**
```json
{
  "models": [
    {
      "name": "tinyllama-chat",
      "size": 1234567890,
      "details": {
        "format": "gguf",
        "parameter_size": "1.1B",
        "quantization_level": "Q2_K"
      }
    }
  ]
}
```

### OpenAI API (совместимость)

#### POST /v1/chat/completions
Chat completion

**Запрос:**
```json
{
  "model": "tinyllama-chat",
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful assistant."
    },
    {
      "role": "user",
      "content": "Hello!"
    }
  ],
  "temperature": 0.7,
  "max_tokens": 100
}
```

**Ответ:**
```json
{
  "id": "chatcmpl-123",
  "object": "chat.completion",
  "created": 1682000000,
  "model": "tinyllama-chat",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Hello! How can I help you?"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 15,
    "completion_tokens": 8,
    "total_tokens": 23
  }
}
```

#### GET /v1/models
Список моделей

**Ответ:**
```json
{
  "object": "list",
  "data": [
    {
      "id": "tinyllama-chat",
      "object": "model",
      "created": 1682000000,
      "owned_by": "aihost"
    }
  ]
}
```

## 🔧 Примеры использования

### cURL (Ollama API)

```bash
# Generate
curl http://localhost:11434/api/generate -d '{
  "model": "tinyllama-chat",
  "prompt": "Why is the sky blue?"
}'

# Chat
curl http://localhost:11434/api/chat -d '{
  "model": "tinyllama-chat",
  "messages": [
    {"role": "user", "content": "Hello!"}
  ]
}'
```

### cURL (OpenAI API)

```bash
curl http://localhost:11434/v1/chat/completions -d '{
  "model": "tinyllama-chat",
  "messages": [
    {"role": "user", "content": "Hello!"}
  ]
}'
```

### Python (OpenAI SDK)

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="dummy"  # Not required
)

response = client.chat.completions.create(
    model="tinyllama-chat",
    messages=[
        {"role": "user", "content": "Hello!"}
    ]
)

print(response.choices[0].message.content)
```

### JavaScript (fetch)

```javascript
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

## 📂 Структура папок

```
AIHost/
├── config/
│   └── server.json              # Конфигурация сервера
├── models/
│   ├── tinyllama-chat/
│   │   ├── model.json          # Конфигурация модели
│   │   ├── system.txt          # Системное сообщение (опционально)
│   │   └── model.gguf          # Файл модели (или URL)
│   └── another-model/
│       └── model.json
```

## ⚙️ Параметры генерации

| Параметр | Ollama | OpenAI | Описание |
|----------|--------|--------|----------|
| temperature | ✅ | ✅ | Температура (0.0-2.0) |
| top_k | ✅ | ❌ | Top-K сэмплирование |
| top_p | ✅ | ✅ | Top-P (nucleus) сэмплирование |
| repeat_penalty | ✅ | ❌ | Штраф за повторения |
| repetition_penalty | ❌ | ❌ | Альтернативное название |
| num_predict | ✅ | ❌ | Максимум токенов (Ollama) |
| max_tokens | ❌ | ✅ | Максимум токенов (OpenAI) |
| seed | ✅ | ✅ | Random seed |
| stop | ✅ | ✅ | Stop sequences |

## 🎛️ Compute Provider

### Доступные провайдеры
- `vulkan` - Vulkan (универсальный, кроссплатформенный)
- `cuda` - NVIDIA CUDA
- `rocm` - AMD ROCm

## 🔧 Management API

AIHost предоставляет полный REST API для управления моделями, мониторинга ресурсов и системной информации.

### Base URL: `/manage`

### Models Management

#### GET `/manage/models`
Список всех загруженных моделей со статистикой

**Response:**
```json
[
  {
    "name": "tinyllama",
    "loaded_at": "2026-05-03T10:00:00Z",
    "total_requests": 42,
    "average_tps": 35.5,
    "last_request_at": "2026-05-03T10:30:00Z",
    "last_prompt": "What is AI?",
    "config": { /* ModelConfig */ }
  }
]
```

#### GET `/manage/models/{name}`
Статус конкретной модели

#### DELETE `/manage/models/{name}`
Выгрузить модель из памяти

**Response:**
```json
{
  "message": "Model 'tinyllama' unloaded"
}
```

#### POST `/manage/models/{name}/reload`
Перезагрузить модель

### System Information

#### GET `/manage/status`
Общий статус сервера

**Response:**
```json
{
  "status": "running",
  "uptime_seconds": 3600,
  "loaded_models": 1,
  "total_requests": 42,
  "memory_mb": 2048.5,
  "thread_count": 24,
  "version": "1.0.0"
}
```

#### GET `/manage/system/resources`
Информация о системных ресурсах

**Response:**
```json
{
  "total_memory_bytes": 17179869184,
  "available_memory_bytes": 8589934592,
  "used_memory_bytes": 8589934592,
  "cpu_usage_percent": 12.5,
  "thread_count": 24,
  "process_memory_bytes": 2147483648,
  "gc_total_memory_bytes": 1073741824
}
```

#### GET `/manage/system/compute`
Информация о compute устройствах

**Response:**
```json
{
  "devices": [
    {
      "model": "tinyllama",
      "device_name": "Vulkan",
      "device_type": "VulkanComputeDevice"
    }
  ],
  "total_models": 1,
  "has_gpu": true
}
```

### Performance Metrics

#### GET `/manage/performance`
Метрики производительности всех моделей

**Response:**
```json
{
  "models": [
    {
      "model": "tinyllama",
      "total_requests": 42,
      "average_tps": 35.5,
      "uptime_seconds": 3600,
      "last_request_seconds_ago": 300
    }
  ],
  "summary": {
    "total_models": 1,
    "total_requests": 42,
    "average_tps": 35.5
  }
}
```

### Logs Management

#### GET `/manage/logs?count=100`
Получить последние логи запросов

**Parameters:**
- `count` (optional, default: 100) - количество записей

**Response:**
```json
[
  {
    "timestamp": "2026-05-03T10:30:00Z",
    "modelName": "tinyllama",
    "endpoint": "/api/generate",
    "prompt": "What is AI?",
    "tokensGenerated": 50,
    "tps": 35.5,
    "durationMs": 1408,
    "success": true
  }
]
```

#### GET `/manage/logs/{modelName}?count=100`
Логи для конкретной модели

#### DELETE `/manage/logs`
Очистить все логи

### Model Configs

#### GET `/manage/configs`
Список всех конфигураций моделей

**Response:**
```json
{
  "tinyllama": {
    "name": "tinyllama",
    "model_path": "./models/tinyllama.gguf",
    "format": "gguf",
    "description": "TinyLlama 1.1B Chat",
    "tags": ["chat", "small"],
    "parameters": { /* GenerationParameters */ }
  }
}
```

#### POST `/manage/configs`
Создать или обновить конфигурацию модели

**Request:**
```json
{
  "name": "my-model",
  "model_path": "./models/my-model.gguf",
  "format": "gguf",
  "description": "My custom model",
  "parameters": {
    "temperature": 0.7,
    "top_k": 40,
    "top_p": 0.9,
    "max_tokens": 512
  }
}
```

#### DELETE `/manage/configs/{name}`
Удалить конфигурацию модели

### Cache Management

#### GET `/manage/cache`
Информация о кэш-директории

**Response:**
```json
{
  "path": "./data/models",
  "file_count": 5,
  "total_size": 5368709120,
  "dir_count": 2
}
```

#### DELETE `/manage/cache`
Очистить весь кэш (осторожно!)

### Direct Chat

#### POST `/manage/chat`
Прямой чат с моделью

**Request:**
```json
{
  "model_name": "tinyllama",
  "message": "What is AI?",
  "system_message": "You are a helpful assistant",
  "temperature": 0.7,
  "top_k": 40,
  "top_p": 0.9,
  "max_tokens": 512
}
```

**Response:**
```json
{
  "model": "tinyllama",
  "response": "AI is artificial intelligence...",
  "tokens": 50,
  "finish_reason": "stop"
}
```

### Model Download

#### POST `/manage/download`
Скачать модель с URL (HuggingFace, etc.)

**Request:**
```json
{
  "name": "my-model",
  "url": "https://huggingface.co/.../model.gguf",
  "format": "gguf"
}
```

**Response:**
```json
{
  "message": "Model 'my-model' downloaded successfully",
  "path": "./data/models/my-model/model.gguf"
}
```

## 🔐 Authentication

Некоторые эндпойнты могут требовать токен управления:

**Header:**
```
Authorization: Bearer <your-management-token>
```

Конфигурация токенов: `data/config/tokens.txt`

## 🌐 Web Management Panel

Полноценная веб-панель управления доступна по адресу: `http://localhost:11434`

### Основные разделы:
- **Dashboard** - общий обзор системы
- **Models** - управление загруженными моделями
- **Resources** - мониторинг системных ресурсов
- **Performance** - метрики производительности
- **Logs** - логи запросов
- **Configs** - конфигурации моделей
- **Cache** - управление кэшем
- **Chat** - тестирование моделей

См. [WEBPANEL.md](WEBPANEL.md) для подробной документации.

## 🔄 Auto-refresh

Web UI автоматически обновляется каждые 5 секунд для разделов:
- Dashboard
- Models
- Resources
- Performance
- Logs

Поддерживаемые провайдеры:
- `vulkan` - AMD/NVIDIA через Vulkan (по умолчанию)
- `cuda` - NVIDIA CUDA
- `rocm` - AMD ROCm/HIP

Установка в `config/server.json`:
```json
{
  "compute_provider": "vulkan",
  "device_index": 0
}
```

## 🔐 Автозагрузка моделей

Если модель не найдена локально и указан URL:

```json
{
  "model": "https://huggingface.co/user/model.gguf",
  "auto_download": true
}
```

При первом запросе модель будет скачана автоматически.

## 📊 Мониторинг

### Health Check
```bash
curl http://localhost:11434/
```

Ответ:
```json
{
  "status": "ok",
  "version": "1.0.0",
  "provider": "Vulkan",
  "device": "AMD Radeon RX 6600 XT",
  "apis": {
    "ollama": true,
    "openai": true
  }
}
```

## 🐛 Troubleshooting

### Модель не загружается
- Проверьте путь в `model.json`
- Убедитесь что файл существует
- Проверьте права доступа

### API не отвечает
- Проверьте порт в `server.json`
- Убедитесь что CORS включен
- Проверьте логи сервера

### GPU не обнаружен
- Установите драйвера (Vulkan/CUDA/ROCm)
- Проверьте `compute_provider` в конфиге
- Попробуйте другой `device_index`
