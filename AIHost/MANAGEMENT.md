# AIHost Management Console

## 🎯 Обзор

AIHost теперь включает полнофункциональную систему управления с веб-интерфейсом, API управления, аутентификацией и логированием запросов.

## 🔐 Система токенов

### Конфигурация

В `config/server.json`:

```json
{
  "manage_token": "your-secret-management-token",
  "tokens_file": "config/tokens.txt"
}
```

### Файл токенов (config/tokens.txt)

```
# Один токен на строку
# Строки начинающиеся с # - комментарии

sk-user1-token-123
sk-user2-token-456
```

**Важно:** Если файл `tokens.txt` не существует, аутентификация отключена для всех запросов.

### Уровни доступа

1. **Публичные эндпоинты** (без токена):
   - `GET /` - Health check
   - `GET /health` - Health check
   - `GET /swagger` - API документация
   - `GET /v1/models` - Список моделей
   - `GET /api/tags` - Список моделей (Ollama)

2. **API эндпоинты** (требуют токен если `tokens_file` настроен):
   - `POST /api/generate` - Генерация текста
   - `POST /api/chat` - Чат
   - `POST /v1/chat/completions` - OpenAI чат

3. **Management эндпоинты** (требуют `manage_token`):
   - `GET /manage/models` - Статистика моделей
   - `POST /manage/models/{name}/reload` - Перезагрузка модели
   - `DELETE /manage/models/{name}` - Выгрузка модели
   - `GET /manage/logs` - Просмотр логов
   - `POST /manage/download` - Загрузка моделей

### Использование токенов

```bash
# API токен
curl -H "Authorization: Bearer sk-user1-token-123" \
  http://localhost:11434/api/generate -d '{"model":"tinyllama-chat","prompt":"Hello"}'

# Management токен
curl -H "Authorization: Bearer your-secret-management-token" \
  http://localhost:11434/manage/models
```

## 🌐 Веб-интерфейс

Откройте в браузере: **http://localhost:11434/**

### Возможности

#### 1. Страница Models
- Просмотр загруженных моделей
- Статистика по каждой модели:
  - Общее количество запросов
  - Средний TPS (токенов в секунду)
  - Время загрузки
  - Последний запрос и промпт
- Действия:
  - 🔄 **Reload** - Перезагрузить модель
  - 🗑️ **Unload** - Выгрузить из памяти

#### 2. Страница Request Logs
- История последних запросов (до 1000)
- Информация:
  - Время запроса
  - Модель
  - Эндпоинт
  - Промпт
  - Количество токенов
  - TPS
  - Длительность
  - Статус (успех/ошибка)
- Автообновление каждые 5 секунд

#### 3. Страница Download Model
- Загрузка моделей из внешних источников
- Поля:
  - **Model Name** - Имя модели (например, `llama2-7b`)
  - **Download URL** - Прямая ссылка на файл
  - **Format** - Формат файла (по умолчанию `gguf`)
  - **Management Token** - Токен доступа (если настроен)

## 📡 Management API

### GET /manage/models

Получить список загруженных моделей с статистикой.

**Требует:** `manage_token`

**Ответ:**
```json
[
  {
    "name": "tinyllama-chat",
    "loaded_at": "2026-04-28T10:30:00Z",
    "total_requests": 42,
    "average_tps": 45.3,
    "last_request_at": "2026-04-28T11:00:00Z",
    "last_prompt": "What is the capital of France?",
    "config": {
      "name": "tinyllama-chat",
      "model": "/path/to/model.gguf",
      "parameters": { ... }
    }
  }
]
```

### GET /manage/models/{name}

Получить детальную информацию о конкретной модели.

**Требует:** `manage_token`

### DELETE /manage/models/{name}

Выгрузить модель из памяти.

**Требует:** `manage_token`

**Пример:**
```bash
curl -X DELETE \
  -H "Authorization: Bearer your-manage-token" \
  http://localhost:11434/manage/models/tinyllama-chat
```

### POST /manage/models/{name}/reload

Перезагрузить модель (выгрузить + загрузить заново).

**Требует:** `manage_token`

**Пример:**
```bash
curl -X POST \
  -H "Authorization: Bearer your-manage-token" \
  http://localhost:11434/manage/models/tinyllama-chat/reload
```

### GET /manage/logs

Получить последние логи запросов.

**Параметры:**
- `count` (int) - Количество логов (по умолчанию 100, максимум 1000)

**Требует:** `manage_token`

**Пример:**
```bash
curl -H "Authorization: Bearer your-manage-token" \
  "http://localhost:11434/manage/logs?count=50"
```

**Ответ:**
```json
[
  {
    "timestamp": "2026-04-28T11:00:00Z",
    "endpoint": "/api/generate",
    "method": "POST",
    "model_name": "tinyllama-chat",
    "prompt": "What is AI?",
    "tokens_generated": 128,
    "duration_ms": 2834.5,
    "tps": 45.2,
    "success": true,
    "error": null
  }
]
```

### GET /manage/logs/{modelName}

Получить логи для конкретной модели.

**Требует:** `manage_token`

### DELETE /manage/logs

Очистить все логи.

**Требует:** `manage_token`

### POST /manage/download

Загрузить модель из URL.

**Требует:** `manage_token`

**Body:**
```json
{
  "name": "my-custom-model",
  "url": "https://example.com/model.gguf",
  "format": "gguf"
}
```

**Ответ:**
```json
{
  "message": "Model 'my-custom-model' downloaded successfully",
  "path": "/path/to/models/my-custom-model/model.gguf"
}
```

### GET /manage/configs

Получить все доступные конфигурации моделей (из `models/*/model.json`).

**Требует:** `manage_token`

## 🤖 HuggingFace интеграция

AIHost включает встроенный HuggingFace downloader для автоматической загрузки моделей.

### Использование через веб-интерфейс

1. Открыть http://localhost:11434
2. Перейти на вкладку **Download Model**
3. Указать:
   - Name: `llama2-7b-chat`
   - URL: `https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf`
   - Format: `gguf`
4. Ввести management token (если настроен)
5. Нажать **Download Model**

### Использование через API

```bash
curl -X POST http://localhost:11434/manage/download \
  -H "Authorization: Bearer your-manage-token" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "llama2-7b-chat",
    "url": "https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf",
    "format": "gguf"
  }'
```

### Прямые ссылки HuggingFace

Формат: `https://huggingface.co/{author}/{repo}/resolve/main/{file}`

**Примеры:**

```bash
# TinyLlama 1.1B Q4_K_M
https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf

# Mistral 7B Q4_K_M
https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf

# CodeLlama 7B Q4_K_M
https://huggingface.co/TheBloke/CodeLlama-7B-Instruct-GGUF/resolve/main/codellama-7b-instruct.Q4_K_M.gguf
```

## ⚙️ Конфигурация модели (model.json)

Каждая модель имеет свой файл конфигурации `data/models/{model-name}/model.json`.

### Полный пример

```json
{
  "name": "tinyllama-chat",
  "model": "D:\\path\\to\\model.gguf",
  "format": "gguf",
  "auto_download": false,
  
  "compute_provider": "vulkan",
  "device_index": 0,
  "keep_alive": 30,
  "num_gpu_layers": -1,
  "batch_size": null,
  "enable_mmap": true,
  "enable_mlock": false,
  
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
  "description": "TinyLlama 1.1B Chat v1.0 Q2_K quantization",
  "tags": ["chat", "instruct", "1.1b"]
}
```

### Основные параметры

| Параметр | Тип | Описание |
|----------|-----|----------|
| `name` | string | Уникальный идентификатор модели |
| `model` | string | Путь к файлу модели или URL |
| `format` | string | Формат модели (по умолчанию "gguf") |
| `auto_download` | bool | Автоматически загрузить если файл не найден |
| `description` | string | Описание модели |
| `tags` | array | Теги для категоризации |

### Настройки вычислений

**НОВОЕ!** Эти параметры позволяют переопределить глобальные настройки сервера для конкретной модели.

| Параметр | Тип | Описание |
|----------|-----|----------|
| `compute_provider` | string? | Драйвер вычислений: "vulkan", "cuda", "rocm" (null = использовать глобальный) |
| `device_index` | int? | Индекс GPU устройства (null = использовать глобальный) |
| `keep_alive` | int? | Минуты до автовыгрузки (null = глобальный, 0 = никогда не выгружать) |
| `num_gpu_layers` | int? | Слои на GPU: -1 = все, 0 = CPU, N = конкретное количество |
| `batch_size` | int? | Предпочитаемый размер батча |
| `enable_mmap` | bool | Использовать memory mapping (экономит RAM) |
| `enable_mlock` | bool | Закрепить в RAM (предотвратить swap) |

#### Примеры использования

**Модель только на CPU:**
```json
{
  "num_gpu_layers": 0
}
```

**Гибридный режим (половина слоев на GPU):**
```json
{
  "num_gpu_layers": 11
}
```

**Выделенное CUDA устройство:**
```json
{
  "compute_provider": "cuda",
  "device_index": 1
}
```

**Никогда не выгружать модель:**
```json
{
  "keep_alive": 0
}
```

**Быстрая выгрузка неиспользуемых моделей:**
```json
{
  "keep_alive": 5
}
```

### Параметры генерации

| Параметр | Тип | Описание |
|----------|-----|----------|
| `temperature` | float | Случайность (0.0-2.0, default 0.7) |
| `top_k` | int | Top-K sampling (default 40) |
| `top_p` | float | Top-P/nucleus sampling (0.0-1.0, default 0.9) |
| `repetition_penalty` | float | Штраф за повторения (default 1.1) |
| `context_size` | int | Размер контекста в токенах (default 2048) |
| `max_tokens` | int | Максимум токенов генерации (default 512) |
| `seed` | int | Random seed (-1 = случайный) |
| `use_kv_cache` | bool | Использовать KV кэш (default true) |
| `kv_cache_quantization` | string | Квантизация KV кэша: "none", "int8", "int4" |
| `stop` | array | Последовательности остановки (например ["\n\n", "User:"]) |

### Системные сообщения

```json
{
  "system_messages": [
    "You are a helpful AI assistant.",
    "You always respond in a professional manner."
  ],
  "system_message_files": [
    "system.txt",
    "additional_instructions.txt"
  ]
}
```

Файлы загружаются из директории модели: `data/models/{model-name}/system.txt`

## 📊 Логирование запросов

### Автоматическое логирование

Все API запросы автоматически логируются с информацией:
- Timestamp
- Эндпоинт и метод
- Имя модели
- Промпт
- Количество сгенерированных токенов
- TPS (токены в секунду)
- Длительность в миллисекундах
- Статус (успех/ошибка)
- Текст ошибки (если есть)

### Просмотр логов

**Веб-интерфейс:** http://localhost:11434 → вкладка **Request Logs**

**API:**
```bash
# Последние 100 логов
curl -H "Authorization: Bearer your-manage-token" \
  http://localhost:11434/manage/logs

# Последние 50 логов
curl -H "Authorization: Bearer your-manage-token" \
  "http://localhost:11434/manage/logs?count=50"

# Логи конкретной модели
curl -H "Authorization: Bearer your-manage-token" \
  http://localhost:11434/manage/logs/tinyllama-chat
```

### Хранение логов

- Логи хранятся **в памяти** (не на диске)
- Максимум: **1000 последних запросов**
- При достижении лимита старые логи удаляются автоматически
- Логи очищаются при перезапуске сервера

### Очистка логов

```bash
curl -X DELETE \
  -H "Authorization: Bearer your-manage-token" \
  http://localhost:11434/manage/logs
```

## 🛠️ Примеры использования

### Полный workflow

1. **Запустить сервер:**
   ```bash
   dotnet run -- --web
   ```

2. **Открыть веб-интерфейс:**
   ```
   http://localhost:11434
   ```

3. **Загрузить модель через UI:**
   - Перейти на Download Model
   - Name: `mistral-7b`
   - URL: `https://huggingface.co/.../mistral-7b.gguf`
   - Скачать

4. **Создать `config/tokens.txt`:**
   ```
   sk-dev-token-123
   sk-prod-token-456
   ```

5. **Настроить `server.json`:**
   ```json
   {
     "manage_token": "admin-secret-token",
     "tokens_file": "config/tokens.txt"
   }
   ```

6. **Использовать API:**
   ```bash
   # Генерация (требует user token)
   curl -H "Authorization: Bearer sk-dev-token-123" \
     http://localhost:11434/api/generate \
     -d '{"model":"mistral-7b","prompt":"Hello"}'

   # Просмотр статистики (требует manage token)
   curl -H "Authorization: Bearer admin-secret-token" \
     http://localhost:11434/manage/models
   ```

## 🔒 Безопасность

### Рекомендации

1. **Всегда устанавливайте manage_token** для production окружения
2. **Используйте HTTPS** для production (настройте reverse proxy: nginx/caddy)
3. **Генерируйте сложные токены:**
   ```bash
   # Linux/Mac
   openssl rand -base64 32

   # PowerShell
   [Convert]::ToBase64String((1..32 | ForEach-Object { Get-Random -Maximum 256 }))
   ```
4. **Ограничьте доступ к tokens_file** (chmod 600)
5. **Регулярно ротируйте токены**
6. **Не храните токены в Git** (добавьте `config/tokens.txt` в `.gitignore`)

### Отключение аутентификации

Для локальной разработки можно отключить аутентификацию:

```json
{
  "manage_token": null,
  "tokens_file": null
}
```

Или просто удалите/переименуйте `config/tokens.txt`.

## 📈 Мониторинг производительности

### Метрики в веб-интерфейсе

- **Average TPS** - Средняя скорость генерации
- **Total Requests** - Общее количество запросов
- **Last Request** - Время последнего запроса
- **Last Prompt** - Последний промпт (первые 100 символов)

### Через API

```bash
curl -H "Authorization: Bearer manage-token" \
  http://localhost:11434/manage/models/tinyllama-chat
```

Ответ включает все метрики производительности.

## 🔧 Troubleshooting

### Веб-интерфейс не открывается

1. Проверьте что сервер запущен с флагом `--web`
2. Проверьте порт в `config/server.json`
3. Убедитесь что папка `wwwroot` существует

### 401 Unauthorized

1. Проверьте наличие `config/tokens.txt`
2. Проверьте токен в заголовке `Authorization: Bearer <token>`
3. Для management API используйте `manage_token` из `server.json`

### Модель не загружается

1. Проверьте путь в `models/{name}/model.json`
2. Убедитесь что файл существует и доступен
3. Проверьте логи в консоли сервера

### Скачивание модели не работает

1. Проверьте что URL корректный (прямая ссылка на файл)
2. Проверьте интернет-соединение
3. Убедитесь что достаточно места на диске
4. Для HuggingFace используйте формат: `https://huggingface.co/{author}/{repo}/resolve/main/{file}`

## 🎉 Готово!

Теперь у вас есть полнофункциональная система управления AIHost с:
- ✅ Веб-интерфейсом
- ✅ Аутентификацией
- ✅ Логированием запросов
- ✅ Мониторингом моделей
- ✅ Загрузкой из HuggingFace
- ✅ Management API

Документация по основным API эндпоинтам: [WEBAPI.md](WEBAPI.md)
