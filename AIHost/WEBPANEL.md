# 🎨 Web Management Panel Documentation

## Overview

AIHost provides a comprehensive web-based management console for monitoring and controlling LLM models, system resources, and performance metrics.

## Access

The web panel is available at: `http://localhost:11434` (or your configured host/port)

## Features

### 📊 Dashboard
- **Real-time Overview**: System status at a glance
- **Key Metrics**:
  - Loaded Models Count
  - Total Requests Processed
  - Memory Usage
  - Server Uptime
- **Auto-refresh**: Updates every 5 seconds

### 🤖 Models
- **Loaded Models**: View all currently loaded models
- **Model Statistics**:
  - Total requests processed
  - Average tokens per second (TPS)
  - Load time and last request time
  - Last prompt preview
- **Model Actions**:
  - 🔄 Reload: Refresh model from disk
  - 🗑️ Unload: Remove model from memory
- **Auto-refresh**: Updates every 5 seconds

### 💻 Resources
- **System Memory**:
  - Total, Used, and Available Memory
  - Process Memory Usage
  - GC Memory Statistics
  - Memory usage percentage
- **System Info**:
  - Thread Count
  - CPU Usage
- **Compute Devices**:
  - Lists all active compute devices
  - Device names and types (Vulkan/CUDA/ROCm)
  - Associated models
- **Auto-refresh**: Updates every 5 seconds

### ⚡ Performance
- **Summary Metrics**:
  - Total models loaded
  - Total requests across all models
  - Average TPS across all models
- **Per-Model Performance**:
  - Request count
  - Average tokens per second
  - Model uptime
  - Time since last request
- **Auto-refresh**: Updates every 5 seconds

### 📝 Logs
- **Request Logs**: View recent API requests
- **Log Details**:
  - Timestamp
  - Model name
  - API endpoint
  - Prompt preview
  - Tokens generated
  - TPS (tokens per second)
  - Duration
  - Success/Error status
- **Auto-refresh**: Updates every 5 seconds
- **Limit**: Shows last 50 requests

### ⚙️ Configs
- **Model Configurations**: View and manage model configs
- **Config Information**:
  - Model format (GGUF, etc.)
  - Model file path
  - Description
  - Tags
- **Actions**:
  - Create new model config
  - Delete existing config
- **Create Config Form**:
  - Model name and path
  - Generation parameters (temperature, top_k, top_p, max_tokens)
  - Format and description

### 💾 Cache
- **Cache Statistics**:
  - Total files in cache
  - Total size (with human-readable format)
  - Number of directories
  - Cache directory path
- **Actions**:
  - Clear all cache (requires confirmation)

### 💬 Chat
- **Direct Model Chat**: Test models directly from the web UI
- **Chat Features**:
  - Select loaded model
  - Optional system message
  - Adjustable temperature
  - Max tokens control
- **Response Display**: Shows model response with token count

## API Endpoints

### Management API (`/manage`)

#### Models
- `GET /manage/models` - List all loaded models with statistics
- `GET /manage/models/{name}` - Get specific model status
- `DELETE /manage/models/{name}` - Unload model from memory
- `POST /manage/models/{name}/reload` - Reload model

#### System
- `GET /manage/status` - Server status and uptime
- `GET /manage/system/resources` - System resource information
- `GET /manage/system/compute` - Compute device information
- `GET /manage/system/buffers` - Buffer pool statistics (planned)

#### Performance
- `GET /manage/performance` - Performance metrics for all models

#### Logs
- `GET /manage/logs?count=N` - Recent request logs (default: 100)
- `GET /manage/logs/{modelName}?count=N` - Logs for specific model
- `DELETE /manage/logs` - Clear all logs

#### Configs
- `GET /manage/configs` - List all model configurations
- `POST /manage/configs` - Create or update model config
- `DELETE /manage/configs/{name}` - Delete model config

#### Cache
- `GET /manage/cache` - Cache directory information
- `DELETE /manage/cache` - Clear all cache

#### Chat
- `POST /manage/chat` - Direct chat with model

#### Download
- `POST /manage/download` - Download model from URL

## Authentication

Some endpoints may require authentication via management token:

1. Configure token in `data/config/tokens.txt`
2. Enter token in "Management Token" field in the UI
3. Token is sent as `Authorization: Bearer <token>` header

## Data Response Format

### Model Status
```json
{
  "name": "tinyllama",
  "loaded_at": "2026-05-03T10:00:00Z",
  "total_requests": 42,
  "average_tps": 35.5,
  "last_request_at": "2026-05-03T10:30:00Z",
  "last_prompt": "What is AI?",
  "config": { /* ModelConfig object */ }
}
```

### System Resources
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

### Performance Metrics
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

## Real-time Updates

The dashboard automatically refreshes data every 5 seconds for the following sections:
- Dashboard
- Models
- Resources
- Performance
- Logs

Manual refresh is available via the 🔄 floating button (bottom-right).

## UI Components

### Status Badges
- 🟢 **LOADED**: Model is loaded and ready
- ✅ **Success**: Request completed successfully
- ❌ **Error**: Request failed

### Actions
- 🔄 **Reload**: Refresh model from disk
- 🗑️ **Unload/Delete**: Remove resource
- 💬 **Chat**: Start conversation
- 📊 **View**: Show details

## Keyboard Shortcuts

- **F5**: Refresh current section
- **Tab Navigation**: Switch between sections

## Browser Compatibility

Tested with:
- Chrome/Edge 90+
- Firefox 88+
- Safari 14+

## Responsive Design

The web panel is fully responsive and works on:
- Desktop (1400px+)
- Tablet (768px - 1399px)
- Mobile (< 768px)

## Troubleshooting

### Models not showing
- Check if server is running (`dotnet run -- --web`)
- Verify models are configured in `data/models/*/model.json`
- Check logs for loading errors

### Authentication errors
- Ensure management token is configured
- Enter token in the UI token field
- Check `data/config/tokens.txt`

### Resource data not loading
- On non-Windows systems, some metrics may be limited
- Requires `System.Management` package on Windows

### Auto-refresh not working
- Check browser console for errors
- Ensure JavaScript is enabled
- Try manual refresh with 🔄 button

## Development

### File Structure
```
wwwroot/
├── index.html    # Main UI structure and styles
└── app.js        # JavaScript logic and API calls
```

### Adding New Sections

1. Add section to navigation in `index.html`:
```html
<button onclick="showSection('newsection')" id="btn-newsection">New Section</button>
```

2. Add section container:
```html
<div id="section-newsection" class="section">
    <h2>New Section</h2>
    <div id="newsection-container"></div>
</div>
```

3. Add load function in `app.js`:
```javascript
async function loadNewSection() {
    // Fetch data and render
}
```

4. Add to `showSection()` switch:
```javascript
else if (section === 'newsection') {
    loadNewSection();
}
```

5. Add to auto-refresh list if needed.

## Security Considerations

- Always use management tokens in production
- Consider HTTPS for external access
- Limit access via firewall/reverse proxy
- Review logs for suspicious activity
- Use rate limiting (built-in via middleware)

## Future Enhancements

- [ ] WebSocket real-time updates
- [ ] Buffer pool statistics
- [ ] GPU utilization graphs
- [ ] Export logs/metrics
- [ ] Dark/Light theme toggle
- [ ] Custom dashboard widgets
- [ ] Multi-user access control
- [ ] Model comparison views
