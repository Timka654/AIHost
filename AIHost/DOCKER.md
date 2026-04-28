# AIHost Docker Deployment

## 🐳 Quick Start

### Build and Run

```bash
# Build Docker image
docker-compose build

# Start container
docker-compose up -d

# View logs
docker-compose logs -f

# Stop container
docker-compose down
```

## 📁 Data Directory Structure

All persistent data is stored in `./data/` which is mounted as a volume:

```
data/
├── config/
│   ├── server.json         # Server configuration
│   └── tokens.txt          # Access tokens
├── models/
│   └── tinyllama-chat/     # Model directories
│       ├── model.json      # Model config
│       ├── system.txt      # System prompts
│       └── *.gguf          # Model files
├── logs/
│   └── requests-*.jsonl    # Request logs by date
└── cache/                  # Temporary cache files
```

**Benefits for Docker:**
- Single volume mount: `-v ./data:/app/data`
- Easy backups: Just backup `./data/`
- Easy migrations: Copy `data/` folder
- Persistent across container restarts

## 🚀 Deployment Options

### Option 1: Docker Compose (Recommended)

```bash
docker-compose up -d
```

Access at: http://localhost:11434

### Option 2: Docker CLI

```bash
# Build image
docker build -t aihost:latest .

# Run container
docker run -d \
  --name aihost \
  -p 11434:11434 \
  -v $(pwd)/data:/app/data \
  --gpus all \
  aihost:latest
```

### Option 3: With NVIDIA GPU

```yaml
# docker-compose.yml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: all
          capabilities: [gpu]
```

```bash
docker-compose up -d
```

### Option 4: With AMD GPU (ROCm)

```yaml
# docker-compose.yml
devices:
  - /dev/kfd:/dev/kfd
  - /dev/dri:/dev/dri
```

Update `data/config/server.json`:
```json
{
  "compute_provider": "rocm"
}
```

## ⚙️ Configuration

### Server Configuration

Edit `data/config/server.json`:

```json
{
  "models_directory": "./data/models",
  "host": "localhost",
  "port": 11434,
  "enable_openai_api": true,
  "enable_ollama_api": true,
  "compute_provider": "vulkan",
  "device_index": 0,
  "enable_cors": true,
  "log_level": "info",
  "manage_token": "your-secret-token",
  "tokens_file": "./data/config/tokens.txt",
  "logs_directory": "./data/logs",
  "cache_directory": "./data/cache",
  "persistent_logs": true,
  "max_log_files": 10,
  "auto_unload_minutes": 30,
  "rate_limit_requests_per_minute": 60
}
```

### Access Tokens

Edit `data/config/tokens.txt`:

```
sk-user1-token-123
sk-user2-token-456
```

## 📦 Model Management

### Download Model Inside Container

```bash
# Get container shell
docker exec -it aihost bash

# Use management API
curl -X POST http://localhost:11434/manage/download \
  -H "Authorization: Bearer your-manage-token" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "llama2-7b",
    "url": "https://huggingface.co/.../llama2.gguf",
    "format": "gguf"
  }'
```

### Or Copy Model from Host

```bash
# Copy model file
cp /path/to/model.gguf ./data/models/my-model/

# Create model.json
cat > ./data/models/my-model/model.json <<EOF
{
  "name": "my-model",
  "model": "./data/models/my-model/model.gguf",
  "format": "gguf",
  "compute_provider": "vulkan",
  "keep_alive": 30,
  "format": "gguf"
}
EOF

# Restart container
docker-compose restart
```

## 🔒 Security

### Production Deployment

1. **Set manage_token:**
   ```json
   {
     "manage_token": "$(openssl rand -base64 32)"
   }
   ```

2. **Configure tokens file:**
   ```bash
   # Generate tokens
   openssl rand -base64 32 > data/config/tokens.txt
   ```

3. **Use HTTPS with reverse proxy:**
   ```yaml
   # docker-compose.yml
   services:
     nginx:
       image: nginx:alpine
       ports:
         - "443:443"
       volumes:
         - ./nginx.conf:/etc/nginx/nginx.conf
         - ./certs:/etc/nginx/certs
       depends_on:
         - aihost
   ```

4. **Limit resources:**
   ```yaml
   deploy:
     resources:
       limits:
         cpus: '4'
         memory: 8G
   ```

## 📊 Monitoring

### View Logs

```bash
# Container logs
docker-compose logs -f

# Request logs
docker exec aihost cat /app/data/logs/requests-2026-04-28.jsonl

# Follow request logs
docker exec aihost tail -f /app/data/logs/requests-$(date +%Y-%m-%d).jsonl
```

### Web Interface

Open: http://localhost:11434

- View loaded models
- Monitor TPS and requests
- View request logs
- Download new models

### Health Check

```bash
curl http://localhost:11434/health
```

## 🔧 Troubleshooting

### Container Won't Start

```bash
# Check logs
docker-compose logs aihost

# Check GPU access
docker exec aihost nvidia-smi  # NVIDIA
docker exec aihost rocm-smi    # AMD
```

### GPU Not Detected

**NVIDIA:**
```bash
# Install nvidia-docker2
sudo apt-get install nvidia-docker2
sudo systemctl restart docker

# Test GPU
docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi
```

**AMD ROCm:**
```bash
# Ensure ROCm drivers installed
rocm-smi

# Add device mappings to docker-compose.yml
devices:
  - /dev/kfd:/dev/kfd
  - /dev/dri:/dev/dri
```

### Port Already in Use

```bash
# Change port in docker-compose.yml
ports:
  - "8080:11434"  # Use 8080 instead
```

### Out of Memory

```bash
# Limit memory usage
deploy:
  resources:
    limits:
      memory: 8G
```

Or unload unused models via web interface.

## 🔄 Updates

### Update Container

```bash
# Pull latest changes
git pull

# Rebuild image
docker-compose build

# Restart with new image
docker-compose up -d

# Old data is preserved in ./data/
```

### Backup Data

```bash
# Backup entire data directory
tar -czf aihost-backup-$(date +%Y%m%d).tar.gz data/

# Restore
tar -xzf aihost-backup-20260428.tar.gz
```

## 🌐 Production Deployment

### With Nginx Reverse Proxy

`nginx.conf`:
```nginx
upstream aihost {
    server aihost:11434;
}

server {
    listen 443 ssl http2;
    server_name ai.example.com;

    ssl_certificate /etc/nginx/certs/fullchain.pem;
    ssl_certificate_key /etc/nginx/certs/privkey.pem;

    location / {
        proxy_pass http://aihost;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### Environment Variables

Override config via environment:

```yaml
environment:
  - AIHOST_PORT=11434
  - AIHOST_COMPUTE_PROVIDER=cuda
  - AIHOST_LOG_LEVEL=debug
```

## 📈 Performance

### Recommended Resources

| Model Size | RAM    | GPU VRAM | CPU Cores |
|-----------|--------|----------|-----------|
| 1-3B      | 4GB    | 4GB      | 2         |
| 7B        | 8GB    | 8GB      | 4         |
| 13B       | 16GB   | 12GB     | 6         |
| 30B+      | 32GB   | 24GB     | 8         |

### Optimization

1. **Use quantized models** (Q4_K_M, Q5_K_M)
2. **Enable KV cache** in model config
3. **Limit concurrent requests** via rate limiting
4. **Auto-unload** unused models (30 min default)

## 🎯 Examples

### Full Production Setup

```bash
# 1. Clone repo
git clone https://github.com/your/aihost.git
cd aihost

# 2. Configure
mkdir -p data/config data/models
cp config/server.json data/config/
nano data/config/server.json  # Set tokens

# 3. Download model
mkdir -p data/models/llama2-7b
wget https://huggingface.co/.../llama2.gguf \
  -O data/models/llama2-7b/model.gguf

# 4. Create model config
cat > data/models/llama2-7b/model.json <<EOF
{
  "name": "llama2-7b",
  "model": "./data/models/llama2-7b/model.gguf",
  "format": "gguf",
  "compute_provider": "vulkan",
  "keep_alive": 30,
  "enable_mmap": true
}
EOF

# 5. Start
docker-compose up -d

# 6. Test
curl -H "Authorization: Bearer your-token" \
  http://localhost:11434/api/generate \
  -d '{"model":"llama2-7b","prompt":"Hello"}'
```

## 📚 Additional Documentation

- [WEBAPI.md](WEBAPI.md) - API reference
- [MANAGEMENT.md](MANAGEMENT.md) - Management console guide
- [QUICKSTART.md](QUICKSTART.md) - Quick start guide

## 🎉 Ready!

Your AIHost instance is now containerized and production-ready! All data persists in `./data/` for easy Docker management.
