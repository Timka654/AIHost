// API base URL
const API_BASE = window.location.origin;

// Current section
let currentSection = 'models';

// Show section
function showSection(section) {
    // Hide all sections
    document.querySelectorAll('.section').forEach(s => s.classList.remove('active'));
    document.querySelectorAll('.nav button').forEach(b => b.classList.remove('active'));

    // Show selected section
    document.getElementById(`section-${section}`).classList.add('active');
    document.getElementById(`btn-${section}`).classList.add('active');

    currentSection = section;

    // Load data for section
    if (section === 'models') {
        loadModels();
    } else if (section === 'logs') {
        loadLogs();
    } else if (section === 'configs') {
        loadConfigs();
    } else if (section === 'cache') {
        loadCacheInfo();
    } else if (section === 'chat') {
        loadChatModels();
    }
}

// Load models
async function loadModels() {
    try {
        const response = await fetch(`${API_BASE}/manage/models`, {
            headers: getAuthHeaders()
        });

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }

        const models = await response.json();
        renderModels(models);
    } catch (error) {
        console.error('Failed to load models:', error);
        document.getElementById('models-container').innerHTML = `
            <div class="empty-state">
                <div>❌</div>
                <p>Failed to load models</p>
                <p style="font-size: 12px; color: #dc3545;">${error.message}</p>
            </div>
        `;
    }
}

// Render models
function renderModels(models) {
    const container = document.getElementById('models-container');

    if (models.length === 0) {
        container.innerHTML = `
            <div class="empty-state">
                <div>📦</div>
                <p>No models loaded</p>
                <p style="font-size: 12px;">Models will appear here when first used</p>
            </div>
        `;
        return;
    }

    container.innerHTML = models.map(model => `
        <div class="model-card">
            <div class="model-header">
                <div class="model-name">${escapeHtml(model.name)}</div>
                <div class="status-badge">LOADED</div>
            </div>
            <div class="stat-row">
                <span class="stat-label">Total Requests:</span>
                <span class="stat-value">${model.total_requests}</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Average TPS:</span>
                <span class="stat-value">${model.average_tps.toFixed(2)}</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Loaded At:</span>
                <span class="stat-value">${formatDate(model.loaded_at)}</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Last Request:</span>
                <span class="stat-value">${model.last_request_at ? formatDate(model.last_request_at) : 'Never'}</span>
            </div>
            ${model.last_prompt ? `
            <div class="stat-row">
                <span class="stat-label">Last Prompt:</span>
                <span class="stat-value" style="font-size: 11px; max-width: 200px; overflow: hidden; text-overflow: ellipsis;">${escapeHtml(model.last_prompt)}</span>
            </div>
            ` : ''}
            <div class="actions">
                <button class="btn btn-reload" onclick="reloadModel('${escapeHtml(model.name)}')">🔄 Reload</button>
                <button class="btn btn-unload" onclick="unloadModel('${escapeHtml(model.name)}')">🗑️ Unload</button>
            </div>
        </div>
    `).join('');
}

// Load configs
async function loadConfigs() {
    try {
        const response = await fetch(`${API_BASE}/manage/configs`, {
            headers: getAuthHeaders()
        });

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }

        const configs = await response.json();
        renderConfigs(configs);
    } catch (error) {
        console.error('Failed to load configs:', error);
        document.getElementById('configs-container').innerHTML = `
            <div class="empty-state">
                <div>❌</div>
                <p>Failed to load configs</p>
                <p style="font-size: 12px; color: #dc3545;">${error.message}</p>
            </div>
        `;
    }
}

// Render configs
function renderConfigs(configs) {
    const container = document.getElementById('configs-container');

    if (Object.keys(configs).length === 0) {
        container.innerHTML = `
            <div class="empty-state">
                <div>📋</div>
                <p>No model configs found</p>
                <p style="font-size: 12px;">Create a new config to get started</p>
            </div>
        `;
        return;
    }

    container.innerHTML = Object.entries(configs).map(([name, config]) => `
        <div class="config-item">
            <div class="config-item-header">
                <div class="config-item-name">${escapeHtml(name)}</div>
                <div class="config-item-actions">
                    <button class="btn btn-secondary" onclick="deleteConfig('${escapeHtml(name)}')">🗑️ Delete</button>
                </div>
            </div>
            <div class="config-item-stats">
                <div class="config-stat">
                    <div class="config-stat-label">Format</div>
                    <div class="config-stat-value">${escapeHtml(config.format || 'N/A')}</div>
                </div>
                <div class="config-stat">
                    <div class="config-stat-label">Path</div>
                    <div class="config-stat-value" style="font-size: 11px; max-width: 150px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">${escapeHtml(config.model_path || 'N/A')}</div>
                </div>
                <div class="config-stat">
                    <div class="config-stat-label">Description</div>
                    <div class="config-stat-value" style="font-size: 11px; max-width: 200px; overflow: hidden; text-overflow: ellipsis;">${escapeHtml(config.description || 'N/A')}</div>
                </div>
                <div class="config-stat">
                    <div class="config-stat-label">Tags</div>
                    <div class="config-stat-value">${escapeHtml(config.tags?.join(', ') || 'N/A')}</div>
                </div>
            </div>
        </div>
    `).join('');
}

// Load cache info
async function loadCacheInfo() {
    try {
        const response = await fetch(`${API_BASE}/manage/cache`, {
            headers: getAuthHeaders()
        });

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }

        const info = await response.json();
        renderCacheInfo(info);
    } catch (error) {
        console.error('Failed to load cache info:', error);
        document.getElementById('cache-container').innerHTML = `
            <div class="empty-state">
                <div>❌</div>
                <p>Failed to load cache info</p>
                <p style="font-size: 12px; color: #dc3545;">${error.message}</p>
            </div>
        `;
    }
}

// Render cache info
function renderCacheInfo(info) {
    const container = document.getElementById('cache-container');

    container.innerHTML = `
        <div class="cache-info">
            <h3>Cache Directory: ${escapeHtml(info.path || 'N/A')}</h3>
            <div class="cache-stats">
                <div class="cache-stat-box">
                    <div class="value">${info.file_count || 0}</div>
                    <div class="label">Files</div>
                </div>
                <div class="cache-stat-box">
                    <div class="value">${formatBytes(info.total_size || 0)}</div>
                    <div class="label">Total Size</div>
                </div>
                <div class="cache-stat-box">
                    <div class="value">${info.dir_count || 0}</div>
                    <div class="label">Directories</div>
                </div>
            </div>
            <div style="margin-top: 20px;">
                <button class="btn btn-danger" onclick="clearCache()">🗑️ Clear All Cache</button>
            </div>
        </div>
    `;
}

// Load chat models
async function loadChatModels() {
    try {
        const response = await fetch(`${API_BASE}/manage/models`, {
            headers: getAuthHeaders()
        });

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }

        const models = await response.json();
        renderChatModels(models);
    } catch (error) {
        console.error('Failed to load chat models:', error);
        document.getElementById('chat-models-container').innerHTML = `
            <div class="empty-state">
                <div>❌</div>
                <p>Failed to load models</p>
                <p style="font-size: 12px; color: #dc3545;">${error.message}</p>
            </div>
        `;
    }
}

// Render chat models
function renderChatModels(models) {
    const container = document.getElementById('chat-models-container');

    if (models.length === 0) {
        container.innerHTML = `
            <div class="empty-state">
                <div>📦</div>
                <p>No models loaded</p>
                <p style="font-size: 12px;">Load a model first to start chatting</p>
            </div>
        `;
        return;
    }

    container.innerHTML = `
        <div class="form-group">
            <label>Select Model:</label>
            <select id="chat-model-select" style="width: 100%; padding: 10px; border: 2px solid #e9ecef; border-radius: 5px; font-size: 14px;">
                ${models.map(m => `<option value="${escapeHtml(m.name)}">${escapeHtml(m.name)}</option>`).join('')}
            </select>
        </div>
        <div class="form-group">
            <label>System Message (optional):</label>
            <textarea id="chat-system-message" rows="3" placeholder="You are a helpful assistant..." style="width: 100%; padding: 10px; border: 2px solid #e9ecef; border-radius: 5px; font-size: 14px;"></textarea>
        </div>
        <div class="form-group">
            <label>Message:</label>
            <textarea id="chat-message" rows="4" placeholder="Type your message here..." style="width: 100%; padding: 10px; border: 2px solid #e9ecef; border-radius: 5px; font-size: 14px;"></textarea>
        </div>
        <div class="form-group">
            <label>Temperature (0.0-2.0):</label>
            <input type="number" id="chat-temperature" value="0.7" step="0.1" min="0" max="2" style="width: 100%; padding: 10px; border: 2px solid #e9ecef; border-radius: 5px; font-size: 14px;">
        </div>
        <div class="form-group">
            <label>Max Tokens:</label>
            <input type="number" id="chat-max-tokens" value="512" step="64" min="1" style="width: 100%; padding: 10px; border: 2px solid #e9ecef; border-radius: 5px; font-size: 14px;">
        </div>
        <button class="btn btn-success" onclick="sendChatMessage()">💬 Send Message</button>
    `;
}

// Load logs
async function loadLogs() {
    try {
        const response = await fetch(`${API_BASE}/manage/logs?count=50`, {
            headers: getAuthHeaders()
        });

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }

        const logs = await response.json();
        renderLogs(logs);
    } catch (error) {
        console.error('Failed to load logs:', error);
        document.getElementById('logs-body').innerHTML = `
            <tr>
                <td colspan="8" style="text-align: center; padding: 40px; color: #dc3545;">
                    Failed to load logs: ${error.message}
                </td>
            </tr>
        `;
    }
}

// Render logs
function renderLogs(logs) {
    const tbody = document.getElementById('logs-body');

    if (logs.length === 0) {
        tbody.innerHTML = `
            <tr>
                <td colspan="8" style="text-align: center; padding: 40px;">No logs available</td>
            </tr>
        `;
        return;
    }

    tbody.innerHTML = logs.map(log => `
        <tr>
            <td>${formatTime(log.timestamp)}</td>
            <td>${escapeHtml(log.modelName)}</td>
            <td>${escapeHtml(log.endpoint)}</td>
            <td style="max-width: 200px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">${escapeHtml(log.prompt)}</td>
            <td>${log.tokensGenerated}</td>
            <td>${log.tps.toFixed(2)}</td>
            <td>${log.durationMs.toFixed(0)}ms</td>
            <td class="${log.success ? 'success' : 'error'}">${log.success ? '✓' : '✗'}</td>
        </tr>
    `).join('');
}

// Reload model
async function reloadModel(name) {
    if (!confirm(`Reload model "${name}"?`)) return;

    try {
        const response = await fetch(`${API_BASE}/manage/models/${encodeURIComponent(name)}/reload`, {
            method: 'POST',
            headers: getAuthHeaders()
        });

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }

        alert(`Model "${name}" reloaded successfully`);
        loadModels();
    } catch (error) {
        alert(`Failed to reload model: ${error.message}`);
    }
}

// Unload model
async function unloadModel(name) {
    if (!confirm(`Unload model "${name}" from memory?`)) return;

    try {
        const response = await fetch(`${API_BASE}/manage/models/${encodeURIComponent(name)}`, {
            method: 'DELETE',
            headers: getAuthHeaders()
        });

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }

        alert(`Model "${name}" unloaded successfully`);
        loadModels();
    } catch (error) {
        alert(`Failed to unload model: ${error.message}`);
    }
}

// Download model
async function downloadModel(event) {
    event.preventDefault();

    const name = document.getElementById('download-name').value;
    const url = document.getElementById('download-url').value;
    const format = document.getElementById('download-format').value;

    if (!confirm(`Download model "${name}" from ${url}?`)) return;

    const submitBtn = event.target.querySelector('.btn-submit');
    submitBtn.disabled = true;
    submitBtn.textContent = 'Downloading...';

    try {
        const response = await fetch(`${API_BASE}/manage/download`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                ...getAuthHeaders()
            },
            body: JSON.stringify({ name, url, format })
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || `HTTP ${response.status}`);
        }

        const result = await response.json();
        alert(`Success!\n${result.message}\nPath: ${result.path}`);

        // Clear form
        document.getElementById('download-name').value = '';
        document.getElementById('download-url').value = '';
    } catch (error) {
        alert(`Download failed: ${error.message}`);
    } finally {
        submitBtn.disabled = false;
        submitBtn.textContent = 'Download Model';
    }
}

// Create config
async function createConfig(event) {
    event.preventDefault();

    const name = document.getElementById('config-name').value;
    const modelPath = document.getElementById('config-model-path').value;
    const format = document.getElementById('config-format').value;
    const description = document.getElementById('config-description').value;
    const temperature = parseFloat(document.getElementById('config-temperature').value) || 0.7;
    const topK = parseInt(document.getElementById('config-top-k').value) || 40;
    const topP = parseFloat(document.getElementById('config-top-p').value) || 0.9;
    const maxTokens = parseInt(document.getElementById('config-max-tokens').value) || 512;

    if (!confirm(`Create model config "${name}"?`)) return;

    const submitBtn = event.target.querySelector('.btn-submit');
    submitBtn.disabled = true;
    submitBtn.textContent = 'Creating...';

    try {
        const config = {
            name: name,
            model_path: modelPath,
            format: format,
            description: description,
            parameters: {
                temperature: temperature,
                top_k: topK,
                top_p: topP,
                max_tokens: maxTokens
            }
        };

        const response = await fetch(`${API_BASE}/manage/configs`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                ...getAuthHeaders()
            },
            body: JSON.stringify(config)
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || `HTTP ${response.status}`);
        }

        alert(`Config "${name}" created successfully`);
        loadConfigs();
    } catch (error) {
        alert(`Failed to create config: ${error.message}`);
    } finally {
        submitBtn.disabled = false;
        submitBtn.textContent = 'Create Config';
    }
}

// Delete config
async function deleteConfig(name) {
    if (!confirm(`Delete config "${name}"?`)) return;

    try {
        const response = await fetch(`${API_BASE}/manage/configs/${encodeURIComponent(name)}`, {
            method: 'DELETE',
            headers: getAuthHeaders()
        });

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }

        alert(`Config "${name}" deleted successfully`);
        loadConfigs();
    } catch (error) {
        alert(`Failed to delete config: ${error.message}`);
    }
}

// Clear cache
async function clearCache() {
    if (!confirm('Clear all model cache? This will delete all downloaded models.')) return;

    try {
        const response = await fetch(`${API_BASE}/manage/cache`, {
            method: 'DELETE',
            headers: getAuthHeaders()
        });

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }

        const result = await response.json();
        alert(`Cache cleared: ${result.message}`);
        loadCacheInfo();
    } catch (error) {
        alert(`Failed to clear cache: ${error.message}`);
    }
}

// Send chat message
async function sendChatMessage() {
    const modelName = document.getElementById('chat-model-select').value;
    const systemMessage = document.getElementById('chat-system-message').value;
    const message = document.getElementById('chat-message').value;
    const temperature = parseFloat(document.getElementById('chat-temperature').value) || 0.7;
    const maxTokens = parseInt(document.getElementById('chat-max-tokens').value) || 512;

    if (!message.trim()) {
        alert('Please enter a message');
        return;
    }

    const submitBtn = document.querySelector('#chat-form button');
    submitBtn.disabled = true;
    submitBtn.textContent = 'Sending...';

    try {
        const response = await fetch(`${API_BASE}/manage/chat`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                ...getAuthHeaders()
            },
            body: JSON.stringify({
                model_name: modelName,
                message: message,
                system_message: systemMessage || null,
                temperature: temperature,
                max_tokens: maxTokens
            })
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || `HTTP ${response.status}`);
        }

        const result = await response.json();
        alert(`Response:\n\n${result.response}`);
    } catch (error) {
        alert(`Chat failed: ${error.message}`);
    } finally {
        submitBtn.disabled = false;
        submitBtn.textContent = '💬 Send Message';
    }
}

// Get auth headers
function getAuthHeaders() {
    const token = document.getElementById('manage-token')?.value;
    if (token) {
        return { 'Authorization': `Bearer ${token}` };
    }
    return {};
}

// Refresh data
function refreshData() {
    if (currentSection === 'models') {
        loadModels();
    } else if (currentSection === 'logs') {
        loadLogs();
    } else if (currentSection === 'configs') {
        loadConfigs();
    } else if (currentSection === 'cache') {
        loadCacheInfo();
    } else if (currentSection === 'chat') {
        loadChatModels();
    }
}

// Format date
function formatDate(dateStr) {
    const date = new Date(dateStr);
    return date.toLocaleString();
}

// Format time
function formatTime(dateStr) {
    const date = new Date(dateStr);
    return date.toLocaleTimeString();
}

// Format bytes
function formatBytes(bytes) {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

// Escape HTML
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Auto-refresh every 5 seconds if on models or logs page
setInterval(() => {
    if (currentSection === 'models' || currentSection === 'logs') {
        refreshData();
    }
}, 5000);

// Initial load
loadModels();