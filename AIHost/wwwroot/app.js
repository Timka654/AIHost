// API base URL
const API_BASE = window.location.origin;

// Current section
let currentSection = 'dashboard';

// Monaco Editor instance
let monacoEditor = null;
let currentConfigSchema = null;

// Show section
function showSection(section) {
    // Hide all sections
    document.querySelectorAll('.section').forEach(s => s.classList.remove('active'));
    document.querySelectorAll('.nav button').forEach(b => b.classList.remove('active'));

    // Show selected section
    document.getElementById(`section-${section}`).classList.add('active');
    document.getElementById(`btn-${section}`).classList.add('active');

    currentSection = section;

    // Update URL hash
    window.location.hash = section;

    // Load data for section
    if (section === 'dashboard') {
        loadDashboard();
    } else if (section === 'models') {
        loadModels();
    } else if (section === 'resources') {
        loadResources();
    } else if (section === 'performance') {
        loadPerformance();
    } else if (section === 'logs') {
        loadLogs();
    } else if (section === 'configs') {
        loadConfigs();
    } else if (section === 'server') {
        loadServerConfig();
    } else if (section === 'tokens') {
        loadTokens();
    } else if (section === 'cache') {
        loadCacheInfo();
    } else if (section === 'downloads') {
        loadDownloads();
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
                    <button class="btn btn-secondary" onclick="initializeModel('${escapeHtml(name)}')">⬇️ Initialize</button>
                    <button class="btn btn-secondary" onclick="editConfig('${escapeHtml(name)}')">✏️ Edit</button>
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
                    <div class="config-stat-value" style="font-size: 11px; max-width: 150px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">${escapeHtml(config.model || config.model_path || 'N/A')}</div>
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
        // Load available configs instead of loaded models
        const response = await fetch(`${API_BASE}/manage/configs`, {
            headers: getAuthHeaders()
        });

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }

        const configs = await response.json();
        renderChatModels(configs);
    } catch (error) {
        console.error('Failed to load chat models:', error);
        document.getElementById('chat-models-container').innerHTML = `
            <div class="empty-state">
                <div>❌</div>
                <p>Failed to load model configs</p>
                <p style="font-size: 12px; color: #dc3545;">${error.message}</p>
            </div>
        `;
    }
}

// Render chat models
function renderChatModels(configs) {
    const container = document.getElementById('chat-models-container');

    if (Object.keys(configs).length === 0) {
        container.innerHTML = `
            <div class="empty-state">
                <div>📋</div>
                <p>No model configs found</p>
                <p style="font-size: 12px;">Create a model config in the Configs section first</p>
                <button class="btn btn-secondary" onclick="showSection('configs')" style="margin-top: 10px;">Go to Configs</button>
            </div>
        `;
        return;
    }

    const configNames = Object.keys(configs);

    container.innerHTML = `
        <div class="form-group">
            <label>Select Model:</label>
            <select id="chat-model-select" style="width: 100%; padding: 10px; border: 2px solid #e9ecef; border-radius: 5px; font-size: 14px;">
                ${configNames.map(name => `<option value="${escapeHtml(name)}">${escapeHtml(name)}</option>`).join('')}
            </select>
            <small style="color: #6c757d;">Model will be auto-loaded on first message if not already loaded</small>
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

// Edit config
async function editConfig(name) {
    try {
        // Fetch current config
        const response = await fetch(`${API_BASE}/manage/configs/${encodeURIComponent(name)}`, {
            headers: getAuthHeaders()
        });

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }

        const config = await response.json();
        
        // Show form with pre-filled data
        showConfigForm(config, true);
    } catch (error) {
        alert(`Failed to load config: ${error.message}`);
    }
}

// Show config form with Monaco Editor
async function showConfigForm(config = null, isEdit = false) {
    const container = document.getElementById('configs-container');
    const title = isEdit ? 'Edit Model Config' : 'Create Model Config';
    const submitText = isEdit ? 'Update Config' : 'Create Config';
    const originalName = config?.name || '';
    
    // Load schema if not already loaded
    if (!currentConfigSchema) {
        try {
            const response = await fetch(`${API_BASE}/manage/configs/schema`, {
                headers: getAuthHeaders()
            });
            currentConfigSchema = await response.json();
        } catch (error) {
            console.error('Failed to load schema:', error);
        }
    }

    // Prepare initial config
    const initialConfig = config || {
        name: '',
        model: '',
        format: 'gguf',
        description: '',
        parameters: {
            temperature: 0.7,
            top_k: 40,
            top_p: 0.9,
            max_tokens: 512
        }
    };

    // Remove $schema from display
    const { $schema, ...displayConfig } = initialConfig;

    container.innerHTML = `
        <div class="config-editor-wrapper">
            <h3>${title}</h3>
            <div class="editor-toolbar">
                <div style="flex: 1;">
                    <label style="display: block; margin-bottom: 5px; font-weight: bold;">Config Name: *</label>
                    <input type="text" id="config-name-input" placeholder="my-model" value="${escapeHtml(originalName)}" ${isEdit ? 'readonly' : ''} required>
                </div>
                <div style="margin-top: auto;">
                    <button class="btn btn-success" onclick="saveConfigFromEditor(${isEdit}, '${escapeHtml(originalName)}')">${submitText}</button>
                    <button class="btn btn-secondary" onclick="loadConfigs()">Cancel</button>
                </div>
            </div>
            <div id="monaco-editor-container" class="monaco-editor-container" style="height: 500px;"></div>
        </div>
    `;

    // Initialize Monaco Editor
    setTimeout(() => {
        require(['vs/editor/editor.main'], function() {
            // Dispose previous editor if exists
            if (monacoEditor) {
                monacoEditor.dispose();
            }

            // Register JSON schema
            if (currentConfigSchema) {
                monaco.languages.json.jsonDefaults.setDiagnosticsOptions({
                    validate: true,
                    schemas: [{
                        uri: 'http://aihost/model-config-schema.json',
                        fileMatch: ['*'],
                        schema: currentConfigSchema
                    }]
                });
            }

            // Create editor
            monacoEditor = monaco.editor.create(document.getElementById('monaco-editor-container'), {
                value: JSON.stringify(displayConfig, null, 2),
                language: 'json',
                theme: 'vs',
                automaticLayout: true,
                minimap: { enabled: false },
                scrollBeyondLastLine: false,
                formatOnPaste: true,
                formatOnType: true
            });
        });
    }, 100);
}

// Save config from Monaco Editor
async function saveConfigFromEditor(isEdit, originalName) {
    const configName = document.getElementById('config-name-input').value.trim();
    
    if (!configName) {
        alert('Please enter a config name');
        return;
    }

    if (!monacoEditor) {
        alert('Editor not initialized');
        return;
    }

    // Get editor content
    let configData;
    try {
        configData = JSON.parse(monacoEditor.getValue());
    } catch (error) {
        alert(`Invalid JSON: ${error.message}`);
        return;
    }

    // Set name from input
    configData.name = configName;

    // Confirm action
    if (!confirm(`${isEdit ? 'Update' : 'Create'} config "${configName}"?`)) {
        return;
    }

    try {
        const url = isEdit 
            ? `${API_BASE}/manage/configs/${encodeURIComponent(originalName)}`
            : `${API_BASE}/manage/configs`;
        
        const response = await fetch(url, {
            method: isEdit ? 'PUT' : 'POST',
            headers: {
                'Content-Type': 'application/json',
                ...getAuthHeaders()
            },
            body: JSON.stringify(configData)
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || `HTTP ${response.status}`);
        }

        alert(`Config "${configName}" ${isEdit ? 'updated' : 'created'} successfully`);
        loadConfigs();
    } catch (error) {
        alert(`Failed to ${isEdit ? 'update' : 'create'} config: ${error.message}`);
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
    const modelName = document.getElementById('chat-model-select')?.value;
    const systemMessage = document.getElementById('chat-system-message')?.value;
    const message = document.getElementById('chat-message')?.value;
    const temperature = parseFloat(document.getElementById('chat-temperature')?.value) || 0.7;
    const maxTokens = parseInt(document.getElementById('chat-max-tokens')?.value) || 512;

    if (!modelName) {
        alert('Please select a model config first.');
        return;
    }

    if (!message || !message.trim()) {
        alert('Please enter a message');
        return;
    }

    // Find the send button
    const submitBtn = document.querySelector('.btn-success[onclick="sendChatMessage()"]');
    if (submitBtn) {
        submitBtn.disabled = true;
        submitBtn.textContent = '⏳ Loading model & sending...';
    }

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
        
        // Display result in a nicer format
        displayChatResponse(message, result);
        
        // Clear message input
        const messageInput = document.getElementById('chat-message');
        if (messageInput) messageInput.value = '';
    } catch (error) {
        alert(`Chat failed: ${error.message}`);
    } finally {
        if (submitBtn) {
            submitBtn.disabled = false;
            submitBtn.textContent = '💬 Send Message';
        }
    }
}

// Display chat response
function displayChatResponse(userMessage, result) {
    const container = document.getElementById('chat-models-container');
    const existingMessages = document.getElementById('chat-messages');
    
    // Create messages container if it doesn't exist
    if (!existingMessages) {
        const messagesHtml = `
            <div id="chat-messages" style="margin-top: 20px; padding: 15px; background: white; border: 2px solid #e9ecef; border-radius: 5px; max-height: 400px; overflow-y: auto;">
                <h4>Conversation</h4>
            </div>
        `;
        container.insertAdjacentHTML('beforeend', messagesHtml);
    }
    
    const messages = document.getElementById('chat-messages');
    const messageHtml = `
        <div class="chat-message user" style="margin: 10px 0; padding: 10px; background: #f0f0f0; border-left: 3px solid #6c757d; border-radius: 5px;">
            <div style="font-weight: bold; font-size: 12px; margin-bottom: 5px; color: #6c757d;">USER</div>
            <div style="font-size: 14px; white-space: pre-wrap; word-wrap: break-word;">${escapeHtml(userMessage)}</div>
        </div>
        <div class="chat-message assistant" style="margin: 10px 0; padding: 10px; background: #e7f3ff; border-left: 3px solid #28a745; border-radius: 5px;">
            <div style="font-weight: bold; font-size: 12px; margin-bottom: 5px; color: #28a745;">ASSISTANT (${escapeHtml(result.model)})</div>
            <div style="font-size: 14px; white-space: pre-wrap; word-wrap: break-word;">${escapeHtml(result.response)}</div>
            <div style="font-size: 11px; color: #6c757d; margin-top: 5px;">Tokens: ${result.tokens || 'N/A'}</div>
        </div>
    `;
    messages.insertAdjacentHTML('beforeend', messageHtml);
    messages.scrollTop = messages.scrollHeight;
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
    if (currentSection === 'dashboard') {
        loadDashboard();
    } else if (currentSection === 'models') {
        loadModels();
    } else if (currentSection === 'resources') {
        loadResources();
    } else if (currentSection === 'performance') {
        loadPerformance();
    } else if (currentSection === 'logs') {
        loadLogs();
    } else if (currentSection === 'configs') {
        loadConfigs();
    } else if (currentSection === 'server') {
        loadServerConfig();
    } else if (currentSection === 'tokens') {
        loadTokens();
    } else if (currentSection === 'cache') {
        loadCacheInfo();
    } else if (currentSection === 'downloads') {
        loadDownloads();
    } else if (currentSection === 'chat') {
        loadChatModels();
    }
}

// Load dashboard
async function loadDashboard() {
    try {
        const response = await fetch(`${API_BASE}/manage/status`, {
            headers: getAuthHeaders()
        });

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }

        const status = await response.json();
        renderDashboard(status);
    } catch (error) {
        console.error('Failed to load dashboard:', error);
    }
}

// Render dashboard
function renderDashboard(status) {
    document.getElementById('dash-models').textContent = status.loaded_models || 0;
    document.getElementById('dash-requests').textContent = status.total_requests || 0;
    document.getElementById('dash-memory').textContent = (status.memory_mb || 0).toFixed(1) + ' MB';
    document.getElementById('dash-uptime').textContent = formatDuration(status.uptime_seconds || 0);
}

// Load resources
async function loadResources() {
    try {
        const [resourcesResp, computeResp] = await Promise.all([
            fetch(`${API_BASE}/manage/system/resources`, { headers: getAuthHeaders() }),
            fetch(`${API_BASE}/manage/system/compute`, { headers: getAuthHeaders() })
        ]);

        if (!resourcesResp.ok || !computeResp.ok) {
            throw new Error('Failed to load resources');
        }

        const resources = await resourcesResp.json();
        const compute = await computeResp.json();

        renderResources(resources, compute);
    } catch (error) {
        console.error('Failed to load resources:', error);
        document.getElementById('resources-container').innerHTML = `
            <div class="empty-state">
                <div>❌</div>
                <p>Failed to load resource information</p>
                <p style="font-size: 12px; color: #dc3545;">${error.message}</p>
            </div>
        `;
    }
}

// Render resources
function renderResources(resources, compute) {
    const container = document.getElementById('resources-container');
    
    const memUsagePercent = resources.total_memory_bytes > 0
        ? (resources.used_memory_bytes / resources.total_memory_bytes * 100).toFixed(1)
        : 0;

    container.innerHTML = `
        <div class="cache-stats">
            <div class="cache-stat-box">
                <div class="value">${formatBytes(resources.total_memory_bytes)}</div>
                <div class="label">Total Memory</div>
            </div>
            <div class="cache-stat-box">
                <div class="value">${formatBytes(resources.used_memory_bytes)}</div>
                <div class="label">Used Memory (${memUsagePercent}%)</div>
            </div>
            <div class="cache-stat-box">
                <div class="value">${formatBytes(resources.available_memory_bytes)}</div>
                <div class="label">Available Memory</div>
            </div>
            <div class="cache-stat-box">
                <div class="value">${formatBytes(resources.process_memory_bytes)}</div>
                <div class="label">Process Memory</div>
            </div>
            <div class="cache-stat-box">
                <div class="value">${formatBytes(resources.gc_total_memory_bytes)}</div>
                <div class="label">GC Memory</div>
            </div>
            <div class="cache-stat-box">
                <div class="value">${resources.thread_count}</div>
                <div class="label">Thread Count</div>
            </div>
        </div>
        
        <div class="cache-info" style="margin-top: 20px;">
            <h3>Compute Devices</h3>
            ${compute.devices && compute.devices.length > 0 ? `
                <div class="config-list">
                    ${compute.devices.map(d => `
                        <div class="config-item">
                            <div class="config-item-header">
                                <div class="config-item-name">${escapeHtml(d.device_name)}</div>
                            </div>
                            <div class="config-item-stats">
                                <div class="config-stat">
                                    <div class="config-stat-label">Model</div>
                                    <div class="config-stat-value">${escapeHtml(d.model)}</div>
                                </div>
                                <div class="config-stat">
                                    <div class="config-stat-label">Device Type</div>
                                    <div class="config-stat-value">${escapeHtml(d.device_type)}</div>
                                </div>
                            </div>
                        </div>
                    `).join('')}
                </div>
            ` : '<p>No compute devices detected</p>'}
        </div>
    `;
}

// Load performance
async function loadPerformance() {
    try {
        const response = await fetch(`${API_BASE}/manage/performance`, {
            headers: getAuthHeaders()
        });

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }

        const perf = await response.json();
        renderPerformance(perf);
    } catch (error) {
        console.error('Failed to load performance:', error);
        document.getElementById('performance-container').innerHTML = `
            <div class="empty-state">
                <div>❌</div>
                <p>Failed to load performance data</p>
                <p style="font-size: 12px; color: #dc3545;">${error.message}</p>
            </div>
        `;
    }
}

// Load downloads
async function loadDownloads() {
    try {
        const response = await fetch(`${API_BASE}/manage/downloads`, {
            headers: getAuthHeaders()
        });

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }

        const downloads = await response.json();
        renderDownloads(downloads);
    } catch (error) {
        console.error('Failed to load downloads:', error);
        document.getElementById('downloads-container').innerHTML = `
            <div class="empty-state">
                <div>❌</div>
                <p>Failed to load downloads</p>
                <p style="font-size: 12px; color: #dc3545;">${error.message}</p>
            </div>
        `;
    }
}

// Render downloads
function renderDownloads(downloads) {
    const container = document.getElementById('downloads-container');

    if (downloads.length === 0) {
        container.innerHTML = `
            <div class="empty-state">
                <div>📦</div>
                <p>No active downloads</p>
                <p style="font-size: 12px;">Click "+ Start Download" to download a model</p>
            </div>
        `;
        return;
    }

    container.innerHTML = downloads.map(d => `
        <div class="config-item">
            <div class="config-item-header">
                <div class="config-item-name">${escapeHtml(d.filename)}</div>
                <div class="config-item-actions">
                    ${d.status === 'downloading' || d.status === 'pending' 
                        ? `<button class="btn btn-secondary" onclick="cancelDownload('${escapeHtml(d.id)}')">🛑 Cancel</button>`
                        : ''}
                </div>
            </div>
            <div class="config-item-stats">
                <div class="config-stat">
                    <div class="config-stat-label">Status</div>
                    <div class="config-stat-value">${getStatusBadge(d.status)}</div>
                </div>
                <div class="config-stat">
                    <div class="config-stat-label">Progress</div>
                    <div class="config-stat-value">${d.progress.toFixed(2)}%</div>
                </div>
                <div class="config-stat">
                    <div class="config-stat-label">Downloaded</div>
                    <div class="config-stat-value">${formatBytes(d.downloaded_bytes)} / ${formatBytes(d.total_bytes)}</div>
                </div>
                <div class="config-stat">
                    <div class="config-stat-label">URL</div>
                    <div class="config-stat-value" style="font-size: 11px; max-width: 300px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">${escapeHtml(d.url)}</div>
                </div>
            </div>
            ${d.status === 'downloading' || d.status === 'pending'
                ? `<div style="margin-top: 10px;"><div style="background: #e9ecef; height: 8px; border-radius: 4px; overflow: hidden;">
                    <div style="background: linear-gradient(90deg, #667eea, #764ba2); height: 100%; width: ${d.progress}%; transition: width 0.3s;"></div>
                </div></div>`
                : ''}
            ${d.error ? `<div style="margin-top: 10px; padding: 10px; background: #f8d7da; border-radius: 5px; color: #721c24; font-size: 12px;">${escapeHtml(d.error)}</div>` : ''}
        </div>
    `).join('');
}

// Show download form
function showDownloadForm() {
    const container = document.getElementById('downloads-container');
    container.innerHTML = `
        <div class="config-editor-wrapper">
            <h3>Start New Download</h3>
            <form onsubmit="startDownload(event)">
                <div class="form-group">
                    <label>Model URL: *</label>
                    <input type="text" id="download-url" placeholder="https://huggingface.co/..." required>
                    <small style="color: #6c757d;">Enter the direct URL to the model file (GGUF)</small>
                </div>
                <div class="form-group">
                    <label>Filename (optional):</label>
                    <input type="text" id="download-filename" placeholder="model.gguf">
                    <small style="color: #6c757d;">Leave empty to use filename from URL</small>
                </div>
                <div style="display: flex; gap: 10px; margin-top: 20px;">
                    <button type="submit" class="btn btn-success">🚀 Start Download</button>
                    <button type="button" class="btn btn-secondary" onclick="loadDownloads()">Cancel</button>
                </div>
            </form>
        </div>
    `;
}

// Start download
async function startDownload(event) {
    event.preventDefault();

    const url = document.getElementById('download-url').value;
    const filename = document.getElementById('download-filename').value || null;

    try {
        const response = await fetch(`${API_BASE}/manage/downloads`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                ...getAuthHeaders()
            },
            body: JSON.stringify({ url, filename })
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || `HTTP ${response.status}`);
        }

        const result = await response.json();
        alert(`Download started! ID: ${result.download_id}`);
        loadDownloads();
    } catch (error) {
        alert(`Failed to start download: ${error.message}`);
    }
}

// Cancel download
async function cancelDownload(downloadId) {
    if (!confirm('Cancel this download?')) return;

    try {
        const response = await fetch(`${API_BASE}/manage/downloads/${encodeURIComponent(downloadId)}`, {
            method: 'DELETE',
            headers: getAuthHeaders()
        });

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }

        alert('Download cancelled');
        loadDownloads();
    } catch (error) {
        alert(`Failed to cancel download: ${error.message}`);
    }
}

// Helper: Get status badge
function getStatusBadge(status) {
    const badges = {
        'pending': '🕐 Pending',
        'downloading': '⬇️ Downloading',
        'completed': '✅ Completed',
        'failed': '❌ Failed',
        'cancelled': '🛑 Cancelled'
    };
    return badges[status] || status;
}

// Helper: Format bytes
function formatBytes(bytes) {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

// Load server configuration
async function loadTokens() {
    try {
        const response = await fetch(`${API_BASE}/manage/tokens`, {
            headers: getAuthHeaders()
        });

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }

        const data = await response.json();
        renderTokens(data.tokens);
    } catch (error) {
        console.error('Failed to load tokens:', error);
        document.getElementById('tokens-container').innerHTML = `
            <div class="empty-state">
                <div>❌</div>
                <p>Failed to load tokens</p>
                <p style="font-size: 12px; color: #dc3545;">${error.message}</p>
            </div>
        `;
    }
}

// Render tokens
function renderTokens(tokens) {
    const container = document.getElementById('tokens-container');

    if (tokens.length === 0) {
        container.innerHTML = `
            <div class="empty-state">
                <div>🔑</div>
                <p>No tokens configured</p>
                <p style="font-size: 12px;">Click "+ Add Token" to create authentication tokens</p>
            </div>
        `;
        return;
    }

    const accessLevelBadge = (level) => {
        const badges = {
            'all': '<span style="background: #28a745; color: white; padding: 3px 8px; border-radius: 3px; font-size: 11px; font-weight: bold;">ALL ACCESS</span>',
            'manage': '<span style="background: #ffc107; color: #333; padding: 3px 8px; border-radius: 3px; font-size: 11px; font-weight: bold;">MANAGE</span>',
            'user': '<span style="background: #6c757d; color: white; padding: 3px 8px; border-radius: 3px; font-size: 11px; font-weight: bold;">USER</span>'
        };
        return badges[level] || badges['user'];
    };

    container.innerHTML = tokens.map(t => `
        <div class="config-item">
            <div class="config-item-header">
                <div class="config-item-name" style="font-family: monospace;">${escapeHtml(t.token)}</div>
                <div class="config-item-actions">
                    <button class="btn btn-secondary" onclick="changeTokenLevel('${escapeHtml(t.token)}', '${t.access_level}')">🔄 Change Level</button>
                    <button class="btn btn-secondary" onclick="deleteToken('${escapeHtml(t.token)}')">🗑️ Delete</button>
                </div>
            </div>
            <div class="config-item-stats">
                <div class="config-stat">
                    <div class="config-stat-label">Access Level</div>
                    <div class="config-stat-value">${accessLevelBadge(t.access_level)}</div>
                </div>
            </div>
        </div>
    `).join('');
}

// Show add token form
function showAddTokenForm() {
    const container = document.getElementById('tokens-container');
    container.innerHTML = `
        <div class="config-editor-wrapper">
            <h3>Add New Token</h3>
            <form onsubmit="addToken(event)">
                <div class="form-group">
                    <label>Token: *</label>
                    <input type="text" id="token-value" placeholder="my-secret-token-12345" required>
                    <small style="color: #6c757d;">Enter a secure random token (recommended 32+ characters)</small>
                </div>
                <div class="form-group">
                    <label>Access Level: *</label>
                    <select id="token-access-level" required>
                        <option value="U">User - API access only</option>
                        <option value="M">Manage - Can use /manage endpoints</option>
                        <option value="A">All - Full access to everything</option>
                    </select>
                </div>
                <div style="display: flex; gap: 10px; margin-top: 20px;">
                    <button type="submit" class="btn btn-success">✅ Add Token</button>
                    <button type="button" class="btn btn-secondary" onclick="loadTokens()">Cancel</button>
                    <button type="button" class="btn btn-secondary" onclick="generateRandomToken()">🎲 Generate Random</button>
                </div>
            </form>
        </div>
    `;
}

// Generate random token
function generateRandomToken() {
    const chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
    let token = '';
    for (let i = 0; i < 32; i++) {
        token += chars.charAt(Math.floor(Math.random() * chars.length));
    }
    document.getElementById('token-value').value = token;
}

// Add token
async function addToken(event) {
    event.preventDefault();

    const token = document.getElementById('token-value').value.trim();
    const accessLevel = document.getElementById('token-access-level').value;

    if (!token) {
        alert('Please enter a token');
        return;
    }

    try {
        const response = await fetch(`${API_BASE}/manage/tokens`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                ...getAuthHeaders()
            },
            body: JSON.stringify({ token, access_level: accessLevel })
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || `HTTP ${response.status}`);
        }

        alert('Token added successfully');
        loadTokens();
    } catch (error) {
        alert(`Failed to add token: ${error.message}`);
    }
}

// Change token level
async function changeTokenLevel(token, currentLevel) {
    const levels = ['user', 'manage', 'all'];
    const levelCodes = { 'user': 'U', 'manage': 'M', 'all': 'A' };
    const levelNames = { 'user': 'User', 'manage': 'Manage', 'all': 'All' };
    
    const currentIndex = levels.indexOf(currentLevel.toLowerCase());
    const nextIndex = (currentIndex + 1) % levels.length;
    const nextLevel = levels[nextIndex];
    const nextCode = levelCodes[nextLevel];

    if (!confirm(`Change token level to ${levelNames[nextLevel]}?`)) return;

    try {
        const response = await fetch(`${API_BASE}/manage/tokens/${encodeURIComponent(token)}`, {
            method: 'PUT',
            headers: {
                'Content-Type': 'application/json',
                ...getAuthHeaders()
            },
            body: JSON.stringify({ access_level: nextCode })
        });

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }

        loadTokens();
    } catch (error) {
        alert(`Failed to update token: ${error.message}`);
    }
}

// Delete token
async function deleteToken(token) {
    if (!confirm(`Delete token "${token}"?`)) return;

    try {
        const response = await fetch(`${API_BASE}/manage/tokens/${encodeURIComponent(token)}`, {
            method: 'DELETE',
            headers: getAuthHeaders()
        });

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }

        alert('Token deleted successfully');
        loadTokens();
    } catch (error) {
        alert(`Failed to delete token: ${error.message}`);
    }
}

// Load server configuration
async function loadServerConfig() {
    const container = document.getElementById('server-container');
    
    try {
        const response = await fetch(`${API_BASE}/manage/server-config`, {
            headers: getAuthHeaders()
        });

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }

        const config = await response.json();
        
        container.innerHTML = `
            <div class="config-editor-wrapper">
                <div class="editor-toolbar">
                    <h3>Server Configuration</h3>
                    <div style="display: flex; gap: 10px;">
                        <button class="btn btn-success" onclick="saveServerConfig()">💾 Save</button>
                        <button class="btn btn-secondary" onclick="loadServerConfig()">🔄 Reload</button>
                    </div>
                </div>
                <div style="background: #fff3cd; border: 1px solid #ffc107; border-radius: 5px; padding: 10px; margin-bottom: 15px; font-size: 12px;">
                    ⚠️ Changes require server restart to take effect
                </div>
                <div id="server-monaco-container" class="monaco-editor-container" style="height: 500px;"></div>
            </div>
            <div style="background: #f8f9fa; border-radius: 5px; padding: 15px;">
                <h4>Configuration Options:</h4>
                <ul style="line-height: 1.8;">
                    <li><strong>models_directory</strong>: Path to models directory</li>
                    <li><strong>cache_directory</strong>: Path to cache directory</li>
                    <li><strong>host</strong>: Server host address</li>
                    <li><strong>port</strong>: Server port (default: 11434)</li>
                    <li><strong>compute_provider</strong>: vulkan, cuda, or rocm</li>
                    <li><strong>auto_unload_minutes</strong>: Auto-unload idle models</li>
                    <li><strong>manage_token</strong>: Admin API authentication token</li>
                </ul>
            </div>
        </div>
    `;

        // Initialize Monaco Editor
        setTimeout(() => {
            require(['vs/editor/editor.main'], function() {
                // Dispose previous editor if exists
                if (window.serverMonacoEditor) {
                    window.serverMonacoEditor.dispose();
                }

                // Create editor
                window.serverMonacoEditor = monaco.editor.create(document.getElementById('server-monaco-container'), {
                    value: JSON.stringify(config, null, 2),
                    language: 'json',
                    theme: 'vs',
                    automaticLayout: true,
                    minimap: { enabled: false },
                    scrollBeyondLastLine: false,
                    formatOnPaste: true,
                    formatOnType: true
                });
            });
        }, 100);
    } catch (error) {
        console.error('Failed to load server config:', error);
        container.innerHTML = `
            <div class="empty-state">
                <div>❌</div>
                <p>Failed to load server configuration</p>
                <p style="font-size: 12px; color: #dc3545;">${error.message}</p>
            </div>
        `;
    }
}

// Save server configuration
async function saveServerConfig() {
    if (!window.serverMonacoEditor) {
        alert('Editor not initialized');
        return;
    }

    try {
        const configJson = window.serverMonacoEditor.getValue();
        const config = JSON.parse(configJson);

        const response = await fetch(`${API_BASE}/manage/server-config`, {
            method: 'PUT',
            headers: {
                'Content-Type': 'application/json',
                ...getAuthHeaders()
            },
            body: configJson
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || `HTTP ${response.status}`);
        }

        const result = await response.json();
        alert(result.message || 'Server config saved successfully. Restart server to apply changes.');
    } catch (error) {
        if (error instanceof SyntaxError) {
            alert(`Invalid JSON: ${error.message}`);
        } else {
            alert(`Failed to save server config: ${error.message}`);
        }
    }
}

// Initialize model (download if needed)
async function initializeModel(name) {
    if (!confirm(`Initialize model from config "${name}"?\n\nThis will download the model if it's not already cached.`)) return;

    try {
        const response = await fetch(`${API_BASE}/manage/configs/${encodeURIComponent(name)}/initialize`, {
            method: 'POST',
            headers: getAuthHeaders()
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || `HTTP ${response.status}`);
        }

        const result = await response.json();
        
        if (result.download_id) {
            alert(`Download started!\n\nFilename: ${result.filename}\nDownload ID: ${result.download_id}\n\nCheck the Downloads section to monitor progress.`);
            // Switch to downloads section
            showSection('downloads');
        } else {
            alert(result.message || 'Model initialized successfully');
        }
    } catch (error) {
        alert(`Failed to initialize model: ${error.message}`);
    }
}

// Render performance
function renderPerformance(perf) {
    const container = document.getElementById('performance-container');

    if (!perf.models || perf.models.length === 0) {
        container.innerHTML = `
            <div class="empty-state">
                <div>📊</div>
                <p>No models loaded</p>
                <p style="font-size: 12px;">Performance data will appear when models are loaded</p>
            </div>
        `;
        return;
    }

    container.innerHTML = `
        <div class="cache-info">
            <h3>Summary</h3>
            <div class="cache-stats">
                <div class="cache-stat-box">
                    <div class="value">${perf.summary.total_models}</div>
                    <div class="label">Total Models</div>
                </div>
                <div class="cache-stat-box">
                    <div class="value">${perf.summary.total_requests}</div>
                    <div class="label">Total Requests</div>
                </div>
                <div class="cache-stat-box">
                    <div class="value">${perf.summary.average_tps.toFixed(2)}</div>
                    <div class="label">Avg TPS</div>
                </div>
            </div>
        </div>

        <div class="config-list" style="margin-top: 20px;">
            ${perf.models.map(m => `
                <div class="config-item">
                    <div class="config-item-header">
                        <div class="config-item-name">${escapeHtml(m.model)}</div>
                    </div>
                    <div class="config-item-stats">
                        <div class="config-stat">
                            <div class="config-stat-label">Total Requests</div>
                            <div class="config-stat-value">${m.total_requests}</div>
                        </div>
                        <div class="config-stat">
                            <div class="config-stat-label">Average TPS</div>
                            <div class="config-stat-value">${m.average_tps.toFixed(2)}</div>
                        </div>
                        <div class="config-stat">
                            <div class="config-stat-label">Uptime</div>
                            <div class="config-stat-value">${formatDuration(m.uptime_seconds)}</div>
                        </div>
                        ${m.last_request_seconds_ago !== null ? `
                        <div class="config-stat">
                            <div class="config-stat-label">Last Request</div>
                            <div class="config-stat-value">${formatDuration(m.last_request_seconds_ago)} ago</div>
                        </div>
                        ` : ''}
                    </div>
                </div>
            `).join('')}
        </div>
    `;
}

// Format duration
function formatDuration(seconds) {
    if (seconds < 60) return `${Math.floor(seconds)}s`;
    if (seconds < 3600) return `${Math.floor(seconds / 60)}m`;
    if (seconds < 86400) return `${Math.floor(seconds / 3600)}h`;
    return `${Math.floor(seconds / 86400)}d`;
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

// Auto-refresh every 5 seconds if on active pages
setInterval(() => {
    if (['dashboard', 'models', 'resources', 'performance', 'logs'].includes(currentSection)) {
        refreshData();
    }
}, 5000);

// Initial load
loadDashboard();