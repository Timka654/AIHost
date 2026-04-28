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
