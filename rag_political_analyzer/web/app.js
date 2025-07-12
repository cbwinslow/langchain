document.addEventListener('DOMContentLoaded', () => {
    const API_BASE_URL = 'http://localhost:8000';

    // --- Utility Functions ---
    function showStatus(message, isError = false) {
        const statusElement = document.getElementById('status-message');
        if (!statusElement) return;
        statusElement.textContent = message;
        statusElement.className = 'status-message'; // Reset classes
        statusElement.classList.add(isError ? 'status-error' : 'status-success');
        statusElement.style.display = 'block';
        setTimeout(() => {
            statusElement.style.display = 'none';
        }, 5000);
    }

    // --- Chat Page Logic (index.html) ---
    if (document.getElementById('chat-container')) {
        const chatBox = document.getElementById('chat-box');
        const chatInput = document.getElementById('chat-input');
        const sendBtn = document.getElementById('send-btn');
        const contextDisplay = document.getElementById('context-content');
        const toggleContextBtn = document.getElementById('toggle-context-btn');

        const addMessage = (message, sender) => {
            const messageElement = document.createElement('div');
            messageElement.classList.add('chat-message', `${sender}-message`);
            // Basic markdown for code blocks
            message = message.replace(/```([\s\S]*?)```/g, '<pre><code>$1</code></pre>');
            messageElement.innerHTML = `<p>${message.replace(/\n/g, '<br>')}</p>`;
            chatBox.appendChild(messageElement);
            chatBox.scrollTop = chatBox.scrollHeight;
        };

        const handleSendQuery = async () => {
            const query = chatInput.value.trim();
            if (!query) return;

            addMessage(query, 'user');
            chatInput.value = '';
            sendBtn.disabled = true;
            sendBtn.innerHTML = '<div class="loader"></div>';

            try {
                const formData = new FormData();
                formData.append('query', query);
                // Can add more form fields here for k_text, k_code etc. if we add inputs for them

                const response = await fetch(`${API_BASE_URL}/query/`, {
                    method: 'POST',
                    body: formData
                });

                if (!response.ok) {
                    const errorData = await response.json();
                    throw new Error(errorData.detail || 'Failed to get a response.');
                }

                const data = await response.json();
                addMessage(data.answer, 'bot');

                // Update context display
                if (data.retrieved_context && data.retrieved_context.length > 0) {
                    contextDisplay.innerHTML = data.retrieved_context.map(ctx => `
                        <div class="context-item">
                            <h4>Source: ${ctx.source_url || 'N/A'} (${ctx.chunk_type})</h4>
                            <p>${ctx.content.replace(/\n/g, '<br>')}</p>
                        </div>
                    `).join('');
                } else {
                    contextDisplay.innerHTML = '<p>No context was retrieved for this query.</p>';
                }

            } catch (error) {
                addMessage(`Error: ${error.message}`, 'bot');
            } finally {
                sendBtn.disabled = false;
                sendBtn.textContent = 'Send';
            }
        };

        sendBtn.addEventListener('click', handleSendQuery);
        chatInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                handleSendQuery();
            }
        });

        toggleContextBtn.addEventListener('click', () => {
            const contextContainer = document.getElementById('context-display');
            const isVisible = contextContainer.style.display !== 'none';
            contextContainer.style.display = isVisible ? 'none' : 'block';
            toggleContextBtn.textContent = isVisible ? 'Show Retrieved Context' : 'Hide Retrieved Context';
        });
    }

    // --- Deep Research Page Logic (research.html) ---
    if (document.getElementById('crawl-form')) {
        const crawlForm = document.getElementById('crawl-form');
        const statusCheckForm = document.getElementById('status-check-form');

        crawlForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const startUrl = document.getElementById('start-url').value;
            const crawlDepth = document.getElementById('crawl-depth').value;
            const maxPages = document.getElementById('max-pages').value;

            const requestBody = {
                start_url: startUrl,
                crawl_depth: parseInt(crawlDepth),
                max_pages: parseInt(maxPages)
            };

            try {
                const response = await fetch(`${API_BASE_URL}/ingest/`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(requestBody)
                });

                if (!response.ok) {
                    const errorData = await response.json();
                    throw new Error(errorData.detail || 'Failed to start ingestion task.');
                }

                const data = await response.json();
                showStatus(`Ingestion task started successfully! Task ID: ${data.task_id}`, false);
                document.getElementById('task-id').value = data.task_id;

            } catch (error) {
                showStatus(`Error: ${error.message}`, true);
            }
        });

        statusCheckForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const taskId = document.getElementById('task-id').value.trim();
            if (!taskId) return;

            const display = document.getElementById('task-status-display');
            display.innerHTML = '<div class="loader"></div> Checking...';

            try {
                const response = await fetch(`${API_BASE_URL}/ingest/status/${taskId}`);
                if (!response.ok) {
                    const errorData = await response.json();
                    throw new Error(errorData.detail || 'Failed to get task status.');
                }
                const data = await response.json();
                display.textContent = `Status for Task ${data.task_id}: ${data.status}`;
            } catch (error) {
                display.textContent = `Error: ${error.message}`;
            }
        });
    }

    // --- Manage Data Page Logic (manage.html) ---
    if (document.getElementById('sources-table-container')) {
        const tableContainer = document.getElementById('sources-table-container');
        const refreshBtn = document.getElementById('refresh-sources-btn');

        const fetchAndRenderSources = async () => {
            tableContainer.innerHTML = '<div class="loader"></div> Loading sources...';
            try {
                const response = await fetch(`${API_BASE_URL}/sources/`);
                if (!response.ok) throw new Error('Failed to fetch data sources.');

                const sources = await response.json();
                if (sources.length === 0) {
                    tableContainer.innerHTML = '<p>No data sources have been ingested yet.</p>';
                    return;
                }

                const table = document.createElement('table');
                table.innerHTML = `
                    <thead>
                        <tr>
                            <th>ID</th>
                            <th>Name</th>
                            <th>Source URL</th>
                            <th>Created At</th>
                            <th>Action</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${sources.map(s => `
                            <tr data-source-id="${s.id}">
                                <td>${s.id}</td>
                                <td>${s.name || 'N/A'}</td>
                                <td>${s.url}</td>
                                <td>${new Date(s.created_at).toLocaleString()}</td>
                                <td><button class="delete-btn" data-id="${s.id}">Delete</button></td>
                            </tr>
                        `).join('')}
                    </tbody>
                `;
                tableContainer.innerHTML = '';
                tableContainer.appendChild(table);

            } catch (error) {
                tableContainer.innerHTML = `<p style="color: var(--primary-text-color);">${error.message}</p>`;
            }
        };

        tableContainer.addEventListener('click', async (e) => {
            if (e.target.classList.contains('delete-btn')) {
                const sourceId = e.target.getAttribute('data-id');
                if (confirm(`Are you sure you want to delete data source ${sourceId} and all its content?`)) {
                    try {
                        const response = await fetch(`${API_BASE_URL}/sources/${sourceId}`, { method: 'DELETE' });
                        if (!response.ok) {
                            const errorData = await response.json();
                            throw new Error(errorData.detail || 'Failed to delete source.');
                        }
                        showStatus(`Source ${sourceId} deleted successfully.`, false);
                        fetchAndRenderSources(); // Refresh the list
                    } catch (error) {
                        showStatus(error.message, true);
                    }
                }
            }
        });

        refreshBtn.addEventListener('click', fetchAndRenderSources);
        fetchAndRenderSources(); // Initial load
    }
});
```
