import React, { useState, useEffect } from 'react';

const LLM_PROVIDERS = ["ollama", "groq", "fakellm"]; // Add other supported providers
const VECTOR_STORES = ["chroma", "elastic", "weaviate"]; // Add other supported stores

export default function SettingsDashboard() {
  const [settings, setSettings] = useState({
    LLM_PROVIDER: '',
    OLLAMA_MODEL: '',
    USE_FAKE_LLM: false,
    EMBEDDING_MODEL: '',
    VECTOR_STORE_TYPE: '',
  });
  const [loading, setLoading] = useState(false);
  const [statusMessage, setStatusMessage] = useState('');

  useEffect(() => {
    fetchCurrentSettings();
  }, []);

  async function fetchCurrentSettings() {
    setLoading(true);
    setStatusMessage('');
    try {
      const resp = await fetch('/settings/current', {
        headers: { 'X-API-Token': 'secret' },
      });
      if (!resp.ok) throw new Error('Failed to fetch current settings');
      const data = await resp.json();
      setSettings({
        LLM_PROVIDER: data.LLM_PROVIDER || '',
        OLLAMA_MODEL: data.OLLAMA_MODEL || '',
        USE_FAKE_LLM: data.USE_FAKE_LLM || false,
        EMBEDDING_MODEL: data.EMBEDDING_MODEL || '',
        VECTOR_STORE_TYPE: data.VECTOR_STORE_TYPE || '',
      });
    } catch (error) {
      setStatusMessage(`Error fetching settings: ${error instanceof Error ? error.message : String(error)}`);
    } finally {
      setLoading(false);
    }
  }

  async function handleUpdateSettings() {
    setLoading(true);
    setStatusMessage('Updating settings...');
    try {
      // Prepare only the settings that are intended to be updated by the user via UI
      const settingsToUpdate = {
        LLM_PROVIDER: settings.USE_FAKE_LLM ? "fakellm" : settings.LLM_PROVIDER, // Ensure fakellm is set if toggled
        OLLAMA_MODEL: settings.LLM_PROVIDER === 'ollama' ? settings.OLLAMA_MODEL : undefined,
        USE_FAKE_LLM: settings.USE_FAKE_LLM,
        EMBEDDING_MODEL: settings.EMBEDDING_MODEL,
        VECTOR_STORE_TYPE: settings.VECTOR_STORE_TYPE,
      };

      const resp = await fetch('/settings/update', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'X-API-Token': 'secret' },
        body: JSON.stringify(settingsToUpdate),
      });
      if (!resp.ok) {
        const errorData = await resp.json().catch(() => ({ detail: 'Unknown error during update' }));
        throw new Error(`Failed to update settings: ${resp.status} - ${errorData.detail}`);
      }
      const data = await resp.json();
      setStatusMessage(data.status || 'Settings updated successfully!');
      // Refresh current settings display
      if (data.new_settings) {
        setSettings({
            LLM_PROVIDER: data.new_settings.LLM_PROVIDER || '',
            OLLAMA_MODEL: data.new_settings.OLLAMA_MODEL || '',
            USE_FAKE_LLM: data.new_settings.USE_FAKE_LLM || false,
            EMBEDDING_MODEL: data.new_settings.EMBEDDING_MODEL || '',
            VECTOR_STORE_TYPE: data.new_settings.VECTOR_STORE_TYPE || '',
        });
      }
    } catch (error) {
      setStatusMessage(`Error updating settings: ${error instanceof Error ? error.message : String(error)}`);
    } finally {
      setLoading(false);
    }
  }

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { name, value, type } = e.target;
    if (type === 'checkbox') {
        const { checked } = e.target as HTMLInputElement;
        setSettings(prev => ({ ...prev, [name]: checked }));
        // If USE_FAKE_LLM is checked, LLM_PROVIDER might be implicitly 'fakellm' on backend
        // or we can explicitly set it here for UI consistency if needed.
        if (name === "USE_FAKE_LLM" && checked) {
            setSettings(prev => ({ ...prev, LLM_PROVIDER: "fakellm"}));
        } else if (name === "USE_FAKE_LLM" && !checked && settings.LLM_PROVIDER === "fakellm") {
            // If unchecked and current provider was fakellm, revert to a default or leave for user to pick
            setSettings(prev => ({ ...prev, LLM_PROVIDER: LLM_PROVIDERS[0] || ''})); // Or fetch current provider if different
        }
    } else {
        setSettings(prev => ({ ...prev, [name]: value }));
    }
  };

  return (
    <div className="space-y-6 max-w-2xl mx-auto">
      <h2 className="text-2xl font-bold text-center">Application Settings</h2>

      <div className="p-6 bg-white shadow rounded-lg space-y-4">
        <div>
          <label htmlFor="USE_FAKE_LLM" className="flex items-center space-x-2">
            <input
              type="checkbox"
              id="USE_FAKE_LLM"
              name="USE_FAKE_LLM"
              checked={settings.USE_FAKE_LLM}
              onChange={handleInputChange}
              className="form-checkbox h-5 w-5 text-blue-600"
              disabled={loading}
            />
            <span className="font-semibold">Use FakeLLM (for testing, no API keys needed)</span>
          </label>
        </div>

        {!settings.USE_FAKE_LLM && (
          <div>
            <label htmlFor="LLM_PROVIDER" className="font-semibold block mb-1">LLM Provider:</label>
            <select
              id="LLM_PROVIDER"
              name="LLM_PROVIDER"
              value={settings.LLM_PROVIDER}
              onChange={handleInputChange}
              className="w-full p-2 border rounded"
              disabled={loading || settings.USE_FAKE_LLM}
            >
              {LLM_PROVIDERS.filter(p => p !== "fakellm").map(provider => (
                <option key={provider} value={provider}>{provider}</option>
              ))}
            </select>
          </div>
        )}

        {settings.LLM_PROVIDER === 'ollama' && !settings.USE_FAKE_LLM && (
          <div>
            <label htmlFor="OLLAMA_MODEL" className="font-semibold block mb-1">Ollama Model Name:</label>
            <input
              type="text"
              id="OLLAMA_MODEL"
              name="OLLAMA_MODEL"
              value={settings.OLLAMA_MODEL}
              onChange={handleInputChange}
              className="w-full p-2 border rounded"
              placeholder="e.g., llama3, mistral"
              disabled={loading}
            />
          </div>
        )}

        <div>
          <label htmlFor="EMBEDDING_MODEL" className="font-semibold block mb-1">Embedding Model:</label>
          <input
            type="text"
            id="EMBEDDING_MODEL"
            name="EMBEDDING_MODEL"
            value={settings.EMBEDDING_MODEL}
            onChange={handleInputChange}
            className="w-full p-2 border rounded"
            placeholder="e.g., sentence-transformers/all-MiniLM-L6-v2 or 'fake'"
            disabled={loading}
          />
           <p className="text-xs text-gray-500 mt-1">Enter model name from HuggingFace or 'fake' for FakeEmbeddings.</p>
        </div>

        <div>
          <label htmlFor="VECTOR_STORE_TYPE" className="font-semibold block mb-1">Vector Store Type:</label>
          <select
            id="VECTOR_STORE_TYPE"
            name="VECTOR_STORE_TYPE"
            value={settings.VECTOR_STORE_TYPE}
            onChange={handleInputChange}
            className="w-full p-2 border rounded"
            disabled={loading}
          >
            {VECTOR_STORES.map(store => (
              <option key={store} value={store}>{store}</option>
            ))}
          </select>
        </div>

        <button
          onClick={handleUpdateSettings}
          disabled={loading}
          className="w-full px-4 py-2 bg-green-500 text-white rounded hover:bg-green-600 disabled:bg-gray-400"
        >
          {loading ? 'Saving...' : 'Save Settings'}
        </button>

        {statusMessage && (
          <p className={`mt-4 text-sm ${statusMessage.startsWith('Error') ? 'text-red-600' : 'text-green-600'}`}>
            {statusMessage}
          </p>
        )}
      </div>
      <p className="text-xs text-gray-500 text-center">
        Note: Changes to settings will attempt to re-initialize components. Some changes, especially to vector stores with existing data, might require an application restart for full effect or could lead to temporary inconsistencies if data is not re-ingested under new embedding models. The ingestion endpoints currently do not fully support dynamic settings changes for the underlying storage/embedding models they use.
      </p>
    </div>
  );
}
