import React, { useState } from 'react';

export default function CookbookDashboard() {
  // State for SQL Agent
  const [sqlQuery, setSqlQuery] = useState('');
  const [sqlAnswer, setSqlAnswer] = useState('');
  const [sqlLoading, setSqlLoading] = useState(false);

  // State for Structured Output
  const [inputText, setInputText] = useState('');
  const [structuredOutput, setStructuredOutput] = useState(null);
  const [structuredLoading, setStructuredLoading] = useState(false);

  async function handleSqlAgentQuery() {
    if (!sqlQuery) {
      setSqlAnswer('Please enter a question for the SQL agent.');
      return;
    }
    setSqlLoading(true);
    setSqlAnswer('');
    try {
      const resp = await fetch('/cookbook/sql-agent', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'X-API-Token': 'secret' },
        body: JSON.stringify({ question: sqlQuery }),
      });
      if (!resp.ok) {
        const errorData = await resp.json().catch(() => ({ detail: 'Unknown error' }));
        throw new Error(`HTTP error! status: ${resp.status} - ${errorData.detail}`);
      }
      const data = await resp.json();
      setSqlAnswer(data.answer);
    } catch (error) {
      console.error('SQL Agent query failed:', error);
      setSqlAnswer(`Failed to get answer: ${error instanceof Error ? error.message : String(error)}`);
    } finally {
      setSqlLoading(false);
    }
  }

  async function handleStructuredOutput() {
    if (!inputText) {
      setStructuredOutput({ error: 'Please enter text to extract from.' });
      return;
    }
    setStructuredLoading(true);
    setStructuredOutput(null);
    try {
      const resp = await fetch('/cookbook/structured-output', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'X-API-Token': 'secret' },
        body: JSON.stringify({ text: inputText }),
      });
      if (!resp.ok) {
        const errorData = await resp.json().catch(() => ({ detail: 'Unknown error' }));
        throw new Error(`HTTP error! status: ${resp.status} - ${errorData.detail}`);
      }
      const data = await resp.json();
      setStructuredOutput(data.answer);
    } catch (error) {
      console.error('Structured output failed:', error);
      setStructuredOutput({ error: `Failed to extract data: ${error instanceof Error ? error.message : String(error)}` });
    } finally {
      setStructuredLoading(false);
    }
  }

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
      {/* SQL Agent Demo */}
      <div className="space-y-4 p-4 border rounded-lg">
        <h2 className="text-xl font-bold mb-2">SQL Agent Demo</h2>
        <p className="text-sm text-gray-600 mb-2">
          Ask questions in natural language about a sample database containing employee and department tables.
        </p>
        <div>
          <label htmlFor="sql-query" className="font-semibold block mb-1">Your Question:</label>
          <textarea
            id="sql-query"
            value={sqlQuery}
            onChange={(e) => setSqlQuery(e.target.value)}
            className="w-full p-2 border rounded mb-2 h-24"
            placeholder="e.g., How many employees are in the Engineering department?"
          />
          <button onClick={handleSqlAgentQuery} disabled={sqlLoading} className="px-4 py-2 bg-teal-500 text-white rounded disabled:bg-gray-400">
            {sqlLoading ? 'Querying DB...' : 'Ask SQL Agent'}
          </button>
        </div>
        {sqlAnswer && (
          <div>
            <h3 className="font-semibold mb-1">SQL Agent's Answer:</h3>
            <pre className="bg-gray-100 p-4 rounded whitespace-pre-wrap">{sqlAnswer}</pre>
          </div>
        )}
      </div>

      {/* Structured Output Demo */}
      <div className="space-y-4 p-4 border rounded-lg">
        <h2 className="text-xl font-bold mb-2">Structured Output Demo</h2>
        <p className="text-sm text-gray-600 mb-2">
          Extract structured data (a person's name and age) from unstructured text. This requires an LLM that supports tool calling (e.g., from Groq, OpenAI, Anthropic).
        </p>
        <div>
          <label htmlFor="structured-input" className="font-semibold block mb-1">Input Text:</label>
          <textarea
            id="structured-input"
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
            className="w-full p-2 border rounded mb-2 h-24"
            placeholder="e.g., John is a 42-year-old engineer."
          />
          <button onClick={handleStructuredOutput} disabled={structuredLoading} className="px-4 py-2 bg-indigo-500 text-white rounded disabled:bg-gray-400">
            {structuredLoading ? 'Extracting...' : 'Extract Person'}
          </button>
        </div>
        {structuredOutput && (
          <div>
            <h3 className="font-semibold mb-1">Extracted JSON:</h3>
            <pre className="bg-gray-100 p-4 rounded whitespace-pre-wrap">
              {JSON.stringify(structuredOutput, null, 2)}
            </pre>
          </div>
        )}
      </div>
    </div>
  );
}
