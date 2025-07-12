import React, { useState } from 'react';

export default function AgentsDashboard() {
  const [query, setQuery] = useState('');
  const [answer, setAnswer] = useState('');
  const [loading, setLoading] = useState(false);

  async function handleAgentQuery() {
    if (!query) {
      setAnswer('Please enter a query for the agent.');
      return;
    }
    setLoading(true);
    setAnswer('');
    try {
      const resp = await fetch('/agent/wikipedia', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'X-API-Token': 'secret' },
        body: JSON.stringify({ question: query }), // FastAPI endpoint expects 'question'
      });
      if (!resp.ok) {
        const errorData = await resp.json().catch(() => ({ detail: 'Unknown error' }));
        throw new Error(`HTTP error! status: ${resp.status} - ${errorData.detail}`);
      }
      const data = await resp.json();
      setAnswer(data.answer);
    } catch (error) {
      console.error('Agent query failed:', error);
      setAnswer(`Failed to get an answer from the agent: ${error instanceof Error ? error.message : String(error)}`);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div>
      <h2 className="text-xl font-bold mb-4">Wikipedia Agent</h2>
      <p className="mb-4 text-sm text-gray-600">
        This agent uses Wikipedia to answer questions. It's a demonstration of a simple ReAct agent.
      </p>

      <div className="space-y-4">
        <div>
          <label htmlFor="agent-query" className="font-semibold block mb-1">Ask the Wikipedia Agent:</label>
          <textarea
            id="agent-query"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            className="w-full p-2 border rounded mb-2 h-24"
            placeholder="e.g., What is the LangChain framework?"
          />
          <button
            onClick={handleAgentQuery}
            disabled={loading || !query}
            className="px-4 py-2 bg-purple-500 text-white rounded disabled:bg-gray-400 hover:bg-purple-600"
          >
            {loading ? 'Agent is thinking...' : 'Ask Agent'}
          </button>
        </div>

        {answer && (
          <div>
            <h3 className="font-semibold mb-1">Agent's Answer:</h3>
            <pre className="bg-gray-100 p-4 rounded whitespace-pre-wrap">{answer}</pre>
          </div>
        )}
      </div>
    </div>
  );
}
