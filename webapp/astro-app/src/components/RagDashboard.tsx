import React, { useState } from 'react';

export default function RagDashboard() {
  const [question, setQuestion] = useState('');
  const [answer, setAnswer] = useState('');

  async function handleQuery() {
    const resp = await fetch('/query', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'X-API-Token': 'secret' },
      body: JSON.stringify({ question }),
    });
    const data = await resp.json();
    setAnswer(data.answer);
  }

  return (
    <div>
      <h2 className="text-xl font-bold mb-2">RAG Query</h2>
      <textarea
        value={question}
        onChange={(e) => setQuestion(e.target.value)}
        className="w-full p-2 border rounded mb-2"
      />
      <button onClick={handleQuery} className="px-4 py-2 bg-blue-500 text-white rounded">
        Ask
      </button>
      {answer && <pre className="mt-4 bg-gray-100 p-2 rounded">{answer}</pre>}
    </div>
  );
}
