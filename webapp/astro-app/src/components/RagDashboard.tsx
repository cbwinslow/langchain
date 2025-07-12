import React, { useState } from 'react';

export default function RagDashboard() {
  const [question, setQuestion] = useState('');
  const [answer, setAnswer] = useState('');
  const [ingestText, setIngestText] = useState('');
  const [pdfFile, setPdfFile] = useState<File | null>(null);
  const [chainType, setChainType] = useState('Standard QA');
  const [loading, setLoading] = useState(false);
  const [ingestStatus, setIngestStatus] = useState('');

  async function handleQuery() {
    if (!question) {
      setAnswer('Please enter a question.');
      return;
    }
    setLoading(true);
    setAnswer('');
    const endpoint = chainType === 'Standard QA' ? '/query' : '/query_graph';
    try {
      const resp = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'X-API-Token': 'secret' },
        body: JSON.stringify({ question }),
      });
      if (!resp.ok) {
        throw new Error(`HTTP error! status: ${resp.status}`);
      }
      const data = await resp.json();
      setAnswer(JSON.stringify(data.answer, null, 2));
    } catch (error) {
      console.error('Query failed:', error);
      setAnswer('Failed to get an answer.');
    } finally {
      setLoading(false);
    }
  }

  async function handleIngestText() {
    if (!ingestText) return;
    setLoading(true);
    setIngestStatus('Ingesting text...');
    try {
      const resp = await fetch('/ingest_text', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'X-API-Token': 'secret' },
        body: JSON.stringify({ content: ingestText }),
      });
      if (!resp.ok) throw new Error(`HTTP error! status: ${resp.status}`);
      await resp.json();
      setIngestStatus('Text ingested successfully!');
      setIngestText('');
    } catch (error) {
      console.error('Text ingestion failed:', error);
      setIngestStatus('Text ingestion failed.');
    } finally {
      setLoading(false);
    }
  }

  async function handleIngestPdf() {
    if (!pdfFile) return;
    setLoading(true);
    setIngestStatus('Ingesting PDF...');
    const formData = new FormData();
    formData.append('file', pdfFile);
    try {
      const resp = await fetch('/ingest_pdf', {
        method: 'POST',
        headers: { 'X-API-Token': 'secret' },
        body: formData,
      });
      if (!resp.ok) throw new Error(`HTTP error! status: ${resp.status}`);
      await resp.json();
      setIngestStatus('PDF ingested successfully!');
      setPdfFile(null);
      // Reset file input
      const fileInput = document.getElementById('pdf-upload') as HTMLInputElement;
      if (fileInput) fileInput.value = '';
    } catch (error) {
      console.error('PDF ingestion failed:', error);
      setIngestStatus('PDF ingestion failed.');
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
      {/* Ingestion Column */}
      <div className="space-y-4">
        <h2 className="text-xl font-bold">Ingestion</h2>

        {/* Text Ingestion */}
        <div>
          <h3 className="font-semibold mb-1">Ingest Raw Text</h3>
          <textarea
            value={ingestText}
            onChange={(e) => setIngestText(e.target.value)}
            className="w-full p-2 border rounded mb-2 h-32"
            placeholder="Paste raw text here to add to the vector store."
          />
          <button onClick={handleIngestText} disabled={loading || !ingestText} className="px-4 py-2 bg-blue-500 text-white rounded disabled:bg-gray-400">
            Submit Text
          </button>
        </div>

        {/* PDF Ingestion */}
        <div>
          <h3 className="font-semibold mb-1">Ingest PDF</h3>
          <input
            id="pdf-upload"
            type="file"
            accept=".pdf"
            onChange={(e) => setPdfFile(e.target.files ? e.target.files[0] : null)}
            className="w-full p-2 border rounded mb-2"
          />
          <button onClick={handleIngestPdf} disabled={loading || !pdfFile} className="px-4 py-2 bg-blue-500 text-white rounded disabled:bg-gray-400">
            Upload PDF
          </button>
        </div>
        {ingestStatus && <p className="mt-2 text-sm text-gray-600">{ingestStatus}</p>}
      </div>

      {/* Query Column */}
      <div className="space-y-4">
        <h2 className="text-xl font-bold">Query</h2>

        {/* Chain Type Selection */}
        <div>
          <label htmlFor="chain-type" className="font-semibold block mb-1">Chain Type</label>
          <select
            id="chain-type"
            value={chainType}
            onChange={(e) => setChainType(e.target.value)}
            className="w-full p-2 border rounded"
          >
            <option>Standard QA</option>
            <option>Graph QA</option>
          </select>
        </div>

        {/* Question Input */}
        <div>
          <label htmlFor="question" className="font-semibold block mb-1">Question</label>
          <textarea
            id="question"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            className="w-full p-2 border rounded mb-2 h-32"
            placeholder="Ask a question about the ingested documents."
          />
          <button onClick={handleQuery} disabled={loading || !question} className="px-4 py-2 bg-green-500 text-white rounded disabled:bg-gray-400">
            {loading ? 'Thinking...' : 'Ask'}
          </button>
        </div>

        {/* Answer Display */}
        {answer && (
          <div>
            <h3 className="font-semibold mb-1">Answer</h3>
            <pre className="bg-gray-100 p-4 rounded whitespace-pre-wrap">{answer}</pre>
          </div>
        )}
      </div>
    </div>
  );
}
