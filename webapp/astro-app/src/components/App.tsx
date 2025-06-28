import React, { useState } from 'react';

const TABS = ['Spreadsheet', 'DB Viewer', 'Query Writer', 'Scripts', 'Settings'];

export default function App() {
  const [tab, setTab] = useState('Spreadsheet');

  return (
    <div className="min-h-screen bg-gray-100 p-4">
      <h1 className="text-2xl font-bold mb-4">Data Analysis Dashboard</h1>
      <div className="flex space-x-4 mb-4">
        {TABS.map((name) => (
          <button
            key={name}
            onClick={() => setTab(name)}
            className={`px-3 py-1 border-b-2 ${tab === name ? 'border-blue-500' : 'border-transparent'}`}
          >
            {name}
          </button>
        ))}
      </div>
      <div className="bg-white rounded shadow p-4">
        {tab === 'Spreadsheet' && <p>Spreadsheet component goes here.</p>}
        {tab === 'DB Viewer' && <p>SQL database viewer goes here.</p>}
        {tab === 'Query Writer' && <p>Query writing tools go here.</p>}
        {tab === 'Scripts' && <p>Script toolkit for API scraping goes here.</p>}
        {tab === 'Settings' && <p>User profile and DB config settings.</p>}
      </div>
    </div>
  );
}
