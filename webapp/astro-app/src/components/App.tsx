import React, { useState } from 'react';
import RagDashboard from './RagDashboard';
import AgentsDashboard from './AgentsDashboard';
import CookbookDashboard from './CookbookDashboard';
import SettingsDashboard from './SettingsDashboard';

const TABS = ['Chat', 'Agents', 'Cookbook', 'Settings'];

export default function App() {
  const [tab, setTab] = useState('Chat');

  return (
    <div className="min-h-screen p-4">
      <div className="max-w-7xl mx-auto">
        <h1 className="text-3xl font-bold mb-6 text-gray-800">LangChain Showcase</h1>
        <div className="flex space-x-1 mb-4 border-b">
          {TABS.map((name) => (
            <button
              key={name}
              onClick={() => setTab(name)}
              className={`px-4 py-2 text-sm font-medium rounded-t-lg ${tab === name ? 'border-b-2 border-blue-500 text-blue-600 bg-white' : 'text-gray-500 hover:text-gray-700'}`}
            >
              {name}
            </button>
          ))}
        </div>
        <div className="bg-white rounded-b-lg rounded-r-lg shadow-md border border-gray-200 p-6">
          {tab === 'Chat' && <RagDashboard />}
          {tab === 'Agents' && <AgentsDashboard />}
          {tab === 'Cookbook' && <CookbookDashboard />}
          {tab === 'Settings' && <SettingsDashboard />}
        </div>
      </div>
    </div>
  );
}
