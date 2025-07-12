"use client";

import { useState, useEffect } from 'react';

// --- Types ---
interface DataSource {
    id: number;
    url: string;
    name: string | null;
    created_at: string;
}

// --- Icons (simple placeholders) ---
const FolderIcon = () => '📁';
const LinkIcon = () => '🔗';
const DeleteIcon = () => '🗑️';

export default function DataSourceExplorer() {
    const [sources, setSources] = useState<DataSource[]>([]);
    const [isLoading, setIsLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    const fetchSources = async () => {
        setIsLoading(true);
        setError(null);
        try {
            const response = await fetch('http://localhost:8000/sources/');
            if (!response.ok) {
                const errData = await response.json();
                throw new Error(errData.detail || 'Failed to fetch data sources.');
            }
            const data: DataSource[] = await response.json();
            setSources(data);
        } catch (err) {
            setError(err instanceof Error ? err.message : 'An unknown error occurred.');
        } finally {
            setIsLoading(false);
        }
    };

    const handleDelete = async (sourceId: number) => {
        if (!confirm(`Are you sure you want to delete data source ${sourceId} and all its content?`)) {
            return;
        }

        try {
            const response = await fetch(`http://localhost:8000/sources/${sourceId}`, {
                method: 'DELETE',
            });
            if (!response.ok) {
                const errData = await response.json();
                throw new Error(errData.detail || 'Failed to delete data source.');
            }
            // Refresh the list after successful deletion
            fetchSources();
        } catch (err) {
            setError(err instanceof Error ? err.message : 'An unknown error occurred.');
        }
    };

    useEffect(() => {
        fetchSources();
    }, []);

    if (isLoading) {
        return <div className="p-4">Loading data sources...</div>;
    }

    if (error) {
        return <div className="p-4 text-red-400">Error: {error}</div>;
    }

    return (
        <div className="p-2 text-gray-300">
            <div className="flex justify-between items-center mb-2">
                <h2 className="text-xs font-bold uppercase tracking-wider">Ingested Sources</h2>
                <button onClick={fetchSources} title="Refresh" className="p-1 hover:bg-gray-700 rounded">🔄</button>
            </div>
            <ul className="space-y-1">
                {sources.length > 0 ? sources.map(source => (
                    <li key={source.id} className="group flex justify-between items-center p-1 rounded hover:bg-gray-700">
                        <div className="flex items-center space-x-2">
                            <span className="text-lg"><FolderIcon /></span>
                            <span className="text-sm">{source.name || source.url}</span>
                        </div>
                        <button
                            onClick={() => handleDelete(source.id)}
                            className="opacity-0 group-hover:opacity-100 transition-opacity text-gray-400 hover:text-red-500"
                            title={`Delete ${source.name}`}
                        >
                            <DeleteIcon />
                        </button>
                    </li>
                )) : (
                    <li className="p-1 text-sm text-gray-500">No data sources found.</li>
                )}
            </ul>
        </div>
    );
}
```
