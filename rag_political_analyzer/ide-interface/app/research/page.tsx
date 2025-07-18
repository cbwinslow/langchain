"use client";

import { useState, FormEvent } from 'react';

export default function ResearchPage() {
    const [isLoading, setIsLoading] = useState(false);
    const [taskId, setTaskId] = useState<string | null>(null);
    const [status, setStatus] = useState<string | null>(null);
    const [error, setError] = useState<string | null>(null);

    const handleCrawlSubmit = async (event: FormEvent<HTMLFormElement>) => {
        event.preventDefault();
        setIsLoading(true);
        setError(null);
        setTaskId(null);
        setStatus('Submitting...');

        const formData = new FormData(event.currentTarget);
        const startUrl = formData.get('start_url') as string;
        const crawlDepth = formData.get('crawl_depth') as string;
        const maxPages = formData.get('max_pages') as string;

        try {
            const response = await fetch('http://localhost:8000/ingest/', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    start_url: startUrl,
                    crawl_depth: parseInt(crawlDepth),
                    max_pages: parseInt(maxPages),
                }),
            });

            if (!response.ok) {
                const errData = await response.json();
                throw new Error(errData.detail || 'Failed to start ingestion task.');
            }

            const data = await response.json();
            setTaskId(data.task_id);
            setStatus(`Task started successfully! ID: ${data.task_id}. You can monitor its status below.`);

        } catch (err) {
            setError(err instanceof Error ? err.message : 'An unknown error occurred.');
            setStatus(null);
        } finally {
            setIsLoading(false);
        }
    };

    const handleStatusCheck = async () => {
        if (!taskId) {
            setError("No task ID to check.");
            return;
        }

        setError(null);
        setStatus(`Checking status for ${taskId}...`);

        try {
            const response = await fetch(`http://localhost:8000/ingest/status/${taskId}`);
            if (!response.ok) {
                const errData = await response.json();
                throw new Error(errData.detail || 'Failed to get task status.');
            }
            const data = await response.json();
            setStatus(`Status for Task ${data.task_id}: ${data.status}`);
        } catch (err) {
            setError(err instanceof Error ? err.message : 'An unknown error occurred.');
            setStatus(null);
        }
    };

    return (
        <div className="p-6">
            <h1 className="text-2xl font-bold mb-4">Deep Research & Ingestion</h1>
            <p className="text-gray-400 mb-6">
                Use this tool to crawl a documentation website. The crawler will extract text and code,
                chunk it, generate embeddings, and store it in the database for querying.
            </p>

            <div className="bg-gray-900 p-6 rounded-lg border border-gray-700">
                <form onSubmit={handleCrawlSubmit}>
                    <div className="mb-4">
                        <label htmlFor="start_url" className="block text-sm font-medium text-gray-300 mb-1">Start URL</label>
                        <input
                            type="url"
                            name="start_url"
                            id="start_url"
                            className="w-full p-2 bg-gray-800 border border-gray-600 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                            placeholder="https://example-docs.com"
                            required
                        />
                    </div>
                    <div className="grid grid-cols-2 gap-4 mb-4">
                        <div>
                            <label htmlFor="crawl_depth" className="block text-sm font-medium text-gray-300 mb-1">Crawl Depth</label>
                            <input
                                type="number"
                                name="crawl_depth"
                                id="crawl_depth"
                                defaultValue="1"
                                min="0"
                                className="w-full p-2 bg-gray-800 border border-gray-600 rounded-md"
                            />
                        </div>
                        <div>
                            <label htmlFor="max_pages" className="block text-sm font-medium text-gray-300 mb-1">Max Pages</label>
                            <input
                                type="number"
                                name="max_pages"
                                id="max_pages"
                                defaultValue="20"
                                min="1"
                                className="w-full p-2 bg-gray-800 border border-gray-600 rounded-md"
                            />
                        </div>
                    </div>
                    <button
                        type="submit"
                        className="w-full px-4 py-2 bg-blue-600 rounded-md hover:bg-blue-700 disabled:bg-gray-500"
                        disabled={isLoading}
                    >
                        {isLoading ? 'Starting...' : 'Start Research Task'}
                    </button>
                </form>
            </div>

            {(status || error) && (
                <div className={`mt-6 p-4 rounded-lg border ${error ? 'bg-red-900/20 border-red-500 text-red-300' : 'bg-green-900/20 border-green-500 text-green-300'}`}>
                    <h3 className="font-bold mb-2">{error ? 'Error' : 'Status'}</h3>
                    <p>{error || status}</p>
                    {taskId && !error && (
                         <button onClick={handleStatusCheck} className="mt-2 px-3 py-1 text-xs bg-gray-600 hover:bg-gray-500 rounded-md">
                            Check Status Again
                         </button>
                    )}
                </div>
            )}
        </div>
    );
}
```
