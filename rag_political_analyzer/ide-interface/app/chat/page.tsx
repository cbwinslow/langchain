"use client";

import { useState, FormEvent, useRef, useEffect } from 'react';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';

// --- Types ---
interface Message {
  sender: 'user' | 'bot';
  text: string;
  context?: any[]; // To store retrieved context
}

interface CodeSnippet {
    language: string | null;
    code: string;
}

// --- Helper Functions ---
function parseBotMessage(message: string): (string | CodeSnippet)[] {
    const parts = message.split(/(```[\s\S]*?```)/g);
    return parts.map(part => {
        const match = part.match(/```(\w*)\n([\s\S]*?)```/);
        if (match) {
            return {
                language: match[1] || 'plaintext',
                code: match[2].trim()
            };
        }
        return part.trim();
    }).filter(part => (typeof part === 'string' && part.length > 0) || typeof part === 'object');
}


// --- Main Component ---
export default function ChatPage() {
  const [messages, setMessages] = useState<Message[]>([
    { sender: 'bot', text: "Hello! Ask me a question about your ingested documentation." }
  ]);
  const [isLoading, setIsLoading] = useState(false);
  const chatContainerRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom
  useEffect(() => {
    if (chatContainerRef.current) {
      chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
    }
  }, [messages]);

  const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const formData = new FormData(event.currentTarget);
    const query = formData.get('query') as string;

    if (!query.trim() || isLoading) return;

    // Add user message to state
    setMessages(prev => [...prev, { sender: 'user', text: query }]);
    setIsLoading(true);
    event.currentTarget.reset();

    try {
        const apiFormData = new FormData();
        apiFormData.append('query', query);

        const response = await fetch('http://localhost:8000/query/', {
            method: 'POST',
            body: apiFormData,
        });

        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.detail || "An error occurred while fetching the response.");
        }

        const data = await response.json();
        setMessages(prev => [...prev, { sender: 'bot', text: data.answer, context: data.retrieved_context }]);

    } catch (error) {
        const errorMessage = error instanceof Error ? error.message : "An unknown error occurred.";
        setMessages(prev => [...prev, { sender: 'bot', text: `Error: ${errorMessage}` }]);
    } finally {
        setIsLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-full bg-gray-800 text-gray-200">
      <div ref={chatContainerRef} className="flex-grow p-4 overflow-y-auto">
        <div className="space-y-4">
          {messages.map((msg, index) => (
            <div key={index} className={`flex ${msg.sender === 'user' ? 'justify-end' : 'justify-start'}`}>
              <div className={`max-w-2xl p-3 rounded-lg ${msg.sender === 'user' ? 'bg-blue-600' : 'bg-gray-700'}`}>
                {parseBotMessage(msg.text).map((part, i) => {
                    if (typeof part === 'string') {
                        return <p key={i} className="whitespace-pre-wrap">{part}</p>;
                    }
                    return (
                        <SyntaxHighlighter key={i} language={part.language ?? 'bash'} style={vscDarkPlus} PreTag="div">
                            {part.code}
                        </SyntaxHighlighter>
                    );
                })}
              </div>
            </div>
          ))}
          {isLoading && (
            <div className="flex justify-start">
              <div className="max-w-2xl p-3 rounded-lg bg-gray-700">
                <div className="flex items-center space-x-2">
                    <div className="w-2 h-2 bg-blue-400 rounded-full animate-pulse"></div>
                    <div className="w-2 h-2 bg-blue-400 rounded-full animate-pulse delay-75"></div>
                    <div className="w-2 h-2 bg-blue-400 rounded-full animate-pulse delay-150"></div>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
      <div className="p-4 border-t border-gray-700">
        <form onSubmit={handleSubmit} className="flex space-x-2">
          <input
            type="text"
            name="query"
            className="flex-grow p-2 bg-gray-900 border border-gray-600 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            placeholder="Send a message..."
            disabled={isLoading}
          />
          <button
            type="submit"
            className="px-4 py-2 bg-blue-600 rounded-md hover:bg-blue-700 disabled:bg-gray-500 disabled:cursor-not-allowed"
            disabled={isLoading}
          >
            Send
          </button>
        </form>
      </div>
    </div>
  );
}
