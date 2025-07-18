import type { Metadata } from "next";
import Link from 'next/link';
import { Inter } from "next/font/google";
import "./globals.css";
import DataSourceExplorer from "@/app/components/DataSourceExplorer"; // Import the explorer

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "CodeRAG IDE",
  description: "An IDE-like interface for a code documentation RAG system.",
};

// Mock data for icons - in a real app, use an icon library like react-icons or SVGs
const FileIcon = () => '📄';
const ResearchIcon = () => '🔬'; // New icon for the research page
const ChatIcon = () => '💬';
const SettingsIcon = () => '⚙️';


export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className={inter.className}>
        <div className="ide-layout">
          <aside className="activity-bar">
            <Link href="/explorer" title="Explorer" className="p-2 text-2xl cursor-pointer hover:bg-gray-700 rounded-md"><FileIcon /></Link>
            <Link href="/research" title="Deep Research" className="p-2 text-2xl cursor-pointer hover:bg-gray-700 rounded-md"><ResearchIcon /></Link>
            <Link href="/chat" title="Chat" className="p-2 text-2xl cursor-pointer hover:bg-gray-700 rounded-md"><ChatIcon /></Link>
            <div className="p-2 text-2xl cursor-pointer hover:bg-gray-700 rounded-md mt-auto mb-2" title="Settings"><SettingsIcon /></div>
          </aside>

          <aside className="sidebar">
            {/* The DataSourceExplorer is now part of the main layout's sidebar */}
            <DataSourceExplorer />
          </aside>

          <main className="main-content">
            <div className="editor-pane">
              {children}
            </div>
            <div className="panel">
              <div className="p-2 border-b border-gray-700">
                <span className="px-2 py-1 text-sm">TERMINAL</span>
                <span className="px-2 py-1 text-sm">OUTPUT</span>
                <span className="px-2 py-1 text-sm">DEBUG CONSOLE</span>
              </div>
              {/* Panel content will go here */}
            </div>
          </main>
        </div>
        <footer className="status-bar">
          <div>Ready</div>
        </footer>
      </body>
    </html>
  );
}
