# app/agents/ingestion_agent.py
import os
import asyncio
from typing import Dict, Any
from urllib.parse import urlparse

from .base_agent import Agent
from app.core.web_crawler import WebCrawler
from app.core.vector_store import CodeDocVectorStore
from app.core.ingestion import get_embeddings_model
from langchain_text_splitters import RecursiveCharacterTextSplitter, PythonCodeTextSplitter

class IngestionAgent(Agent):
    """
    Handles the ingestion of documents from various sources, including web crawls.
    """
    def __init__(
        self,
        name: str = "IngestionAgent",
        vector_store: CodeDocVectorStore,
        embedding_model: Any
    ):
        super().__init__(name)
        self.vector_store = vector_store
        self.embedding_model = embedding_model

        # Initialize text and code splitters
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        self.code_splitter = PythonCodeTextSplitter(chunk_size=500, chunk_overlap=50)
        print(f"{self.name} initialized.")

    async def _ingest_url(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handles the 'ingest_from_url' task."""
        start_url = data.get("start_url")
        if not start_url:
            return {"status": "error", "message": "start_url missing from task data."}

        crawl_depth = data.get("crawl_depth", 1)
        max_pages = data.get("max_pages", 20)

        print(f"{self.name}: Starting ingestion for {start_url}")

        # 1. Crawl the website
        crawler = WebCrawler(max_depth=crawl_depth, max_pages=max_pages)
        crawled_data = await crawler.run(start_url)

        if not crawled_data:
            msg = "No data was crawled. The website might be inaccessible, empty, or block crawlers."
            print(f"{self.name}: {msg}")
            return {"status": "success", "message": msg, "chunks_stored": 0}

        # 2. Add data source entry
        source_name = urlparse(start_url).netloc
        source_id = self.vector_store.add_data_source(url=start_url, name=source_name)
        print(f"{self.name}: Data source '{start_url}' registered with ID: {source_id}")

        # 3. Process each crawled page
        total_chunks_processed = 0
        for page in crawled_data:
            page_chunks = []

            # Process text content
            if page.get('text_content'):
                text_docs = self.text_splitter.create_documents([page['text_content']])
                for doc in text_docs:
                    page_chunks.append({
                        "content": doc.page_content, "chunk_type": "text",
                        "metadata": {"source_url": page['url'], "title": page['title']}
                    })

            # Process code snippets
            if page.get('code_snippets'):
                for snippet in page['code_snippets']:
                    code_docs = self.code_splitter.create_documents([snippet])
                    for doc in code_docs:
                        page_chunks.append({
                            "content": doc.page_content, "chunk_type": "code",
                            "metadata": {"source_url": page['url'], "title": page['title'], "language": "python"}
                        })

            if not page_chunks: continue

            # Generate embeddings and store
            chunk_contents = [chunk['content'] for chunk in page_chunks]
            chunk_embeddings = self.embedding_model.embed_documents(chunk_contents)
            for i, chunk in enumerate(page_chunks):
                chunk['embedding'] = chunk_embeddings[i]

            self.vector_store.add_content_chunks(source_id, page_chunks)
            total_chunks_processed += len(page_chunks)

        final_msg = f"Ingestion for {start_url} complete. Stored {total_chunks_processed} chunks."
        print(f"{self.name}: {final_msg}")
        return {"status": "success", "message": final_msg, "chunks_stored": total_chunks_processed}

    async def execute(self, task: Dict[str, Any], **kwargs) -> Any:
        """
        Executes an ingestion task based on its type.
        - "ingest_from_url": Crawls and ingests a website.
        """
        task_type = task.get("type")
        data = task.get("data", {})

        try:
            if task_type == "ingest_from_url":
                return await self._ingest_url(data)
            # Future task types like "ingest_from_file" could be added here
            else:
                return {"status": "error", "message": f"Unknown task type for IngestionAgent: {task_type}"}
        except Exception as e:
            error_msg = f"An unexpected error occurred in {self.name}: {e}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            return {"status": "error", "message": error_msg}

# Example Usage
if __name__ == '__main__':
    from dotenv import load_dotenv

    load_dotenv(dotenv_path='../../.env')

    async def test_ingestion_agent():
        print("Testing unified IngestionAgent...")
        try:
            emb_model = get_embeddings_model()
            try: emb_dim = len(emb_model.embed_query("test"))
            except: emb_dim = 384

            vector_db = CodeDocVectorStore(embedding_dimension=emb_dim)
            agent = IngestionAgent(vector_store=vector_db, embedding_model=emb_model)

            task = {
                "type": "ingest_from_url",
                "data": {
                    "start_url": "https://python.langchain.com/v0.2/docs/introduction/",
                    "crawl_depth": 0, # Just crawl the single start page for a quick test
                    "max_pages": 1
                }
            }
            result = await agent.execute(task)
            print("\n--- Agent Execution Result ---")
            print(result)

        except Exception as e:
            print(f"\nError during test: {e}")
            print("Ensure DB is running and .env is configured.")

    asyncio.run(test_ingestion_agent())
```
