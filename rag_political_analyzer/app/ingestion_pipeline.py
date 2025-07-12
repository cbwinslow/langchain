# app/ingestion_pipeline.py
import asyncio
from typing import List, Dict, Any

# Adjust imports to match the project structure
from app.core.web_crawler import WebCrawler
from app.core.vector_store import CodeDocVectorStore
from app.core.ingestion import get_embeddings_model # Re-using the embedding model loader
from langchain_text_splitters import RecursiveCharacterTextSplitter, PythonCodeTextSplitter # Example for python

class IngestionPipeline:
    """
    Orchestrates the full process of crawling, chunking, embedding, and storing documentation.
    """
    def __init__(self):
        print("Initializing Ingestion Pipeline...")
        self.embedding_model = get_embeddings_model()

        # Determine embedding dimension from the model
        try:
            sample_embedding = self.embedding_model.embed_query("test")
            emb_dim = len(sample_embedding)
        except Exception as e:
            print(f"Warning: Could not dynamically determine embedding dimension. Defaulting to 384. Error: {e}")
            emb_dim = 384 # Fallback

        self.vector_store = CodeDocVectorStore(embedding_dimension=emb_dim)

        # Initialize text and code splitters
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        # Using Python as an example, this could be extended for more languages
        self.code_splitter = PythonCodeTextSplitter(
            chunk_size=500, chunk_overlap=50
        )
        print("Ingestion Pipeline initialized.")

    async def run_for_url(self, start_url: str, crawl_depth: int = 2, max_pages: int = 50):
        """
        Executes the entire ingestion pipeline for a given starting URL.
        """
        print(f"\n--- Starting ingestion for {start_url} ---")

        # 1. Crawl the website
        crawler = WebCrawler(max_depth=crawl_depth, max_pages=max_pages)
        crawled_data = await crawler.run(start_url)

        if not crawled_data:
            print("No data crawled. Ending ingestion pipeline.")
            return

        # 2. Add a single data source entry for the entire crawl
        # We use the start_url as the main identifier for this batch.
        source_name = urlparse(start_url).netloc
        source_id = self.vector_store.add_data_source(url=start_url, name=source_name)
        print(f"Data source '{start_url}' registered with ID: {source_id}")

        # 3. Process each crawled page
        total_chunks_processed = 0
        for page in crawled_data:
            print(f"Processing page: {page['url']}")
            page_chunks = []

            # 3a. Process and chunk text content
            if page['text_content']:
                text_docs = self.text_splitter.create_documents([page['text_content']])
                for doc in text_docs:
                    page_chunks.append({
                        "content": doc.page_content,
                        "chunk_type": "text",
                        "metadata": {"source_url": page['url'], "title": page['title']}
                    })

            # 3b. Process and chunk code snippets
            if page['code_snippets']:
                for snippet in page['code_snippets']:
                    # Assuming python for now, a more advanced system would detect language
                    code_docs = self.code_splitter.create_documents([snippet])
                    for doc in code_docs:
                        page_chunks.append({
                            "content": doc.page_content,
                            "chunk_type": "code",
                            "metadata": {"source_url": page['url'], "title": page['title'], "language": "python"}
                        })

            if not page_chunks:
                print(f"No chunks generated for page: {page['url']}")
                continue

            # 4. Generate embeddings for all chunks from the page
            chunk_contents = [chunk['content'] for chunk in page_chunks]
            chunk_embeddings = self.embedding_model.embed_documents(chunk_contents)

            # Add embeddings to our chunk dictionaries
            for i, chunk in enumerate(page_chunks):
                chunk['embedding'] = chunk_embeddings[i]

            # 5. Store chunks in the vector database
            try:
                self.vector_store.add_content_chunks(source_id, page_chunks)
                total_chunks_processed += len(page_chunks)
                print(f"Successfully stored {len(page_chunks)} chunks for page {page['url']}")
            except Exception as e:
                print(f"Error storing chunks for page {page['url']}: {e}")

        print(f"\n--- Ingestion for {start_url} complete. Total chunks processed: {total_chunks_processed} ---")


# Main execution block for running this script from the command line
if __name__ == '__main__':
    from urllib.parse import urlparse
    import argparse

    parser = argparse.ArgumentParser(description="Run the RAG ingestion pipeline for a documentation website.")
    parser.add_argument("start_url", type=str, help="The starting URL to crawl (e.g., 'https://python.langchain.com/v0.2/docs/introduction/').")
    parser.add_argument("--depth", type=int, default=1, help="Maximum depth to crawl from the start URL.")
    parser.add_argument("--max_pages", type=int, default=10, help="Maximum number of pages to crawl and process.")

    args = parser.parse_args()

    # Create and run the pipeline
    pipeline = IngestionPipeline()

    # Run the async main function
    asyncio.run(pipeline.run_for_url(
        start_url=args.start_url,
        crawl_depth=args.depth,
        max_pages=args.max_pages
    ))
```
