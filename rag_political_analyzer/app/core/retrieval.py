# app/core/retrieval.py
from typing import List, Dict, Any, Tuple
import spacy # Import spacy
from .vector_store import PGVectorStore
from .ingestion import get_embeddings_model, get_spacy_model # Import get_spacy_model

def enrich_query_with_spacy(query: str, nlp_model: Any) -> str:
    """
    Enriches a query by extracting key terms (entities, noun chunks) using spaCy.
    For now, it appends unique key terms to the original query.
    More sophisticated strategies could be to generate multiple queries or use synonyms.
    """
    if not query.strip():
        return query

    doc = nlp_model(query)

    # Extract entities and important nouns/noun phrases
    key_terms = set() # Use a set to avoid duplicates

    for ent in doc.ents:
        key_terms.add(ent.text.lower())

    for noun_chunk in doc.noun_chunks:
        # Filter out very short or common noun chunks if needed
        if len(noun_chunk.text.split()) > 1 or noun_chunk.root.pos_ in ['NOUN', 'PROPN']:
            key_terms.add(noun_chunk.text.lower())

    # Add significant individual nouns not already covered
    for token in doc:
        if token.pos_ in ['NOUN', 'PROPN'] and not token.is_stop and len(token.lemma_) > 2:
            key_terms.add(token.lemma_.lower())

    if not key_terms:
        return query

    # Combine original query with extracted key terms
    # A simple strategy: append them.
    # More advanced: reweighting, generating sub-queries, etc.
    enriched_query_parts = [query]
    # Add key terms not already in the query string (simple check)
    original_query_lower = query.lower()
    for term in sorted(list(key_terms)): # Sorted for consistent ordering if it matters
        if term not in original_query_lower:
            enriched_query_parts.append(term)

    return " ".join(enriched_query_parts)


class Retriever:
    def __init__(self, vector_store: PGVectorStore, embedding_model: Any, spacy_model: Any):
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.spacy_model = spacy_model # Store the spaCy model

    def retrieve(self, query: str, k: int = 5, use_query_enrichment: bool = True) -> List[Dict[str, Any]]:
        """
        Retrieves the top k most relevant document chunks for a given query.
        Returns a list of document chunks (content and metadata).
        Optionally enriches the query using spaCy.
        """
        if not query:
            return []

        processed_query = query
        if use_query_enrichment and self.spacy_model:
            processed_query = enrich_query_with_spacy(query, self.spacy_model)
            print(f"Original Query: {query}")
            print(f"Enriched Query: {processed_query}")

        # 1. Generate embedding for the (potentially enriched) query
        query_embedding = self.embedding_model.embed_query(processed_query)

        # 2. Perform similarity search in the vector store
        # similarity_search_with_scores returns List[Tuple[Dict[str, Any], float]]
        # Each tuple is (document_data, similarity_score)
        search_results = self.vector_store.similarity_search_with_scores(
            query_embedding=query_embedding,
            k=k
        )

        # 3. Format results: Extract documents, optionally include scores if needed downstream
        retrieved_docs = []
        for doc_data, score in search_results:
            # doc_data is expected to be {"id": ..., "content": ..., "metadata": ...}
            # We can add the score to the metadata if desired
            doc_data_with_score = doc_data.copy()
            doc_data_with_score['metadata']['retrieval_score'] = score
            retrieved_docs.append(doc_data_with_score)

        return retrieved_docs

# Example Usage (for testing this file directly)
if __name__ == '__main__':
    # This requires a running PGVector instance and the PGVectorStore class defined.
    # It also requires the ingestion module for get_embeddings_model.
    # Ensure .env file is set up in rag_political_analyzer directory for DB connection.

    print("Testing Retriever...")

    # Initialize components (these would typically be managed by the application's setup)
    try:
        # 1. Initialize Embedding Model
        print("Initializing embedding model...")
        embeddings = get_embeddings_model()
        print("Embedding model initialized.")

        # 2. Initialize Vector Store (assuming 'test_collection' and dimension 384 from previous example)
        # For this test, we assume the default embedding dimension of all-MiniLM-L6-v2 (384)
        # If you ran vector_store.py example with dimension 3, this would mismatch.
        # Ensure consistency or re-initialize the table with the correct dimension.
        # For a robust test, we should ensure the table matches MODEL_NAME's dimension.
        # MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2" has 384 dimensions.
        print("Initializing vector store...")
        # Use a consistent collection name and dimension
        # The dimension should match the output of `embeddings.embed_query("test")`
        # For "sentence-transformers/all-MiniLM-L6-v2", it's 384.
        test_collection_name = "retriever_test_docs"
        embedding_dim = len(embeddings.embed_query("test query")) # Get actual dimension

        vector_db = PGVectorStore(collection_name=test_collection_name, embedding_dimension=embedding_dim)
        print(f"Vector store for '{test_collection_name}' (dim: {embedding_dim}) initialized.")

        # 3. Add some dummy data to the vector store for testing
        print("Adding dummy documents to vector store for retriever test...")
        sample_docs_content = [
            "The impact of fiscal policy on economic growth is a complex topic.",
            "Monetary policy decisions are made by the central bank.",
            "Recent election results show a shift in voter preferences towards environmental issues.",
            "International trade agreements can significantly affect domestic industries.",
            "The role of technology in modern political campaigns has grown substantially."
        ]

        docs_to_add = []
        for i, content in enumerate(sample_docs_content):
            embedding_vector = embeddings.embed_query(content) # Use query embedding for consistency here
            docs_to_add.append({
                "content": content,
                "embedding": embedding_vector,
                "metadata": {"source": f"dummy_source_{i+1}.txt", "doc_id": f"doc_{i+1}"}
            })

        vector_db.add_documents(docs_to_add)
        print(f"{len(docs_to_add)} dummy documents added.")

        # 4. Initialize Retriever
        retriever_instance = Retriever(vector_store=vector_db, embedding_model=embeddings)
        print("Retriever initialized.")

        # 5. Perform a retrieval
        test_query = "What are the effects of economic policies?"
        print(f"\nRetrieving documents for query: '{test_query}'")
        retrieved_documents = retriever_instance.retrieve(query=test_query, k=3)

        if retrieved_documents:
            print("\nRetrieved documents:")
            for i, doc in enumerate(retrieved_documents):
                print(f"  Rank {i+1}:")
                print(f"    Content: {doc['content'][:100]}...")
                print(f"    Metadata: {doc['metadata']}")
                # The score is now part of metadata: doc['metadata']['retrieval_score']
                print(f"    Retrieval Score: {doc['metadata'].get('retrieval_score', 'N/A'):.4f}")
        else:
            print("No documents retrieved for the query.")

    except Exception as e:
        print(f"An error occurred during retriever test: {e}")
        print("Ensure PostgreSQL/PGVector is running and accessible, and .env is configured.")
        print("You might need to create the 'retriever_test_docs' table or adjust connection settings.")

    # Optional: Clean up the test table (manual or via a script)
    # conn = vector_db.get_db_connection()
    # cur = conn.cursor()
    # cur.execute(f"DROP TABLE IF EXISTS {test_collection_name};")
    # conn.commit()
    # cur.close()
    # conn.close()
    # print(f"Test table '{test_collection_name}' dropped.")
