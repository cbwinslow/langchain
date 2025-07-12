# app/core/ingestion.py
import os
from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings # Using SentenceTransformer directly is also fine

# Placeholder for actual SentenceTransformer model loading
# For now, using HuggingFaceEmbeddings wrapper which uses sentence-transformers
import spacy
from langchain_core.documents import Document as LangchainDocument

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2" # A common default
embeddings_model = None
spacy_nlp = None

def get_spacy_model():
    """Loads the spaCy model."""
    global spacy_nlp
    if spacy_nlp is None:
        try:
            spacy_nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Downloading en_core_web_sm spaCy model...")
            spacy.cli.download("en_core_web_sm")
            spacy_nlp = spacy.load("en_core_web_sm")
    return spacy_nlp

def get_embeddings_model():
    global embeddings_model
    if embeddings_model is None:
        embeddings_model = HuggingFaceEmbeddings(
            model_name=MODEL_NAME,
            model_kwargs={'device': 'cpu'}, # Explicitly use CPU if GPU is not available/needed
            encode_kwargs={'normalize_embeddings': False} # Normalization can be done if required by distance metric
        )
    return embeddings_model

def extract_spacy_features(text: str, nlp_model: Any) -> Dict[str, Any]:
    """
    Extracts features like entities and noun chunks from text using spaCy.
    :param text: The text to process.
    :param nlp_model: An initialized spaCy model instance.
    """
    if not nlp_model: # Should not happen if initialized correctly
        return {"entities": [], "noun_chunks": [], "keywords": []}

    doc = nlp_model(text)

    entities = []
    for ent in doc.ents:
        entities.append({"text": ent.text, "label": ent.label_, "start": ent.start_char, "end": ent.end_char})

    noun_chunks = [chunk.text for chunk in doc.noun_chunks]

    # Example: Extracting keywords based on POS tagging (e.g., proper nouns, nouns)
    keywords = [token.lemma_ for token in doc if token.pos_ in ("PROPN", "NOUN") and not token.is_stop]

    return {
        "entities": entities,
        "noun_chunks": noun_chunks,
        "keywords": list(set(keywords)) # Unique keywords
    }

def load_document(file_path: str) -> List[LangchainDocument]:
    """
    Loads a document based on its extension.
    Supported: .txt, .pdf
    """
    _, ext = os.path.splitext(file_path)
    docs_data = []

    if ext.lower() == ".txt":
        loader = TextLoader(file_path, encoding="utf-8")
        docs_data = loader.load()
    elif ext.lower() == ".pdf":
        loader = PyPDFLoader(file_path)
        docs_data = loader.load_and_split() # PyPDFLoader can do basic splitting
    # TODO: Add .docx support using python-docx
    # from docx import Document
    # if ext.lower() == ".docx":
    #     document = Document(file_path)
    #     full_text = "\n".join([para.text for para in document.paragraphs])
    #     # For DOCX, we manually create a Document object if Langchain's loader is not used
    #     # from langchain_core.documents import Document as LangchainDocument
    #     # docs_data = [LangchainDocument(page_content=full_text, metadata={"source": file_path})]
    #     pass # Placeholder for DOCX
    else:
        print(f"Unsupported file type: {ext}")
        return []

    # Convert Langchain Document objects to a simpler dict structure for now if needed
    # Or keep them as Document objects if the rest of the pipeline uses them.
    # For now, assuming they are Langchain Document objects.
    return docs_data


def chunk_documents(documents: List[Any], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Any]:
    """
    Splits documents into smaller chunks.
    Accepts Langchain Document objects.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    split_docs = text_splitter.split_documents(documents)

    # Enhance metadata with spaCy features
    enriched_chunks = []
    for chunk_doc in split_docs:
        spacy_features = extract_spacy_features(chunk_doc.page_content)
        # Merge spaCy features into existing metadata
        # Ensure metadata exists
        if chunk_doc.metadata is None:
            chunk_doc.metadata = {}
        chunk_doc.metadata.update(spacy_features) # Add entities, noun_chunks, keywords
        enriched_chunks.append(chunk_doc)

    return enriched_chunks


def generate_embeddings_for_chunks(chunks: List[Any]) -> List[List[float]]:
    """
    Generates embeddings for a list of text chunks.
    Accepts Langchain Document objects, extracts their page_content.
    """
    model = get_embeddings_model()
    chunk_texts = [chunk.page_content for chunk in chunks]
    embeddings = model.embed_documents(chunk_texts)
    return embeddings


async def process_and_store_document(file_path: str, vector_store_client: Any):
    """
    Main ingestion workflow for a single document.
    """
    print(f"Processing document: {file_path}")
    documents = load_document(file_path)
    if not documents:
        print(f"No documents loaded from {file_path}")
        return

    chunks = chunk_documents(documents)
    if not chunks:
        print(f"No chunks created for {file_path}")
        return

    chunk_embeddings = generate_embeddings_for_chunks(chunks)
    print(f"Generated {len(chunk_embeddings)} embeddings for {len(chunks)} chunks.")

    # Store chunks and their embeddings in the vector store
    # Assuming vector_store_client has an `add_documents` or similar method
    # This part will heavily depend on the vector_store.py implementation

    # For pgvector, this would involve constructing SQL queries or using an ORM
    # Example:
    # for i, chunk in enumerate(chunks):
    #     vector_store_client.add_item(
    #         content=chunk.page_content,
    #         embedding=chunk_embeddings[i],
    #         metadata=chunk.metadata
    #     )
    # print(f"Stored {len(chunks)} chunks in vector store.")

    # For Langchain's PGVector wrapper:
    # from langchain_community.vectorstores.pgvector import PGVector
    # PGVector.from_documents(
    # documents=chunks, # Langchain PGVector expects Document objects
    # embedding=get_embeddings_model(),
    # connection_string=YOUR_DB_CONNECTION_STRING, # This should come from config
    # collection_name="political_documents" # Example collection name
    # )

    # This function will be more fleshed out when vector_store.py is defined.
    # For now, we'll simulate this part.

    # Simulate storing by printing what would be stored
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i+1} Metadata: {chunk.metadata}, Content Preview: {chunk.page_content[:100]}...")
        # In a real scenario: vector_store_client.add(text=chunk.page_content, embedding=chunk_embeddings[i], metadata=chunk.metadata)

    print(f"Document {file_path} processed and 'stored' (simulated).")


if __name__ == '__main__':
    # Example Usage (requires a vector store client and a sample file)
    # This is for testing purposes; actual calls will be from an API or orchestrator.

    # Create a dummy file for testing
    sample_txt_path = "sample_document.txt"
    with open(sample_txt_path, "w") as f:
        f.write("This is a sample document about politics.\nIt contains several sentences and topics.\nLangchain and sentence transformers are cool.")

    # Simulate a vector store client
    class DummyVectorStoreClient:
        def __init__(self):
            self.items = []
        def add_item(self, content, embedding, metadata):
            self.items.append({"content": content, "embedding": "dummy_embedding_vector", "metadata": metadata})
            print(f"DummyVectorStore: Added item with metadata {metadata}")

    dummy_client = DummyVectorStoreClient()

    import asyncio
    asyncio.run(process_and_store_document(sample_txt_path, dummy_client))

    # Clean up dummy file
    os.remove(sample_txt_path)
