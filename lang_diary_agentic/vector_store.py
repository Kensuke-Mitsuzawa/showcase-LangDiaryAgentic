import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

from .models.vector_store_entry import ErrorRecord


PERSIST_DIRECTORY = "./data/chroma_db"

def get_vector_store():
    # Uses a free, local model (runs fast on CPU)
    # 'all-MiniLM-L6-v2' is the industry standard for lightweight embeddings
    embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    vector_store = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embedding_function,
        collection_name="lingualog_errors"
    )
    return vector_store

def add_error_log(record: ErrorRecord):
    db = get_vector_store()
    
    # 1. Content: Used for similarity search (Embeddings)
    # We use the helper we defined in schemas.py
    text_content = record.to_string()
    
    # 2. Metadata: specific fields you can filter on later
    metadata = {
        "category": record.category,
        "rule": record.error_rule,
        "correction": record.correction
    }
    
    # Add to Chroma
    db.add_texts(
        texts=[text_content],
        metadatas=[metadata]
    )
    db.persist()


def query_past_errors(query_text: str, k: int = 3):
    db = get_vector_store()
    results = db.similarity_search(query_text, k=k)
    return [doc.page_content for doc in results]
