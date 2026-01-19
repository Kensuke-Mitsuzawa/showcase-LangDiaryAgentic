import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

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

def add_error_log(error_text: str):
    db = get_vector_store()
    db.add_texts([error_text])
    db.persist()

def query_past_errors(query_text: str, k=3):
    db = get_vector_store()
    results = db.similarity_search(query_text, k=k)
    return [doc.page_content for doc in results]
