import os
import logging
from typing import List

from langchain_chroma import Chroma

from .llm_custom_api_wrapper import RemoteServerEmbeddings
from .models.vector_store_entry import ErrorRecord
from .configs import settings

logger = logging.getLogger(__name__)


def get_vector_store(client_embedding_model_server: RemoteServerEmbeddings):
    # Uses a free, local model (runs fast on CPU)
    # 'all-MiniLM-L6-v2' is the industry standard for lightweight embeddings
    vector_store = Chroma(
        persist_directory=settings.ErrorVectorDB_PATH,
        embedding_function=client_embedding_model_server,
        collection_name="lingualog_errors"
    )
    return vector_store


def add_error_logs(records: List[ErrorRecord],
                   client_embedding_model_server: RemoteServerEmbeddings):
    """
    Save a batch of error logs to memory.
    Efficiently inserts multiple records in one DB transaction.
    """
    if not records:
        return

    db = get_vector_store(client_embedding_model_server)
    
    # Prepare lists for batch insertion
    batch_texts = []
    batch_metadatas = []
    
    for record in records:
        # 1. Content: The text to be embedded
        record_string = record.to_string()

        batch_texts.append(record_string)
        
        # 2. Metadata: Structured data for filtering
        batch_metadatas.append({
            "category": record.category,
            "error_rule": record.error_rule,
            "correction": record.correction,
            "example_phrase": record.example_phrase,
            "primary_id_DiaryEntry": record.primary_id_DiaryEntry,
            "language_diary_text": record.language_diary_text,
            "language_annotation_text": record.language_annotation_text,\
            "model_id_embedding": record.model_id_embedding
        })
    # end for
    
    # 3. Batch Insert (One call to the DB)
    logger.info(f"💾 Saving {len(batch_texts)} errors to ChromaDB...")

    db.add_texts(
        texts=batch_texts,
        metadatas=batch_metadatas,
)    
    # In newer Chroma versions, persist is automatic, but keeping it is safe
    if hasattr(db, "persist"):
        db.persist()
    # end if
# end def


def query_past_errors(query_text: str,
                      lang_annotation: str,
                      lang_diary_body: str,
                      model_id_embedding: str, 
                      client_embedding_model_server: RemoteServerEmbeddings,
                      k: int = 10):
    """
    Args:
        query_text: Grammatical Error Information.
    """
    db = get_vector_store(client_embedding_model_server)
    
    filter_dict={
        "$and": [
            {"language_annotation_text": {"$eq": lang_annotation}},
            {"language_diary_text": {"$eq": lang_diary_body}},
            {"model_id_embedding": {"$eq": model_id_embedding}}
        ]
    }
    results = db.similarity_search(query_text, filter=filter_dict, k=k)
    
    return [doc.page_content for doc in results]
