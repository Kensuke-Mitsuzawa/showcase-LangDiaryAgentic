import logging
import copy
import typing as ty
import pandas as pd
from pydantic import ValidationError

from langchain_chroma import Chroma

from .configs import settings
from .db_handler import HandlerDairyDB
from .models.generation_records import DiaryEntry, UnknownExpressionEntry
from .models.vector_store_entry import ErrorRecord


def fetch_grammatical_errors(
        primary_key_diary_table: str,
        vector_db: Chroma) -> ty.List[ErrorRecord]:

    filter_dict={
        "primary_id_DiaryEntry": { "$in": [primary_key_diary_table] }
    }
    d_record_corrections = vector_db.get(where=filter_dict)
    seq_metadata = d_record_corrections['metadatas']
    seq_documents = d_record_corrections['documents']
    assert len(seq_metadata) == len(seq_documents)
    n_document = len(seq_metadata)

    seq_correction_records = []
    for i in range(n_document):
        try:
            _record = ErrorRecord(**seq_metadata[i])
            seq_correction_records.append(_record)
        except Exception:
            pass
        # end try
    # end for

    return seq_correction_records



# def fetch_records_language(
#         language_daiary_body: str,
#         handler: HandlerDairyDB,
#         vector_db: Chroma,
#         language_annotation: ty.Optional[str] = None, 
#         ) -> ty.Optional[ty.Dict[str, pd.DataFrame]]:
#     """Fetching the records for the viewer."""

#     seq_records_dairy = handler.fetch_dairy_entry_language(language_daiary_body, language_annotation)
#     if seq_records_dairy is None:
#         return None
#     # end if
#     seq_records_diary_dict = [_r.model_dump() for _r in seq_records_dairy]
#     df_records_diary = pd.DataFrame(seq_records_diary_dict)

#     # ---- prepating the records for the correction db ----
#     seq_primary_id = [_r.primary_id for _r in seq_records_dairy]

#     filter_dict={
#         "primary_id_DiaryEntry": { "$in": seq_primary_id }
#     }
#     d_record_corrections = vector_db.get(where=filter_dict)
#     seq_metadata = d_record_corrections['metadatas']
#     seq_documents = d_record_corrections['documents']
#     assert len(seq_metadata) == len(seq_documents)
#     n_document = len(seq_metadata)

#     seq_correction_records = []
#     for i in range(n_document):
#         try:
#             _record = ErrorRecord(**seq_metadata[i])
#             seq_correction_records.append(_record)
#         except Exception:
#             pass
#         # end try
#     # end for
#     df_records_correction = pd.DataFrame([_r.model_dump() for _r in seq_correction_records])
#     df_records_correction = df_records_correction.rename(columns={"primary_id_DiaryEntry": "primary_id"})

#     df_merged = pd.merge(df_records_diary, df_records_correction, on="primary_id")

#     return {
#         "df_diary": df_records_diary, 
#         "df_error": df_records_correction,
#         "df_merged": df_merged
#     }

# # end def