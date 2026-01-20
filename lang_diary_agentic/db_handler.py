import typing as ty
import logging
import duckdb
import datetime

from .models.generation_records import DiaryEntry, UnknownExpressionEntry
from .logging_configs import apply_logging_suppressions

apply_logging_suppressions()

logger = logging.getLogger(__name__)



class HandlerDairyDB():
    def __init__(self, db_path: str):
        self.db_path = db_path

    def init_db(self):
        self.init_table_diary()
        self.init_table_unknown_expressions()

    def init_table_unknown_expressions(self):
        """Create the table if it doesn't exist."""
        conn = duckdb.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS unknown_expressions (
                primary_id VARCHAR PRIMARY KEY,
                created_at TIMESTAMP,
                expression VARCHAR,
                language_source VARCHAR,
                language_annotation VARCHAR
            );
        """)
        conn.close()

    def init_table_diary(self):
        """Create the table if it doesn't exist."""
        conn = duckdb.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS diary_entries (
                primary_id VARCHAR PRIMARY KEY,
                created_at TIMESTAMP,
                date_diary DATE,
                language_source VARCHAR,
                language_annotation VARCHAR,
                diary_original TEXT,
                diary_replaced TEXT,
                diary_corrected TEXT
            );
        """)
        conn.close()

    def save_unknown_expression(self, entry_unknown: UnknownExpressionEntry):
        """Insert a new record."""
        conn = duckdb.connect(self.db_path)

        query = """
        INSERT INTO unknown_expressions VALUES (
            ?, ?, ?, ?, ?
        )
        """

        conn.execute(query, (
            entry_unknown.primary_id, # primary_id
            entry_unknown.created_at, # created_at
            entry_unknown.expression, # expression
            entry_unknown.language_source, # language_source
            entry_unknown.language_annotation # language_annotation
        ))

        conn.commit()
        conn.close()
        logger.info("✅ Unknown expression saved to DuckDB.")
    # end def

    def save_diary_entry(self, entry_diary: DiaryEntry):
        """Insert a new record."""
        conn = duckdb.connect(self.db_path)
        
        # We use a parameterized query for security (prevents SQL injection)
        query = """
        INSERT INTO diary_entries VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?
        )
        """
            
        conn.execute(query, (
            entry_diary.primary_id, # primary_id
            entry_diary.created_at, # created_at
            entry_diary.date_diary, # date_diary
            entry_diary.language_source, # language_source
            entry_diary.language_annotation, # language_annotation
            entry_diary.diary_original, # diary_original
            entry_diary.diary_replaced, # diary_replaced
            entry_diary.diary_corrected # diary_corrected
        ))
        
        conn.commit()
        conn.close()
        logger.info("✅ Diary entry saved to DuckDB.")
    # end def

    # ----- fetch -----
    def fetch_dairy_entry_language(self,  
                                   language_daiary_body: str,
                                   language_annotation: ty.Optional[str] = None
                                   ) -> ty.Optional[ty.List[DiaryEntry]]:
        conn = duckdb.connect(self.db_path)
        if language_annotation is None:
            query = "SELECT * FROM diary_entries WHERE language_source = ?"
            seq_result = conn.execute(query, (language_daiary_body,)).fetchall()
        else:
            query = "SELECT * FROM diary_entries WHERE language_source = ? AND language_annotation = ?"
            seq_result = conn.execute(query, (language_daiary_body, language_annotation)).fetchall()
        # end if
        conn.close()

        if seq_result is None:
            return None
        # end for

        stack = []
        for _entry in seq_result:
            if isinstance(_entry[1], datetime.datetime):
                date_diary = _entry[2].isoformat()
            else:
                date_diary = _entry[2]
            # end if

            _entry = DiaryEntry(
                primary_id=_entry[0],
                created_at=_entry[1],
                date_diary=date_diary,
                language_source=_entry[3],
                language_annotation=_entry[4],
                diary_original=_entry[5],
                diary_replaced=_entry[6],
                diary_corrected=_entry[7]
            )
            stack.append(_entry)
        # end for

        return stack
    # end def




