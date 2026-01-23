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
                primary_id_DiaryEntry VARCHAR,
                created_at TIMESTAMP,
                expression VARCHAR,
                expression_translation VARCHAR,
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
                diary_rewritten TEXT,
                level_rewriting VARCHAR,
                model_id_tutor VARCHAR,
                title_diary VARCHAR
            );
        """)
        conn.close()

    def save_unknown_expression(self, entry_unknown: UnknownExpressionEntry):
        """Insert a new record."""
        conn = duckdb.connect(self.db_path)

        query = """
        INSERT INTO unknown_expressions VALUES (
            ?, ?, ?, ?, ?, ?, ?
        )
        """

        conn.execute(query, (
            entry_unknown.primary_id, # primary_id
            entry_unknown.primary_id_DiaryEntry, # primary_id_DiaryEntry
            entry_unknown.created_at, # created_at
            entry_unknown.expression, # expression
            entry_unknown.expression_translation,
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
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
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
            entry_diary.diary_rewritten, # diary_corrected
            entry_diary.level_rewriting,
            entry_diary.model_id_tutor,
            entry_diary.title_diary
        ))
        
        conn.commit()
        conn.close()
        logger.info("✅ Diary entry saved to DuckDB.")
    # end def

    # ----- fetch -----
    def fetch_dairy_entry_language(self,
                                   language_daiary_body: ty.Optional[str] = None,
                                   language_annotation: ty.Optional[str] = None,
                                   daiary_primary_key: ty.Optional[str] = None
                                   ) -> ty.Optional[ty.List[DiaryEntry]]:
        conn = duckdb.connect(self.db_path, read_only=True)
        query_base = "SELECT * FROM diary_entries"
        query_vars = []
        where_clause = []

        if language_daiary_body is not None:
            query_vars.append(language_daiary_body)
            where_clause.append("language_source = ?")
        if language_annotation is not None:
            query_vars.append(language_annotation)
            where_clause.append("language_annotation = ?")
        if daiary_primary_key is not None:
            query_vars.append(daiary_primary_key)
            where_clause.append("primary_id = ?")
        
        query_final = query_base + " WHERE " + " AND ".join(where_clause) 
        # end if

        if len(query_vars) == 0:
            seq_result = conn.execute(query_base).fetchall()
        else:
            seq_result = conn.execute(query_final, tuple(query_vars)).fetchall()
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
                diary_rewritten=_entry[7],
                level_rewriting=_entry[8],
                model_id_tutor=_entry[9],
                title_diary=_entry[10]
            )
            stack.append(_entry)
        # end for

        return stack
    # end def

    def fetch_unknown_expression(self, daiary_primary_key: ty.Optional[str] = None) -> ty.Optional[ty.List[UnknownExpressionEntry]]:
        conn = duckdb.connect(self.db_path, read_only=True)
        query_base = "SELECT * FROM unknown_expressions"
        query_vars = []
        where_clause = []

        if daiary_primary_key is not None:
            query_vars.append(daiary_primary_key)
            where_clause.append("primary_id_DiaryEntry = ?")
        

        query_final = query_base + " WHERE " + " AND ".join(where_clause) 
        # end if
        seq_result = conn.execute(query_final, tuple(query_vars)).fetchall()
        conn.close()

        if seq_result is None:
            return None
        # end for

        stack = []
        for _entry in seq_result:

            _entry = UnknownExpressionEntry(
                primary_id=_entry[0],
                primary_id_DiaryEntry=_entry[1],
                created_at=_entry[2],
                expression=_entry[3],
                expression_translation=_entry[4],
                language_source=_entry[5],
                language_annotation=_entry[6]
            )
            stack.append(_entry)
        # end for

        return stack
 


