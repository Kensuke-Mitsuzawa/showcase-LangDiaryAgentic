import typing as ty
import logging
import duckdb
import datetime
import json
from pathlib import Path

from .models.generation_records import DiaryEntry, UnknownExpressionEntry, HistoryRecord
from .logging_configs import apply_logging_suppressions

apply_logging_suppressions()

logger = logging.getLogger(__name__)



class HandlerDairyDB():
    def __init__(self, db_path: str):
        self.db_path = db_path
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self.init_db()

    def init_db(self):
        self.init_table_diary()
        self.init_table_unknown_expressions()
        self.init_table_history_record()

    def init_table_history_record(self):
        conn = duckdb.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS diary_version_history (
                history_id VARCHAR PRIMARY KEY,
                primary_id_DiaryEntry VARCHAR,
                version_from INTEGER,
                version_to INTEGER,
                created_at TIMESTAMP,
                
                -- UNIFIED COLUMN: Stores a JSON blob
                -- Example content: { "diary_rewritten": "...", "expressions": {...} }
                changes JSON 
            );
        """)
        conn.commit()
        conn.close()

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
                span_original VARCHAR,
                span_translation VARCHAR,
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
                title_diary VARCHAR,
                current_version INTEGER,
                is_show BOOLEAN
            );
        """)
        conn.close()

    # ---- POST or UPDATE methods ----

    def save_diary_version_history(self, history_record: HistoryRecord):
        conn = duckdb.connect(self.db_path)

        d_obj = history_record.model_dump()
        # We use a parameterized query for security (prevents SQL injection)
        _col_names = []
        _values = []
        for _k, _v in d_obj.items():
            _col_names.append(_k)
            _values.append(_v)
        # end for
        place_holder = ', '.join(['?'] * len(_col_names))
        query = f"INSERT INTO diary_version_history ({', '.join(_col_names)}) VALUES ({place_holder})"
        conn.execute(query, tuple(_values))
    
        conn.commit()
        conn.close()

    def save_unknown_expression(self, entry_unknown: UnknownExpressionEntry):
        """Insert a new record."""
        conn = duckdb.connect(self.db_path)

        d_obj = entry_unknown.model_dump()
        # We use a parameterized query for security (prevents SQL injection)
        _col_names = []
        _values = []
        for _k, _v in d_obj.items():
            _col_names.append(_k)
            _values.append(_v)
        # end for
        place_holder = ', '.join(['?'] * len(_col_names))
        query = f"INSERT INTO unknown_expressions ({', '.join(_col_names)}) VALUES ({place_holder})"
        
        logger.info(query)
        conn.execute(query, tuple(_values))

        conn.commit()
        conn.close()
        logger.info("✅ Unknown expression saved to DuckDB.")
    # end def

    def save_diary_entry(self, entry_diary: DiaryEntry):
        """Insert a new record."""
        conn = duckdb.connect(self.db_path)
        
        d_obj = entry_diary.model_dump()
        # We use a parameterized query for security (prevents SQL injection)
        _col_names = []
        _values = []
        for _k, _v in d_obj.items():
            _col_names.append(_k)
            _values.append(_v)
        # end for

        place_holder = ', '.join(['?'] * len(_col_names))
        query = f"INSERT INTO diary_entries ({', '.join(_col_names)}) VALUES ({place_holder})"
        
        conn.execute(query, tuple(_values))
        
        conn.commit()
        conn.close()
        logger.info("✅ Diary entry saved to DuckDB.")
    # end def

    # ----- fetch -----
    def fetch_dairy_entry_language(self,
                                   language_daiary_body: ty.Optional[str] = None,
                                   language_annotation: ty.Optional[str] = None,
                                   daiary_primary_key: ty.Optional[str] = None,
                                   is_show_only: bool = True,
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
        if is_show_only:
            query_vars.append(True)
            where_clause.append("is_show = ?")
        
        query_final = query_base + " WHERE " + " AND ".join(where_clause) 
        # end if

        if len(query_vars) == 0:
            seq_result = conn.execute(query_base).fetchall()
        else:
            seq_result = conn.execute(query_final, tuple(query_vars)).fetchall()
        # end if

        columns = [desc[0] for desc in conn.description]
        dict_results = [dict(zip(columns, row)) for row in seq_result]        
        conn.close()

        if seq_result is None:
            return None
        # end for

        stack = []
        for _entry in dict_results:
            if isinstance(_entry['date_diary'], (datetime.datetime, datetime.date)):
                _entry['date_diary'] = _entry['date_diary'].isoformat()
            else:
                _entry['date_diary'] = _entry['date_diary']
            # end if

            if _entry['current_version'] not in _entry:
                _entry['current_version'] = 0
            # end if
            if _entry['current_version'] is None:
                _entry['current_version'] = 0
            # end if

            _entry = DiaryEntry(**_entry)
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
                
        if len(where_clause) > 0:
            query_final = query_base + " WHERE " + " AND ".join(where_clause) 
        else:
            query_final = query_base
        # end if

        seq_result = conn.execute(query_final, tuple(query_vars)).fetchall()
        columns = [desc[0] for desc in conn.description]
        dict_results = [dict(zip(columns, row)) for row in seq_result]
        conn.close()

        if dict_results is None:
            return None
        # end for

        stack = []
        for _entry in dict_results:
            if isinstance(_entry['span_original'], str):
                _entry['span_original'] = tuple(json.loads(_entry['span_original']))
            if isinstance(_entry['span_translation'], str):
                _entry['span_translation'] = tuple(json.loads(_entry['span_translation']))
            
            _entry = UnknownExpressionEntry(**_entry)
            stack.append(_entry)
        # end for

        return stack
 


