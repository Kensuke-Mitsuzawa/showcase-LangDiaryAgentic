import typing as ty
import logging
import duckdb

from pydantic import BaseModel, Field
from datetime import datetime

from .logging_configs import apply_logging_suppressions

apply_logging_suppressions()

logger = logging.getLogger(__name__)




class DiaryEntry(BaseModel):
    date_diary: str
    language_source: str
    language_annotation: str
    diary_original: str
    diary_replaced: str
    diary_corrected: str
    created_at: datetime = Field(default_factory=datetime.now)    
    primary_id: ty.Optional[str] = None

    def model_post_init(self, context: ty.Any) -> None:
        if self.primary_id is None:
            datetime_str = self.created_at.isoformat()
            self.primary_id = f"{self.date_diary}_{datetime_str}"
        # end if
# end class


class UnknownExpressionEntry(BaseModel):
    expression: str
    language_source: str
    language_annotation: str
    created_at: datetime = Field(default_factory=datetime.now)    
    primary_id: ty.Optional[str] = None

    def model_post_init(self, context: ty.Any) -> None:
        if self.primary_id is None:
            datetime_str = self.created_at.isoformat()
            self.primary_id = f"{self.expression}_{datetime_str}"
        # end if
# end class


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


