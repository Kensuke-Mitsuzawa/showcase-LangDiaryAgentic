import logging
import duckdb

from ..models import AgentState, DiaryEntry, UnknownExpressionEntry, TranslationReplacementInformation
from ..db_handler import HandlerDairyDB
from ..configs import SettingsVariables

logger = logging.getLogger(__name__)

def node_save_duckdb(state: AgentState, settings: SettingsVariables):
    """New Node: Save everything to DuckDB"""
    logger.info("--- [4] Saving to DuckDB ---")
    
    # Use today's date if not provided
    diary_date = state["diary_date"]
    created_at = state["created_at"]

    language_source = state.get("lang_diary_body", "Unknown")
    language_source = "Unknown" if language_source is None else language_source

    language_annotation = state.get("lang_annotation", "Unknown")
    language_annotation = "Unknown" if language_annotation is None else language_annotation

    diary_entry: DiaryEntry = state["diary_entry"]
    
    diary_entry_primary_key = diary_entry.primary_id
    assert diary_entry_primary_key is not None

    seq_unknown_expression_entry = []
    seq_bracket_text = state['translation_pair_extracted']
    _d_expression: TranslationReplacementInformation
    for _d_expression in seq_bracket_text:
        _unknown_expression_entry = UnknownExpressionEntry(
            expression=_d_expression.expression_original,
            expression_translation=_d_expression.expression_translation,
            span_original=_d_expression.span_original,
            span_translation=_d_expression.span_translation,
            language_source=language_source,
            language_annotation=language_annotation,
            created_at=created_at,
            primary_id_DiaryEntry=diary_entry_primary_key,
            primary_id=None
        )
        seq_unknown_expression_entry.append(_unknown_expression_entry)
    
    assert settings.GENERATION_DB_PATH is not None
    handler = HandlerDairyDB(settings.GENERATION_DB_PATH)

    conn = duckdb.connect(settings.GENERATION_DB_PATH)
    conn.execute("UPDATE diary_entries SET diary_replaced = ?, diary_rewritten = ? WHERE primary_id = ?", (
        state["final_response"],
        state["suggestion_response"],
        diary_entry.primary_id
    ))

    for _entry in seq_unknown_expression_entry:
        handler.save_unknown_expression(_entry)

    return {}    
