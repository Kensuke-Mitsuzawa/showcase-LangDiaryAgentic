import logging
import duckdb

from ..models import AgentState, DiaryEntry, UnknownExpressionEntry, TranslationReplacementInformation
from ..db_handler import HandlerDairyDB
from ..configs import SettingsVariables

logger = logging.getLogger(__name__)

def node_save_duckdb(state: AgentState, settings: SettingsVariables) -> AgentState:
    """New Node: Save everything to DuckDB"""
    logger.info("--- [4] Saving to DuckDB ---")
    
    diary_entry = state.diary_entry_input
    diary_entry_primary_key = diary_entry.primary_id
    assert diary_entry_primary_key is not None

    created_at = diary_entry.created_at
    language_source = diary_entry.language_source or "Unknown"
    language_annotation = diary_entry.language_annotation or "Unknown"

    processed = state.processed_output
    if processed is None:
        logger.warning("No processed output to save.")
        return state

    seq_unknown_expression_entry = []
    seq_bracket_text = processed.translation_pair_extracted or []
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
        processed.diary_replaced,
        processed.diary_rewritten,
        diary_entry_primary_key
    ))
    conn.commit()
    conn.close()

    for _entry in seq_unknown_expression_entry:
        handler.save_unknown_expression(_entry)

    return state    
