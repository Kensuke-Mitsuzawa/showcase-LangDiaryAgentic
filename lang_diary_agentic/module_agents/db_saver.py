import logging
import duckdb

from ..models import (
    AgentState, 
    DiaryEntry, 
    UnknownExpressionEntry, 
    TranslationReplacementInformation,
    PhraseRewritingEntry,
    HistoryRecord
)
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
    # end for

    seq_phrase_rewriting_entry = []
    seq_phrase_rewriting = processed.phrases_rewritten or []
    for _phrase_rewriting in seq_phrase_rewriting:
        _phrase_rewriting_entry = PhraseRewritingEntry(
            expression_source=_phrase_rewriting.phrase_target,
            expression_rewritten=_phrase_rewriting.phrase_rewritten,
            remark_field=_phrase_rewriting.explanation,
            language_source=language_source,
            language_annotation=language_annotation,
            created_at=created_at,
            primary_id_DiaryEntry=diary_entry_primary_key,
            primary_id=None
        )
        seq_phrase_rewriting_entry.append(_phrase_rewriting_entry)
    # end for
    
    assert settings.GENERATION_DB_PATH is not None
    handler = HandlerDairyDB(settings.GENERATION_DB_PATH)
    handler.init_table_diary()

    conn = duckdb.connect(settings.GENERATION_DB_PATH)
    conn.execute("UPDATE diary_entries SET diary_replaced = ?, diary_rewritten = ?, evaluation_current_level = ? WHERE primary_id = ?", (
        processed.diary_replaced,
        processed.diary_rewritten,
        processed.evaluation_current_level,
        diary_entry_primary_key
    ))
    conn.commit()
    conn.close()

    for _entry in seq_unknown_expression_entry:
        handler.save_unknown_expression(_entry)
    # end for

    for _entry in seq_phrase_rewriting_entry:
        handler.save_phrase_rewriting(_entry)
    # end for

    return state    
