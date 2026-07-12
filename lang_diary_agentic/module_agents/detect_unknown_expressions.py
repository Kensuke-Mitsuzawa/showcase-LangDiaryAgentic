import logging
import re
from ..models import AgentState
from ..static import Languages_Code
from ..utils import check_language

from ..models.states import ProcessedOutputInformation

logger = logging.getLogger(__name__)

def node_detect_unknown_expressions(state: AgentState) -> AgentState:
    """
    Task: Identifying unknown expressions embedded in the draft text.
    LLM Used: SMALL (Fast)
    """
    logger.info("--- Detecting unknown expressions ---")
    
    # draft_text = state["draft_text"]
    draft_text = state.diary_entry_input.diary_original
    seq_text_blanket = [x.group() for x in re.finditer(r'\[[^]]+\]', draft_text)]    
    
    # If user already provided them via UI, skip detection
    language_diary = state.diary_entry_input.language_source.strip()
    language_annotation = state.diary_entry_input.language_annotation.strip()

    assert language_diary in Languages_Code, f"The language code {language_diary} is not valid. Check the language code in ISO 693-3."
    assert language_annotation in Languages_Code, f"The language code {language_annotation} is not valid. Check the language code in ISO 693-3."

    processed_output_info = ProcessedOutputInformation(
        diary_replaced="",
        diary_rewritten="",
        total_review="",
        unkown_expressions=seq_text_blanket,
        translation_pair_extracted=[],
        retrieved_context="",
        grammatical_errors_extracted=[]
    )

    state.processed_output = processed_output_info

    # Update and return the sub-models using the top-level keys of AgentState
    return state.model_copy(update={
        "diary_entry_input": state.diary_entry_input.model_copy(update={
            "lang_annotation": language_annotation,
            "lang_diary_body": language_diary,
        }),
        "processed_output": state.processed_output
    })

