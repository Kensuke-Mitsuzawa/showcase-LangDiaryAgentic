import logging
import re
from ..models import AgentState
from ..static import Languages_Code
from ..utils import check_language

logger = logging.getLogger(__name__)

def node_language_detect(state: AgentState):
    """
    Task: Identify languages if not provided.
    LLM Used: SMALL (Fast)
    """
    logger.info("--- [0] Detecting Language ---")
    
    draft_text = state["draft_text"]
    seq_text_blanket = [x.group() for x in re.finditer(r'\[[^]]+\]', draft_text)]    
    
    # If user already provided them via UI, skip detection
    if state.get("lang_annotation") and state.get("lang_diary_body"):
        _language_diary = state.get("lang_diary_body").strip()  # type: ignore
        _language_annotation = state.get("lang_annotation").strip()  # type: ignore

        assert _language_diary in Languages_Code, f"The language code {_language_diary} is not valid. Check the language code in ISO 693-3."
        assert _language_annotation in Languages_Code, f"The language code {_language_annotation} is not valid. Check the language code in ISO 693-3."

        return {
            "lang_annotation": _language_annotation,
            "lang_diary_body": _language_diary,
            "unkown_expressions": seq_text_blanket
        }
    # end if

    # Otherwise, ask the Small LLM
    logger.info("Missing languages. Asking Small LLM...")

    # Since this part is supposed to be shorter. So, I use the traditional ML model.
    language_annotation = check_language.detect_language(' '.join(seq_text_blanket))

    draft_text_without_blanket = re.sub(r'\[.+\]', '', draft_text)
    language_target = check_language.detect_language(' '.join(seq_text_blanket))

    return {
        "lang_diary_body": language_target,
        "lang_annotation": language_annotation,
        "unkown_expressions": seq_text_blanket
    }
