import logging
import re
import typing as ty

from ..models import AgentState
from ..static import Iso693_code2natural_name
from ..llm_clients import llm_large, create_compatible_chain
from ..utils import check_language

logger = logging.getLogger(__name__)

PossibleReturnRoutineNodeRewriter = ty.Literal['success', 'insufficient_length', 'incorrect_language', 'xml_error']

def _func_routine_node_rewriter(prompt_content: str, state: AgentState) -> ty.Tuple[str, str, str]:
    """
    Return: (rewritten-text, full-response, error-code).
    """
    tast_config = state["config_rewriter"]

    lang_code_diary = state['lang_diary_body']
    lang_name_natural_lan: str = Iso693_code2natural_name[lang_code_diary]

    template = [
        ("system", f"You are an expert {lang_name_natural_lan} editor."),
        ("user", prompt_content)
    ]
    chain = create_compatible_chain(template, llm_large.bind(
        max_length=tast_config.max_tokens, 
        enable_thinking=tast_config.enable_thinking))

    response = chain.invoke({
        "user_text": state['final_response'], 
        "target_lang": lang_name_natural_lan, 
        "level_rewriting": state['level_rewriting']
    })

    logger.info(f"Rewriter response: {response}")
    logger.debug(f"dialy-lang={state['lang_diary_body']}. Level-rewiritng={state['level_rewriting']}")
    response_text: str = response.content

    group_replaced = re.findall(r'<rewriting>(.*?)</rewriting>', response_text, re.DOTALL)
        
    if group_replaced == []:
        logger.warning(f"Regex error. Return the full response. Response={response_text}")
        return response_text, response_text, 'xml_error'
    elif response.response_metadata.get('finish_reason') == "length":
        return response_text, response_text, 'insufficient_length'
    else:
        text_rewriting = group_replaced[-1]
        _detected_language = check_language.detect_language(text_rewriting)

        if _detected_language != lang_code_diary:
            logger.warning(f"Unmatched Language code. Expected code={lang_code_diary}, Rewriting-text={_detected_language}. Retry.")
            return response_text, response_text, 'incorrect_language'
        else:
            text_rewriting = text_rewriting.replace('[', '').replace(']', '')
            return text_rewriting, response_text, 'success'


def node_rewriter(state: AgentState, max_try: int = 5, default_max_length: int = 512) -> ty.Dict:
    """Node: Rewritting"""
    logger.info("--- Node: Rewriting ---")
    task_config = state["config_rewriter"]

    if not state["is_processor_success"]:
        return {
            "suggestion_response": ""
        }
    if task_config.is_execute is False:
        logger.info("SKip the tast since is_execute = False.")
        return {
            "suggestion_response": ""
        }
    
    lang_code_diary = state['lang_diary_body']

    prompt_content = (
        "Task:\n"
        "1. Rewrite the following text (in the {level_rewriting} level of the CEFR). The rewriting language must stick with language={target_lang}\n"
        "IMPORTANT: Return the result ONLY as XML in the following structure:\n"
        "<rewriting>rewritten text</rewriting>\n\n"
        "Input: {user_text}"
    )

    _current_try = 0
    _validation_status = False
    while True:
        if _current_try == max_try:
            break
        if _validation_status is True:
            break

        _response_rewriting, _full_response, _flag_error = _func_routine_node_rewriter(prompt_content, state)

        if _flag_error == 'xml_error':
            logger.warning("failed to extract XML. retry.")
            _msg_addition = "IMPORTANT: Return the result ONLY as XML in the following structure:\n<rewriting>rewritten text</rewriting>\n"
            prompt_content += _msg_addition
            _current_try += 1
            continue

        if _flag_error == 'insufficient_length':
            default_max_length += 100

        if _flag_error == 'incorrect_language':
            _msg_addition = "IMPORTANT: Rewriting language must be {target_lang}. Rewrite the input text to match the {level_rewriting} CEFR level."
            _current_try += 1
            continue

        _validation_status = True

    return {
        "suggestion_response": _response_rewriting
    }
