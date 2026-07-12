import logging
import re
import typing as ty

from ..models import AgentState, ErrorRecord
from ..llm_clients import llm_large, client_embedding_model_server, create_compatible_chain
from ..vector_store import add_error_logs
from ..configs import settings

logger = logging.getLogger(__name__)

def __extract_xml_errors_node_archivist(text: str, is_skip_1st_error_tag: bool = True):
    """
    Parses multiple <error> blocks from the text.
    Returns a list of dicts: [{'rule': '...', 'phrase': '...', ...}, {...}]
    """
    errors = []
    
    no_errors_tag = re.findall(r"<no_errors/>", text, re.DOTALL)
    if len(no_errors_tag) > 1:
        return []

    error_blocks = re.findall(r"<error>(.*?)</error>", text, re.DOTALL)
    if is_skip_1st_error_tag and len(error_blocks) == 1:
        return []

    for _i_error, block in enumerate(error_blocks):
        if _i_error == 0 and is_skip_1st_error_tag:
            continue

        rule = re.search(r"<rule>(.*?)</rule>", block, re.DOTALL)
        phrase = re.search(r"<phrase>(.*?)</phrase>", block, re.DOTALL)
        correction = re.search(r"<correction>(.*?)</correction>", block, re.DOTALL)
        category = re.search(r"<category>(.*?)</category>", block, re.DOTALL)
        
        if rule and correction:
            errors.append({
                "error_rule": rule.group(1).strip(),
                "example_phrase": phrase.group(1).strip() if phrase else "",
                "correction": correction.group(1).strip(),
                "category": category.group(1).strip() if category else "None"
            })
            
    return errors


def node_archivist(state: AgentState) -> ty.Dict:
    """Node 3: Archivist"""
    logger.info("--- Node: Archive ---")
    tast_config = state["config_archivist"]

    if not state["is_processor_success"]:
        return {}

    lang_diary_body = state['lang_diary_body']

    template = [
        ("system", f"You are a strict language grammarian of {lang_diary_body}." ),
        ("user", (
                "Task: Identify ALL grammatical, vocabulary, or spelling errors in the user's draft.\n"
                "For EACH error, output an XML block exactly like this:\n\n"
                "<error>\n"
                "  <rule>The specific rule violated</rule>\n"
                "  <phrase>The incorrect phrase from text</phrase>\n"
                "  <correction>The corrected phrase</correction>\n"
                "  <category>Grammar OR Vocabulary OR Spelling</category>\n"
                "</error>\n\n"
                "If there are no errors, output: <no_errors/>\n\n"
                f"Draft: {state['final_response']}")
            )
    ]

    chain = create_compatible_chain(template, llm_large.bind(max_tokens=tast_config.max_tokens, 
                                                                     enable_thinking=tast_config.enable_thinking))
    response = chain.invoke({})

    error_list = __extract_xml_errors_node_archivist(response.content)

    __error_list_obj = []
    for err in error_list:
        err['primary_id_DiaryEntry'] = state["primary_id_DiaryEntry"]
        err['language_diary_text'] = state['lang_diary_body']
        err['language_annotation_text'] = state['lang_annotation']
        err['model_id_embedding'] = settings.MODEL_NAME_Embedding
        try:
            record = ErrorRecord(**err)
            __error_list_obj.append(record)
            logger.debug(f"Grammatical-Error: {record}")
        except Exception as e:
            logger.error(e)

    error_list_obj = []
    for err in __error_list_obj:
        if err.example_phrase == err.correction:
            continue
        error_list_obj.append(err)

    if len(error_list_obj) > 0:
        logger.debug(f"Found {len(error_list)} errors.")
        add_error_logs(error_list_obj, client_embedding_model_server)
    else:
        logger.debug("No errors found.")
    
    return {
        "grammatical_errors_extracted": error_list, 
    }
