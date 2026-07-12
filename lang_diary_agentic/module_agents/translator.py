import logging
import json
import copy
import re
import typing as ty

from ..models import AgentState, TranslationReplacementInformation
from ..static import Iso693_code2natural_name
from ..llm_clients import llm_large, create_compatible_chain

logger = logging.getLogger(__name__)

def __extract_xml_errors_node_processor(text: str, 
                                        expected_extractions: ty.List[str], 
                                        is_skip_1st_error_tag: bool = True):
    """
    Parses multiple <error> blocks from the text.
    Returns a list of dicts: [{'rule': '...', 'phrase': '...', ...}, {...}]
    """
    errors = []
    
    no_errors_tag = re.findall(r"<no_errors/>", text, re.DOTALL)
    if len(no_errors_tag) > 1:
        return []

    seq_translations = re.findall(r'<bracket>(.*?)</bracket>[\s\n]?+<translation>(.*?)</translation>', text, re.DOTALL)

    if len(expected_extractions) == len(seq_translations):
       is_skip_1st_error_tag = False 

    if is_skip_1st_error_tag and len(seq_translations) == 1:
        return []

    if is_skip_1st_error_tag:
        seq_candaidate = seq_translations[1:]
    else:
        seq_candaidate = seq_translations

    for _i_block, block in enumerate(seq_candaidate):
        if len(block) != 2:
            continue

        errors.append({
            "expression_original": block[0].strip(),
            "expression_translation": block[1]
        })

    return errors


def node_translator(state: AgentState) -> ty.Dict:
    """Node 2: Coach"""
    logger.info("--- Node: translator ---")

    assert state["config_translator"].is_execute is True, "is_execute must be set True."
    tast_config = state["config_translator"]

    is_processor_success = True

    sub_phrase_language_pair: str = ""
    lang_annotation = state["lang_annotation"]
    lang_diary_body = state["lang_diary_body"]
    lang_annotation_natural_name = Iso693_code2natural_name[lang_annotation]
    lang_diary_body_natural_name = Iso693_code2natural_name[lang_diary_body]

    if lang_annotation is None or lang_diary_body is None:
        sub_phrase_language_pair = ""
    else:
        sub_phrase_language_pair = f"The bracketed text is written in {lang_annotation_natural_name}. The translation target language is {lang_diary_body_natural_name}."

    xml_schema = """
        <bracket>[text]</bracket><translation>corresponding translation</translation>
    """

    user_content = (
        "Task:\n"
        f"Translate text in bracketed [text] one by one. {sub_phrase_language_pair}\n"
        "IMPORTANT: Return the result ONLY as XML in the following structure:\n"
        f"{xml_schema}\n\n"
        "INPUT: {unkown_expressions}"
    )

    template = [
        ("system", f"You are a translator from {lang_annotation_natural_name} to {lang_diary_body_natural_name}."),
        ("user", user_content)
    ]

    chain = create_compatible_chain(template, llm_large.bind(max_tokens=tast_config.max_tokens, enable_thinking=tast_config.enable_thinking))
    response = chain.invoke({
        "unkown_expressions": json.dumps(state['unkown_expressions']), 
        "lang_annotation": lang_annotation,
    })

    # Simple cleanup to remove the prompt from the output if the model echos it
    clean_response = response.content.split("<|assistant|>")[-1]
    
    seq_translations = __extract_xml_errors_node_processor(clean_response, expected_extractions=state['unkown_expressions'])

    # ---- replace the bracketed [text] one-by-one ----
    draft_text = copy.deepcopy(state['draft_text'])

    # record position before the replacement
    regex_position_original = []
    for _d_pair in seq_translations:
        _regex_pattern = _d_pair["expression_original"].replace('[', '').replace(']', '').replace(' ', '\\s')
        regex_position_original += [(_d_pair, _o) for _o in re.finditer(f'{_regex_pattern}', draft_text, re.DOTALL)]

    if len(seq_translations) != len(regex_position_original):
        logger.error(f"Fail to extract the XML tag. Check the LLM's response: {clean_response}")
        return {"final_response": "", "is_processor_success": False, "translation_pair_extracted": []}

    assert len(seq_translations) == len(regex_position_original), f"Invalid extraction {seq_translations}, {regex_position_original}"

    # replacement
    _t_regex: ty.Tuple[ty.Dict, re.Match]
    for _t_regex in regex_position_original:
        draft_text = draft_text.replace(_t_regex[0]['expression_original'], _t_regex[0]['expression_translation'])

    # record position after the replacement
    regex_position_replacement = []
    for _d_pair in seq_translations:
        regex_position_replacement += [(_d_pair, _o) for _o in re.finditer(f'{_d_pair["expression_translation"]}', draft_text, re.DOTALL)]

    assert len(seq_translations) == len(regex_position_replacement), f"Invalid extraction {seq_translations}, {regex_position_replacement}"
    
    # merge two `regex_position_original` and `regex_position_replacement`
    seq_replacement_before_after = []
    for _i_regex, _d_pair in enumerate(seq_translations):
        _position_before =  regex_position_original[_i_regex][1].span()
        _position_after = regex_position_replacement[_i_regex][1].span()
        seq_replacement_before_after.append(TranslationReplacementInformation(
            expression_original=_d_pair['expression_original'],
            expression_translation=_d_pair['expression_translation'],
            span_original=_position_before,
            span_translation=_position_after))

    logger.debug(f"After translation: {draft_text}")
    return {
        "final_response": draft_text,
        "is_processor_success": is_processor_success,
        "translation_pair_extracted": seq_replacement_before_after
    }
