import logging
import json
import copy
import re
import typing as ty

import google.generativeai as genai
import instructor
from pydantic import BaseModel

from ..models import AgentState, TranslationReplacementInformation
from ..static import Iso693_code2natural_name

logger = logging.getLogger(__name__)


class ResponseTranslationPair(BaseModel):
    expression_original: str
    expression_translation: str
    

class ResponseObjectAgent(BaseModel):
    response: ty.List[ResponseTranslationPair]


# def __extract_xml_errors_node_processor(text: str, 
#                                         expected_extractions: ty.List[str], 
#                                         is_skip_1st_error_tag: bool = True):
#     """
#     Parses multiple <error> blocks from the text.
#     Returns a list of dicts: [{'rule': '...', 'phrase': '...', ...}, {...}]
#     """
#     errors = []
    
#     no_errors_tag = re.findall(r"<no_errors/>", text, re.DOTALL)
#     if len(no_errors_tag) > 1:
#         return []

#     seq_translations = re.findall(r'<bracket>(.*?)</bracket>[\s\n]?+<translation>(.*?)</translation>', text, re.DOTALL)

#     if len(expected_extractions) == len(seq_translations):
#        is_skip_1st_error_tag = False 

#     if is_skip_1st_error_tag and len(seq_translations) == 1:
#         return []

#     if is_skip_1st_error_tag:
#         seq_candaidate = seq_translations[1:]
#     else:
#         seq_candaidate = seq_translations

#     for _i_block, block in enumerate(seq_candaidate):
#         if len(block) != 2:
#             continue

#         errors.append({
#             "expression_original": block[0].strip(),
#             "expression_translation": block[1]
#         })

#     return errors


from ..configs import SettingsVariables


def node_translator(state: AgentState, settings: SettingsVariables) -> AgentState:
    """Node 2: Coach"""
    logger.info("--- Node: translator ---")

    # forming the system message.
    system_prompt_message = """You are an expert translator. Your task is to translate the input text into the target language."""

    # forming the user message.

    assert state.parameter_config_llm.config_translator.is_execute is True, "is_execute must be set True."
    tast_config = state.parameter_config_llm.config_translator

    is_processor_success = True

    sub_phrase_language_pair: str = ""
    lang_annotation = state.diary_entry_input.lang_annotation
    lang_diary_body = state.diary_entry_input.lang_diary_body
    lang_annotation_natural_name = Iso693_code2natural_name[lang_annotation]
    lang_diary_body_natural_name = Iso693_code2natural_name[lang_diary_body]

    if lang_annotation is None or lang_diary_body is None:
        sub_phrase_language_pair = ""
    else:
        sub_phrase_language_pair = f"The bracketed text is written in {lang_annotation_natural_name}. The translation target language is {lang_diary_body_natural_name}."

    seq_unkown_expressions = [
        {"expression_original": _exp_phrase} for _exp_phrase in state.processed_output.unkown_expressions
    ]

    user_content = (
        "# Task:\n"
        f"Translate text in bracketed [text] one by one. {sub_phrase_language_pair}\n"
        f"# INPUT\n"
        f"{json.dumps(seq_unkown_expressions)}"
    )

    # Configure Gemini API key
    genai.configure(api_key=settings.Cloud_API_Token)

    client = instructor.from_gemini(
        client=genai.GenerativeModel(
            model_name=settings.MODEL_NAME_Primary,
        ),
        mode=instructor.Mode.GEMINI_JSON,
    )

    response_obj = client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_prompt_message},
            {"role": "user", "content": user_content}
        ],
        response_model=ResponseObjectAgent,
        generation_config={
            "temperature": state.parameter_config_llm.config_translator.temperature,
            "top_p": state.parameter_config_llm.config_translator.top_p,
        },
    )

    # Extract the translations from response_obj
    seq_translations = [
        {
            "expression_original": item.expression_original,
            "expression_translation": item.expression_translation.replace('[', '').replace(']', '')
        }
        for item in response_obj.response
    ]

    # ---- replace the bracketed [text] one-by-one ----
    diary_replaced: str = copy.deepcopy(state.diary_entry_input.diary_original)

    # record position before the replacement
    regex_position_original = []
    for _d_pair in seq_translations:
        _regex_pattern = _d_pair["expression_original"].replace('[', '').replace(']', '').replace(' ', '\\s')
        regex_position_original += [(_d_pair, _o) for _o in re.finditer(f'{_regex_pattern}', diary_replaced, re.DOTALL)]
    # end for

    if len(seq_translations) != len(regex_position_original):
        logger.error(f"Fail to match expressions. Expected {len(seq_translations)}, found {len(regex_position_original)}")
        return {
            "processed_output": state.processed_output.model_copy(update={
                "diary_replaced": "",
                "diary_rewritten": "",
                "total_review": "",
                "unkown_expressions": [],
                "translation_pair_extracted": []
            }),
            "is_processor_success": False
        }
    # end if

    assert len(seq_translations) == len(regex_position_original), f"Invalid extraction {seq_translations}, {regex_position_original}"

    # replacement
    _t_regex: ty.Tuple[ty.Dict, re.Match]
    for _t_regex in regex_position_original:
        diary_replaced = diary_replaced.replace(_t_regex[0]['expression_original'], _t_regex[0]['expression_translation'].replace('[', '').replace(']', ''))
    # end for

    # record position after the replacement
    regex_position_replacement = []
    for _d_pair in seq_translations:
        _regex_pattern: str = _d_pair["expression_translation"].replace('[', '').replace(']', '')
        regex_position_replacement += [(_d_pair, _o) for _o in re.finditer(_regex_pattern, diary_replaced, re.DOTALL)]
    # end for

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
    # end for

    logger.debug(f"After translation: {diary_replaced}")
    return state.model_copy(update={
        "processed_output": state.processed_output.model_copy(update={
            "diary_replaced": diary_replaced,
            "translation_pair_extracted": seq_replacement_before_after,
        }),
        "is_processor_success": is_processor_success
    })

