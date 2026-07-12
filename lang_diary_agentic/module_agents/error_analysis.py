from sqlalchemy import desc
import logging
import re
import json
import typing as ty

import google.generativeai as genai
import instructor

from pydantic import BaseModel, Field

from ..models import AgentState, ErrorRecord
# from ..llm_clients import llm_large, client_embedding_model_server, create_compatible_chain
from ..vector_store import add_error_logs
from ..configs import SettingsVariables

logger = logging.getLogger(__name__)


# def __extract_xml_errors_node_archivist(text: str, is_skip_1st_error_tag: bool = True):
#     """
#     Parses multiple <error> blocks from the text.
#     Returns a list of dicts: [{'rule': '...', 'phrase': '...', ...}, {...}]
#     """
#     errors = []
    
#     no_errors_tag = re.findall(r"<no_errors/>", text, re.DOTALL)
#     if len(no_errors_tag) > 1:
#         return []

#     error_blocks = re.findall(r"<error>(.*?)</error>", text, re.DOTALL)
#     if is_skip_1st_error_tag and len(error_blocks) == 1:
#         return []

#     for _i_error, block in enumerate(error_blocks):
#         if _i_error == 0 and is_skip_1st_error_tag:
#             continue

#         rule = re.search(r"<rule>(.*?)</rule>", block, re.DOTALL)
#         phrase = re.search(r"<phrase>(.*?)</phrase>", block, re.DOTALL)
#         correction = re.search(r"<correction>(.*?)</correction>", block, re.DOTALL)
#         category = re.search(r"<category>(.*?)</category>", block, re.DOTALL)
        
#         if rule and correction:
#             errors.append({
#                 "error_rule": rule.group(1).strip(),
#                 "example_phrase": phrase.group(1).strip() if phrase else "",
#                 "correction": correction.group(1).strip(),
#                 "category": category.group(1).strip() if category else "None"
#             })
            
#     return errors


class ErrorObject(BaseModel):
    category_error: ty.Literal["Grammar", "Spelling"] = Field(description="Grammar OR Spelling")
    error_rule: ty.Literal['gender-error', 'number-error', 'verb-conjugation-error', 'preposition-error', 'article-error', 'other-error'] = Field(description="The sub-category of the error.")
    text_error: str = Field(description="The incorrect text in the input text.")
    text_correction: str = Field(description="The corrected text.")
    text_explanation: str = Field(description="The explanation of the error.")
    

class ResponseObjectAgent(BaseModel):
    seq_error_objects: ty.List[ErrorObject]


def node_error_analysis(
    state: AgentState,
    settings: SettingsVariables
) -> AgentState:
    """Node 3: Archivist"""

    # forming the system message.
    system_prompt_message = f"You are an expert editor of the {state.diary_entry_input.lang_diary_body} language."

    logger.info("--- Node: Archive ---")
    # task_config = state.task_config.error_analysis

    if not state.is_processor_success:
        return state
    # end if

    lang_diary_body = state.diary_entry_input.lang_diary_body

    user_prompt = (
        "# Task\n"
        "Identify errors in the given input text.\n"
        "Correct the errors and explain the reason of errors.\n"
        f"The explanation of errors should be in {state.diary_entry_input.language_annotation} language."
        
        "# Input\n"
        f"{state.processed_output.diary_replaced}"
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
            {"role": "user", "content": user_prompt}
        ],
        response_model=ResponseObjectAgent,
        generation_config={
            "temperature": state.parameter_config_llm.config_translator.temperature,
            "top_p": state.parameter_config_llm.config_translator.top_p,
        },
    )

    seq_error_records: ty.List[ErrorRecord] = []
    for err in response_obj.seq_error_objects:
        err_record = ErrorRecord(
            model_id_embedding=settings.MODEL_NAME_Embedding,
            language_diary_text=state.diary_entry_input.language_source,
            language_annotation_text=state.diary_entry_input.language_annotation,
            error_rule=err.error_rule,
            example_phrase=err.text_error,
            correction=err.text_correction,
            category=err.category_error,
            explanation=err.text_explanation,
        )

        seq_error_records.append(err_record)
    # end for

    return state.model_copy(update={
        "processed_output": state.processed_output.model_copy(
            update={
                "grammatical_errors_extracted": seq_error_records,
            })
    })


    # chain = create_compatible_chain(template, llm_large.bind(max_tokens=tast_config.max_tokens, 
    #                                                                  enable_thinking=tast_config.enable_thinking))
    # response = chain.invoke({})

    # error_list = __extract_xml_errors_node_archivist(response.content)

    # __error_list_obj = []
    # for err in error_list:
    #     err['primary_id_DiaryEntry'] = state["primary_id_DiaryEntry"]
    #     err['language_diary_text'] = state['lang_diary_body']
    #     err['language_annotation_text'] = state['lang_annotation']
    #     err['model_id_embedding'] = settings.MODEL_NAME_Embedding
    #     try:
    #         record = ErrorRecord(**err)
    #         __error_list_obj.append(record)
    #         logger.debug(f"Grammatical-Error: {record}")
    #     except Exception as e:
    #         logger.error(e)

    # error_list_obj = []
    # for err in __error_list_obj:
    #     if err.example_phrase == err.correction:
    #         continue
    #     error_list_obj.append(err)

    # if len(error_list_obj) > 0:
    #     logger.debug(f"Found {len(error_list)} errors.")
    #     add_error_logs(error_list_obj, client_embedding_model_server)
    # else:
    #     logger.debug("No errors found.")
    
    # return {
    #     "grammatical_errors_extracted": error_list, 
    # }
