from openai import BaseModel
import logging
import re
import json
import typing as ty

import google.generativeai as genai
import instructor

from ..models import (
    AgentState,
    PossibleLevelRewriting,
    ElementRewritingObject
)
from ..static import Iso693_code2natural_name
from ..configs import SettingsVariables

from ..utils import check_language

logger = logging.getLogger(__name__)


class ResponseObjectAgent(BaseModel):
    evaluation_current_level: PossibleLevelRewriting
    text_rewritten: str
    seq_element_rewriting: list[ElementRewritingObject]


def node_rewriter(state: AgentState, settings: SettingsVariables) -> AgentState:
    """Node: Rewritting"""

    system_prompt_message = f"You are an expert editor of the {state.diary_entry_input.lang_diary_body} language."

    logger.info("--- Node: Rewriting ---")

    if not state.is_processor_success:
        return state
    # end if

    prompt_content = (
        "# Task\n"
        f"Rewrite the given draft text in the {state.diary_entry_input.level_rewriting} level of the CEFR. \n" 
        f"The rewriting language must stick with language={state.diary_entry_input.language_source}\n"
        f'# Instruction\n'
        f'1. evaluate the current level of the given text and set the evaluated level to `evaluation_current_level`.\n'
        f'2. scan the input text and extract the phrase that needs to be rewritten. Set the extracted phrase to `phrase_target`.\n'
        f'3. rewrite the extracted phrase to be more natural and fluent in the level of {state.diary_entry_input.level_rewriting}. Set the rewritten phrase to `phrase_rewritten`.\n'
        f'4. provide the explanation of the rewriting. Set the explanation to `explanation` written in {state.diary_entry_input.language_annotation}.\n'
        f'5. Finally, set the rewritten text to the variable `text_rewritten`.'
        "# Input\n"
        f"{json.dumps(state.processed_output.diary_replaced)}"
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
            {"role": "user", "content": prompt_content}
        ],
        response_model=ResponseObjectAgent,
        generation_config={
            "temperature": state.parameter_config_llm.config_translator.temperature,
            "top_p": state.parameter_config_llm.config_translator.top_p,
        },
    )


    return state.model_copy(update={
        "processed_output": state.processed_output.model_copy(
            update={
                "evaluation_current_level": response_obj.evaluation_current_level,
                "diary_rewritten": response_obj.text_rewritten,
                "phrases_rewritten": response_obj.seq_element_rewriting,
            })
    })

