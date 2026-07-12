import logging
import re
import typing as ty

from ..models import AgentState
from ..llm_clients import llm_large, create_compatible_chain

logger = logging.getLogger(__name__)

def node_reviewer(state: AgentState) -> ty.Dict:
    """Node: Reviewer"""
    logger.info("--- Node: Reviewer ---")
    task_config = state["config_reviewer"]

    if task_config.is_execute is False:
        logger.info("SKip the task since is_execute is False.")
        return {
            "total_review": ""
        }

    if not state["is_processor_success"]:
        return {
            "total_review": ""
        }
    
    template = [
        ("system", 'You are a "Memory Coach" for a language learner. Your goal is to analyze the user CURRENT MISTAKES and compare them against their ERROR HISTORY.'),
        ("user", (
                "1. **Current Mistakes:** A list of errors found in the user's latest diary entry.\n"
                "2. **Error History:** A list of similar errors the user made in the past (retrieved from database). \n"
                "### INSTRUCTIONS\n"
                'Step 1: Compare the "Current Mistakes" with the "Error History".\n'
                'Step 2: Classify the situation into one of two categories:\n'
                '   - **"RECURRING"**: The user made a mistake similar to one in history (e.g., gender agreement again, same vocabulary word).\n'
                '   - **"NEW"**: These are fresh mistakes not seen in the provided history.\n'
                'Step 3: Generate a short, helpful message.\n'
                '   - If RECURRING: Be firm but encouraging. Remind them of the specific rule they forgot.\n'
                '   - If NEW: Be gentle. Explain the new concept briefly.\n'
                '   - The user\'s target learning level is {level_rewriting}\n'
                "IMPORTANT: Return the result ONLY as XML in the following structure:\n"
                "<review>review contents</review>\n\n"
                "Current Mistakes: {current_mistakes}\n"
                "Error History: {error_history}\n"
                "Vocabularies that user does not know: {unkown_expressions}"
            )
        )
    ]

    chain = create_compatible_chain(template, llm_large.bind(max_length=1024, enable_thinking=True))

    response = chain.invoke({
        "current_mistakes": state['grammatical_errors_extracted'], 
        "error_history": state['retrieved_context'], 
        "level_rewriting": state['level_rewriting'],
        "unkown_expressions": state['unkown_expressions']
    })

    response_text: str = response.content
    group_replaced = re.findall(r'<review>(.*?)</review>', response_text, re.DOTALL)
    if group_replaced == []:
        logger.warning(f"Regex error. Return the full response. Response={response}")
        text_review = response
    else:
        text_review = group_replaced[-1]

    return {
        "total_review": text_review
    }
