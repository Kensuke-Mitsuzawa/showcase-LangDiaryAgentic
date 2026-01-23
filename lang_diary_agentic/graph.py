import logging
import json
import re
import copy
from datetime import date, datetime

import typing as ty
from typing import TypedDict
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, END
from langchain_core.language_models import BaseLanguageModel
from langchain_openai import ChatOpenAI

# from langdetect import detect

from .utils import check_language
from .llm_custom_api_wrapper import CustomHFServerLLM, RemoteServerEmbeddings
from .vector_store import add_error_logs
from .logging_configs import apply_logging_suppressions
from .models.vector_store_entry import ErrorRecord
from .db_handler import HandlerDairyDB, UnknownExpressionEntry, DiaryEntry
from .static import (
    Languages_Code,
    Iso693_code2natural_name,
    PossibleChoiceModeDeployment,
    PossibleCloudLLMProvider,
    PossibleLevelRewriting
)
from .configs import settings


apply_logging_suppressions()

logger = logging.getLogger(__name__)


# --- Define State dictionary ---
class AgentState(TypedDict):
    draft_text: str
    retrieved_context: str
    final_response: str
    suggestion_response: str
    # new_errors: str
    unkown_expressions: ty.List[ty.Dict[str, str]]
    total_review: str
    grammatical_errors_extracted: ty.List[ErrorRecord]
    # meta-information
    lang_annotation: ty.Optional[str]
    lang_diary_body: ty.Optional[str]
    level_rewriting: str    
    primary_id_DiaryEntry: str
    diary_date: str
    created_at: datetime
    # signal to convey the task status
    is_processor_success: bool
    is_archivist_success: bool


# ---- API-setups ----


def server_llm() -> CustomHFServerLLM:
    """Helper to load a model pipeline"""
    # Initialize your custom connection
    assert settings.Server_API_Endpoint is not None
    llm = CustomHFServerLLM(api_url=settings.Server_API_Endpoint)
    
    if llm.check_connection() is False:
        raise RuntimeError(f"The server is not available at {settings.Server_API_Endpoint}.")
    # end if

    return llm


def cloud_llm(
    api_key: str, 
    model_name: ty.Optional[str] = None,
    provider: PossibleCloudLLMProvider = "openai", 
    temperature: float = 0.7
) -> BaseLanguageModel:
    """
    Factory function to return a Cloud-based LLM (OpenAI/Gemini).
    
    Args:
        provider: "openai" or "google"
        api_key: The secret token (defaults to env vars OPENAI_API_KEY or GOOGLE_API_KEY)
        model_name: Specific model (e.g. "gpt-4o", "gemini-pro")
        temperature: Creativity parameter
    """
    
    # 1. OpenAI Implementation
    if provider.lower() == "openai":
        # Uses OPENAI_API_KEY environment variable by default if api_key is None
        llm = ChatOpenAI(
            api_key=api_key,
            model=model_name or "gpt-4o",
            temperature=temperature,
            max_retries=2,
            # 'streaming=True' is often preferred for cloud APIs
            streaming=True 
        )
        return llm

    # 2. Google Gemini Implementation
    elif provider.lower() == "google":
        raise NotImplementedError()
        # # Uses GOOGLE_API_KEY environment variable by default if api_key is None
        # llm = ChatGoogleGenerativeAI(
        #     google_api_key=api_key,
        #     model=model_name or "gemini-1.5-flash",
        #     temperature=temperature,
        #     convert_system_message_to_human=True # Helps with older Gemini quirks
        # )
        # return llm
    else:
        raise ValueError(f"Unsupported provider: {provider}")


if settings.Mode_Deployment == "cloud_api":
    assert settings.Cloud_API_Token is not None
    llm_large = cloud_llm(api_key=settings.Cloud_API_Token, model_name=settings.MODEL_NAME_Primary)
    tokenizer = None
elif settings.Mode_Deployment == "server":
    logger.info(f"connecting to the API endpoint: {settings.Server_API_Endpoint}")
    llm_large = server_llm()
    logger.info("API is ready.")
else:
    raise ValueError(f"Invalid Mode_Deployment: {settings.Mode_Deployment}")
# end if

client_embedding_model_server = RemoteServerEmbeddings(api_url=settings.Server_API_Endpoint)

# ---- END: API-setups ----




def create_compatible_chain(formatted_input: ty.List[ty.Tuple], 
                            llm):
    """
    Dynamically builds the chain based on whether input is String or List.
    """
    if settings.Mode_Deployment == "server":
        # # 1. Local Model Path (String Input)
        prompt = ChatPromptTemplate.from_messages(formatted_input)

    elif settings.Mode_Deployment == "cloud_api":
        # 2. API Path (List Input)
        # We must convert dicts -> LangChain Message Objects
        messages = []
        for msg in formatted_input:
            if msg['role'] == 'user':
                messages.append(HumanMessage(content=msg['content']))
            elif msg['role'] == 'system':
                messages.append(SystemMessage(content=msg['content']))
            elif msg['role'] == 'assistant':
                messages.append(AIMessage(content=msg['content']))
        
        # Create a ChatPromptTemplate from these messages
        prompt = ChatPromptTemplate.from_messages(messages)

    # Build and return the chain
    # Note: We invoke with empty dict {} because the prompt is already fully populated
    chain = prompt | llm
    return chain


# --- Define Nodes ---

def node_validation(state: AgentState):
    level_rewriting = state.get("level_rewriting")
    assert level_rewriting is not None
    assert level_rewriting in ty.get_args(PossibleLevelRewriting)
    

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
        _language_diary = state.get("lang_diary_body").strip()
        _language_annotation = state.get("lang_annotation").strip()

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


    # Since this part is supposed to be shorter. So, I use the tranditional ML model.
    language_annotation = check_language.detect_language(' '.join(seq_text_blanket))

    draft_text_without_blanket = re.sub(r'\[.+\]', '', draft_text)
    language_target = check_language.detect_language(' '.join(seq_text_blanket))

    return {
        "lang_diary_body": language_target,
        "lang_annotation": language_annotation,
        "unkown_expressions": seq_text_blanket
    }
    

# def node_retriever(state: AgentState) -> ty.Dict:
#     """Node 1: Retrieve
    
#     Operation: fetch entries from the vector DB.
#     Dependency: vector DB.
#     """
#     logging.info("--- Node: Retrieve ---")

#     past_errors = query_past_errors(
#         query_text=, 
#         lang_annotation=state["lang_annotation"], 
#         lang_diary_body=state["lang_diary_body"],
#         model_id_embedding=settings.MODEL_NAME_Embedding,
#         client_embedding_model_server=client_embedding_model_server
#     )
#     context_str = "\n".join([f"- {err}" for err in past_errors]) if past_errors else "None"
#     return {"retrieved_context": context_str}



def __extract_xml_errors_node_processor(text: str, is_skip_1st_error_tag: bool = True):
    """
    Parses multiple <error> blocks from the text.
    Returns a list of dicts: [{'rule': '...', 'phrase': '...', ...}, {...}]
    """
    errors = []
    
    no_errors_tag = re.findall(r"<no_errors/>", text, re.DOTALL)
    if len(no_errors_tag) > 1:
        return []
    # end if

    # 1. Find all content inside <error>...</error> tags
    # re.DOTALL allows the dot (.) to match newlines
    # seq_translations = re.findall(r'<translations>(.*?)</translations>', text, re.DOTALL)
    seq_translations = re.findall(r'<bracket>(.*?)</bracket>[\s\n]?+<translation>(.*?)</translation>', text, re.DOTALL)

    if is_skip_1st_error_tag and len(seq_translations) == 1:
        return []
    # end if

    if is_skip_1st_error_tag:
        seq_candaidate = seq_translations[1:]
    else:
        seq_candaidate = seq_translations
    # end if

    for _i_block, block in enumerate(seq_candaidate):
        if len(block) != 2:
            continue
        # end if

        errors.append({
            "expression_original": block[0].replace('[', '').replace(']', '').strip(),
            "expression_translation": block[1]
        })
    # end for

    return errors


def node_translator(state: AgentState) -> ty.Dict:
    """Node 2: Coach"""
    logging.info("--- Node: translator ---")

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
    # end if

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
    chain = create_compatible_chain(template, llm_large.bind(max_tokens=512, enable_thinking=False))
    response = chain.invoke({
        "unkown_expressions": json.dumps(state['unkown_expressions']), 
        "lang_annotation": lang_annotation,
    })

    # Simple cleanup to remove the prompt from the output if the model echos it
    clean_response = response.content.split("<|assistant|>")[-1]
    seq_translations = __extract_xml_errors_node_processor(clean_response)

    # replace the bracketed [text] one-by-one
    draft_text = copy.deepcopy(state['draft_text'])
    for _translation_obj in seq_translations:
        draft_text = draft_text.replace(_translation_obj['expression_original'], _translation_obj['expression_translation'])
    # end for

    logger.debug(f"Correction: {draft_text}")
    return {
        "final_response": draft_text,
        "is_processor_success": is_processor_success,
        "unkown_expressions": seq_translations
    }



def __extract_xml_errors_node_archivist(text: str, is_skip_1st_error_tag: bool = True):
    """
    Parses multiple <error> blocks from the text.
    Returns a list of dicts: [{'rule': '...', 'phrase': '...', ...}, {...}]
    """
    errors = []
    
    no_errors_tag = re.findall(r"<no_errors/>", text, re.DOTALL)
    if len(no_errors_tag) > 1:
        return []
    # end if

    # 1. Find all content inside <error>...</error> tags
    # re.DOTALL allows the dot (.) to match newlines
    error_blocks = re.findall(r"<error>(.*?)</error>", text, re.DOTALL)
    if is_skip_1st_error_tag and len(error_blocks)== 1:
        return []
    # end if


    for _i_error, block in enumerate(error_blocks):
        if _i_error == 0 and is_skip_1st_error_tag:
            continue
        # end if

        # 2. Extract fields from within each block
        rule = re.search(r"<rule>(.*?)</rule>", block, re.DOTALL)
        phrase = re.search(r"<phrase>(.*?)</phrase>", block, re.DOTALL)
        correction = re.search(r"<correction>(.*?)</correction>", block, re.DOTALL)
        category = re.search(r"<category>(.*?)</category>", block, re.DOTALL)
        
        # Only add if we found the critical fields
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
    logging.info("--- Node: Archive ---")

    # set the meta-info first
    diary_date = state.get("date_diary", str(date.today()))
    created_at = datetime.now()
    datetime_str = created_at.isoformat()
    primary_id_DiaryEntry = f"{diary_date}_{datetime_str}"

    # do nothing if `is_processor_success` is False
    if not state["is_processor_success"]:
        return {
        "primary_id_DiaryEntry": primary_id_DiaryEntry,
        "diary_date": diary_date,
        "created_at": created_at
    }
    # end if

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

    chain = chain = create_compatible_chain(template, llm_large.bind(max_tokens=1024, enable_thinking=True))
    response  = chain.invoke({})

    # Extract List using Regex
    error_list = __extract_xml_errors_node_archivist(response.content)
    # Save to DB (Loop through found errors)
    error_list_obj = []
    for err in error_list:
        # Create your Pydantic object or Dict here
        err['primary_id_DiaryEntry'] = primary_id_DiaryEntry
        err['language_diary_text'] = state['lang_diary_body']
        err['language_annotation_text'] = state['lang_annotation']
        err['model_id_embedding'] = settings.MODEL_NAME_Embedding
        try:
            record = ErrorRecord(**err)
            error_list_obj.append(record)
            logger.debug(f"Grammatical-Error: {record}")
        except Exception as e:
            logger.error(e)
        # end try
    # end for

    if len(error_list_obj) > 0:
        logger.debug(f"Found {len(error_list)} errors.")
        add_error_logs(error_list_obj, client_embedding_model_server)
    else:
        logger.debug("No errors found.")
    # end if
    
    return {
        "grammatical_errors_extracted": error_list, 
        "primary_id_DiaryEntry": primary_id_DiaryEntry,
        "diary_date": diary_date,
        "created_at": created_at
    }


PossibleReturnRoutineNodeRewriter = ty.Literal['success', 'insufficient_length', 'incorrect_language', 'xml_error']
def _func_routine_node_rewriter(prompt_content: str, state: AgentState, default_max_length: int) -> ty.Tuple[str, str, str]:
    """
    
    Return: (rewritten-text, full-response, error-code).
    """
    lang_code_diary = state['lang_diary_body']
    lang_name_natural_lan: str = Iso693_code2natural_name[lang_code_diary]

    template = [
        ("system", f"You are an expert {lang_name_natural_lan} editor."),
        ("user", prompt_content)
    ]
    chain = create_compatible_chain(template, llm_large.bind(max_length=default_max_length, enable_thinking=False))

    response  = chain.invoke({
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
    elif response.response_metadata['finish_reason'] == "length":
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
    # end if

    


def node_rewriter(state: AgentState, max_try: int = 5, default_max_length: int = 512) -> ty.Dict:
    """Node: Rewritting"""
    logging.info("--- Node: Rewriting ---")

    # do nothing if `is_processor_success` is False
    if not state["is_processor_success"]:
        return {}
    # end if
    
    lang_code_diary = state['lang_diary_body']

    prompt_content = (
        "Task:\n"
        "1. Rewrite the following text (in the {level_rewriting} level of the CEFR). The rewriting language must stick with language={target_lang}\n"
        "IMPORTANT: Return the result ONLY as XML in the following structure:\n"
        "<rewriting>rewritten text</rewriting>\n\n"
        "Input: {user_text}"
    )
    # ---- loop: LLM-call and validation ----
    _current_try = 0
    _validation_status = False
    while True:
        if _current_try == max_try:
            break
        if _validation_status is True:
            break
        # end if

        _response_rewriting, _full_response, _flag_error = _func_routine_node_rewriter(prompt_content, state, default_max_length)

        if _flag_error == 'xml_error':
            # case: XML does not exist.
            logger.warning("failed to extract XML. retry.")
            _msg_addition = "IMPORTANT: Return the result ONLY as XML in the following structure:\n<rewriting>rewritten text</rewriting>\n"
            prompt_content += _msg_addition
            _current_try += 1
            continue
        # end if

        if _flag_error == 'insufficient_length':
            default_max_length += 100

        if _flag_error == 'incorrect_language':
            _msg_addition = "IMPORTANT: Rewriting language must be {target_lang}. Rewrite the input text to match the {level_rewriting} CEFR level."
            _current_try += 1
            continue
            # end if

        _validation_status = True
    # end
    # ----


    return {
        "suggestion_response": _response_rewriting
    }


def node_reviewer(state: AgentState) -> ty.Dict:
    """Node: Reviewer"""
    logging.info("--- Node: Reviewer ---")

    # do nothing if `is_processor_success` is False
    if not state["is_processor_success"]:
        return {}
    # end if
    

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
    # # Chain: Prompt -> LLM
    chain = create_compatible_chain(template, llm_large.bind(max_length=1024, enable_thinking=True))

    response  = chain.invoke({
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
    # end if

    return {
        "total_review": text_review
    }



def node_save_duckdb(state: AgentState):
    """New Node: Save everything to DuckDB"""
    logger.info("--- [4] Saving to DuckDB ---")
    
    # Use today's date if not provided
    diary_date = state["diary_date"]
    created_at = state["created_at"]

    language_source = state.get("lang_diary_body", "Unknown")
    language_source = "Unknown" if language_source is None else language_source

    language_annotation = state.get("lang_annotation", "Unknown")
    language_annotation = "Unknown" if language_annotation is None else language_annotation

    diary_entry = DiaryEntry(
        date_diary=diary_date,
        language_source=language_source,
        language_annotation=language_annotation,
        diary_original=state["draft_text"],
        diary_replaced=state["final_response"],
        diary_corrected=state["suggestion_response"],
        created_at=created_at,
        primary_id=state["primary_id_DiaryEntry"]
    )

    seq_unknown_expression_entry = []
    seq_bracket_text = state['unkown_expressions']
    for _d_expression in seq_bracket_text:
        _unknown_expression_entry = UnknownExpressionEntry(
            expression=_d_expression['expression_original'],
            expression_translation=_d_expression['expression_translation'],
            language_source=language_source,
            language_annotation=language_annotation,
            created_at=created_at,
            primary_id=None
        )
        seq_unknown_expression_entry.append(_unknown_expression_entry)
    # end for
    
    assert settings.GENERATION_DB_PATH is not None
    handler = HandlerDairyDB(settings.GENERATION_DB_PATH)
    handler.init_db()

    handler.save_diary_entry(diary_entry)
    for _entry in seq_unknown_expression_entry:
        handler.save_unknown_expression(_entry)
    # end for

    return {}    

def init_graph():
    # --- 4. Build Graph ---
    workflow = StateGraph(AgentState)
    workflow.add_node("validator", node_validation)
    workflow.add_node("detector", node_language_detect)
    workflow.add_node("translator", node_translator)
    workflow.add_node("archivist", node_archivist)
    workflow.add_node("rewriter", node_rewriter)
    workflow.add_node("reviewer", node_reviewer)
    workflow.add_node("db_saver", node_save_duckdb)

    workflow.set_entry_point("validator")
    workflow.add_edge("validator", "detector")    
    workflow.add_edge("detector", "translator")
    workflow.add_edge("translator", "archivist")
    workflow.add_edge("archivist", "rewriter")
    workflow.add_edge('rewriter', 'reviewer')
    workflow.add_edge("reviewer", "db_saver")
    workflow.add_edge("db_saver", END)

    app_graph = workflow.compile()

    return app_graph