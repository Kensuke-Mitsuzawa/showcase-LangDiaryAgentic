import torch
import logging
import json
import re
from datetime import date, datetime

import typing as ty
from typing import TypedDict
from langchain_community.llms import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from langdetect import detect

from .vector_store import query_past_errors, add_error_logs
from .logging_configs import apply_logging_suppressions
from .models.vector_store_entry import ErrorRecord
from .db_handler import HandlerDairyDB, UnknownExpressionEntry, DiaryEntry
from .configs import GENERATION_DB_PATH, languages_code

apply_logging_suppressions()

logger = logging.getLogger(__name__)


# model_id_small = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
model_id_large = "microsoft/Phi-3-mini-4k-instruct"
# model_id_large = "Qwen/Qwen2.5-7B-Instruct"




# --- Define State dictionary ---
class AgentState(TypedDict):
    draft_text: str
    retrieved_context: str
    final_response: str
    suggestion: str
    new_errors: str
    lang_annotation: ty.Optional[str]  # e.g., "French"
    lang_diary_body: ty.Optional[str]  # e.g., "English"    
    bracket_text: ty.List[str]
    primary_id_DiaryEntry: str
    diary_date: str
    created_at: datetime


def load_local_llm(model_id, max_tokens=500):
    """Helper to load a model pipeline"""
    logger.info(f"⏳ Loading Model: {model_id}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # Define the IDs that stop generation. 
    # 32000 = <|endoftext|>, 32007 = <|end|> (Phi-3 specific)
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|end|>"),
        tokenizer.convert_tokens_to_ids("<|assistant|>") # Safety net
    ]    
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype="auto",
        trust_remote_code=False, 
    )
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_tokens,
        temperature=0.1,
        eos_token_id=terminators,
        pad_token_id=tokenizer.eos_token_id
    )
    return HuggingFacePipeline(pipeline=pipe)



# --- Configuration ---
# SMALL MODEL (Task: Language Detection) -> Fast, Low RAM
# Suggestion: "TinyLlama/TinyLlama-1.1B-Chat-v1.0" or "google/gemma-2b-it"
# llm_small = load_local_llm(model_id_small, max_tokens=100)

# LARGE MODEL (Task: Grammar Coaching) -> Smart, Higher RAM
# Suggestion: "microsoft/Phi-3-mini-4k-instruct" or "Qwen/Qwen2.5-7B-Instruct"
llm_large = load_local_llm(model_id_large, max_tokens=600)


# --- 3. Define Nodes ---


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

        assert _language_diary in languages_code, f"The language code {_language_diary} is not valid. Check the language code in 2 character."
        assert _language_annotation in languages_code, f"The language code {_language_annotation} is not valid. Check the language code in 2 character."

        return {
            "lang_annotation": _language_annotation,
            "lang_diary_body": _language_diary,
            "bracket_text": seq_text_blanket
        }
    # end if

    # Otherwise, ask the Small LLM
    logger.info("Missing languages. Asking Small LLM...")


    # Since this part is supposed to be shorter. So, I use the tranditional ML model.
    language_annotation = detect(' '.join(seq_text_blanket))

    draft_text_without_blanket = re.sub(r'\[.+\]', '', draft_text)
    language_target = detect(draft_text_without_blanket)

    return {
        "lang_diary_body": language_target,
        "lang_annotation": language_annotation,
        "bracket_text": seq_text_blanket
    }
    

def node_retriever(state: AgentState) -> ty.Dict:
    """Node 1: Retrieve
    
    Operation: fetch entries from the vector DB.
    Dependency: vector DB.
    """
    logging.info("--- Node 1: Retrieve ---")
    past_errors = query_past_errors(
        query_text=state["draft_text"], 
        lang_annotation=state["lang_annotation"], 
        lang_diary_body=state["lang_diary_body"])
    context_str = "\n".join([f"- {err}" for err in past_errors]) if past_errors else "None"
    return {"retrieved_context": context_str}


def node_processor(state: AgentState) -> ty.Dict:
    """Node 2: Coach"""
    logging.info("--- Node 2: Process ---")
    
    # We use a PromptTemplate designed for instruction-tuned models (Phi-3 format)
    # Phi-3 uses <|user|> and <|assistant|> tokens

    sub_phrase_language_pair: str = ""
    source_lang = state["lang_annotation"]
    target_lang = state["lang_diary_body"]

    if source_lang is None or target_lang is None:
        sub_phrase_language_pair = ""
    else:
        sub_phrase_language_pair = f"The bracketed text is written in {source_lang}. The translation target language is {target_lang}."
    # end if

    json_schema = """
        "<replaced>corrected draft</replaced>"
        "<explanation>warning messages.</explanation>"
        }"""

    user_content = (
        "You are a strict language tutor.\n"
        "Context (Past Errors): {context}\n\n"
        "Task:\n"
        "1. Translate bracketed [text] in the Draft to the correct language. {sub_phrase_language_pair}\n"
        "2. Correct grammar.\n"
        "3. If the user repeats a mistake from Context, warn them.\n"
        "IMPORTANT: Return the result ONLY as XML in the following structure:\n"
        "{json_schema}\n\n"
        "Draft: {draft}"
    )

    chat = [
        {"role": "user", "content": user_content},
    ]

    tokenizer = llm_large.pipeline.tokenizer
    template = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

    prompt = PromptTemplate(
        template=template,
        input_variables=["context", "draft"]
    )
    
    chain = prompt | llm_large | StrOutputParser()
    
    response = chain.invoke({
        "draft": state["draft_text"],
        "context": state["retrieved_context"],
        "sub_phrase_language_pair": sub_phrase_language_pair,
        "json_schema": json_schema
    })
    
    # Simple cleanup to remove the prompt from the output if the model echos it
    clean_response = response.split("<|assistant|>")[-1]

    group_replaced = re.search(r'<replaced>(.+)</replaced>', clean_response)
    group_explanation = re.search(r'<explanation>(.+)</explanation>', clean_response)

    if group_replaced is None:
        final_response = clean_response
    else:
        final_response = group_replaced.group(1)
    # end if

    if group_explanation is None:
        explanation = clean_response
    else:
        explanation = group_explanation.group(1)
    # end if

    return {"final_response": final_response, "suggestion": explanation}


def extract_xml_errors(text: str):
    """
    Parses multiple <error> blocks from the text.
    Returns a list of dicts: [{'rule': '...', 'phrase': '...', ...}, {...}]
    """
    errors = []
    
    # 1. Find all content inside <error>...</error> tags
    # re.DOTALL allows the dot (.) to match newlines
    error_blocks = re.findall(r"<error>(.*?)</error>", text, re.DOTALL)
    
    for block in error_blocks:
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
    logging.info("--- Node 3: Archive ---")
        
    chat = [
        {
            "role": "user", 
            "content": (
                "You are a strict language grammarian.\n"
                "Task: Identify ALL grammatical, vocabulary, or spelling errors in the user's draft.\n"
                "For EACH error, output an XML block exactly like this:\n\n"
                "<error>\n"
                "  <rule>The specific rule violated</rule>\n"
                "  <phrase>The incorrect phrase from text</phrase>\n"
                "  <correction>The corrected phrase</correction>\n"
                "  <category>Grammar OR Vocabulary OR Spelling</category>\n"
                "</error>\n\n"
                "If there are no errors, output: <no_errors/>\n\n"
                f"Draft: {state['draft_text']}\n"
                f"Reference Correction: {state['final_response']}"
            )
        }
    ]

    tokenizer = llm_large.pipeline.tokenizer
    template = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

    prompt = PromptTemplate(template=template, input_variables=[])
    
    # Chain: Prompt -> LLM
    chain = prompt | llm_large
    response  = chain.invoke({})

    # 4. Extract List using Regex
    error_list = extract_xml_errors(response)

    diary_date = state.get("date_diary", str(date.today()))
    created_at = datetime.now()
    datetime_str = created_at.isoformat()
    primary_id_DiaryEntry = f"{diary_date}_{datetime_str}"
    
    # 5. Save to DB (Loop through found errors)
    error_list_obj = []
    for err in error_list:
        # Create your Pydantic object or Dict here
        err['primary_id_DiaryEntry'] = primary_id_DiaryEntry
        err['language_diary_text'] = state['lang_diary_body']
        err['language_annotation_text'] = state['lang_annotation']
        record = ErrorRecord(**err)
        error_list_obj.append(record)
    # end for

    if len(error_list_obj) > 0:
        logger.debug(f"Found {len(error_list)} errors.")
        add_error_logs(error_list_obj)
    else:
        logger.debug("No errors found.")
    # end if
    
    return {
        "new_errors": json.dumps(error_list), 
        "primary_id_DiaryEntry": primary_id_DiaryEntry,
        "diary_date": diary_date,
        "created_at": created_at
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
        diary_replaced=state.get("replaced_text", ""),
        diary_corrected=state["final_response"],
        created_at=created_at,
        primary_id=state["primary_id_DiaryEntry"]
    )

    seq_unknown_expression_entry = []
    seq_bracket_text = state['bracket_text']
    for _expression in seq_bracket_text:
        _unknown_expression_entry = UnknownExpressionEntry(
            expression=_expression,
            language_source=language_source,
            language_annotation=language_annotation,
            created_at=created_at,
            primary_id=None
        )
        seq_unknown_expression_entry.append(_unknown_expression_entry)
    # end for

    handler = HandlerDairyDB(GENERATION_DB_PATH)
    handler.init_db()

    handler.save_diary_entry(diary_entry)
    for _entry in seq_unknown_expression_entry:
        handler.save_unknown_expression(_entry)
    # end for

    return {}    

def init_graph():
    # --- 4. Build Graph ---
    workflow = StateGraph(AgentState)
    workflow.add_node("detector", node_language_detect)
    workflow.add_node("retriever", node_retriever)
    workflow.add_node("processor", node_processor)
    workflow.add_node("archivist", node_archivist)
    workflow.add_node("db_saver", node_save_duckdb)

    workflow.set_entry_point("detector")
    workflow.add_edge("detector", "retriever")
    workflow.add_edge("retriever", "processor")
    workflow.add_edge("processor", "archivist")
    workflow.add_edge("archivist", "db_saver") 
    workflow.add_edge("db_saver", END)

    app_graph = workflow.compile()

    return app_graph