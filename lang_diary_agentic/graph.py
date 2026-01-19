import torch
import logging
import json
import re

import typing as ty
from typing import TypedDict
from langchain_community.llms import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langgraph.graph import StateGraph, END
from langchain_core.output_parsers import PydanticOutputParser

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from langdetect import detect

from .vector_store import query_past_errors, add_error_log
from .logging_configs import apply_logging_suppressions
from .models.vector_store_entry import ErrorRecord

parser = PydanticOutputParser(pydantic_object=ErrorRecord)

apply_logging_suppressions()

logger = logging.getLogger(__name__)

# MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"
# model_id_small = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
model_id_large = "Qwen/Qwen2.5-7B-Instruct"



# --- Define State dictionary ---
class AgentState(TypedDict):
    draft_text: str
    target_lang: ty.Optional[str]  # e.g., "French"
    source_lang: ty.Optional[str]  # e.g., "English"    
    retrieved_context: str
    final_response: str
    new_errors: str



def load_local_llm(model_id, max_tokens=500):
    """Helper to load a model pipeline"""
    logger.info(f"⏳ Loading Model: {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
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
    
    # If user already provided them via UI, skip detection
    if state.get("target_lang") and state.get("source_lang"):
        logger.info(f"   Using provided: {state['target_lang']} (Target)")
        return {} # No state update needed
    # end if

    # Otherwise, ask the Small LLM
    logger.info("Missing languages. Asking Small LLM...")

    draft_text = state["draft_text"]

    # Since this part is supposed to be shorter. So, I use the tranditional ML model.
    seq_text_blanket = [x.group() for x in re.finditer(r'\[.+\]', draft_text)]
    language_annotation = detect(' '.join(seq_text_blanket))

    draft_text_without_blanket = re.sub(r'\[.+\]', '', draft_text)
    language_target = detect(draft_text_without_blanket)

    # # ---- language detection for the main body ----
    # template = """<|user|>
    # Analyze this text. What language is the main text written in, except for the words in [brackets] written in? 

    # Return ONLY a JSON like: {{"lang": "French"}}
    
    # Text: {draft}<|end|>
    # <|assistant|>"""
    
    # prompt = PromptTemplate(template=template, input_variables=["draft"])
    # # We use a JSON parser to ensure safety
    # chain = prompt | llm_small

    # result_text = chain.invoke({"draft": state["draft_text"]})

    # # ---- extract the json part ----
    # res_json_text = re.search(r'\{.+\}', result_text)
    
    # if res_json_text is None:
    #     logger.warning(f"Failed to detect the language. The response: {result_text}")
    #     return {"target_lang": None, "source_lang": None}
    # # end if

    # result_text_json_part = res_json_text.group(0)
    # # ---- END: language detection for the main body ----
    
    # try:
    #     result = json.loads(result_text_json_part)
    #     target_lang = result.get("lang", "Unknown") 
    #     logger.debug(f'Detected language: {target_lang} (Target) and {language_annotation} (Source)')

    #     return {
    #         "target_lang": target_lang, 
    #         "source_lang": language_annotation
    #     }
    # except Exception:
    #     # Fallback if detection fails
    #     logger.warning(f"Failed to detect the language. The response: {result_text_json_part}")
    #     return {"target_lang": None, "source_lang": None}

    return {
        "target_lang": language_target,
        "source_lang": language_annotation
    }
    

def node_retriever(state: AgentState) -> ty.Dict:
    """Node 1: Retrieve
    
    Operation: fetch entries from the vector DB.
    Dependency: vector DB.
    """
    logging.info("--- Node 1: Retrieve ---")
    past_errors = query_past_errors(state["draft_text"])
    context_str = "\n".join([f"- {err}" for err in past_errors]) if past_errors else "None"
    return {"retrieved_context": context_str}

def node_processor(state: AgentState) -> ty.Dict:
    """Node 2: Coach"""
    logging.info("--- Node 2: Process ---")
    
    # We use a PromptTemplate designed for instruction-tuned models (Phi-3 format)
    # Phi-3 uses <|user|> and <|assistant|> tokens

    sub_phrase_language_pair: str = ""
    source_lang = state["source_lang"]
    target_lang = state["target_lang"]

    if source_lang is None or target_lang is None:
        sub_phrase_language_pair = ""
    else:
        sub_phrase_language_pair = "The bracketed text is written in {source_lang}. The translation target language is {target_lang}."
    # end if


    template = """<|user|>
You are a strict language tutor.
Context (Past Errors): {context}

Task:
1. Translate bracketed [text] in the Draft to the correct language. {sub_phrase_language_pair}
2. Correct grammar.
3. If the user repeats a mistake from Context, warn them.

Draft: {draft} <|end|>
<|assistant|>"""
    
    prompt = PromptTemplate(
        template=template,
        input_variables=["context", "draft", "source_lang", "target_lang"]
    )
    
    chain = prompt | llm_large | StrOutputParser()
    
    response = chain.invoke({
        "draft": state["draft_text"],
        "context": state["retrieved_context"],
        "sub_phrase_language_pair": sub_phrase_language_pair
    })
    
    # Simple cleanup to remove the prompt from the output if the model echos it
    clean_response = response.split("<|assistant|>")[-1].strip()
    
    return {"final_response": clean_response}

def node_archivist(state: AgentState) -> ty.Dict:
    """Node 3: Archivist"""
    logging.info("--- Node 3: Archive ---")
    
    template = """<|user|>
Extract grammatical errors from this text as a list. If none, say "None".
Text: {response} <|end|>
<|assistant|>"""
    
    prompt = PromptTemplate(
        template=template,
        input_variables=["response"],
        # This auto-generates the "Return JSON with keys..." instructions
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    
    # Chain: Prompt -> LLM -> JSON Parser
    chain = prompt | llm_large | parser
    
    try:
        # result will be an ErrorRecord OBJECT, not a string
        error_record: ErrorRecord = chain.invoke({"response": state["final_response"]})
        
        # Only save if it's a real error
        if error_record.category != "None":
            # Save the object, not just text
            add_error_log(error_record)
            return {"new_errors": f"Saved: {error_record.error_rule}"}
        # end if   
    except Exception as e:
        logger.error(f"Parsing failed: {e}")
        return {"new_errors": "None"}
        
    return {"new_errors": "None"}

def init_graph():
    # --- 4. Build Graph ---
    workflow = StateGraph(AgentState)
    workflow.add_node("detector", node_language_detect)
    workflow.add_node("retriever", node_retriever)
    workflow.add_node("processor", node_processor)
    workflow.add_node("archivist", node_archivist)

    workflow.set_entry_point("detector")
    workflow.add_edge("detector", "retriever")
    workflow.add_edge("retriever", "processor")
    workflow.add_edge("processor", "archivist")
    workflow.add_edge("archivist", END)

    app_graph = workflow.compile()

    return app_graph