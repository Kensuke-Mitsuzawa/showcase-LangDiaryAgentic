import torch
import logging

import typing as ty
from typing import TypedDict
from langchain_community.llms import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from .vector_store import query_past_errors, add_error_log
from .logging_configs import apply_logging_suppressions

apply_logging_suppressions()

logger = logging.getLogger(__name__)

MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"


# --- Define State dictionary ---
class AgentState(TypedDict):
    draft_text: str
    retrieved_context: str
    final_response: str
    new_errors: str



def init_llm(model_id: str = MODEL_ID) -> HuggingFacePipeline:
    """initializing the base LLM."""

    # --- 1. Setup Local LLM ---
    # We use a model small enough for CPU but smart enough for logic
    logger.info(f"Loading local model: {model_id}...")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",  # Uses GPU if available, else CPU
        torch_dtype=torch.float32, 
        trust_remote_code=False
    )

    # Create the Hugging Face Pipeline
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        temperature=0.1,
        top_p=0.95,
        repetition_penalty=1.15
    )

    # Wrap it in LangChain
    llm = HuggingFacePipeline(pipeline=pipe)

    return llm
# end def

llm = init_llm()

# --- 3. Define Nodes ---

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
    template = """<|user|>
You are a strict language tutor.
Context (Past Errors): {context}

Task:
1. Translate bracketed [text] in the Draft to the correct language.
2. Correct grammar.
3. If the user repeats a mistake from Context, warn them.

Draft: {draft} <|end|>
<|assistant|>"""
    
    prompt = PromptTemplate(
        template=template,
        input_variables=["context", "draft"]
    )
    
    chain = prompt | llm | StrOutputParser()
    
    response = chain.invoke({
        "draft": state["draft_text"],
        "context": state["retrieved_context"]
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
    
    prompt = PromptTemplate(template=template, input_variables=["response"])
    chain = prompt | llm | StrOutputParser()
    
    extracted = chain.invoke({"response": state["final_response"]})
    clean_extracted = extracted.split("<|assistant|>")[-1].strip()
    
    if "None" not in clean_extracted and len(clean_extracted) > 5:
        add_error_log(clean_extracted)
        
    return {"new_errors": clean_extracted}


def init_graph():
    # --- 4. Build Graph ---
    workflow = StateGraph(AgentState)
    workflow.add_node("retriever", node_retriever)
    workflow.add_node("processor", node_processor)
    workflow.add_node("archivist", node_archivist)

    workflow.set_entry_point("retriever")
    workflow.add_edge("retriever", "processor")
    workflow.add_edge("processor", "archivist")
    workflow.add_edge("archivist", END)

    app_graph = workflow.compile()

    return app_graph