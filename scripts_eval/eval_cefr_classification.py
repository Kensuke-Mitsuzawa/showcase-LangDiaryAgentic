import itertools
import duckdb
import torch
import uuid
import time
import typing as ty
import pydantic
import json
import re
from typing import Optional, List, Dict
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

import logging
logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# 1. DATABASE SCHEMA & PYDANTIC MODELS
# --------------------------------------------------------------------------

PossibleStrategyDecoding = ty.Literal['argmax', 'top-p', 'greedy']
PossibleCefrLevel = ty.Literal['A1', 'A2', 'B1', 'B2', 'C1', 'C2']

class LLMConfiguration(pydantic.BaseModel):
    llm_configuration_id: str
    model_id_primary: str
    model_id_assistant: Optional[str]
    strategy_decoding: PossibleStrategyDecoding


class RecordDocument(pydantic.BaseModel):
    benchmark_id: str
    document_id: str
    language: str
    level_cefr_truth: str
    text_content: str


class RecordResponse(pydantic.BaseModel):
    response_id: str = pydantic.Field(default_factory=lambda: str(uuid.uuid4()))
    benchmark_id: str
    document_id: str
    prompt_id: str
    llm_configuration_id: str
    
    response_llm: str
    level_cefr_estimated: Optional[str]
    is_xml_success: bool
    time_execution: float


class PromptTemplate(pydantic.BaseModel):
    prompt_id: str
    template_text: str


# Connect to DuckDB (Persistent)
con = duckdb.connect("cefr_benchmark.duckdb")

def init_db():
    con.execute("""
        CREATE TABLE IF NOT EXISTS llm_configurations (
            llm_configuration_id VARCHAR PRIMARY KEY,
            model_id_primary VARCHAR,
            model_id_assistant VARCHAR,
            strategy_decoding VARCHAR
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            benchmark_id VARCHAR,
            document_id VARCHAR PRIMARY KEY,
            language VARCHAR,
            level_cefr_truth VARCHAR,
            text_content VARCHAR
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS prompts (
            prompt_id VARCHAR PRIMARY KEY,
            template_text VARCHAR
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS responses (
            response_id VARCHAR PRIMARY KEY,
            benchmark_id VARCHAR,
            document_id VARCHAR,
            prompt_id VARCHAR,
            llm_configuration_id VARCHAR,
            response_llm VARCHAR,
            level_cefr_estimated VARCHAR,
            is_xml_success BOOLEAN,
            time_execution DOUBLE
        )
    """)

# --------------------------------------------------------------------------
# 2. DATASET LOADING (UniversalCEFR)
# --------------------------------------------------------------------------

def set_datasets():
    logger.info("📥 Downloading UniversalCEFR dataset...")
    try:
        # Load UniversalCEFR (using a small split for testing if available, else train)
        # Note: 'UniversalCEFR/UniversalCEFR' might need specific config or split
        ds = load_dataset("UniversalCEFR/UniversalCEFR", split="test[:50]") # Limit to 50 for demo
    except Exception:
        logger.warning("⚠️  Dataset not found or error. Generating dummy data for demo.")
        return

    # Filter for target languages
    target_langs = ['en', 'fr', 'de']
    
    for row in ds:
        lang = row.get('lang') or row.get('language')
        if lang not in target_langs:
            continue
            
        doc = RecordDocument(
            benchmark_id="UniversalCEFR_v1",
            document_id=str(uuid.uuid4()),
            language=lang,
            level_cefr_truth=row['cefr_level'],
            text_content=row['text']
        )
        
        # Insert into DuckDB
        con.execute(
            "INSERT OR IGNORE INTO documents VALUES (?, ?, ?, ?, ?)",
            (doc.benchmark_id, doc.document_id, doc.language, doc.level_cefr_truth, doc.text_content)
        )
    logger.info("✅ Datasets loaded into DuckDB.")

# --------------------------------------------------------------------------
# 3. LLM EXECUTION LOGIC (Speculative Decoding)
# --------------------------------------------------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"

def set_hf_model(model_id_primary: str, model_id_assistant: Optional[str]):
    logger.info(f"⚙️ Loading Main: {model_id_primary} | Draft: {model_id_assistant}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_id_primary)
    model = AutoModelForCausalLM.from_pretrained(
        model_id_primary, 
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    assistant_model = None
    if model_id_assistant:
        # Crucial: Draft model must use same tokenizer usually, or you need 'universal assisted decoding'
        assistant_model = AutoModelForCausalLM.from_pretrained(
            model_id_assistant, 
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
    return tokenizer, model, assistant_model


def set_prompt_template() -> ty.List[PromptTemplate]:
    # 4. The Prompt Template
    PROMPT_TEMPLATE = """You are a CEFR evaluator.
Analyze the following text and return the CEFR level in XML format:
<analysis>
 <cefr>LEVEL</cefr>
</analysis>

Text: "{text}"
"""
    prompt_id = "prompt_v1_xml"

    # Insert into DB
    con.execute("INSERT OR REPLACE INTO prompts VALUES (?, ?)", (prompt_id, PROMPT_TEMPLATE))

    return [PromptTemplate(prompt_id=prompt_id, template_text=PROMPT_TEMPLATE)]


def set_model_configurations() -> ty.List[LLMConfiguration]:
    # 5. Model Configurations
    plans = [
        LLMConfiguration(
            llm_configuration_id="conf_qwen_pure", 
            model_id_primary="Qwen/Qwen2.5-7B-Instruct", 
            model_id_assistant=None, 
            strategy_decoding="greedy"
        ),
        LLMConfiguration(
            llm_configuration_id="conf_qwen_speculative", 
            model_id_primary="Qwen/Qwen2.5-7B-Instruct", 
            model_id_assistant="Qwen/Qwen2.5-0.5B-Instruct", 
            strategy_decoding="greedy"
        )
    ]

    # 2. Insert Configs to DB
    for p in plans:
        con.execute("INSERT OR REPLACE INTO llm_configurations VALUES (?, ?, ?, ?)", 
                   (p.llm_configuration_id, p.model_id_primary, p.model_id_assistant, p.strategy_decoding))

    return plans



def execute_hf_model(prompt, tokenizer, model, assistant_model, decoding_config: Dict):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    start_time = time.time()
    
    # Hugging Face generate() handles the speculative loop internally
    # when assistant_model is passed.
    outputs = model.generate(
        **inputs,
        assistant_model=assistant_model, 
        max_new_tokens=512,
        pad_token_id=tokenizer.eos_token_id,
        **decoding_config
    )
    
    duration = time.time() - start_time
    decoded_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return decoded_text, duration

# --------------------------------------------------------------------------
# 4. PARSING & LOGIC
# --------------------------------------------------------------------------

def extract_xml_cefr(text: str) -> tuple[Optional[str], bool]:
    # Look for <cefr>...</cefr>
    match = re.search(r"<cefr>(.*?)</cefr>", text, re.IGNORECASE)
    if match:
        return match.group(1).strip().upper(), True
    return None, False


def execute_plan(
    llm_config: LLMConfiguration,
    prompt_conf: PromptTemplate,
    docs: ty.List[RecordDocument]
):    
    # Load models once per config
    tokenizer, model, assistant_model = set_hf_model(llm_config.model_id_primary, llm_config.model_id_assistant)

    # Decoding args
    # TODO
    gen_kwargs = {"do_sample": False} if llm_config.strategy_decoding == "greedy" else {"do_sample": True, "top_p": 0.9}

    doc_row: RecordDocument
    for doc_row in docs:
        # Unpack duckdb row
        # bench_id, doc_id, lang, truth, content = doc_row

        full_prompt = prompt_conf.template_text.format(text=doc_row.text_content)

        # RUN INFERENCE
        logger.info(f"🚀 Running {llm_config.llm_configuration_id} on {doc_row.document_id}...")
        response_text, duration = execute_hf_model(full_prompt, tokenizer, model, assistant_model, gen_kwargs)
        
        # PARSE
        estimated_level, xml_success = extract_xml_cefr(response_text)
        
        # SAVE RECORD
        record = RecordResponse(
            benchmark_id=doc_row.benchmark_id,
            document_id=doc_row.document_id,
            prompt_id=prompt_conf.prompt_id,
            llm_configuration_id=llm_config.llm_configuration_id,
            response_llm=response_text,
            level_cefr_estimated=estimated_level,
            is_xml_success=xml_success,
            time_execution=duration
        )
        
        _d_obj = record.model_dump()
        _clause_placeholder = ", ".join(["?"] * len(_d_obj))

        con.execute(
            f"INSERT INTO responses VALUES ({_clause_placeholder})",
            tuple(_d_obj.values())
        )
        logger.info(f"   Saved. Time: {duration:.2f}s | Level: {estimated_level}")


def main():
    exec_plans = set_model_configurations()
    prompt_templates = set_prompt_template()

    # Fetch benchmark-Documents
    _docs = con.execute("SELECT * FROM documents").fetchall()
    _cols = [desc[0] for desc in con.description]
    docs = [RecordDocument(**dict(zip(_cols, doc))) for doc in _docs]

    seq_plans = list(itertools.product(exec_plans, prompt_templates))
    for llm_config, prompt_conf in seq_plans:
        execute_plan(llm_config, prompt_conf, docs)


# --------------------------------------------------------------------------
# MAIN ENTRY
# --------------------------------------------------------------------------
if __name__ == "__main__":
    init_db()
    set_datasets() # Downloads and populates DB
    main()

    # Verification
    logger.info("\n📊 Results Summary:")
    con.sql("SELECT llm_configuration_id, AVG(time_execution) as avg_time, COUNT(*) as count FROM responses GROUP BY 1").show()
    con.close()
    logger.info("✅ Evaluation complete.")
