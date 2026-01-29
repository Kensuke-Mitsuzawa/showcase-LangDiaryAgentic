import itertools
import duckdb
import torch
import uuid
import time
import typing as ty
import pydantic
import json
import hashlib
import jsonlines
import re
import sys
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
    is_use_think_assistant: bool = False
    max_new_token: int = 512
    temperature: float = 0.0
    top_p: float = 0.9
    top_k: int = 50

    class Config:
        arbitrary_types_allowed = True

    @pydantic.model_validator(mode='after')
    def set_validation(self):
        if self.strategy_decoding == 'greedy':
            self.temperature = 0.0
            self.top_p = 1.0
            self.top_k = 0
        
        return self


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
    is_eos_success: bool
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
            strategy_decoding VARCHAR,
            is_use_think_assistant BOOLEAN,
            max_new_token INTEGER,
            temperature DOUBLE,
            top_p DOUBLE,
            top_k INTEGER
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
            is_eos_success BOOLEAN,
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
        # See: https://huggingface.co/UniversalCEFR
        ds_eng = load_dataset("UniversalCEFR/elle_et")['train']
        ds_fra = load_dataset("UniversalCEFR/kwiqiz_fr")['train']
        ds_deu = load_dataset("UniversalCEFR/merlin_de")['train']
    except Exception:
        logger.warning("⚠️  Dataset not found or error. Generating dummy data for demo.")
        return

    # Filter for target languages
    target_langs = ['en', 'fr', 'de']
    
    ds_all = list(ds_eng) + list(ds_fra) + list(ds_deu)

    for row in ds_all:
        lang = row.get('lang') or row.get('language')
        if lang not in target_langs:
            continue
        
        _d_elem_hash_id = dict(
            benchmark_id="UniversalCEFR_v1",
            language=lang,
            text_content=row['text'],
            level_cefr_truth=row['cefr_level']
        )
        doc_hash = hashlib.sha256(json.dumps(_d_elem_hash_id, sort_keys=True).encode()).hexdigest()
        doc = RecordDocument(
            benchmark_id="UniversalCEFR_v1",
            document_id=f'UniversalCEFR_v1/{doc_hash}',
            language=lang,
            level_cefr_truth=row['cefr_level'],
            text_content=row['text']
        )
        
        # Insert into DuckDB
        con.execute(
            "INSERT OR IGNORE INTO documents VALUES (?, ?, ?, ?, ?)",
            (doc.benchmark_id, doc.document_id, doc.language, doc.level_cefr_truth, doc.text_content)
        )
        con.commit()
    logger.info("✅ Datasets loaded into DuckDB.")

# --------------------------------------------------------------------------
# 3. LLM EXECUTION LOGIC (Speculative Decoding)
# --------------------------------------------------------------------------


def set_hf_model(model_id_primary: str, 
                 model_id_assistant: Optional[str]):
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
    #     # 4. The Prompt Template
    seq_records = []
    reader = jsonlines.open('setting_prompt_records.jsonl')
    for obj in reader:
        PROMPT_TEMPLATE = obj['prompt_template']
        prompt_id = obj['prompt_name']

        # Insert into DB
        con.execute("INSERT OR REPLACE INTO prompts VALUES (?, ?)", (prompt_id, PROMPT_TEMPLATE))
        con.commit()

        _r = PromptTemplate(prompt_id=prompt_id, template_text=PROMPT_TEMPLATE)
        seq_records.append(_r)
    # end for
    return seq_records



def set_model_configurations() -> ty.List[LLMConfiguration]:
    # Model Configurations
    seq_records = []
    reader = jsonlines.open('setting_configuration.jsonl')
    for obj in reader:
        _r = LLMConfiguration(**obj)
        seq_records.append(_r)
    # end for

    plans = seq_records

    # 2. Insert Configs to DB
    for p in plans:
        _d_obj = p.model_dump()
        _clause_placeholder = ", ".join(["?"] * len(_d_obj))
        con.execute(f"INSERT OR REPLACE INTO llm_configurations VALUES ({_clause_placeholder})", 
                   tuple(_d_obj.values()))
        con.commit()
    # end for
    return plans



def execute_hf_model(prompt: str, 
                     tokenizer: AutoTokenizer, 
                     model: AutoModelForCausalLM, 
                     assistant_model: Optional[AutoModelForCausalLM], 
                     decoding_config: Dict,
                     max_new_tokens: int = 512) -> ty.Tuple[str, float]:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    n_current_tokens = inputs.input_ids.shape[-1]

    decoding_config['max_new_tokens'] = n_current_tokens + max_new_tokens
    
    start_time = time.time()
    
    # Hugging Face generate() handles the speculative loop internally
    # when assistant_model is passed.
    outputs = model.generate(
        **inputs,
        assistant_model=assistant_model, 
        pad_token_id=tokenizer.pad_token_type_id,
        eos_token_id=tokenizer.eos_token_id,
        **decoding_config
    )
    
    duration = time.time() - start_time

    input_length = inputs.input_ids.shape[-1]
    new_tokens = outputs[0][input_length:]

    generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    
    is_eos_with_end_token = False
    if tokenizer.eos_token_id in new_tokens:
        is_eos_with_end_token = True
    # end if
    
    if is_eos_with_end_token is False:
        logger.warning("⚠️  Warning: Output may be incomplete (no EOS token generated).")
    # end if
    
    return generated_text, duration, is_eos_with_end_token

# --------------------------------------------------------------------------
# 4. PARSING & LOGIC
# --------------------------------------------------------------------------

def extract_xml_cefr(text: str) -> tuple[Optional[str], bool]:
    # Look for <cefr>...</cefr>
    match = re.findall(r"<cefr>(.*?)</cefr>", text, re.IGNORECASE)
    if len(match) > 1:
        return match[-1].strip().upper(), True
    elif len(match) == 1:
        _extracted = match[0].strip().upper()
        if _extracted == 'LEVEL':
            return None, False
        elif _extracted in ['A1', 'A2', 'B1', 'B2', 'C1', 'C2']:
            return _extracted, True
        else:
            return None, False
    elif len(match) == 0:
        return None, False
    else:
        return None, False


def execute_plan(
    llm_config: LLMConfiguration,
    prompt_conf: PromptTemplate,
    docs: ty.List[RecordDocument],
    model: ty.Optional[AutoModelForCausalLM] = None, 
    assistant_model: Optional[AutoModelForCausalLM] = None, 
    tokenizer: ty.Optional[AutoTokenizer] = None,
    is_dry_run: bool = False
    ) -> ty.Tuple[AutoTokenizer, AutoModelForCausalLM, Optional[AutoModelForCausalLM]]:
    if model is None or tokenizer is None:
        assert llm_config.model_id_primary is not None, "Model ID must be provided."
        # assert llm_config.model_id_assistant is not None, "Assistant Model ID must be provided."
        # Load models once per config
        tokenizer, model, assistant_model = set_hf_model(llm_config.model_id_primary, llm_config.model_id_assistant)  # type: ignore
    # end if

    # Decoding args
    gen_kwargs = {}
    if llm_config.strategy_decoding == "greedy":
        gen_kwargs = {
            "do_sample": False,
            "temperature": llm_config.temperature,
            "top_p": llm_config.top_p,
            "top_k": llm_config.top_k
        }
    else:
        gen_kwargs = {
            "do_sample": True,
            "temperature": llm_config.temperature,
            "top_p": llm_config.top_p,
            "top_k": llm_config.top_k
        }
    # end if

    doc_row: RecordDocument
    for doc_row in docs:
        con.execute(f"SELECT count(*) from responses WHERE document_id = '{doc_row.document_id}' AND prompt_id = '{prompt_conf.prompt_id}' AND llm_configuration_id = '{llm_config.llm_configuration_id}'",)
        
        n_existing = con.fetchone()

        if n_existing is None:
            continue
        if n_existing[0] > 0 and is_dry_run is False:
            logger.info(f"⏭️  Skipping {doc_row.document_id} for {llm_config.llm_configuration_id} | {prompt_conf.prompt_id} (already exists).")
            continue
        # end if

        full_prompt = prompt_conf.template_text.format(text=doc_row.text_content)

        # RUN INFERENCE
        logger.info(f"🚀 Running {llm_config.llm_configuration_id} on {doc_row.document_id}...")
        response_text, duration, is_eos_success = execute_hf_model(full_prompt, tokenizer, model, assistant_model, gen_kwargs, max_new_tokens=llm_config.max_new_token)
        
        # PARSE
        estimated_level, is_xml_success = extract_xml_cefr(response_text)
        if is_xml_success is False:
            estimated_level = None
        # end if
        
        # SAVE RECORD
        record = RecordResponse(
            benchmark_id=doc_row.benchmark_id,
            document_id=doc_row.document_id,
            prompt_id=prompt_conf.prompt_id,
            llm_configuration_id=llm_config.llm_configuration_id,
            response_llm=response_text,
            level_cefr_estimated=estimated_level,
            is_xml_success=is_xml_success,
            is_eos_success=is_eos_success,
            time_execution=duration
        )

        logger.info(f"  🚀 Time: {duration:.2f}s | Level: {estimated_level} | Truth: {doc_row.level_cefr_truth}")
        if is_dry_run is False:
            _d_obj = record.model_dump()
            _clause_placeholder = ", ".join(["?"] * len(_d_obj))

            con.execute(
                f"INSERT INTO responses VALUES ({_clause_placeholder})",
                tuple(_d_obj.values())
            )
            con.commit()
        else:
            logger.warning("  🧪 Dry run mode: Not saving to DB.")
        # end if
    # end for

    return tokenizer, model, assistant_model


def get_top_n_lang_groups(seq_docs: ty.List[RecordDocument], n: int) -> List[RecordDocument]:
    seq_stack = []
    # Count occurrences of each language
    for _, lang_group in itertools.groupby(sorted(seq_docs, key=lambda x: x.language), key=lambda x: x.language):
        _seq_group = list(lang_group)
        if len(_seq_group) <= n:
            filtered_docs = _seq_group
            seq_stack += filtered_docs
        else:
            filtered_docs = _seq_group[:n]
            seq_stack += filtered_docs
    # end for

    return seq_stack


def main(is_dry_run: bool = False, n_docs_per_lang: int = 10):
    exec_plans = set_model_configurations()
    prompt_templates = set_prompt_template()

    # Fetch benchmark-Documents
    _docs = con.execute("SELECT * FROM documents").fetchall()
    _cols = [desc[0] for desc in con.description]
    docs = [RecordDocument(**dict(zip(_cols, doc))) for doc in _docs]
    docs = get_top_n_lang_groups(docs, n=n_docs_per_lang)

    logger.info(f"📝 Total documents to process: {len(docs)}")

    seq_plans = list(itertools.product(exec_plans, prompt_templates))
    _tokenizer, _model, _assistant_model = None, None, None
    for llm_config, prompt_conf in seq_plans:
        logger.info(f"=== Starting Execution: Config={llm_config.llm_configuration_id} | Prompt={prompt_conf.prompt_id} ===")
        _tokenizer, _model, _assistant_model = execute_plan(
            llm_config, 
            prompt_conf, 
            docs,
            model=_model,
            assistant_model=_assistant_model,
            tokenizer=_tokenizer,
            is_dry_run=is_dry_run)
    # end for

# --------------------------------------------------------------------------
# MAIN ENTRY
# --------------------------------------------------------------------------
if __name__ == "__main__":
    from pathlib import Path
    assert Path().cwd().name == "eval_cefr_classification", f"Please run the script from the 'eval_cefr_classification' directory. Current dir: {Path().cwd()}"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.basicConfig(level=logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.getLogger().addHandler(logging.FileHandler("eval_cefr_classification.log"))

    logger.info("----- 🧪 Starting CEFR Classification Evaluation -----")
    logger.info(f"Using device: {device}")
    
    init_db()
    set_datasets() # Downloads and populates DB
    main(is_dry_run=False, n_docs_per_lang=10)

    # Verification
    logger.info("\n📊 Results Summary:")
    con.sql("SELECT llm_configuration_id, AVG(time_execution) as avg_time, COUNT(*) as count FROM responses GROUP BY 1").show()
    con.close()
    logger.info("✅ Evaluation complete.")
