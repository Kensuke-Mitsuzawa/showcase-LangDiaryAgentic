import typing
import torch
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain_community.embeddings import HuggingFaceEmbeddings

import logging

from lang_diary_agentic.logging_configs import apply_logging_suppressions

from lang_diary_agentic.configs import settings

apply_logging_suppressions()

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


# 1. Setup the App
app = FastAPI(title="LinguaLog Model API")

# 2. Configuration

device = "cuda" if torch.cuda.is_available() else "cpu"

logger.info(f"Loading model {settings.MODEL_NAME_Primary} on {device}...")

# 3. Load Model & Tokenizer (Global variables)
tokenizer = AutoTokenizer.from_pretrained(settings.MODEL_NAME_Primary, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    settings.MODEL_NAME_Primary,
    device_map="auto", # Automatically distributes to GPU
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    trust_remote_code=False
)

terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|end|>"),
        tokenizer.convert_tokens_to_ids("<|assistant|>") # Safety net
]

logger.info(f"Model {settings.MODEL_NAME_Primary} loaded successfully.")


embedding_function = HuggingFaceEmbeddings(model_name=settings.MODEL_NAME_Embedding)
logger.info(f"Model {settings.MODEL_NAME_Embedding} loaded successfully.")

logger.info("Ready.")

# 4. Define the Request Data Structure
class GenerateRequest(BaseModel):
    prompt: str
    max_length: int = 512
    temperature: float = 0.7
    enable_thinking: bool = True


class EmbeddingRequest(BaseModel):
    text: str


def determine_finish_reason(output_ids, 
                            max_new_tokens,
                            input_ids,
                            eos_token_id) -> str:
    """
    output_ids: The list of token IDs generated
    max_new_tokens: The limit you set (e.g., 512)
    eos_token_id: The ID of the model's stop token (e.g., 151645 for Qwen)
    """
    
    # 1. Check if the LAST token is the EOS token
    if output_ids[-1] == eos_token_id:
        return "stop"
    
    # 2. Check if we hit the length limit
    # Note: len(output_ids) includes the prompt tokens usually, so be careful.
    # A safer check is: Did we generate exactly 'max_new_tokens'?
    generated_count = len(output_ids) - len(input_ids[0]) # Calculate only NEW tokens
    
    if generated_count >= max_new_tokens:
        return "length"
    
    # Fallback (rare)
    return "unknown"


@app.post("/embedding")
async def generate_embedding(request: EmbeddingRequest):
    try:
        # Generate embedding (returns List[float])
        vector = embedding_function.embed_query(request.text)
        return {"embedding": vector}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate")
async def generate_text(request: GenerateRequest):
    try:
        logger.debug(f"Generation args: max-length: {request.max_length}, temperature: {request.temperature}, enable_thinking: {request.enable_thinking}")
        # prompt may be in the JSON format.
        try:
            prompt_json = json.loads(request.prompt)
            prompt_text = tokenizer.apply_chat_template(prompt_json, 
                                                        tokenize=False, 
                                                        add_generation_prompt=True, 
                                                        enable_thinking=request.enable_thinking)
        except Exception as e:
            logger.error(f'Error = {e}')
            prompt_text = request.prompt
        # end if

        # Format input (Simple chat template logic)
        inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
        logger.debug("Executing the generation...")
        # Generate
        outputs = model.generate(
            **inputs,
            max_new_tokens=request.max_length,
            temperature=request.temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
        logger.debug("Generation completed")
        
        # Decode
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        finish_reason = determine_finish_reason(outputs[0], request.max_length, inputs["input_ids"], tokenizer.eos_token_id)
        
        # Clean up the prompt from the response if needed
        response_text = generated_text.replace(request.prompt, "").strip()
        
        return {
            "generated_text": response_text,
            "finish_reason": finish_reason
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
# end def

@app.get("/generate-model-id")
async def return_model_id():
    return {"model_id": settings.MODEL_NAME_Primary}
# end

@app.get("/alive")
async def return_status():
    return {"status": "OK"}
# end


# Run with: uvicorn server:app --host 0.0.0.0 --port 8000