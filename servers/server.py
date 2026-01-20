import typing
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging

from lang_diary_agentic.logging_configs import apply_logging_suppressions
from lang_diary_agentic import configs

apply_logging_suppressions()
logger = logging.getLogger(__name__)


MODEL_NAME = configs.MODEL_NAME_Primary

# 1. Setup the App
app = FastAPI(title="LinguaLog Model API")

# 2. Configuration

device = "cuda" if torch.cuda.is_available() else "cpu"

logger.info(f"Loading model {MODEL_NAME} on {device}...")

# 3. Load Model & Tokenizer (Global variables)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto", # Automatically distributes to GPU
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    trust_remote_code=False
)

logger.info(f"Model {MODEL_NAME} loaded successfully.")
logger.info("Ready.")

# 4. Define the Request Data Structure
class GenerateRequest(BaseModel):
    prompt: str
    max_length: int = 512
    temperature: float = 0.7


# 5. Define the Endpoint
@app.post("/generate")
async def generate_text(request: GenerateRequest):
    try:
        # Format input (Simple chat template logic)
        inputs = tokenizer(request.prompt, return_tensors="pt").to(device)
        
        # Generate
        outputs = model.generate(
            **inputs,
            max_new_tokens=request.max_length,
            temperature=request.temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        
        # Decode
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Clean up the prompt from the response if needed
        response_text = generated_text.replace(request.prompt, "").strip()
        
        return {"generated_text": response_text}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
# end def


@app.get("/alive")
async def return_status():
    return {"status": "OK"}
# end


# Run with: uvicorn server:app --host 0.0.0.0 --port 8000