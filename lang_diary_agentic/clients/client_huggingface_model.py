from typing import Any, List, Optional
from langchain_core.language_models.llms import LLM
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatResult, ChatGeneration

from langchain_core.embeddings import Embeddings
import requests
import logging
import json

from .base import ClientEmbeddingModel, ClientLLM

from lang_diary_agentic.logging_configs import apply_logging_suppressions
apply_logging_suppressions()

logger = logging.getLogger(__name__)




class CustomHFServerEmbeddings(ClientEmbeddingModel, Embeddings):
    def __init__(self, api_url: str):
        self.api_url = api_url

    def check_connection(self) -> bool:
        try:
            response = requests.get(f"{self.api_url}/alive")
            response.raise_for_status()
            return True
        except Exception as e:
            return False

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # This function is used when you call vector_store.add_texts()
        embeddings = []
        for text in texts:
            response = requests.post(f"{self.api_url}/embedding", json={"text": text})
            response.raise_for_status()
            embeddings.append(response.json()["embedding"])
        return embeddings

    def embed_query(self, text: str) -> list[float]:
        # This function is used when you call vector_store.similarity_search()
        response = requests.post(f"{self.api_url}/embedding", json={"text": text})
        response.raise_for_status()
        return response.json()["embedding"]


class CustomHFServerLLM(ClientLLM, BaseChatModel):
    api_url: str
    
    @property
    def _llm_type(self) -> str:
        return "custom_hf_server"
    
    def check_connection(self) -> bool:
        try:
            response = requests.get(f"{self.api_url}/alive")
            response.raise_for_status()
            return True
        except Exception as e:
            return False
        
    def get_available_models(self) -> List[str]:
        try:
            response = requests.get(f"{self.api_url}/generate-model-id")
            response.raise_for_status()
            response_json = response.json()
            model_id = response_json.get("model_id", "")
        except Exception as e:
            generated_text = f"Error connecting to server: {e}"
        # end try

        return [model_id]

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
        ) -> ChatResult:
        
        # -------------------------------------------------------
        # STEP 1: Convert LangChain Messages -> HuggingFace JSON
        # -------------------------------------------------------
        hf_formatted_chat = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                role = "system"
            elif isinstance(msg, HumanMessage):
                role = "user"
            elif isinstance(msg, AIMessage):
                role = "assistant"
            else:
                role = "user" # fallback
            
            hf_formatted_chat.append({"role": role, "content": msg.content})
        # end for

        payload = {
            "prompt": json.dumps(hf_formatted_chat),
            "max_length": kwargs.get("max_length", 512),
            "temperature": kwargs.get("temperature", 0.7),
            "enable_thinking": kwargs.get("enable_thinking", True)
        }
        
        finish_reason = "unknown"
        token_usage = {}
        try:
            response = requests.post(f"{self.api_url}/generate", json=payload)
            response.raise_for_status()
            response_json = response.json()
            generated_text = response_json.get("generated_text", "")
            finish_reason = response_json.get("finish_reason", "unknown")
            token_usage = response_json.get("usage", {})
        except Exception as e:
            generated_text = f"Error connecting to server: {e}"
        # end
        # 2. Create AIMessage with EXTRA DATA
        message = AIMessage(
            content=generated_text,
            response_metadata={
                "finish_reason": finish_reason,
                "token_usage": token_usage
            }
        )
    
        return ChatResult(generations=[ChatGeneration(message=message)])