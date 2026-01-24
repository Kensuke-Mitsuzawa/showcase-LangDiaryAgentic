import requests
import json
import logging
from typing import Any, List, Optional, Dict
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from langchain_core.embeddings import Embeddings

from ..configs import settings
from .base import ClientEmbeddingModel, ClientLLM

assert settings.Server_API_Endpoint is not None


logger = logging.getLogger(__name__)


class CustomOllamaEmbeddings(ClientEmbeddingModel, Embeddings):
    def __init__(self, 
                 base_url: str = settings.Server_API_Endpoint, 
                 model: str = "all-minilm"):
        self.base_url = base_url
        self.model = model

    def check_connection(self) -> bool:
        """Checks if Ollama is running."""
        try:
            # Ollama root endpoint returns "Ollama is running"
            response = requests.get(f"{self.base_url}/")
            response.raise_for_status()
            return True
        except Exception:
            return False

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embeds a list of texts. 
        Note: The native Ollama /api/embeddings endpoint processes one prompt at a time.
        """
        embeddings = []
        for text in texts:
            # Call the internal helper to avoid code duplication
            embedding = self._get_embedding(text)
            embeddings.append(embedding)
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """Embeds a single query text."""
        return self._get_embedding(text)

    def _get_embedding(self, text: str) -> List[float]:
        """Internal helper to make the API request."""
        payload = {
            "model": self.model,
            "prompt": text
        }
        
        try:
            response = requests.post(f"{self.base_url}/api/embeddings", json=payload)
            response.raise_for_status()
            # Ollama returns {"embedding": [0.1, ...]}
            return response.json()["embedding"]
        except Exception as e:
            # In production, you might want to log this error
            logger.error(f"Error embedding text: {e}")
            return []


class CustomOllamaServerLLM(ClientLLM, BaseChatModel):
    api_url: str = settings.Server_API_Endpoint # Default Ollama URL
    model_name: str = settings.MODEL_NAME_Primary # You must specify the model tag here

    @property
    def _llm_type(self) -> str:
        return "custom_ollama_server"

    def check_connection(self) -> bool:
        """Checks if Ollama is running."""
        try:
            # Ollama root endpoint usually returns "Ollama is running"
            response = requests.get(f"{self.api_url}/")
            response.raise_for_status()
            return True
        except Exception:
            return False

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> ChatResult:
        
        # -------------------------------------------------------
        # STEP 1: Convert LangChain Messages -> Ollama JSON
        # -------------------------------------------------------
        ollama_formatted_chat = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                role = "system"
            elif isinstance(msg, HumanMessage):
                role = "user"
            elif isinstance(msg, AIMessage):
                role = "assistant"
            else:
                role = "user" 

            ollama_formatted_chat.append({"role": role, "content": msg.content})

        # -------------------------------------------------------
        # STEP 2: Prepare Payload (Ollama Specifics)
        # -------------------------------------------------------
        # Ollama puts hyperparameters inside an 'options' dict
        options = {
            "temperature": kwargs.get("temperature", 0.7),
            # Note: Ollama uses 'num_predict' for max tokens, not 'max_length'
            "num_predict": kwargs.get("max_tokens", kwargs.get("max_length", 512)), 
            "top_p": kwargs.get("top_p", 0.9),
        }

        # Add stop tokens if provided
        if stop:
            options["stop"] = stop

        payload = {
            "model": self.model_name,
            "messages": ollama_formatted_chat,
            "stream": False,  # We want a single JSON response
            "options": options
        }
        
        # -------------------------------------------------------
        # STEP 3: Send Request
        # -------------------------------------------------------
        finish_reason = "unknown"
        token_usage = {}
        generated_text = ""

        try:
            # Note the endpoint is /api/chat
            response = requests.post(f"{self.api_url}/api/chat", json=payload)
            response.raise_for_status()
            
            response_json = response.json()
            
            # Extract content from Ollama's specific response structure
            # Structure: { "message": { "role": "assistant", "content": "..." }, "done": true, ... }
            generated_text = response_json.get("message", {}).get("content", "")
            
            # Map Ollama's 'done_reason' to standard finish reasons
            done_reason = response_json.get("done_reason", "stop")
            finish_reason = done_reason if done_reason else "stop"

            # Extract Usage stats (Ollama provides these fields at root)
            token_usage = {
                "prompt_tokens": response_json.get("prompt_eval_count", 0),
                "completion_tokens": response_json.get("eval_count", 0),
                "total_tokens": response_json.get("prompt_eval_count", 0) + response_json.get("eval_count", 0)
            }

        except Exception as e:
            generated_text = f"Error connecting to Ollama: {e}"

        # -------------------------------------------------------
        # STEP 4: Return Result
        # -------------------------------------------------------
        message = AIMessage(
            content=generated_text,
            response_metadata={
                "finish_reason": finish_reason,
                "token_usage": token_usage,
                "model_name": self.model_name
            }
        )

        return ChatResult(generations=[ChatGeneration(message=message)])
    

# if __name__ == '__main__':
#     CustomOllamaEmbeddings(base_url="192.168.2.200:11434").check_connection()
#     CustomOllamaServerLLM(api_url="192.168.2.200:11434").check_connection()