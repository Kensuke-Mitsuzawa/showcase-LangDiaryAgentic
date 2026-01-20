from typing import Any, List, Optional
from langchain_core.language_models.llms import LLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
import requests
import logging

from lang_diary_agentic.logging_configs import apply_logging_suppressions
apply_logging_suppressions()

logger = logging.getLogger(__name__)



class CustomHFServerLLM(LLM):
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

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        # Payload matching the Pydantic model on server
        payload = {
            "prompt": prompt,
            "max_length": kwargs.get("max_length", 512),
            "temperature": kwargs.get("temperature", 0.7)
        }
        
        try:
            response = requests.post(f"{self.api_url}/generate", json=payload)
            response.raise_for_status()
            return response.json()["generated_text"]
        except Exception as e:
            return f"Error connecting to server: {e}"
