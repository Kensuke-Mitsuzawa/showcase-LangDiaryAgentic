import logging
import typing as ty
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.embeddings import Embeddings
import google.generativeai as genai

from .clients import (
    CustomOllamaEmbeddings,
    CustomHFServerEmbeddings,
    CustomHFServerLLM,
    CustomOllamaServerLLM
)
from .configs import settings

logger = logging.getLogger(__name__)

def server_custom_hf() -> ty.Tuple[CustomHFServerLLM, CustomHFServerEmbeddings]:
    """Helper to load a model pipeline"""
    assert settings.Server_API_Endpoint is not None
    llm = CustomHFServerLLM(api_url=settings.Server_API_Endpoint)
    embedding = CustomHFServerEmbeddings(api_url=settings.Server_API_Endpoint)
    if llm.check_connection() is False:
        raise RuntimeError(f"The server is not available at {settings.Server_API_Endpoint}.")
    if embedding.check_connection() is False:
        raise RuntimeError(f"The server is not available at {settings.Server_API_Endpoint}.")        
    return llm, embedding

def server_ollama() -> ty.Tuple[CustomOllamaServerLLM, CustomOllamaEmbeddings]:
    assert settings.Server_API_Endpoint is not None
    llm = CustomOllamaServerLLM(api_url=settings.Server_API_Endpoint)
    embedding = CustomOllamaEmbeddings(settings.Server_API_Endpoint)
    if llm.check_connection() is False:
        raise RuntimeError(f"The server is not available at {settings.Server_API_Endpoint}.")
    if embedding.check_connection() is False:
        raise RuntimeError(f"The server is not available at {settings.Server_API_Endpoint}.")        
    return llm, embedding


class GeminiChatOpenAI(ChatOpenAI):
    def bind(self, **kwargs):
        cleaned_kwargs = {}
        for k, v in kwargs.items():
            if k == "max_length":
                cleaned_kwargs["max_tokens"] = max(v, 1024)
            elif k == "max_tokens":
                cleaned_kwargs["max_tokens"] = max(v, 1024)
            elif k == "enable_thinking":
                pass
            else:
                cleaned_kwargs[k] = v
        if "max_tokens" not in cleaned_kwargs:
            cleaned_kwargs["max_tokens"] = 1024
        return super().bind(**cleaned_kwargs)


class GeminiEmbeddings(Embeddings):
    def __init__(self, api_key: str, model: str):
        self.gemini_model = model
        if self.gemini_model == "all-MiniLM-L6-v2":
            self.gemini_model = "models/gemini-embedding-2"
        elif not self.gemini_model.startswith("models/"):
            self.gemini_model = f"models/{self.gemini_model}"
        genai.configure(api_key=api_key)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        embeddings = []
        for text in texts:
            result = genai.embed_content(model=self.gemini_model, content=text)
            embeddings.append(result['embedding'])
        return embeddings

    def embed_query(self, text: str) -> list[float]:
        result = genai.embed_content(model=self.gemini_model, content=text)
        return result['embedding']


if settings.Mode_Deployment == "cloud_api":
    assert settings.Cloud_API_Token is not None
    primary_model = settings.MODEL_NAME_Primary
    if not primary_model or "Qwen" in primary_model:
        primary_model = "gemini-3.5-flash"
    
    llm_large = GeminiChatOpenAI(
        openai_api_key=settings.Cloud_API_Token,
        openai_api_base="https://generativelanguage.googleapis.com/v1beta/openai/",
        model_name=primary_model,
        temperature=0.0
    )
    client_embedding_model_server = GeminiEmbeddings(
        api_key=settings.Cloud_API_Token,
        model=settings.MODEL_NAME_Embedding or "models/gemini-embedding-2"
    )
    tokenizer = None
elif settings.Mode_Deployment == "server_custom_hf":
    logger.info(f"connecting to the API endpoint: {settings.Server_API_Endpoint}")
    llm_large, client_embedding_model_server = server_custom_hf()
    logger.info("API is ready.")
elif settings.Mode_Deployment == "server_ollama":
    logger.info(f"connecting to the API endpoint: {settings.Server_API_Endpoint}")
    llm_large, client_embedding_model_server = server_ollama()
    logger.info("API is ready.")
else:
    raise ValueError(f"Invalid Mode_Deployment: {settings.Mode_Deployment}")


def create_compatible_chain(formatted_input: ty.List[ty.Tuple], 
                            llm):
    """
    Dynamically builds the chain based on whether input is String or List.
    """
    if settings.Mode_Deployment in ("server_custom_hf", "server_ollama", "cloud_api"):
        prompt = ChatPromptTemplate.from_messages(formatted_input)
    else:
        raise ValueError(f"Unsupported Mode_Deployment: {settings.Mode_Deployment}")
    
    chain = prompt | llm
    return chain
