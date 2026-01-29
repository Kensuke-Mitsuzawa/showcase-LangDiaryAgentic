import abc
import typing as ty

from pydantic import BaseModel, Field


class ClientEmbeddingModel(abc.ABC):
    pass


class ClientLLM(abc.ABC):
    @abc.abstractmethod
    def get_available_models(self) -> ty.List[str]:
        raise NotImplementedError()


class GenerationParameter(BaseModel):
    model_name: str = Field(..., description="The name of the model to use for generation.")

    temperature: float = Field(0.7, description="Sampling temperature to use.")
    max_tokens: int = Field(512, description="Maximum number of tokens to generate.")

    top_p: float = Field(0.9, description="Nucleus sampling parameter.")

    enable_thinking: bool = Field(True, description="Whether to enable thinking.")

    stop: ty.Optional[ty.List[str]] = Field(None, description="List of stop sequences.")

    # fields for speculative decoding
    draft_model_name: ty.Optional[str] = Field(None, description="The name of the draft model to use for speculative decoding.")
    num_assistant_tokens: int = Field(5, description="Number of speculative tokens to generate.")