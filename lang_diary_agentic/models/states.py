import typing as ty
from pydantic import BaseModel, Field
from ..static import PossibleLevelRewriting
from .generation_records import DiaryEntry
from .vector_store_entry import ErrorRecord

class ElementRewritingObject(BaseModel):
    phrase_target: str
    phrase_rewritten: str
    explanation: str

class TaskParameterConfig(BaseModel):
    is_execute: bool = True
    max_tokens: int = 512
    enable_thinking: bool = False
    temperature: float = 0.5
    top_p: float = 0.2
    model_name: ty.Optional[str] = Field(description="Model name", default_factory=lambda: None)


class TranslationReplacementInformation(BaseModel):
    expression_original: str
    expression_translation: str
    span_original: ty.Tuple[int, int]
    span_translation: ty.Tuple[int, int]



class ParameterConfig(BaseModel):
    # task config
    config_translator: TaskParameterConfig = Field(default_factory=TaskParameterConfig)
    config_archivist: TaskParameterConfig = Field(default_factory=TaskParameterConfig)
    config_rewriter: TaskParameterConfig = Field(default_factory=TaskParameterConfig)
    config_reviewer: TaskParameterConfig = Field(default_factory=TaskParameterConfig)
    

class ProcessedOutputInformation(BaseModel):
    diary_replaced: str = Field(description="Draft text of the diary entry. A copy of the original input text.")
    diary_rewritten: str = Field(description="Draft text of the diary entry after the rewriting process.")
    
    total_review: str

    # fields used for the managing the unknown expressions.
    unkown_expressions: ty.List[str] = Field(description="Unknown expressions in the draft text.")
    translation_pair_extracted: ty.List[TranslationReplacementInformation]
    
    # fields used for the RAG DB.
    retrieved_context: str = Field(description="Retrieved context from the vector store.")
    grammatical_errors_extracted: ty.List[ErrorRecord]
    phrases_rewritten: ty.List[ElementRewritingObject]
    
    evaluation_current_level: PossibleLevelRewriting

# --- Define State dictionary ---
class AgentState(BaseModel):
    diary_entry_input: DiaryEntry
    parameter_config_llm: ParameterConfig = Field(default_factory=ParameterConfig)
    
    processed_output: ty.Optional[ProcessedOutputInformation] = None

    # signal to convey the task status
    is_processor_success: bool = False
    is_archivist_success: bool = False
