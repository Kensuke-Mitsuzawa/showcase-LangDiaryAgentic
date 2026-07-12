import typing as ty
from datetime import datetime
from pydantic import BaseModel, Field
from typing import TypedDict
from ..static import PossibleLevelRewriting
from .generation_records import DiaryEntry
from .vector_store_entry import ErrorRecord

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

    # final_response: str = Field(description="Final response from the diary-agents.")
    # suggestion_response: str = Field(description="Suggestion response from the translator.")
    
    total_review: str

    # fields used for the managing the unknown expressions.
    unkown_expressions: ty.List[str] = Field(description="Unknown expressions in the draft text.")
    translation_pair_extracted: ty.List[TranslationReplacementInformation]
    
    # fields used for the RAG DB.
    retrieved_context: str = Field(description="Retrieved context from the vector store.")
    grammatical_errors_extracted: ty.List[ErrorRecord]
    

# --- Define State dictionary ---
class AgentState(BaseModel):
    diary_entry_input: DiaryEntry
    parameter_config_llm: ParameterConfig = Field(default_factory=ParameterConfig)
    
    processed_output: ty.Optional[ProcessedOutputInformation] = None

    # signal to convey the task status
    is_processor_success: bool = False
    is_archivist_success: bool = False

    # draft_text: str
    # retrieved_context: str
    # final_response: str
    # suggestion_response: str
    # unkown_expressions: ty.List[str]
    # total_review: str
    # translation_pair_extracted: ty.List[TranslationReplacementInformation]
    # grammatical_errors_extracted: ty.List[ErrorRecord]
    # # meta-information
    # lang_annotation: ty.Optional[str]
    # lang_diary_body: ty.Optional[str]
    # level_rewriting: PossibleLevelRewriting
    # diary_date: str
    # title_diary: str
    # primary_id_DiaryEntry: str
    # created_at: datetime
    # diary_entry: DiaryEntry
    # # task config
    # config_translator: TaskParameterConfig
    # config_archivist: TaskParameterConfig
    # config_rewriter: TaskParameterConfig
    # config_reviewer: TaskParameterConfig
