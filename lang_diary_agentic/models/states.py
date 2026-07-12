import typing as ty
from datetime import datetime
from pydantic import BaseModel
from typing import TypedDict
from ..static import PossibleLevelRewriting
from .generation_records import DiaryEntry
from .vector_store_entry import ErrorRecord

class TaskParameterConfig(BaseModel):
    is_execute: bool = True
    max_tokens: int = 512
    enable_thinking: bool = False


class TranslationReplacementInformation(BaseModel):
    expression_original: str
    expression_translation: str
    span_original: ty.Tuple[int, int]
    span_translation: ty.Tuple[int, int]


# --- Define State dictionary ---
class AgentState(TypedDict):
    draft_text: str
    retrieved_context: str
    final_response: str
    suggestion_response: str
    unkown_expressions: ty.List[str]
    total_review: str
    translation_pair_extracted: ty.List[TranslationReplacementInformation]
    grammatical_errors_extracted: ty.List[ErrorRecord]
    # meta-information
    lang_annotation: ty.Optional[str]
    lang_diary_body: ty.Optional[str]
    level_rewriting: PossibleLevelRewriting
    diary_date: str
    title_diary: str
    primary_id_DiaryEntry: str
    created_at: datetime
    diary_entry: DiaryEntry
    # signal to convey the task status
    is_processor_success: bool
    is_archivist_success: bool
    # task config
    config_translator: TaskParameterConfig
    config_archivist: TaskParameterConfig
    config_rewriter: TaskParameterConfig
    config_reviewer: TaskParameterConfig
