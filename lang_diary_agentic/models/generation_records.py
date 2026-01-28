import typing as ty
import hashlib
from pydantic import BaseModel, Field
from datetime import datetime

from ..static import PossibleLevelRewriting

class DiaryEntry(BaseModel):
    date_diary: str
    language_source: str
    language_annotation: str
    diary_original: str
    diary_replaced: str
    diary_rewritten: str
    level_rewriting: PossibleLevelRewriting
    model_id_tutor: str
    title_diary: str
    current_version: int
    created_at: datetime = Field(default_factory=datetime.now)    
    primary_id: ty.Optional[str] = None
    is_show: bool

    def model_post_init(self, context: ty.Any) -> None:
        if self.primary_id is None:
            datetime_str = self.created_at.isoformat()
            self.primary_id = f"{self.date_diary}-{datetime_str}"
        # end if
# end class


class UnknownExpressionEntry(BaseModel):
    expression: str
    expression_translation: str
    span_original: ty.Tuple[int, int]
    span_translation: ty.Tuple[int, int]
    language_source: str
    language_annotation: str
    remark_field: ty.Optional[str]
    created_at: datetime = Field(default_factory=datetime.now)
    primary_id_DiaryEntry: str = Field(description="primary_key of DiaryEntry table.")
    primary_id: ty.Optional[str] = None

    def model_post_init(self, context: ty.Any) -> None:
        if self.primary_id is None:
            datetime_str = self.created_at.isoformat()
            key_combination = f"{self.primary_id_DiaryEntry}_{self.expression}_{datetime_str}"
            hashlib_object = hashlib.sha256(key_combination.encode())
            hex_dig = hashlib_object.hexdigest()
            self.primary_id = hex_dig
        # end if
# end class


class HistoryRecord(BaseModel):
    history_id: str
    primary_id_DiaryEntry: str
    version_from: int
    version_to: int
    created_at: datetime
    changes: ty.Dict
