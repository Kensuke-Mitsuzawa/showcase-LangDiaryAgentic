import typing as ty

from pydantic import BaseModel, Field
from datetime import datetime


class DiaryEntry(BaseModel):
    date_diary: str
    language_source: str
    language_annotation: str
    diary_original: str
    diary_replaced: str
    diary_corrected: str
    created_at: datetime = Field(default_factory=datetime.now)    
    primary_id: ty.Optional[str] = None

    def model_post_init(self, context: ty.Any) -> None:
        if self.primary_id is None:
            datetime_str = self.created_at.isoformat()
            self.primary_id = f"{self.date_diary}_{datetime_str}"
        # end if
# end class


class UnknownExpressionEntry(BaseModel):
    expression: str
    expression_translation: str
    language_source: str
    language_annotation: str
    created_at: datetime = Field(default_factory=datetime.now)    
    primary_id: ty.Optional[str] = None

    def model_post_init(self, context: ty.Any) -> None:
        if self.primary_id is None:
            datetime_str = self.created_at.isoformat()
            self.primary_id = f"{self.expression}_{datetime_str}"
        # end if
# end class