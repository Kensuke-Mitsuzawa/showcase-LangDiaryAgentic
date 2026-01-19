from langchain_core.pydantic_v1 import BaseModel, Field
from typing import Literal, Any
from ..configs import languages_code


class ErrorRecord(BaseModel):
    """Schema for an error entry in the database."""
    language_diary_text: str = Field()
    
    language_annotation_text: str = Field()
    
    error_rule: str = Field(
        description="The specific grammatical rule violated (e.g. 'Gender agreement for furniture')."
    )
    
    example_phrase: str = Field(
        description="The short phrase from the user's text containing the error."
    )
    
    correction: str = Field(
        description="The correct version of the phrase."
    )
    
    category: Literal["Grammar", "Vocabulary", "Spelling", "None"] = Field(
        description="The linguistic category of the error."
    )

    def to_string(self):
        """Helper to format this for embedding."""
        return f"{self.category} Error: {self.error_rule}. Example: '{self.example_phrase}' -> '{self.correction}'"

    
    def model_post_init(self, __context: Any) -> None:
        # manipulation on fields
        assert self.language_annotation_text in languages_code, f"The language code {self.language_annotation_text} is not valid. Check the language code in 2 character."
        assert self.language_diary_text in languages_code, f"The language code {self.language_diary_text} is not valid. Check the language code in 2 character."
