from langchain_core.pydantic_v1 import BaseModel, Field
from typing import Literal


class ErrorRecord(BaseModel):
    """Schema for an error entry in the database."""
    
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
