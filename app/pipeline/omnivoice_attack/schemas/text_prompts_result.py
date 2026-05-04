"""
Schema for text prompt preparation result.
"""
from pathlib import Path
from pydantic import BaseModel, Field


class TextPromptsResult(BaseModel):
    """Result from text prompt preparation (Step 2).

    Attributes:
        prompts_metadata_path: Path to JSON file with speaker -> text prompts mapping.
        total_prompts: Total number of text prompts assigned across all speakers.
    """

    prompts_metadata_path: Path = Field(
        ...,
        description="Path to text_prompts.json with speaker text prompt assignments"
    )
    total_prompts: int = Field(
        ...,
        description="Total number of text prompts assigned across all speakers"
    )

    class Config:
        """Pydantic model configuration."""
        frozen = False
        arbitrary_types_allowed = True
