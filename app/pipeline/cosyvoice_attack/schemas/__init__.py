"""
Pydantic schemas for CosyVoice Attack Pipeline.
"""
from app.pipeline.cosyvoice_attack.schemas.pipeline_config import CosyVoicePipelineConfig
from app.pipeline.cosyvoice_attack.schemas.reference_result import ReferenceResult
from app.pipeline.cosyvoice_attack.schemas.text_prompts_result import TextPromptsResult
from app.pipeline.cosyvoice_attack.schemas.generation_result import GenerationResult
from app.pipeline.cosyvoice_attack.schemas.validation_result import ValidationResult
from app.pipeline.cosyvoice_attack.schemas.formatting_result import FormattingResult

__all__ = [
    "CosyVoicePipelineConfig",
    "ReferenceResult",
    "TextPromptsResult",
    "GenerationResult",
    "ValidationResult",
    "FormattingResult",
]
