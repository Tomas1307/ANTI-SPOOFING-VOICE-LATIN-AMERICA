"""
Pipeline configuration schema for Partial Spoof Pipeline.
"""
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field


class PartialSpoofPipelineConfig(BaseModel):
    """Runtime configuration for the Partial Spoof Pipeline.

    Allows overriding default settings and controlling which steps execute.
    The attack_system field is required and determines which voice cloning
    backend is used for generating cloned speech.

    Attributes:
        attack_system: Voice cloning system identifier.
        run_step_1: Execute Step 1 (transcribe bonafide audio).
        run_step_2: Execute Step 2 (generate cloned speech).
        run_step_3: Execute Step 3 (forced alignment).
        run_step_4: Execute Step 4 (select words to replace).
        run_step_5: Execute Step 5 (splice audio).
        run_step_6: Execute Step 6 (validate splice quality).
        run_step_7: Execute Step 7 (format output to LA).
        tiers: Word replacement tiers to generate.
        device_override: Override compute device from settings.
        random_seed_override: Override random seed from settings.
        output_dir_override: Override output directory from settings.
        skip_existing: Skip already-generated files during resume.
    """

    attack_system: str = Field(
        default="fishgram",
        description="Voice cloning system: fishgram, qwen, cosyvoice, outetts, chatterbox, openvoice",
    )
    run_step_1: bool = Field(default=True, description="Execute Step 1: transcribe bonafide audio")
    run_step_2: bool = Field(default=True, description="Execute Step 2: generate cloned speech")
    run_step_3: bool = Field(default=True, description="Execute Step 3: forced alignment")
    run_step_4: bool = Field(default=True, description="Execute Step 4: select words to replace")
    run_step_5: bool = Field(default=True, description="Execute Step 5: splice audio")
    run_step_6: bool = Field(default=True, description="Execute Step 6: validate splice quality")
    run_step_7: bool = Field(default=True, description="Execute Step 7: format output to LA")
    tiers: List[str] = Field(
        default=["W1", "W2", "W3"],
        description="Word replacement tiers to generate",
    )
    device_override: Optional[str] = Field(
        default=None,
        description="Override compute device (e.g., 'cuda:1')",
    )
    random_seed_override: Optional[int] = Field(
        default=None,
        description="Override random seed for reproducibility",
    )
    output_dir_override: Optional[Path] = Field(
        default=None,
        description="Override output directory path",
    )
    skip_existing: bool = Field(
        default=False,
        description="Skip already-generated files during resume runs",
    )

    class Config:
        """Pydantic model configuration."""

        frozen = False
        arbitrary_types_allowed = True
