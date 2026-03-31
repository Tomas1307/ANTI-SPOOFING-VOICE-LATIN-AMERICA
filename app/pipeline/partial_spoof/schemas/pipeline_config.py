"""Pipeline configuration schema for the Partial Spoof pipeline."""
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field


class PartialSpoofPipelineConfig(BaseModel):
    """Runtime configuration for a Partial Spoof pipeline execution.

    Provides per-run overrides for the attack system, step execution flags,
    tier selection, and resource settings. Defaults allow running the full
    pipeline without any explicit configuration.

    Attributes:
        attack_system: Voice cloning system identifier.
        run_step_1: Execute Step 1 (transcribe bonafide audio).
        run_step_2: Execute Step 2 (generate cloned speech).
        run_step_3: Execute Step 3 (forced alignment).
        run_step_4: Execute Step 4 (select words to replace).
        run_step_5: Execute Step 5 (splice audio).
        run_step_6: Execute Step 6 (validate splice quality).
        run_step_7: Execute Step 7 (format output to LA).
        tiers: List of tier identifiers to generate.
        device_override: Override the default PyTorch device.
        random_seed_override: Override the default random seed.
        output_dir_override: Override the default output directory.
        skip_existing: Skip generation for samples that already exist on disk.
    """

    attack_system: str = Field(
        default="fishgram",
        description="Voice cloning system: fishgram, qwen, cosyvoice, outetts, chatterbox, openvoice.",
    )
    run_step_1: bool = Field(default=True, description="Execute Step 1: transcribe bonafide.")
    run_step_2: bool = Field(default=True, description="Execute Step 2: generate cloned speech.")
    run_step_3: bool = Field(default=True, description="Execute Step 3: forced alignment.")
    run_step_4: bool = Field(default=True, description="Execute Step 4: select words.")
    run_step_5: bool = Field(default=True, description="Execute Step 5: splice audio.")
    run_step_6: bool = Field(default=True, description="Execute Step 6: validate splice quality.")
    run_step_7: bool = Field(default=True, description="Execute Step 7: format output to LA.")
    tiers: List[str] = Field(
        default=["W1", "W2", "W3"],
        description="Tier identifiers to generate.",
    )
    device_override: Optional[str] = Field(
        default=None,
        description="Override PyTorch device (e.g., 'cuda:1').",
    )
    random_seed_override: Optional[int] = Field(
        default=None,
        description="Override random seed for word selection.",
    )
    output_dir_override: Optional[Path] = Field(
        default=None,
        description="Override output directory path.",
    )
    skip_existing: bool = Field(
        default=False,
        description="Skip samples that already have generated output files.",
    )
