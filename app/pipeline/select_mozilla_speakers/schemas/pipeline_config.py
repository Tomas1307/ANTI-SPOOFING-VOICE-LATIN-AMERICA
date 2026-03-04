"""
Pipeline configuration schema for Mozilla Common Voice speaker selection.
"""
from pathlib import Path
from pydantic import BaseModel, Field
from typing import Optional


class MozillaSpeakerPipelineConfig(BaseModel):
    """Configuration for running the Mozilla Common Voice speaker selection pipeline.

    This schema defines the runtime configuration that can be passed to the pipeline.
    Most defaults are defined in settings.py and can be overridden here.

    Attributes:
        run_step_1: Whether to run Step 1 (HABLA embedding extraction)
        run_step_2: Whether to run Step 2 (CV embedding extraction)
        run_step_3: Whether to run Step 3 (similarity filtering)
        run_step_4: Whether to run Step 4 (balanced sampling)
        run_step_5: Whether to run Step 5 (dataset integration)

        similarity_threshold_override: Optional override for similarity threshold
        random_seed_override: Optional override for random seed
        output_dir_override: Optional override for output directory
    """

    run_step_1: bool = Field(
        default=True,
        description="Run Step 1: Extract HABLA embeddings"
    )
    run_step_2: bool = Field(
        default=True,
        description="Run Step 2: Extract Common Voice embeddings"
    )
    run_step_3: bool = Field(
        default=True,
        description="Run Step 3: Filter by similarity threshold"
    )
    run_step_4: bool = Field(
        default=True,
        description="Run Step 4: Balanced stratified sampling"
    )
    run_step_5: bool = Field(
        default=True,
        description="Run Step 5: Integrate into HABLA v2 dataset"
    )

    similarity_threshold_override: Optional[float] = Field(
        default=None,
        description="Override similarity threshold (default: from settings.py)"
    )
    random_seed_override: Optional[int] = Field(
        default=None,
        description="Override random seed (default: from settings.py)"
    )
    output_dir_override: Optional[Path] = Field(
        default=None,
        description="Override output directory (default: from settings.py)"
    )

    class Config:
        """Pydantic model configuration."""
        frozen = False
