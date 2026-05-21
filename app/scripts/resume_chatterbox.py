"""
Resume Chatterbox attack pipeline from Step 3 only.

Skips Step 1 (reference preparation) and Step 2 (text prompt assignment)
which are non-idempotent: Step 2 in particular overwrites
`text_prompts.json` with a fresh random sample on each run. Re-running
Step 2 on resume risks shifting the text_id -> text mapping if the
Common Voice corpus or HABLA speaker list has changed since the original
invocation, which would break the resume contract in Step 3 (sample_id
keys would no longer match generation_metadata.json).

This wrapper sets `run_step_1=False, run_step_2=False, run_step_3=True,
run_step_4=True, run_step_5=True` so Step 3 picks up exactly where the
April run left off, then Step 4 (validation) and Step 5 (LA formatting)
finalize the corpus.

Usage on ml-server03:
    export CUDA_VISIBLE_DEVICES=3
    source ~/ANTI-SPOOFING-VOICE-LATIN-AMERICA/envs/chatterbox_env/bin/activate
    python -m app.scripts.resume_chatterbox
"""
from loguru import logger

from app.pipeline.chatterbox_attack.pipeline_facade import ChatterboxAttackPipeline
from app.pipeline.chatterbox_attack.schemas.pipeline_config import (
    ChatterboxPipelineConfig,
)


class ChatterboxResumeRunner:
    """One-shot runner that invokes the Chatterbox facade starting at Step 3.

    Skips Step 1 + Step 2 so resume preserves the original text_prompts
    mapping. Relies on the existing reference_metadata.json,
    text_prompts.json, and generation_metadata.json on disk.

    Attributes:
        config: ChatterboxPipelineConfig with Steps 1 and 2 disabled.
    """

    def __init__(self) -> None:
        """Initialise with Steps 1 and 2 disabled."""
        self.config = ChatterboxPipelineConfig(
            run_step_1=False,
            run_step_2=False,
            run_step_3=True,
            run_step_4=True,
            run_step_5=True,
        )

    def run(self) -> None:
        """Invoke the Chatterbox facade with the resume-friendly config."""
        logger.info("=" * 80)
        logger.info("CHATTERBOX RESUME RUNNER")
        logger.info("Skipping Step 1 (references) and Step 2 (text prompts);")
        logger.info("starting at Step 3 (generate speech) with on-disk state.")
        logger.info("=" * 80)
        pipeline = ChatterboxAttackPipeline(config=self.config)
        pipeline.run()


if __name__ == "__main__":
    ChatterboxResumeRunner().run()
