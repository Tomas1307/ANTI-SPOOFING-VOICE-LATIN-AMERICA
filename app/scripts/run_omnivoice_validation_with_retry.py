"""
Run the OmniVoice pipeline in validation mode (3 speakers, 6 samples) with
the same retry loop as the production runner.

This tests the full QA path end-to-end on a small set:
    1. Steps 1-5 run once, exactly as production would.
    2. Any sample rejected by Step 4 (including non-verbal prefix artifacts)
       has its WAV deleted and is regenerated up to MAX_GENERATION_RETRIES
       times. Each retry round re-runs Steps 3, 4, and 5.
    3. After the loop, the final pass/reject counts are printed and any
       still-rejected samples are listed.

Usage on ml-server03:
    source envs/omnivoice_env/bin/activate
    export CUDA_VISIBLE_DEVICES=1
    python -m app.scripts.run_omnivoice_validation_with_retry
    deactivate

The bonafide listening test for TEXT_00001 and TEXT_00002 should be done
manually after this script completes; the new clean WAVs replace the old
artifact-bearing ones in data/omnivoice_output/generated/.
"""
import json
import sys
from pathlib import Path

from loguru import logger

from app.pipeline.omnivoice_attack import (
    OmniVoiceAttackPipeline,
    OmniVoicePipelineConfig,
    settings,
)


def _find_rejected_sample_ids(output_dir: Path) -> list:
    """Compute rejected sample IDs from generation vs validation metadata.

    Args:
        output_dir: Pipeline output directory containing the metadata JSONs.

    Returns:
        List of sample IDs that exist in generation_metadata.json but not in
        validated_samples.json. Empty list if either metadata file is missing.
    """
    gen_meta_path = output_dir / "generation_metadata.json"
    val_meta_path = output_dir / "validated_samples.json"

    if not gen_meta_path.exists() or not val_meta_path.exists():
        return []

    with open(gen_meta_path, "r", encoding="utf-8") as f:
        generated = json.load(f)
    with open(val_meta_path, "r", encoding="utf-8") as f:
        validated = json.load(f)

    return list(set(generated.keys()) - set(validated.keys()))


def main() -> int:
    """Execute validation-mode pipeline with retry loop and report results.

    Returns:
        Process exit code (0 if pipeline ran to completion, 1 on missing inputs).
    """
    settings.VALIDATION_MODE = True
    settings.SAMPLES_PER_SPEAKER = 2
    settings.MATCH_BONAFIDE_COUNT = False

    logger.info("=" * 80)
    logger.info("OMNIVOICE VALIDATION + RETRY TEST")
    logger.info("=" * 80)
    logger.info(f"  Validation mode  : {settings.VALIDATION_MODE}")
    logger.info(f"  Samples/speaker  : {settings.SAMPLES_PER_SPEAKER}")
    logger.info(f"  Speakers         : {settings.VALIDATION_SPEAKERS}")
    logger.info(f"  Max retries      : {settings.MAX_GENERATION_RETRIES}")
    logger.info(f"  Output dir       : {settings.OUTPUT_DIR}")
    logger.info("")

    initial_config = OmniVoicePipelineConfig(
        run_step_1=True,
        run_step_2=True,
        run_step_3=True,
        run_step_4=True,
        run_step_5=True,
        skip_existing_step_3=True,
    )
    OmniVoiceAttackPipeline(config=initial_config).run()

    output_dir = settings.OUTPUT_DIR
    gen_dir = output_dir / "generated"
    max_retries = settings.MAX_GENERATION_RETRIES
    system_id = settings.OMNIVOICE_SYSTEM_ID

    for retry_round in range(1, max_retries + 1):
        rejected_ids = _find_rejected_sample_ids(output_dir)

        if not rejected_ids:
            logger.info("")
            logger.info("All samples passed validation. No more retries needed.")
            break

        logger.info("")
        logger.info("=" * 80)
        logger.info(f"RETRY ROUND {retry_round}/{max_retries}")
        logger.info("=" * 80)
        logger.info(f"  Rejected samples to regenerate: {len(rejected_ids)}")
        for sid in rejected_ids:
            logger.info(f"    - {sid}")

        deleted_count = 0
        for sample_id in rejected_ids:
            wav_path = gen_dir / f"{system_id}_{sample_id}.wav"
            if wav_path.exists():
                wav_path.unlink()
                deleted_count += 1
        logger.info(f"  Deleted {deleted_count} rejected WAV files")

        retry_config = OmniVoicePipelineConfig(
            run_step_1=False,
            run_step_2=False,
            run_step_3=True,
            run_step_4=True,
            run_step_5=True,
            skip_existing_step_3=True,
        )
        OmniVoiceAttackPipeline(config=retry_config).run()
    else:
        final_rejected = _find_rejected_sample_ids(output_dir)
        if final_rejected:
            logger.warning("")
            logger.warning(
                f"After {max_retries} retry rounds, "
                f"{len(final_rejected)} samples remain rejected:"
            )
            for sid in final_rejected:
                logger.warning(f"    - {sid}")
            logger.warning("These were excluded from the LA/ output.")

    val_path = output_dir / "validated_samples.json"
    gen_path = output_dir / "generation_metadata.json"
    if val_path.exists() and gen_path.exists():
        with open(val_path, "r", encoding="utf-8") as f:
            validated = json.load(f)
        with open(gen_path, "r", encoding="utf-8") as f:
            generated = json.load(f)

        logger.info("")
        logger.info("=" * 80)
        logger.info("FINAL RESULTS")
        logger.info("=" * 80)
        logger.info(f"  Total samples in metadata : {len(generated)}")
        logger.info(f"  Passed validation         : {len(validated)}")
        logger.info(f"  Final rejections          : {len(generated) - len(validated)}")
        logger.info(f"  Output (LA/)              : {output_dir / 'LA'}")
        logger.info("")
        logger.info("Listen to these to confirm bleed is gone:")
        for sample_id in sorted(validated.keys()):
            wav = generated.get(sample_id, {}).get("audio_path")
            if wav:
                logger.info(f"    {wav}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
