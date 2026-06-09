"""
Chatterbox Attack Pipeline Facade

Orchestrates all 5 steps of the Chatterbox Multilingual TTS voice cloning
attack pipeline. Fourth attack pipeline providing a distinct flow-matching
architecture alongside FishGram (VQGAN), Qwen (Dual-Track), and OpenVoice
(VITS + tone converter).
"""
from pathlib import Path
from loguru import logger

from app.pipeline.chatterbox_attack.schemas.pipeline_config import ChatterboxPipelineConfig
from app.pipeline.chatterbox_attack.steps import (
    ReferenceAudioPreparator,
    TextPromptPreparator,
    SpeechGenerator,
    QualityValidator,
    OutputFormatter,
)
from app.pipeline.chatterbox_attack.settings import settings


class ChatterboxAttackPipeline:
    """Facade for Chatterbox Multilingual TTS voice cloning attack pipeline.

    Orchestrates 5 steps:
    1. Prepare reference audio (10s clips from training samples).
    2. Prepare text prompts (Spanish transcripts from Mozilla CV).
    3. Generate synthetic speech (ChatterboxMultilingualTTS, language_id='es',
       watermark bypassed via NoOpWatermarker for research validity).
    4. Validate quality (Parakeet TDT STT + WER/CER + spurious prefix trimming).
    5. Format output (ASVspoof2019 LA format, CHATTERBOX system ID, 6000000+ IDs).

    Chatterbox runs entirely in-process. No HTTP server is required.
    The model is loaded from HuggingFace Hub on first call (auto-cached),
    then held in GPU memory for the duration of Step 3.

    Supports validation mode (3 speakers, 6 samples) and production mode
    (162 speakers, 810 samples) via settings.VALIDATION_MODE toggle.

    Attributes:
        config: Pipeline runtime configuration with step toggles and overrides.
    """

    def __init__(self, config: ChatterboxPipelineConfig | None = None):
        """Initialize Chatterbox attack pipeline.

        Args:
            config: Pipeline configuration (default: run all steps).
        """
        self.config = config or ChatterboxPipelineConfig()
        logger.info("ChatterboxAttackPipeline initialized")

        if self.config.samples_per_speaker_override is not None:
            settings.SAMPLES_PER_SPEAKER = self.config.samples_per_speaker_override

        if self.config.random_seed_override is not None:
            settings.RANDOM_SEED = self.config.random_seed_override

        if self.config.output_dir_override is not None:
            settings.OUTPUT_DIR = self.config.output_dir_override

        if self.config.device_override is not None:
            settings.DEVICE = self.config.device_override

    def run(self) -> Path:
        """Execute the full Chatterbox attack pipeline.

        Returns:
            Path to output directory (LA/) with ASVspoof2019 format.

        Raises:
            Exception: If any step fails.
        """
        logger.info("=" * 80)
        logger.info("CHATTERBOX ATTACK PIPELINE - START")
        logger.info("=" * 80)

        mode_str = "VALIDATION" if settings.VALIDATION_MODE else "PRODUCTION"
        logger.info(f"Mode            : {mode_str}")
        logger.info(f"Samples/speaker : {settings.SAMPLES_PER_SPEAKER}")
        logger.info(f"Output directory: {settings.OUTPUT_DIR}")
        logger.info(f"Device          : {settings.DEVICE}")
        logger.info(f"Language        : {settings.LANGUAGE_ID}")
        logger.info(f"Exaggeration    : {settings.EXAGGERATION}")
        logger.info(f"CFG weight      : {settings.CFG_WEIGHT}")
        logger.info("")

        # =========================
        # STEP 1: Prepare References
        # =========================
        if self.config.run_step_1:
            logger.info("STEP 1/5: Prepare Reference Audio")
            logger.info("-" * 80)
            result_1 = ReferenceAudioPreparator().execute()
            logger.info(f"References prepared: {result_1.reference_count} speakers")
            logger.info(f"  Split breakdown: {result_1.split_breakdown}")
            logger.info("")
        else:
            logger.warning("STEP 1 skipped (run_step_1=False)")

        # =========================
        # STEP 2: Prepare Text Prompts
        # =========================
        if self.config.run_step_2:
            logger.info("STEP 2/5: Prepare Text Prompts")
            logger.info("-" * 80)
            result_2 = TextPromptPreparator().execute()
            logger.info(f"Text prompts assigned: {result_2.total_prompts} prompts")
            logger.info("")
        else:
            logger.warning("STEP 2 skipped (run_step_2=False)")

        # =========================
        # STEP 3: Generate Speech
        # =========================
        if self.config.run_step_3:
            logger.info("STEP 3/5: Generate Synthetic Speech (ChatterboxMultilingualTTS)")
            logger.info("-" * 80)
            result_3 = SpeechGenerator().execute()
            logger.info(f"Generated: {result_3.total_generated} samples")
            logger.info(f"  Failed: {len(result_3.failed_generations)}")
            logger.info(f"  Average RTF: {result_3.avg_rtf:.2f}")
            logger.info("")
        else:
            logger.warning("STEP 3 skipped (run_step_3=False)")

        # =========================
        # STEP 4: Validate Quality
        # =========================
        if self.config.run_step_4:
            logger.info("STEP 4/5: Validate Quality (Parakeet STT + WER/CER)")
            logger.info("-" * 80)
            result_4 = QualityValidator().execute()
            logger.info("Validation complete")
            logger.info(
                f"  Passed: {result_4.validation_stats['passed']}/"
                f"{result_4.validation_stats['total']}"
            )
            logger.info(f"  Rejected: {result_4.validation_stats['rejected']}")
            logger.info(f"  Avg WER: {result_4.avg_wer:.4f}")
            logger.info(f"  Avg CER: {result_4.avg_cer:.4f}")
            logger.info(f"  Prefix trims: {result_4.prefix_trim_count}")
            logger.info("")
        else:
            logger.warning("STEP 4 skipped (run_step_4=False)")

        # =========================
        # STEP 5: Format Output
        # =========================
        if self.config.run_step_5:
            logger.info("STEP 5/5: Format Output to ASVspoof2019 LA")
            logger.info("-" * 80)
            result_5 = OutputFormatter().execute()
            logger.info("Output formatted")
            logger.info(f"  Output directory: {result_5.output_directory}")
            logger.info(f"  Total samples: {sum(result_5.total_samples.values())}")
            logger.info(f"  Train: {result_5.total_samples.get('train', 0)}")
            logger.info(f"  Dev  : {result_5.total_samples.get('dev', 0)}")
            logger.info(f"  Eval : {result_5.total_samples.get('eval', 0)}")
            logger.info("")
        else:
            logger.warning("STEP 5 skipped (run_step_5=False)")
            result_5 = None

        logger.info("=" * 80)
        logger.info("CHATTERBOX ATTACK PIPELINE - COMPLETE")
        logger.info("=" * 80)

        if self.config.run_step_5 and result_5 is not None:
            return result_5.output_directory
        return settings.OUTPUT_DIR


if __name__ == "__main__":
    pipeline = ChatterboxAttackPipeline()
    result = pipeline.run()
    logger.info(f"Pipeline complete. Output: {result}")
