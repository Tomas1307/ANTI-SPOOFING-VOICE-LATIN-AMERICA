"""
Qwen Attack Pipeline Facade

Orchestrates all 5 steps of the Qwen3-TTS voice cloning attack pipeline.
Secondary attack pipeline providing codec architecture diversity alongside FishGram.
"""
from loguru import logger
from pathlib import Path
from app.pipeline.qwen_attack.schemas.pipeline_config import QwenPipelineConfig
from app.pipeline.qwen_attack.steps import (
    ReferenceAudioPreparator,
    TextPromptPreparator,
    SpeechGenerator,
    QualityValidator,
    OutputFormatter,
)
from app.pipeline.qwen_attack.settings import settings


class QwenAttackPipeline:
    """Facade for Qwen3-TTS voice cloning attack pipeline.

    Orchestrates 5 steps:
    1. Prepare reference audio with STT transcription (15s clips + Whisper)
    2. Prepare text prompts (Spanish transcripts from Mozilla CV, max 40 words)
    3. Generate synthetic speech (local Qwen3-TTS 1.7B model inference)
    4. Validate quality (DNSMOS + speaker similarity + artifact detection)
    5. Format output (ASVspoof2019 LA format)

    Unlike FishGram (HTTP API), Qwen3-TTS runs as a local model loaded
    directly into GPU memory during Step 3. The model is released after
    generation completes.

    Supports validation mode (3 speakers, 6 samples) and production mode
    (162 speakers, 810 samples) via settings.VALIDATION_MODE toggle.

    Attributes:
        config: Pipeline runtime configuration with step toggles and overrides.
    """

    def __init__(self, config: QwenPipelineConfig | None = None):
        """Initialize Qwen attack pipeline.

        Args:
            config: Pipeline configuration (default: run all steps).
        """
        self.config = config or QwenPipelineConfig()
        logger.info("QwenAttackPipeline initialized")

        # Apply config overrides to settings
        if self.config.samples_per_speaker_override is not None:
            settings.SAMPLES_PER_SPEAKER = self.config.samples_per_speaker_override

        if self.config.random_seed_override is not None:
            settings.RANDOM_SEED = self.config.random_seed_override

        if self.config.output_dir_override is not None:
            settings.OUTPUT_DIR = self.config.output_dir_override

        if self.config.device_override is not None:
            settings.DEVICE = self.config.device_override

    def run(self) -> Path:
        """Execute the full Qwen attack pipeline.

        Returns:
            Path to output directory (LA/) with ASVspoof2019 format.

        Raises:
            Exception: If any step fails.
        """
        logger.info("=" * 80)
        logger.info("QWEN ATTACK PIPELINE - START")
        logger.info("=" * 80)

        mode_str = "VALIDATION" if settings.VALIDATION_MODE else "PRODUCTION"
        logger.info(f"Mode: {mode_str}")
        logger.info(f"Model: {settings.QWEN_MODEL_ID}")
        logger.info(f"Samples per speaker: {settings.SAMPLES_PER_SPEAKER}")
        logger.info(f"Output directory: {settings.OUTPUT_DIR}")
        logger.info(f"Device: {settings.DEVICE}")
        logger.info(f"Text length range: {settings.TEXT_LENGTH_RANGE}")
        logger.info("")

        # =========================
        # STEP 1: Prepare References
        # =========================
        if self.config.run_step_1:
            logger.info("STEP 1/5: Prepare Reference Audio + STT Transcription")
            logger.info("-" * 80)

            step_1 = ReferenceAudioPreparator()
            result_1 = step_1.execute()

            logger.info(f"References prepared: {result_1.reference_count} speakers")
            logger.info(f"  Split breakdown: {result_1.split_breakdown}")
            logger.info("")
        else:
            logger.warning("STEP 1 skipped (run_step_1=False)")
            result_1 = None

        # =========================
        # STEP 2: Prepare Text Prompts
        # =========================
        if self.config.run_step_2:
            logger.info("STEP 2/5: Prepare Text Prompts")
            logger.info("-" * 80)

            step_2 = TextPromptPreparator()
            result_2 = step_2.execute()

            logger.info(f"Text prompts assigned: {result_2.total_prompts} prompts")
            logger.info("")
        else:
            logger.warning("STEP 2 skipped (run_step_2=False)")
            result_2 = None

        # =========================
        # STEP 3: Generate Speech
        # =========================
        if self.config.run_step_3:
            logger.info("STEP 3/5: Generate Synthetic Speech (Qwen3-TTS Local)")
            logger.info("-" * 80)

            step_3 = SpeechGenerator()
            result_3 = step_3.execute()

            logger.info(f"Generated: {result_3.total_generated} samples")
            logger.info(f"  Failed: {len(result_3.failed_generations)}")
            logger.info(f"  Average RTF: {result_3.avg_rtf:.2f}")
            logger.info("")
        else:
            logger.warning("STEP 3 skipped (run_step_3=False)")
            result_3 = None

        # =========================
        # STEP 4: Validate Quality
        # =========================
        if self.config.run_step_4:
            logger.info("STEP 4/5: Validate Quality + Artifact Detection")
            logger.info("-" * 80)

            step_4 = QualityValidator()
            result_4 = step_4.execute()

            logger.info("Validation complete")
            logger.info(
                f"  Passed: {result_4.validation_stats['passed']}/"
                f"{result_4.validation_stats['total']}"
            )
            logger.info(f"  Rejected: {result_4.validation_stats['rejected']}")
            logger.info(f"  Avg DNSMOS: {result_4.avg_dnsmos:.2f}")
            logger.info(f"  Avg similarity: {result_4.avg_similarity:.2f}")
            logger.info("")
        else:
            logger.warning("STEP 4 skipped (run_step_4=False)")
            result_4 = None

        # =========================
        # STEP 5: Format Output
        # =========================
        if self.config.run_step_5:
            logger.info("STEP 5/5: Format Output to ASVspoof2019 LA")
            logger.info("-" * 80)

            step_5 = OutputFormatter()
            result_5 = step_5.execute()

            logger.info("Output formatted")
            logger.info(f"  Output directory: {result_5.output_directory}")
            logger.info(f"  Total samples: {sum(result_5.total_samples.values())}")
            logger.info(f"  Train: {result_5.total_samples.get('train', 0)}")
            logger.info(f"  Dev: {result_5.total_samples.get('dev', 0)}")
            logger.info(f"  Eval: {result_5.total_samples.get('eval', 0)}")
            logger.info("")
        else:
            logger.warning("STEP 5 skipped (run_step_5=False)")
            result_5 = None

        # =========================
        # COMPLETE
        # =========================
        logger.info("=" * 80)
        logger.info("QWEN ATTACK PIPELINE - COMPLETE")
        logger.info("=" * 80)

        if result_5 is not None:
            return result_5.output_directory
        else:
            return settings.OUTPUT_DIR
