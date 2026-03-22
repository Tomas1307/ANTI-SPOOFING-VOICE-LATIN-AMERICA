"""
FishGram Attack Pipeline Facade

Orchestrates all 5 steps of the Fish Speech voice cloning attack pipeline.
"""
from loguru import logger
from pathlib import Path
from app.pipeline.fishgram_attack.schemas.pipeline_config import FishGramPipelineConfig
from app.pipeline.fishgram_attack.steps import (
    ReferenceAudioPreparator,
    TextPromptPreparator,
    SpeechGenerator,
    QualityValidator,
    OutputFormatter
)
from app.pipeline.fishgram_attack.settings import settings


class FishGramAttackPipeline:
    """Facade for FishGram voice cloning attack pipeline.

    Orchestrates 5 steps:
    1. Prepare reference audio (15s clips per speaker)
    2. Prepare text prompts (Spanish transcripts from Mozilla CV)
    3. Generate synthetic speech (via Fish Speech HTTP API)
    4. Validate quality (DNSMOS + speaker similarity)
    5. Format output (ASVspoof2019 LA format)

    The Fish Speech TTS model runs as an external HTTP API server on ml-server03.
    The server must be started before running this pipeline.

    Supports validation mode (3 speakers, 6 samples) and production mode
    (162 speakers, 810 samples) via settings.VALIDATION_MODE toggle.
    """

    def __init__(self, config: FishGramPipelineConfig | None = None):
        """Initialize FishGram attack pipeline.

        Args:
            config: Pipeline configuration (default: run all steps)
        """
        self.config = config or FishGramPipelineConfig()
        logger.info("FishGramAttackPipeline initialized")

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
        """Execute the full FishGram attack pipeline.

        Returns:
            Path to output directory (LA/) with ASVspoof2019 format

        Raises:
            Exception: If any step fails
        """
        logger.info("=" * 80)
        logger.info("FISHGRAM ATTACK PIPELINE - START")
        logger.info("=" * 80)

        mode_str = "VALIDATION" if settings.VALIDATION_MODE else "PRODUCTION"
        logger.info(f"Mode: {mode_str}")
        logger.info(f"Samples per speaker: {settings.SAMPLES_PER_SPEAKER}")
        logger.info(f"Output directory: {settings.OUTPUT_DIR}")
        logger.info(f"Device: {settings.DEVICE}")
        logger.info("")

        # =========================
        # STEP 1: Prepare References
        # =========================
        if self.config.run_step_1:
            logger.info("STEP 1/5: Prepare Reference Audio")
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
            logger.info("STEP 3/5: Generate Synthetic Speech")
            logger.info("-" * 80)

            step_3 = SpeechGenerator(skip_existing=self.config.skip_existing_step_3)
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
            logger.info("STEP 4/5: Validate Quality")
            logger.info("-" * 80)

            step_4 = QualityValidator()
            result_4 = step_4.execute()

            logger.info(f"Validation complete")
            logger.info(f"  Passed: {result_4.validation_stats['passed']}/{result_4.validation_stats['total']}")
            logger.info(f"  Rejected: {result_4.validation_stats['rejected']}")
            logger.info(f"  Avg WER: {result_4.avg_wer:.4f}")
            logger.info(f"  Avg CER: {result_4.avg_cer:.4f}")
            logger.info(f"  Prefix trims: {result_4.prefix_trim_count}")
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

            logger.info(f"Output formatted")
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
        logger.info("FISHGRAM ATTACK PIPELINE - COMPLETE")
        logger.info("=" * 80)

        if result_5 is not None:
            return result_5.output_directory
        else:
            return settings.OUTPUT_DIR
