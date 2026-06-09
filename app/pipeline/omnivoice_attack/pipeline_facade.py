"""
OmniVoice Attack Pipeline Facade

Orchestrates all 5 steps of the OmniVoice voice cloning attack pipeline.
"""
from pathlib import Path

from loguru import logger

from app.pipeline.omnivoice_attack.schemas.pipeline_config import OmniVoicePipelineConfig
from app.pipeline.omnivoice_attack.steps import (
    ReferenceAudioPreparator,
    TextPromptPreparator,
    SpeechGenerator,
    QualityValidator,
    OutputFormatter,
)
from app.pipeline.omnivoice_attack.settings import settings


class OmniVoiceAttackPipeline:
    """Facade for OmniVoice voice cloning attack pipeline.

    Orchestrates 5 steps:
        1. Prepare reference audio (10s clips per speaker, Parakeet ref_text).
        2. Prepare text prompts (Spanish transcripts from Mozilla CV).
        3. Generate synthetic speech (OmniVoice in-process, 24 kHz native).
        4. Validate quality (Parakeet TDT WER/CER + NISQA + ECAPA-TDNN similarity).
        5. Format output (ASVspoof2019 LA, FLAC at 16 kHz).

    OmniVoice (k2-fsa) is a 646-language zero-shot TTS model based on a
    diffusion language model architecture. Spanish is supported with 27,559
    hours of training data.

    Supports validation mode (3 speakers) and production mode (all speakers)
    via settings.VALIDATION_MODE toggle.

    Attributes:
        config: Pipeline run configuration.
    """

    def __init__(self, config: OmniVoicePipelineConfig | None = None) -> None:
        """Initialize OmniVoice attack pipeline.

        Args:
            config: Pipeline configuration (default: run all steps with settings defaults).
        """
        self.config = config or OmniVoicePipelineConfig()
        logger.info("OmniVoiceAttackPipeline initialized")

        if self.config.samples_per_speaker_override is not None:
            settings.SAMPLES_PER_SPEAKER = self.config.samples_per_speaker_override

        if self.config.random_seed_override is not None:
            settings.RANDOM_SEED = self.config.random_seed_override

        if self.config.output_dir_override is not None:
            settings.OUTPUT_DIR = self.config.output_dir_override

        if self.config.device_override is not None:
            settings.DEVICE = self.config.device_override

    def run(self) -> Path:
        """Execute the full OmniVoice attack pipeline.

        Returns:
            Path to output directory (LA/) with ASVspoof2019 format.

        Raises:
            Exception: If any step fails.
        """
        logger.info("=" * 80)
        logger.info("OMNIVOICE ATTACK PIPELINE - START")
        logger.info("=" * 80)

        mode_str = "VALIDATION" if settings.VALIDATION_MODE else "PRODUCTION"
        logger.info(f"Mode: {mode_str}")
        logger.info(f"Model: {settings.OMNIVOICE_MODEL_ID}")
        logger.info(f"Samples per speaker: {settings.SAMPLES_PER_SPEAKER}")
        logger.info(f"Output directory: {settings.OUTPUT_DIR}")
        logger.info(f"Device: {settings.DEVICE}")
        logger.info(f"Native sample rate: {settings.OMNIVOICE_NATIVE_SAMPLE_RATE} Hz")
        logger.info(f"Target sample rate: {settings.SAMPLE_RATE} Hz")
        logger.info("")

        if self.config.run_step_1:
            logger.info("STEP 1/5: Prepare Reference Audio (with Parakeet transcription)")
            logger.info("-" * 80)

            step_1 = ReferenceAudioPreparator()
            result_1 = step_1.execute()

            logger.info(f"References prepared: {result_1.reference_count} speakers")
            logger.info(f"  Transcribed: {result_1.transcribed_count}/{result_1.reference_count}")
            logger.info(f"  Split breakdown: {result_1.split_breakdown}")
            logger.info("")
        else:
            logger.warning("STEP 1 skipped (run_step_1=False)")

        if self.config.run_step_2:
            logger.info("STEP 2/5: Prepare Text Prompts")
            logger.info("-" * 80)

            step_2 = TextPromptPreparator()
            result_2 = step_2.execute()

            logger.info(f"Text prompts assigned: {result_2.total_prompts} prompts")
            logger.info("")
        else:
            logger.warning("STEP 2 skipped (run_step_2=False)")

        if self.config.run_step_3:
            logger.info("STEP 3/5: Generate Synthetic Speech")
            logger.info("-" * 80)

            step_3 = SpeechGenerator(skip_existing=self.config.skip_existing_step_3)
            result_3 = step_3.execute()

            logger.info(f"Generated: {result_3.total_generated} samples")
            logger.info(f"  Failed: {len(result_3.failed_generations)}")
            logger.info(f"  Average RTF: {result_3.avg_rtf:.3f}")
            logger.info("")
        else:
            logger.warning("STEP 3 skipped (run_step_3=False)")

        if self.config.run_step_4:
            logger.info("STEP 4/5: Validate Quality")
            logger.info("-" * 80)

            step_4 = QualityValidator()
            result_4 = step_4.execute()

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

        result_5 = None
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

        logger.info("=" * 80)
        logger.info("OMNIVOICE ATTACK PIPELINE - COMPLETE")
        logger.info("=" * 80)

        if result_5 is not None:
            return result_5.output_directory
        return settings.OUTPUT_DIR
