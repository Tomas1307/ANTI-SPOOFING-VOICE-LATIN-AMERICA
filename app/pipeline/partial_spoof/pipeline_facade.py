"""
Partial Spoof Pipeline Facade

Orchestrates the 7-step partial spoof generation pipeline that creates
partially spoofed Latin American Spanish audio by replacing individual
words in bonafide HABLA utterances with voice-cloned versions.

This pipeline receives an attack system as a Strategy and produces
ASVspoof2019 LA formatted output with the 'partial_spoof' label.
"""
from pathlib import Path

from loguru import logger

from app.pipeline.partial_spoof.settings import settings
from app.pipeline.partial_spoof.schemas.pipeline_config import PartialSpoofPipelineConfig
from app.pipeline.partial_spoof.steps.step_01_transcribe_bonafide import BonafideTranscriber
from app.pipeline.partial_spoof.steps.step_02_generate_cloned_speech import ClonedSpeechGenerator
from app.pipeline.partial_spoof.steps.step_03_forced_alignment import ForcedAligner
from app.pipeline.partial_spoof.steps.step_04_select_words import WordSelector
from app.pipeline.partial_spoof.steps.step_05_splice_audio import AudioSplicer
from app.pipeline.partial_spoof.steps.step_06_validate_splice import SpliceQualityValidator
from app.pipeline.partial_spoof.steps.step_07_format_output import OutputFormatter
from app.pipeline.partial_spoof.utils.strategy_factory import create_attack_strategy


class PartialSpoofPipeline:
    """Facade for the Partial Spoof generation pipeline.

    Orchestrates 7 sequential steps to produce partially spoofed audio:
    1. Transcribe bonafide audio (Parakeet TDT)
    2. Generate cloned speech (attack Strategy)
    3. Forced alignment on both versions
    4. Select words to replace per tier (W1/W2/W3)
    5. Splice cloned word segments into bonafide audio
    6. Validate splice quality (placeholder)
    7. Format output to ASVspoof2019 LA structure

    Attributes:
        config: Pipeline run configuration.
    """

    def __init__(self, config: PartialSpoofPipelineConfig | None = None) -> None:
        """Initialize the Partial Spoof pipeline.

        Applies configuration overrides to module-level settings and
        sets the output directory based on the attack system name.

        Args:
            config: Optional runtime configuration. Uses defaults if None.
        """
        self.config = config or PartialSpoofPipelineConfig()
        self._apply_config_overrides()
        logger.info(f"PartialSpoofPipeline initialized for {self.config.attack_system}")

    def _apply_config_overrides(self) -> None:
        """Apply runtime config overrides to module-level settings."""
        settings.ATTACK_SYSTEM = self.config.attack_system
        settings.ENABLED_TIERS = self.config.tiers

        if self.config.device_override:
            settings.DEVICE = self.config.device_override
        if self.config.random_seed_override is not None:
            settings.RANDOM_SEED = self.config.random_seed_override
        if self.config.output_dir_override:
            settings.OUTPUT_DIR = self.config.output_dir_override
        else:
            settings.OUTPUT_DIR = Path(f"data/{self.config.attack_system}_partial_spoof")

    def run(self) -> Path:
        """Execute the full partial spoof pipeline.

        Returns:
            Path to the output LA/ directory.

        Raises:
            Exception: If any pipeline step fails.
        """
        logger.info("=" * 80)
        logger.info("PARTIAL SPOOF PIPELINE - START")
        logger.info(f"Attack system: {self.config.attack_system}")
        logger.info(f"Tiers: {self.config.tiers}")
        logger.info(f"Output: {settings.OUTPUT_DIR}")
        logger.info("=" * 80)

        try:
            strategy = create_attack_strategy(self.config.attack_system)

            # === STEP 1: Transcribe bonafide audio ===
            if self.config.run_step_1:
                logger.info("-" * 40)
                step_1 = BonafideTranscriber()
                result_1 = step_1.execute()
                logger.info(
                    f"Step 1 result: {result_1.total_transcribed} transcribed, "
                    f"{result_1.skipped_short} skipped. "
                    f"Distribution: {result_1.word_count_distribution}"
                )

            # === STEP 2: Generate cloned speech ===
            if self.config.run_step_2:
                logger.info("-" * 40)
                step_2 = ClonedSpeechGenerator(
                    strategy=strategy,
                    skip_existing=self.config.skip_existing,
                )
                result_2 = step_2.execute()
                logger.info(
                    f"Step 2 result: {result_2.total_generated} generated, "
                    f"{len(result_2.failed_generations)} failed, "
                    f"avg RTF={result_2.avg_rtf:.3f}"
                )

            # === STEP 3: Forced alignment ===
            if self.config.run_step_3:
                logger.info("-" * 40)
                step_3 = ForcedAligner()
                result_3 = step_3.execute()
                logger.info(
                    f"Step 3 result: {result_3.total_aligned} aligned, "
                    f"{len(result_3.failed_alignments)} failed, "
                    f"avg {result_3.avg_words_per_utterance:.1f} words/utt"
                )

            # === STEP 4: Select words ===
            if self.config.run_step_4:
                logger.info("-" * 40)
                step_4 = WordSelector(
                    enabled_tiers=self.config.tiers,
                )
                result_4 = step_4.execute()
                logger.info(
                    f"Step 4 result: {result_4.total_selections} selections. "
                    f"Tiers: {result_4.tier_counts}"
                )

            # === STEP 5: Splice audio ===
            if self.config.run_step_5:
                logger.info("-" * 40)
                step_5 = AudioSplicer(
                    attack_system_name=strategy.name(),
                )
                result_5 = step_5.execute()
                logger.info(
                    f"Step 5 result: {result_5.total_spliced} spliced, "
                    f"{len(result_5.failed_splices)} failed. "
                    f"Avg spoof ratio: {result_5.avg_spoof_duration_ratio:.3f}"
                )

            # === STEP 6: Validate splice quality ===
            if self.config.run_step_6:
                logger.info("-" * 40)
                step_6 = SpliceQualityValidator()
                result_6 = step_6.execute()
                logger.info(
                    f"Step 6 result: {result_6.total_validated} validated. "
                    f"Avg flux={result_6.avg_spectral_flux:.4f}, "
                    f"F0={result_6.avg_f0_delta:.2f}, "
                    f"Energy={result_6.avg_energy_delta:.4f}"
                )

            # === STEP 7: Format output ===
            if self.config.run_step_7:
                logger.info("-" * 40)
                step_7 = OutputFormatter(
                    system_id_prefix=strategy.name(),
                )
                result_7 = step_7.execute()
                logger.info(
                    f"Step 7 result: LA output at {result_7.output_directory}. "
                    f"Samples: {result_7.total_samples}"
                )

            logger.info("=" * 80)
            logger.info("PARTIAL SPOOF PIPELINE - COMPLETE")
            logger.info("=" * 80)

            return settings.OUTPUT_DIR / "LA"

        except Exception as exc:
            logger.exception(f"Partial spoof pipeline failed: {exc}")
            raise
