"""
Partial Spoof Pipeline Facade

Orchestrates the 7-step partial spoof generation pipeline that creates
partially spoofed Latin American Spanish audio by replacing individual
words in bonafide HABLA utterances with voice-cloned versions.

Includes a two-level retry system:
  Level 1 (free): Smart word re-selection within the same clone.
  Level 2 (expensive): Regenerate the clone with a different TTS seed,
    re-align, and retry splicing. Up to MAX_REGENERATIONS attempts.
"""
import json
from pathlib import Path
from typing import Optional

import librosa
import numpy as np
from loguru import logger

from app.pipeline.partial_spoof.settings import settings
from app.pipeline.partial_spoof.schemas.pipeline_config import PartialSpoofPipelineConfig
from app.pipeline.partial_spoof.steps.step_01_transcribe_bonafide import BonafideTranscriber
from app.pipeline.partial_spoof.steps.step_02_generate_cloned_speech import ClonedSpeechGenerator
from app.pipeline.partial_spoof.steps.step_03_forced_alignment import ForcedAligner
from app.pipeline.partial_spoof.steps.step_04_select_words import WordSelector
from app.pipeline.partial_spoof.steps.step_05_splice_audio import AudioSplicer
from app.pipeline.partial_spoof.steps.step_05b_apply_boundary_jitter import BoundaryJitterApplier
from app.pipeline.partial_spoof.steps.step_06_validate_splice import SpliceQualityValidator
from app.pipeline.partial_spoof.steps.step_07_format_output import OutputFormatter
from app.pipeline.partial_spoof.utils.checkpoint_manager import CheckpointManager
from app.pipeline.partial_spoof.utils.manifest_loader import ManifestLoader
from app.pipeline.partial_spoof.utils.strategy_factory import create_attack_strategy

MAX_REGENERATIONS = 3


class PartialSpoofPipeline:
    """Facade for the Partial Spoof generation pipeline.

    Orchestrates 7 sequential steps with a regeneration loop:
    1. Transcribe bonafide audio (Parakeet TDT) — runs once
    2-5. Generate -> Align -> Select -> Splice — runs in a loop
         with regeneration for samples that fail word matching
    6. Validate splice quality
    7. Format output to ASVspoof2019 LA structure

    Attributes:
        config: Pipeline run configuration.
    """

    def __init__(self, config: PartialSpoofPipelineConfig | None = None) -> None:
        """Initialize the Partial Spoof pipeline.

        Args:
            config: Optional runtime configuration. Uses defaults if None.
        """
        self.config = config or PartialSpoofPipelineConfig()
        self._apply_config_overrides()
        self.manifest_loader: Optional[ManifestLoader] = self._load_manifest()
        self.checkpoint: Optional[CheckpointManager] = self._build_checkpoint()
        logger.info(f"PartialSpoofPipeline initialized for {self.config.attack_system}")

    def _apply_config_overrides(self) -> None:
        """Apply runtime config overrides to module-level settings."""
        settings.ATTACK_SYSTEM = self.config.attack_system
        settings.ENABLED_TIERS = self.config.tiers

        if self.config.device_override:
            settings.DEVICE = self.config.device_override
        if self.config.random_seed_override is not None:
            settings.RANDOM_SEED = self.config.random_seed_override
        if self.config.enable_boundary_jitter_override is not None:
            settings.ENABLE_BOUNDARY_JITTER = self.config.enable_boundary_jitter_override
        if self.config.bonafide_file_partition_override is not None:
            settings.BONAFIDE_FILE_PARTITION = self.config.bonafide_file_partition_override
        if self.config.manifest_path_override is not None:
            settings.MANIFEST_PATH = self.config.manifest_path_override
        if self.config.manifest_slice_attack_override is not None:
            settings.MANIFEST_SLICE_ATTACK = self.config.manifest_slice_attack_override
        if self.config.manifest_slice_partition_override is not None:
            settings.MANIFEST_SLICE_PARTITION = (
                self.config.manifest_slice_partition_override
            )

        if self.config.output_dir_override:
            settings.OUTPUT_DIR = self.config.output_dir_override
        elif self.config.use_manifest:
            partition = settings.BONAFIDE_FILE_PARTITION
            settings.OUTPUT_DIR = (
                Path("data/partial_spoof_output")
                / self.config.attack_system
                / partition
            )
        else:
            base_name = f"data/{self.config.attack_system}_partial_spoof"
            if settings.ENABLE_BOUNDARY_JITTER:
                base_name = f"{base_name}_jitter"
            settings.OUTPUT_DIR = Path(base_name)

    def _load_manifest(self) -> Optional[ManifestLoader]:
        """Load the dispatch manifest if manifest-driven mode is enabled.

        Returns:
            ManifestLoader with entries loaded from settings.MANIFEST_PATH,
            or None when use_manifest is False or the manifest file is
            absent.
        """
        if not self.config.use_manifest:
            return None
        if not settings.MANIFEST_PATH.exists():
            logger.warning(
                f"use_manifest=True but {settings.MANIFEST_PATH} does not exist. "
                "Falling back to legacy mode."
            )
            return None
        loader = ManifestLoader()
        loader.load(settings.MANIFEST_PATH)
        return loader

    def _build_checkpoint(self) -> Optional[CheckpointManager]:
        """Construct the per-(attack, partition) CheckpointManager.

        Returns:
            CheckpointManager keyed by (attack_system, partition) and
            writing into OUTPUT_DIR/.checkpoint.json, or None when
            ENABLE_CHECKPOINT_RESUME is False in settings.
        """
        if not settings.ENABLE_CHECKPOINT_RESUME:
            return None
        return CheckpointManager(
            attack=self.config.attack_system,
            partition=settings.BONAFIDE_FILE_PARTITION,
            output_dir=settings.OUTPUT_DIR,
        )

    def _resolve_manifest_slice_attack(self) -> str:
        """Return the manifest slice attack key for this run.

        Order of precedence: explicit settings.MANIFEST_SLICE_ATTACK,
        then the config attack_system.

        Returns:
            Attack identifier to filter the manifest by.
        """
        if settings.MANIFEST_SLICE_ATTACK is not None:
            return settings.MANIFEST_SLICE_ATTACK
        return self.config.attack_system

    def _resolve_manifest_slice_partition(self) -> str:
        """Return the manifest slice partition key for this run.

        Order of precedence: explicit settings.MANIFEST_SLICE_PARTITION,
        then settings.BONAFIDE_FILE_PARTITION.

        Returns:
            Partition identifier to filter the manifest by.
        """
        if settings.MANIFEST_SLICE_PARTITION is not None:
            return settings.MANIFEST_SLICE_PARTITION
        return settings.BONAFIDE_FILE_PARTITION

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
        logger.info(f"Bonafide partition: {settings.BONAFIDE_FILE_PARTITION}")
        logger.info(f"Boundary jitter: {settings.ENABLE_BOUNDARY_JITTER}")
        logger.info(f"Max regenerations: {MAX_REGENERATIONS}")
        logger.info("=" * 80)

        try:
            strategy = create_attack_strategy(self.config.attack_system)

            # === STEP 1: Transcribe bonafide audio (runs once) ===
            if self.config.run_step_1:
                logger.info("-" * 40)
                if self.manifest_loader is not None:
                    step_1 = BonafideTranscriber(
                        manifest_loader=self.manifest_loader,
                        manifest_attack=self._resolve_manifest_slice_attack(),
                        manifest_partition=self._resolve_manifest_slice_partition(),
                    )
                else:
                    step_1 = BonafideTranscriber()
                result_1 = step_1.execute()
                logger.info(
                    f"Step 1 result: {result_1.total_transcribed} transcribed, "
                    f"{result_1.skipped_short} skipped. "
                    f"Distribution: {result_1.word_count_distribution}"
                )

            # === STEPS 2-5: Generate -> Align -> Select -> Splice ===
            # Runs in a regeneration loop for failed samples
            if self.config.run_step_2:
                self._run_generation_loop(strategy)

            # === STEP 5b: Apply boundary jitter (optional) ===
            if (
                settings.ENABLE_BOUNDARY_JITTER
                and self.config.run_step_5b
            ):
                logger.info("-" * 40)
                step_5b = BoundaryJitterApplier()
                result_5b = step_5b.execute()
                logger.info(
                    f"Step 5b result: {result_5b.total_processed} processed, "
                    f"{result_5b.total_skipped} skipped, "
                    f"{result_5b.total_boundaries_seen} boundaries. "
                    f"Operations: {result_5b.operation_counts}. "
                    f"Avg drift: {result_5b.avg_duration_drift_ms:.1f} ms."
                )

            # === STEP 6: Validate splice quality ===
            if self.config.run_step_6:
                logger.info("-" * 40)
                step_6 = SpliceQualityValidator()
                result_6 = step_6.execute()
                logger.info(
                    f"Step 6 result: {result_6.total_validated} passed, "
                    f"{result_6.total_rejected} rejected. "
                    f"Avg WER={result_6.avg_wer:.4f}, CER={result_6.avg_cer:.4f}, "
                    f"NISQA={result_6.avg_nisqa:.2f}, Sim={result_6.avg_speaker_similarity:.3f}"
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

    def _run_generation_loop(self, strategy) -> None:
        """Run Steps 2-5 with regeneration for failed samples.

        Level 1 retry (smart selection) happens inside Step 5.
        Level 2 retry (regeneration) happens here — re-runs Steps 2-3-4-5
        for samples that exhausted all word selection candidates.

        Args:
            strategy: The loaded attack strategy for TTS generation.
        """
        all_regeneration_history = {}

        for regen_round in range(MAX_REGENERATIONS + 1):
            is_retry = regen_round > 0
            round_label = f"[Regen {regen_round}] " if is_retry else ""

            # --- Step 2: Generate cloned speech ---
            logger.info("-" * 40)
            logger.info(f"{round_label}STEP 2: Generate Cloned Speech")

            seed_offset = regen_round * 1000
            step_2 = ClonedSpeechGenerator(
                strategy=strategy,
                skip_existing=not is_retry,
                seed_offset=seed_offset,
                regenerate_keys=self._get_regen_keys(regen_round) if is_retry else None,
                checkpoint=self.checkpoint,
            )
            result_2 = step_2.execute()
            logger.info(
                f"{round_label}Step 2: {result_2.total_generated} generated, "
                f"{len(result_2.failed_generations)} failed, "
                f"avg RTF={result_2.avg_rtf:.3f}"
            )

            # --- Clone Similarity Gate (between Steps 2 and 3) ---
            if settings.ENABLE_CLONE_SIMILARITY_GATE:
                rejected_count = self._filter_clones_by_similarity(
                    round_label=round_label,
                )
                if rejected_count > 0:
                    logger.info(
                        f"{round_label}Similarity gate: {rejected_count} clones rejected "
                        f"(threshold={settings.MIN_CLONE_SIMILARITY})"
                    )

            # --- Step 3: Forced alignment ---
            if self.config.run_step_3:
                logger.info("-" * 40)
                logger.info(f"{round_label}STEP 3: Forced Alignment")
                step_3 = ForcedAligner()
                result_3 = step_3.execute()
                logger.info(
                    f"{round_label}Step 3: {result_3.total_aligned} aligned, "
                    f"{len(result_3.failed_alignments)} failed, "
                    f"avg {result_3.avg_words_per_utterance:.1f} words/utt"
                )

            # --- Step 4: Select words ---
            if self.config.run_step_4:
                logger.info("-" * 40)
                logger.info(f"{round_label}STEP 4: Select Words")
                step_4 = WordSelector(enabled_tiers=self.config.tiers)
                result_4 = step_4.execute()
                logger.info(
                    f"{round_label}Step 4: {result_4.total_selections} selections. "
                    f"Tiers: {result_4.tier_counts}"
                )

            # --- Step 5: Splice audio ---
            if self.config.run_step_5:
                logger.info("-" * 40)
                logger.info(f"{round_label}STEP 5: Splice Audio")
                step_5 = AudioSplicer(
                    attack_system_name=strategy.name(),
                    checkpoint=self.checkpoint,
                )
                result_5 = step_5.execute()
                logger.info(
                    f"{round_label}Step 5: {result_5.total_spliced} spliced, "
                    f"{len(result_5.failed_splices)} failed. "
                    f"Avg spoof ratio: {result_5.avg_spoof_duration_ratio:.3f}"
                )

            # Check if any samples need regeneration
            rejected_path = settings.OUTPUT_DIR / "splice_rejected.json"
            if rejected_path.exists():
                with open(rejected_path, "r", encoding="utf-8") as f:
                    rejected = json.load(f)

                needs_regen = {
                    k: v for k, v in rejected.items()
                    if v.get("best_achieved", 0) < v.get("expected_words", 1)
                }

                if not needs_regen:
                    logger.info(f"{round_label}No samples need regeneration. Done.")
                    break

                all_regeneration_history[f"round_{regen_round}"] = {
                    "rejected_count": len(needs_regen),
                    "sample_ids": list(needs_regen.keys()),
                }

                if regen_round < MAX_REGENERATIONS:
                    sample_keys = set()
                    for key in needs_regen:
                        base_key = key.rsplit("_W", 1)[0]
                        sample_keys.add(base_key)

                    logger.info(
                        f"{round_label}{len(needs_regen)} samples need regeneration "
                        f"({len(sample_keys)} unique utterances). "
                        f"Starting regen round {regen_round + 1}..."
                    )
                    self._save_regen_keys(regen_round + 1, list(sample_keys))
                else:
                    logger.warning(
                        f"Max regenerations ({MAX_REGENERATIONS}) reached. "
                        f"{len(needs_regen)} samples remain unresolved."
                    )
            else:
                logger.info(f"{round_label}No rejections. Done.")
                break

        # Save regeneration history
        if all_regeneration_history:
            history_path = settings.OUTPUT_DIR / "regeneration_history.json"
            with open(history_path, "w", encoding="utf-8") as f:
                json.dump(all_regeneration_history, f, ensure_ascii=False, indent=2)
            logger.info(f"Regeneration history saved to {history_path}")

    def _save_regen_keys(self, round_num: int, sample_keys: list) -> None:
        """Save sample keys that need regeneration for the next round.

        Args:
            round_num: The regeneration round number.
            sample_keys: List of base sample keys (without tier suffix).
        """
        regen_path = settings.OUTPUT_DIR / f"regen_keys_round_{round_num}.json"
        with open(regen_path, "w", encoding="utf-8") as f:
            json.dump(sample_keys, f)

    def _filter_clones_by_similarity(self, round_label: str = "") -> int:
        """Filter cloned audio by ECAPA-TDNN speaker similarity.

        Loads each clone and its bonafide reference, computes cosine
        similarity, and removes entries below MIN_CLONE_SIMILARITY from
        the cloned_generation_metadata.json. Saves a filter log for
        traceability.

        Args:
            round_label: Prefix for log messages (e.g. "[Regen 1] ").

        Returns:
            Number of clones rejected.
        """
        from app.utils.ecapa_similarity import EcapaSimilarity

        gen_meta_path = settings.OUTPUT_DIR / "cloned_generation_metadata.json"
        transcripts_path = settings.OUTPUT_DIR / "bonafide_transcripts.json"

        if not gen_meta_path.exists() or not transcripts_path.exists():
            logger.warning(f"{round_label}Similarity gate: metadata not found, skipping.")
            return 0

        with open(gen_meta_path, "r", encoding="utf-8") as f:
            gen_meta = json.load(f)
        with open(transcripts_path, "r", encoding="utf-8") as f:
            transcripts = json.load(f)

        ecapa = EcapaSimilarity()
        ecapa.load(device=settings.DEVICE)

        ref_embeddings = {}
        filter_log = {}
        rejected_keys = []
        similarities = []

        for sample_key, entry in list(gen_meta.items()):
            speaker_id = entry.get("speaker_id", sample_key.split("_")[0])

            if speaker_id not in ref_embeddings:
                bf_path = None
                if sample_key in transcripts:
                    bf_path = transcripts[sample_key].get("audio_path")
                if bf_path and Path(bf_path).exists():
                    ref_embeddings[speaker_id] = ecapa.extract_embedding(Path(bf_path))
                else:
                    continue

            cloned_path = entry.get("audio_path", "")
            if not Path(cloned_path).exists():
                continue

            sim = ecapa.compute_similarity_from_embedding(
                ref_embeddings[speaker_id], Path(cloned_path)
            )
            similarities.append(sim)

            passed = sim >= settings.MIN_CLONE_SIMILARITY
            filter_log[sample_key] = {
                "similarity": round(sim, 4),
                "passed": passed,
                "speaker_id": speaker_id,
            }

            if not passed:
                rejected_keys.append(sample_key)
                del gen_meta[sample_key]

        with open(gen_meta_path, "w", encoding="utf-8") as f:
            json.dump(gen_meta, f, ensure_ascii=False, indent=2)

        filter_path = settings.OUTPUT_DIR / "clone_similarity_filter.json"
        with open(filter_path, "w", encoding="utf-8") as f:
            json.dump(filter_log, f, ensure_ascii=False, indent=2)

        avg_sim = float(np.mean(similarities)) if similarities else 0.0
        logger.info(
            f"{round_label}Similarity gate: {len(similarities)} evaluated, "
            f"{len(rejected_keys)} rejected, avg SIM={avg_sim:.3f}"
        )

        return len(rejected_keys)

    def _get_regen_keys(self, round_num: int) -> list | None:
        """Load sample keys that need regeneration for this round.

        Args:
            round_num: The current regeneration round number.

        Returns:
            List of sample keys, or None if file not found.
        """
        regen_path = settings.OUTPUT_DIR / f"regen_keys_round_{round_num}.json"
        if regen_path.exists():
            with open(regen_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return None


if __name__ == "__main__":
    settings.MAX_SAMPLES = 50
    config = PartialSpoofPipelineConfig(attack_system="qwen")
    pipeline = PartialSpoofPipeline(config=config)
    pipeline.run()
