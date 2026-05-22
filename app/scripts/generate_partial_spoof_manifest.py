"""
Pre-flight: generate the partial spoof dispatch manifest.

Runs Parakeet TDT on the FULL HABLA v2 bonafide corpus (both
not_jittered and jittered partitions), then writes the dispatch
manifest CSV plus its summary JSON. The manifest is the single
source of truth consumed by all 12 per-attack pipeline runs
(6 attacks x 2 partitions).

Idempotent: re-running with the same seeds and the same bonafide
data produces byte-identical CSV and summary outputs.

Usage on ml-server03:
    export CUDA_VISIBLE_DEVICES=1
    source ~/ANTI-SPOOFING-VOICE-LATIN-AMERICA/envs/fishgram_env/bin/activate
    python -m app.scripts.generate_partial_spoof_manifest
"""
import json
from pathlib import Path
from typing import Dict, List

from loguru import logger
from tqdm import tqdm

from app.pipeline.partial_spoof.settings import settings
from app.pipeline.partial_spoof.utils.manifest_generator import ManifestGenerator
from app.pipeline.partial_spoof.utils.manifest_loader import ManifestLoader
from app.pipeline.partial_spoof.utils.sample_key_builder import SampleKeyBuilder
from app.pipeline.partial_spoof.utils.tier_eligibility import TierEligibilityComputer
from app.utils.parakeet_transcriber import ParakeetTranscriber


class ManifestPreflightRunner:
    """Run the one-shot pre-flight that produces the dispatch manifest.

    Single responsibility: transcribe the full HABLA v2 bonafide corpus
    (ignoring the not_jittered/jittered partition filter so both halves
    are covered in one pass), feed the transcripts to the
    ManifestGenerator, and persist the manifest CSV + summary JSON +
    full-transcript JSON. The full-transcript JSON is reused by every
    per-attack pipeline run via the manifest-aware Step 1 short-circuit,
    so Parakeet TDT only runs once on the entire corpus.

    Attributes:
        bonafide_dir: Path to the HABLA v2 speaker root.
        manifest_path: Path to the output manifest CSV.
        summary_path: Path to the output summary JSON.
        transcripts_path: Path to the cached full-transcript JSON.
    """

    TRANSCRIPTS_FILENAME = "bonafide_transcripts_full.json"

    def __init__(self) -> None:
        """Initialise from the partial spoof settings singleton."""
        self.bonafide_dir = settings.BONAFIDE_DIR
        self.manifest_path = settings.MANIFEST_PATH
        self.summary_path = settings.MANIFEST_SUMMARY_PATH
        self.transcripts_path = (
            settings.MANIFEST_PATH.parent / self.TRANSCRIPTS_FILENAME
        )

    def run(self) -> None:
        """Execute the full pre-flight: transcribe, plan, persist.

        Idempotent and side-effect-bounded to the manifest directory.
        """
        logger.info("=" * 80)
        logger.info("PARTIAL SPOOF MANIFEST PRE-FLIGHT - START")
        logger.info(f"Bonafide root        : {self.bonafide_dir}")
        logger.info(f"Manifest CSV         : {self.manifest_path}")
        logger.info(f"Summary JSON         : {self.summary_path}")
        logger.info(f"Cached transcripts   : {self.transcripts_path}")
        logger.info(f"Attack weights       : {settings.ATTACK_WEIGHTS}")
        logger.info(f"Assignment seed      : {settings.ATTACK_ASSIGNMENT_SEED}")
        logger.info(f"Partition seed       : {settings.BONAFIDE_PARTITION_SEED}")
        logger.info("=" * 80)

        transcripts = self._transcribe_full_corpus()
        self._save_transcripts(transcripts)

        generator = ManifestGenerator(
            attack_weights=settings.ATTACK_WEIGHTS,
            attack_assignment_seed=settings.ATTACK_ASSIGNMENT_SEED,
            bonafide_partition_seed=settings.BONAFIDE_PARTITION_SEED,
            tier_computer=TierEligibilityComputer(
                min_words_w1=settings.MIN_WORDS_W1,
                min_words_w2=settings.MIN_WORDS_W2,
                min_words_w3=settings.MIN_WORDS_W3,
            ),
        )
        entries, summary = generator.generate(transcripts)

        loader = ManifestLoader()
        loader.save(entries, self.manifest_path)
        loader.save_summary(summary, self.summary_path)

        self._log_summary(summary)

        logger.info("=" * 80)
        logger.info("PARTIAL SPOOF MANIFEST PRE-FLIGHT - COMPLETE")
        logger.info("=" * 80)

    def _transcribe_full_corpus(self) -> Dict[str, Dict]:
        """Transcribe every bonafide file with Parakeet TDT.

        Bypasses the not_jittered/jittered partition filter so a single
        pass covers both halves of the corpus. Returns transcripts keyed
        by sample_key in the same shape as Step 1's
        bonafide_transcripts.json output.

        Returns:
            Dict from sample_key to entry dict with speaker_id, split,
            audio_path, transcript, word_count, word_timestamps.
        """
        if self.transcripts_path.exists():
            logger.info(
                f"Cached transcripts found at {self.transcripts_path}; "
                "loading instead of re-transcribing."
            )
            with open(self.transcripts_path, "r", encoding="utf-8") as handle:
                return json.load(handle)

        logger.info("Loading Parakeet TDT model...")
        transcriber = ParakeetTranscriber()
        transcriber.load(
            model_id=settings.PARAKEET_MODEL_ID,
            device=settings.DEVICE,
        )

        speaker_dirs = self._collect_speaker_dirs()
        logger.info(f"Transcribing {len(speaker_dirs)} speakers...")

        transcripts: Dict[str, Dict] = {}
        skipped_short = 0
        for speaker_dir in tqdm(speaker_dirs, desc="Speakers"):
            speaker_id = speaker_dir.name
            for split in ("train", "val", "test"):
                split_dir = speaker_dir / split
                if not split_dir.exists():
                    continue
                audio_files = self._collect_audio_files(split_dir)
                for audio_path in audio_files:
                    text, timestamps = transcriber.transcribe_with_timestamps(
                        audio_path
                    )
                    word_count = len(text.split())
                    if word_count < settings.MIN_WORDS_W1:
                        skipped_short += 1
                        continue
                    sample_key = SampleKeyBuilder.build(speaker_id, audio_path.stem)
                    transcripts[sample_key] = {
                        "speaker_id": speaker_id,
                        "split": split,
                        "audio_path": str(audio_path),
                        "transcript": text,
                        "word_count": word_count,
                        "word_timestamps": [
                            {"word": wt.word, "start": wt.start, "end": wt.end}
                            for wt in timestamps
                        ],
                    }

        logger.info(
            f"Transcription complete: {len(transcripts)} kept, "
            f"{skipped_short} skipped (< {settings.MIN_WORDS_W1} words)."
        )
        return transcripts

    def _collect_speaker_dirs(self) -> List[Path]:
        """Return the list of speaker directories under the bonafide root.

        Returns:
            Sorted list of speaker directory paths. Honours
            settings.VALIDATION_MODE for quick local sanity checks but
            production runs should always set VALIDATION_MODE=False.
        """
        if settings.VALIDATION_MODE:
            dirs: List[Path] = []
            for speaker_id in settings.VALIDATION_SPEAKERS:
                candidate = self.bonafide_dir / speaker_id
                if candidate.exists():
                    dirs.append(candidate)
                else:
                    logger.warning(f"Validation speaker missing: {candidate}")
            return dirs

        return sorted(
            d for d in self.bonafide_dir.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        )

    def _collect_audio_files(self, split_dir: Path) -> List[Path]:
        """Return audio files under split_dir matching wav/flac/mp3 patterns.

        Args:
            split_dir: train/val/test directory under one speaker.

        Returns:
            Sorted list of audio paths.
        """
        return sorted(
            f
            for ext in ("*.wav", "*.flac", "*.mp3")
            for f in split_dir.glob(ext)
        )

    def _save_transcripts(self, transcripts: Dict[str, Dict]) -> None:
        """Persist the full transcripts JSON for per-attack runs to reuse.

        Args:
            transcripts: Full sample_key -> entry mapping.
        """
        self.transcripts_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.transcripts_path, "w", encoding="utf-8") as handle:
            json.dump(transcripts, handle, ensure_ascii=False, indent=2)
        logger.info(
            f"Saved full transcripts to {self.transcripts_path} "
            f"({len(transcripts)} entries)."
        )

    def _log_summary(self, summary) -> None:
        """Pretty-print the manifest summary to the log.

        Args:
            summary: ManifestSummary instance.
        """
        logger.info("-" * 60)
        logger.info("Manifest summary")
        logger.info(f"  Total entries     : {summary.total_entries}")
        logger.info(f"  Speakers (total)  : {summary.speakers_total}")
        logger.info(
            f"  Speakers covering all attacks : "
            f"{summary.speakers_with_all_attacks}"
        )
        logger.info(
            f"  Speakers with single attack   : "
            f"{summary.speakers_with_single_attack}"
        )
        logger.info("  Attack counts (target vs actual):")
        for attack in summary.attack_weights_target:
            target = summary.attack_weights_target[attack]
            actual = summary.attack_weights_actual.get(attack, 0.0)
            count = summary.per_attack_count.get(attack, 0)
            logger.info(
                f"    {attack:<12} target={target:.3f}  "
                f"actual={actual:.3f}  n={count}"
            )
        logger.info("  Partition counts:")
        for part, count in summary.per_partition_count.items():
            logger.info(f"    {part:<14} n={count}")
        logger.info("  Tier potential counts (planned slots):")
        for tier, count in summary.per_tier_potential_count.items():
            logger.info(f"    {tier:<4} n={count}")
        logger.info("-" * 60)


if __name__ == "__main__":
    ManifestPreflightRunner().run()
