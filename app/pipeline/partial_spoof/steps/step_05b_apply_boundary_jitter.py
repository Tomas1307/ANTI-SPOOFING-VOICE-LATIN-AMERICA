"""
Step 5b: Apply Boundary Jitter

Post-processing step that runs after Step 5 (splice) and before Step 6
(quality validation). For each spliced utterance, every internal word
boundary is independently subjected to a coin flip with probability
JITTER_PROBABILITY. Heads -> randomly choose one of three structural
manipulations (truncate, overlap, bleed) and apply at that boundary.
Tails -> leave the boundary natural.

Spoof boundaries (those surrounding cloned words) receive the same coin
flip on top of the splice that was already applied in Step 5. This
homogenizes boundary artifacts so the splice does not stand out as the
only manipulated boundary, attacking the "find the noisy boundary"
detector shortcut documented in Negroni et al. (2024) and the
generalization-shortcut analysis in Muller (2024).

Boundaries are processed right-to-left so each manipulation only affects
later (un-touched) audio; left-side boundary timestamps remain valid
across iterations. The total audio length may drift modestly (positive
for bleed, negative for truncate/overlap); the drift is recorded per
utterance for later analysis.
"""
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Tuple

import librosa
import numpy as np
import soundfile as sf
from loguru import logger
from tqdm import tqdm

from app.pipeline.partial_spoof.settings import settings
from app.pipeline.partial_spoof.schemas.boundary_jitter_result import BoundaryJitterResult
from app.pipeline.partial_spoof.utils.word_bleed import bleed_at_boundary
from app.pipeline.partial_spoof.utils.word_overlap import overlap_at_boundary
from app.pipeline.partial_spoof.utils.word_truncate import truncate_at_boundary


OPERATIONS = ("truncate", "overlap", "bleed")


class BoundaryJitterApplier:
    """Applies random boundary manipulations to spliced utterances.

    Reads splice_metadata.json and alignment_metadata.json from the
    main pipeline directory, applies the jitter plan per utterance, writes
    new WAV files to ``jittered/`` next to ``spliced/``, and saves a
    boundary_jitter_metadata.json that mirrors splice_metadata.json (so
    Step 6 / Step 7 can transparently consume it as the new spliced output).

    Attributes:
        output_dir: Pipeline output directory containing splice metadata
            and the spliced/ subdirectory.
    """

    def __init__(self, output_dir: Path | None = None) -> None:
        """Initialize boundary jitter applier.

        Args:
            output_dir: Output directory (default: from settings).
        """
        self.output_dir = output_dir or settings.OUTPUT_DIR

    def execute(self) -> BoundaryJitterResult:
        """Apply boundary jitter to all spliced utterances.

        Returns:
            BoundaryJitterResult with metadata path and processing statistics.

        Raises:
            FileNotFoundError: If splice_metadata.json or alignment metadata
                is missing.
        """
        logger.info("Step 5b: Applying boundary jitter to spliced utterances...")

        splice_path = self.output_dir / "splice_metadata.json"
        alignment_path = self.output_dir / "alignment_metadata.json"

        if not splice_path.exists():
            raise FileNotFoundError(
                f"splice_metadata.json not found at {splice_path}. Run Step 5 first."
            )
        if not alignment_path.exists():
            raise FileNotFoundError(
                f"alignment_metadata.json not found at {alignment_path}. Run Step 3 first."
            )

        with open(splice_path, "r", encoding="utf-8") as f:
            splice_metadata = json.load(f)
        with open(alignment_path, "r", encoding="utf-8") as f:
            alignment_data = json.load(f)

        jittered_dir = self.output_dir / "jittered"
        jittered_dir.mkdir(parents=True, exist_ok=True)

        operation_counts: Dict[str, int] = {op: 0 for op in OPERATIONS}
        operation_counts["none"] = 0
        total_processed = 0
        total_skipped = 0
        total_boundaries_seen = 0
        drift_samples: List[int] = []
        failed_utterances: List[str] = []

        new_metadata: Dict[str, dict] = {}
        jitter_plans: Dict[str, dict] = {}

        for splice_key, entry in tqdm(splice_metadata.items(), desc="Boundary jitter"):
            spliced_audio_path = Path(entry["spliced_audio_path"])
            if not spliced_audio_path.exists():
                logger.warning(f"Spliced audio missing for {splice_key}: {spliced_audio_path}")
                total_skipped += 1
                failed_utterances.append(splice_key)
                continue

            sample_key = self._sample_key_from_splice_key(splice_key)
            if sample_key not in alignment_data:
                logger.warning(f"Alignment missing for {splice_key} (sample {sample_key})")
                total_skipped += 1
                failed_utterances.append(splice_key)
                continue

            try:
                audio, _ = librosa.load(
                    str(spliced_audio_path), sr=settings.SAMPLE_RATE, mono=True
                )
            except Exception as exc:
                logger.error(f"Failed to load spliced audio for {splice_key}: {exc}")
                total_skipped += 1
                failed_utterances.append(splice_key)
                continue

            bonafide_words = alignment_data[sample_key].get("bonafide_words", [])
            if len(bonafide_words) < 2:
                jittered_path = jittered_dir / spliced_audio_path.name
                sf.write(str(jittered_path), audio, settings.SAMPLE_RATE)
                new_entry = dict(entry)
                new_entry["spliced_audio_path"] = str(jittered_path)
                new_entry["jitter_applied"] = False
                new_entry["jitter_drift_samples"] = 0
                new_metadata[splice_key] = new_entry
                jitter_plans[splice_key] = {"boundaries": [], "drift_samples": 0}
                total_processed += 1
                continue

            seed = settings.JITTER_SEED + self._stable_hash(splice_key)
            rng = np.random.RandomState(seed)

            boundaries = self._compute_boundary_samples(bonafide_words)
            total_boundaries_seen += len(boundaries)

            audio_jittered, plan, drift = self._apply_jitter_plan(
                audio=audio, boundaries=boundaries, rng=rng
            )

            for op_label in plan["op_labels"]:
                operation_counts[op_label] = operation_counts.get(op_label, 0) + 1

            jittered_path = jittered_dir / spliced_audio_path.name
            sf.write(str(jittered_path), audio_jittered, settings.SAMPLE_RATE)

            new_entry = dict(entry)
            new_entry["spliced_audio_path"] = str(jittered_path)
            new_entry["total_duration_s"] = float(len(audio_jittered) / settings.SAMPLE_RATE)
            new_entry["jitter_applied"] = True
            new_entry["jitter_drift_samples"] = int(drift)
            new_metadata[splice_key] = new_entry

            jitter_plans[splice_key] = {
                "boundaries": plan["boundary_records"],
                "drift_samples": int(drift),
            }

            drift_samples.append(drift)
            total_processed += 1

        new_metadata_path = self.output_dir / "splice_metadata.json"
        with open(new_metadata_path, "w", encoding="utf-8") as f:
            json.dump(new_metadata, f, ensure_ascii=False, indent=2)

        plan_path = self.output_dir / "boundary_jitter_metadata.json"
        with open(plan_path, "w", encoding="utf-8") as f:
            json.dump(jitter_plans, f, ensure_ascii=False, indent=2)

        avg_drift_ms = 0.0
        if drift_samples:
            avg_drift_ms = float(
                np.mean([abs(d) / settings.SAMPLE_RATE * 1000.0 for d in drift_samples])
            )

        logger.info(
            f"Step 5b complete: {total_processed} processed, "
            f"{total_skipped} skipped. Boundaries seen: {total_boundaries_seen}. "
            f"Operation counts: {operation_counts}. "
            f"Avg |drift|: {avg_drift_ms:.1f} ms."
        )

        return BoundaryJitterResult(
            jitter_metadata_path=plan_path,
            total_processed=total_processed,
            total_skipped=total_skipped,
            total_boundaries_seen=total_boundaries_seen,
            operation_counts=operation_counts,
            avg_duration_drift_ms=avg_drift_ms,
            failed_utterances=failed_utterances,
        )

    def _sample_key_from_splice_key(self, splice_key: str) -> str:
        """Derive the alignment sample key from the splice key.

        Splice keys follow the pattern ``{sample_key}_{tier}`` where tier
        is one of W1/W2/W3. The alignment metadata is keyed by sample_key
        (without tier).

        Args:
            splice_key: Splice metadata key, e.g. ``arf_00295_clip01_W2``.

        Returns:
            Underlying sample key, e.g. ``arf_00295_clip01``.
        """
        for tier in ("_W1", "_W2", "_W3"):
            if splice_key.endswith(tier):
                return splice_key[: -len(tier)]
        return splice_key

    def _stable_hash(self, key: str) -> int:
        """Compute a deterministic 32-bit integer hash of a string key.

        Python's built-in hash() is randomized per process, so we use a
        SHA-256 prefix for reproducibility across runs.

        Args:
            key: String to hash.

        Returns:
            32-bit unsigned integer derived from the SHA-256 digest.
        """
        digest = hashlib.sha256(key.encode("utf-8")).digest()[:4]
        return int.from_bytes(digest, byteorder="little")

    def _compute_boundary_samples(self, bonafide_words: List[dict]) -> List[int]:
        """Convert word-level alignments to boundary sample indices.

        Internal boundaries are the start times of words 1..N-1 (the first
        word's start and the last word's end are NOT internal boundaries).

        Args:
            bonafide_words: List of word alignment entries with start_seconds
                and end_seconds keys.

        Returns:
            List of boundary sample indices, one per internal boundary.
        """
        sr = settings.SAMPLE_RATE
        boundaries: List[int] = []
        for i in range(1, len(bonafide_words)):
            start_s = bonafide_words[i].get(
                "start_seconds", bonafide_words[i].get("start", 0.0)
            )
            boundaries.append(int(start_s * sr))
        return boundaries

    def _apply_jitter_plan(
        self,
        audio: np.ndarray,
        boundaries: List[int],
        rng: np.random.RandomState,
    ) -> Tuple[np.ndarray, dict, int]:
        """Apply the jitter plan to one utterance.

        Iterates boundaries right-to-left so each manipulation only affects
        later (un-touched) audio; the absolute sample index of earlier
        boundaries remains valid across iterations.

        Args:
            audio: Spliced audio (1-D float32, at settings.SAMPLE_RATE).
            boundaries: Internal boundary sample indices (left-to-right).
            rng: Seeded numpy RandomState for reproducible decisions.

        Returns:
            Tuple of:
                - Modified audio.
                - Plan dict with 'op_labels' (per-boundary op or 'none')
                  and 'boundary_records' (detailed per-boundary log).
                - Total length delta in samples (positive=grew, negative=shrank).
        """
        sr = settings.SAMPLE_RATE
        op_labels: List[str] = []
        boundary_records: List[dict] = []
        total_delta = 0
        truncate_min, truncate_max = settings.JITTER_TRUNCATE_RANGE_MS
        overlap_min, overlap_max = settings.JITTER_OVERLAP_RANGE_MS
        bleed_min, bleed_max = settings.JITTER_BLEED_RANGE_MS

        for i in reversed(range(len(boundaries))):
            boundary_sample = boundaries[i]

            if rng.uniform(0.0, 1.0) >= settings.JITTER_PROBABILITY:
                op_labels.append("none")
                boundary_records.append({
                    "boundary_index": i,
                    "boundary_sample": boundary_sample,
                    "operation": "none",
                })
                continue

            op = OPERATIONS[rng.randint(0, len(OPERATIONS))]
            record: dict = {
                "boundary_index": i,
                "boundary_sample": boundary_sample,
                "operation": op,
            }

            if op == "truncate":
                ms = float(rng.uniform(truncate_min, truncate_max))
                duration_samples = int(ms / 1000.0 * sr)
                side = "left_tail" if rng.randint(0, 2) == 0 else "right_head"
                audio, delta = truncate_at_boundary(
                    audio, boundary_sample, duration_samples, side=side
                )
                record.update({"side": side, "duration_ms": ms, "delta_samples": delta})
            elif op == "overlap":
                ms = float(rng.uniform(overlap_min, overlap_max))
                overlap_samples = int(ms / 1000.0 * sr)
                audio, delta = overlap_at_boundary(
                    audio,
                    boundary_sample,
                    overlap_samples,
                    fade=settings.JITTER_OVERLAP_FADE,
                )
                record.update({
                    "duration_ms": ms,
                    "fade": settings.JITTER_OVERLAP_FADE,
                    "delta_samples": delta,
                })
            else:
                ms = float(rng.uniform(bleed_min, bleed_max))
                duration_samples = int(ms / 1000.0 * sr)
                direction = (
                    "right_to_left" if rng.randint(0, 2) == 0 else "left_to_right"
                )
                audio, delta = bleed_at_boundary(
                    audio, boundary_sample, duration_samples, direction=direction
                )
                record.update({
                    "direction": direction,
                    "duration_ms": ms,
                    "delta_samples": delta,
                })

            op_labels.append(op)
            boundary_records.append(record)
            total_delta += delta

        boundary_records.reverse()
        op_labels.reverse()

        return (
            audio,
            {"op_labels": op_labels, "boundary_records": boundary_records},
            total_delta,
        )
