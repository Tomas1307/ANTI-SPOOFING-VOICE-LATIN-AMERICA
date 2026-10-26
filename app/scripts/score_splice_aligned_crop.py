"""
Re-score partial-spoof clips with a crop window aligned to the splice.

WHY
---
DF-Arena's published feature extractor truncates every clip to a fixed
64,600-sample (4.0375 s) window taken from the geometric centre of the clip.
``measure_crop_confound.py`` showed that this window misses every spliced
word on 7.7% of partial-spoof clips overall, and on 12.4% of W1 clips, purely
from clip geometry -- those clips cannot be correctly classified as spoof by
construction under the zero-shot DF-Arena baseline, regardless of detector
quality.

This script answers the follow-up question that geometry alone cannot: if
the crop window is placed so the spliced word is actually inside it, how
well does DF-Arena detect the splice it was shown? It does not change how
the published zero-shot baseline is scored, and it does not touch
``app/pipeline/training`` or any file that baseline depends on. It reads the
partial-spoof pipeline's own splice metadata and audio directly, builds a new
crop window per clip centred on the spliced-word span instead of on the whole
clip, runs the published DF-Arena weights on that window, and reports EER by
tier against the bonafide score pool from an existing zero-shot run.

Comparing this "aligned-crop EER" to the original zero-shot EER (18-47%
across tiers, per ``measure_crop_confound.py``'s docstring) separates two
explanations that are currently confounded: the attack being genuinely hard
to detect, versus the detector never being shown the manipulated region.

METHOD
------
For each spliced clip longer than the crop window, the spliced-word span is
reconstructed from ``bonafide_start_s``/``bonafide_end_s`` and
``cloned_natural_duration_s`` in the pipeline's own ``splice_metadata.json``
convention (clips carry natural-duration splices with no time-stretch, so
later words shift by the cumulative duration delta of every earlier splice).
A crop window of exactly 64,600 samples is then centred on that span and
clamped to the clip, so the window covers as much of the splice as physically
possible. Clips at or under the crop length need no realignment: the model's
own tiling already keeps every sample. The audio is loaded from
``spliced_audio_path`` as recorded in the metadata at splice time; a clip
whose file has since moved or been cleaned up is skipped and counted, not
silently dropped.

Bonafide scores are not recomputed. They are read from an existing zero-shot
DF-Arena score file (produced by
``python -m app.scripts.run_detector_training --eval-only``), since bonafide
clips carry no splice and nothing about this experiment changes how they
should be scored.

USAGE
-----
    cd ~/ANTI-SPOOFING-VOICE-LATIN-AMERICA
    source envs/dfarena_env/bin/activate
    export CUDA_VISIBLE_DEVICES=1
    python -m app.scripts.score_splice_aligned_crop \\
        --partial-spoof-root data/partial_spoof_output \\
        --bonafide-scores-file data/training_runs/dfarena_zeroshot/scores/scores_eval.txt \\
        --model-id Speech-Arena-2025/DF_Arena_1B_V_1 \\
        --output data/training_runs/dfarena_zeroshot/scores/scores_aligned_crop.txt
    deactivate
"""
import argparse
import glob
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import soundfile as sf
import torch
from loguru import logger

from app.pipeline.training.dfarena.dfarena_detector import DFArenaDetector
from app.pipeline.training.utils import metrics

CROP_SAMPLES = 64600
SAMPLE_RATE = 16000
CROP_SECONDS = CROP_SAMPLES / SAMPLE_RATE


def reconstruct_final_positions(spoofed_words: List[dict]) -> List[Tuple[float, float]]:
    """Reconstruct each spliced word's position within the final spliced file.

    Duplicated from ``measure_crop_confound.py`` deliberately, so this script
    stays runnable from a single file with no project-internal dependency
    beyond the detector and metrics utilities it needs for inference. The
    pipeline splices with natural duration and no time-stretch, so each word
    after the first shifts by the cumulative duration delta of every earlier
    splice in the same clip.

    Args:
        spoofed_words: The clip's spoofed_words list, in transcript order.

    Returns:
        One (start_s, end_s) tuple per word, in final-file coordinates.
    """
    positions: List[Tuple[float, float]] = []
    cumulative_offset = 0.0

    for word in spoofed_words:
        bonafide_start = word["bonafide_start_s"]
        bonafide_end = word["bonafide_end_s"]
        cloned_duration = word.get("cloned_natural_duration_s")
        if cloned_duration is None:
            cloned_duration = bonafide_end - bonafide_start

        final_start = bonafide_start + cumulative_offset
        final_end = final_start + cloned_duration
        positions.append((final_start, final_end))

        bonafide_duration = bonafide_end - bonafide_start
        cumulative_offset += cloned_duration - bonafide_duration

    return positions


def _overlap_seconds(
    window_start: float, window_end: float, positions: List[Tuple[float, float]]
) -> float:
    """Sum the overlap between a window and every spliced-word interval.

    Args:
        window_start: Start of the candidate window, in seconds.
        window_end: End of the candidate window, in seconds.
        positions: Spliced-word intervals in final-file coordinates.

    Returns:
        Total overlapping duration, in seconds.
    """
    return sum(
        max(0.0, min(end, window_end) - max(start, window_start))
        for start, end in positions
    )


def _whole_word_count(
    window_start: float, window_end: float, positions: List[Tuple[float, float]]
) -> int:
    """Count spliced words that fall entirely inside a window.

    Args:
        window_start: Start of the candidate window, in seconds.
        window_end: End of the candidate window, in seconds.
        positions: Spliced-word intervals in final-file coordinates.

    Returns:
        Number of intervals fully contained in the window.
    """
    return sum(1 for start, end in positions if start >= window_start and end <= window_end)


def aligned_crop_window(
    positions: List[Tuple[float, float]], total_duration_s: float
) -> Tuple[float, float]:
    """Compute the crop window that best covers the spliced words.

    Unlike DF-Arena's own centre-of-clip crop, this searches for the window
    placement that keeps the most spliced content visible. When every
    spliced word fits inside one 64,600-sample window (the common case: W1
    is a single word, and most W2/W3 words land close together), this finds
    the placement that contains all of them whole. When the words are spread
    further apart than the window allows -- possible in W2/W3, whose words
    can sit far apart in a long clip -- centring on the full span from the
    first word's start to the last word's end is not guaranteed to be best:
    it can clip pieces of every word instead of keeping some of them whole.
    This instead picks the placement that keeps the most words entirely
    inside the window, breaking ties by the placement with the largest total
    overlap, so a left-over word is dropped whole rather than left half-cut.

    The search only needs to check window starts aligned to each word's own
    start or end, plus the two clip-edge positions: the optimal window for a
    fixed-width overlap maximisation over intervals always has one edge
    touching an interval endpoint, so no other placement can do better.

    Args:
        positions: Spliced-word intervals in final-file coordinates.
        total_duration_s: Duration of the final spliced clip.

    Returns:
        A (start_s, end_s) tuple of exactly ``CROP_SECONDS`` duration, or
        (0.0, total_duration_s) when the clip is at or under the crop length
        and needs no cropping at all.
    """
    if total_duration_s <= CROP_SECONDS:
        return 0.0, total_duration_s

    max_start = total_duration_s - CROP_SECONDS
    candidates = {0.0, max_start}
    for start, end in positions:
        candidates.add(start)
        candidates.add(end - CROP_SECONDS)
    candidates = {max(0.0, min(candidate, max_start)) for candidate in candidates}

    best_start = 0.0
    best_key = (-1, -1.0)
    for window_start in candidates:
        window_end = window_start + CROP_SECONDS
        whole = _whole_word_count(window_start, window_end, positions)
        overlap = _overlap_seconds(window_start, window_end, positions)
        key = (whole, overlap)
        if key > best_key:
            best_key = key
            best_start = window_start

    return best_start, best_start + CROP_SECONDS


def coverage_stats(
    positions: List[Tuple[float, float]], window_start: float, window_end: float
) -> Tuple[int, int]:
    """Report how many spliced words survive a chosen window intact.

    Args:
        positions: Spliced-word intervals in final-file coordinates.
        window_start: Start of the chosen window, in seconds.
        window_end: End of the chosen window, in seconds.

    Returns:
        A tuple of (whole_word_count, total_word_count).
    """
    return _whole_word_count(window_start, window_end, positions), len(positions)


def load_aligned_crop(audio_path: Path, window_start_s: float, window_end_s: float) -> np.ndarray:
    """Load one clip and extract or tile it to exactly CROP_SAMPLES samples.

    Args:
        audio_path: Path to the spliced audio file.
        window_start_s: Start of the crop window, in seconds.
        window_end_s: End of the crop window, in seconds.

    Returns:
        A waveform of exactly ``CROP_SAMPLES`` samples at ``SAMPLE_RATE``.

    Raises:
        ValueError: If the file's sample rate is not ``SAMPLE_RATE``. The
            project convention is to fail loudly on a rate mismatch rather
            than silently resample, matching ``MarsaAudioDataset``.
    """
    waveform, rate = sf.read(str(audio_path), dtype="float32", always_2d=False)
    if rate != SAMPLE_RATE:
        raise ValueError(
            f"{audio_path.name} is {rate} Hz, expected {SAMPLE_RATE} Hz"
        )
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)
    length = waveform.shape[0]

    if length <= CROP_SAMPLES:
        repeats = int(np.ceil(CROP_SAMPLES / max(length, 1)))
        tiled = np.tile(waveform, repeats)
        return tiled[:CROP_SAMPLES]

    start_sample = int(round(window_start_s * SAMPLE_RATE))
    start_sample = max(0, min(start_sample, length - CROP_SAMPLES))
    return waveform[start_sample : start_sample + CROP_SAMPLES]


def collect_spoof_clips(partial_spoof_root: str) -> List[dict]:
    """Scan every system/partition's splice metadata for usable clips.

    Args:
        partial_spoof_root: Root of the partial-spoof output tree.

    Returns:
        One dict per clip with its tier, audio path and word positions.
    """
    pattern = f"{partial_spoof_root}/*/*/LA/partial_spoof_metadata.json"
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No files matched: {pattern}")

    clips: List[dict] = []
    missing_audio = 0
    for path_str in paths:
        path = Path(path_str)
        system = path.parts[-4]
        partition = path.parts[-3]
        data = json.loads(path.read_text(encoding="utf-8"))

        for splice_key, sample in data.items():
            total_duration = sample["total_duration_s"]
            if total_duration <= CROP_SECONDS:
                continue  # Tiled, not cropped: already fully visible by construction.

            audio_path = Path(sample["spliced_audio_path"])
            if not audio_path.exists():
                missing_audio += 1
                continue

            positions = reconstruct_final_positions(sample.get("spoofed_words", []))
            if not positions:
                continue

            window_start, window_end = aligned_crop_window(positions, total_duration)
            whole_words, total_words = coverage_stats(positions, window_start, window_end)

            clips.append(
                {
                    "splice_key": splice_key,
                    "system": system,
                    "partition": partition,
                    "tier": sample.get("tier", "UNKNOWN"),
                    "audio_path": audio_path,
                    "total_duration_s": total_duration,
                    "positions": positions,
                    "window_start_s": window_start,
                    "window_end_s": window_end,
                    "whole_words": whole_words,
                    "total_words": total_words,
                }
            )

    logger.info(
        f"Collected {len(clips)} croppable spliced clips; "
        f"{missing_audio} skipped for missing spliced_audio_path."
    )
    return clips


def load_bonafide_scores(scores_path: Path) -> np.ndarray:
    """Read the bonafide scores out of an existing DF-Arena score file.

    Args:
        scores_path: Path to a score file in the project's
            ``audio_id attack_id key score`` convention.

    Returns:
        Bonafide scores as a one-dimensional array.
    """
    scores: List[float] = []
    with open(scores_path, "r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) != 4:
                continue
            if parts[2] == "bonafide":
                scores.append(float(parts[3]))
    return np.array(scores, dtype=np.float32)


def score_clips(
    clips: List[dict], model: DFArenaDetector, device: torch.device, batch_size: int
) -> np.ndarray:
    """Score every clip with its aligned crop window.

    Args:
        clips: Clips collected by ``collect_spoof_clips``.
        model: Loaded DF-Arena detector, in evaluation mode.
        device: Compute device.
        batch_size: Clips scored per forward pass.

    Returns:
        One countermeasure score per clip, in the same order as ``clips``.
    """
    model.eval()
    scores: List[float] = []

    with torch.no_grad():
        for start in range(0, len(clips), batch_size):
            batch_clips = clips[start : start + batch_size]
            waveforms = []
            for clip in batch_clips:
                waveforms.append(
                    load_aligned_crop(
                        clip["audio_path"], clip["window_start_s"], clip["window_end_s"]
                    )
                )

            batch = torch.from_numpy(np.stack(waveforms)).to(device)
            lengths = torch.full((batch.shape[0],), CROP_SAMPLES, device=device)
            logits = model(batch, lengths)
            scores.extend(model.score_from_logits(logits).cpu().numpy().tolist())

            if (start // batch_size) % 20 == 0:
                logger.debug(f"  scored {start + len(batch_clips):,}/{len(clips):,} clips")

    return np.array(scores, dtype=np.float32)


def report_coverage_by_tier(clips: List[dict]) -> None:
    """Print how many clips keep every spliced word fully inside the window.

    Only clips that needed cropping are reported here; clips at or under the
    crop length are never partial, since the model tiles them whole.

    Args:
        clips: Clips collected by ``collect_spoof_clips``.
    """
    tiers = sorted({clip["tier"] for clip in clips})
    for tier in tiers:
        tier_clips = [clip for clip in clips if clip["tier"] == tier]
        full = sum(1 for clip in tier_clips if clip["whole_words"] == clip["total_words"])
        partial = len(tier_clips) - full
        logger.info(
            f"{tier}: {full:,}/{len(tier_clips):,} clips keep every spliced word "
            f"fully inside the aligned window ({partial:,} have at least one word "
            "partially cut or dropped because the words span more than 4.0375s)."
        )


def report_eer_by_tier(
    clips: List[dict], spoof_scores: np.ndarray, bonafide_scores: np.ndarray
) -> None:
    """Compute and print aligned-crop EER pooled by tier.

    Args:
        clips: Clips collected by ``collect_spoof_clips``, in score order.
        spoof_scores: Aligned-crop scores, aligned with ``clips``.
        bonafide_scores: Bonafide score pool from the existing zero-shot run.
    """
    tiers = sorted({clip["tier"] for clip in clips})
    for tier in tiers:
        mask = np.array([clip["tier"] == tier for clip in clips])
        tier_scores = spoof_scores[mask]
        eer, _threshold = metrics.compute_eer(bonafide_scores, tier_scores)
        logger.info(f"{tier}: n={mask.sum():,}  aligned-crop EER={eer:.3f}%")

    eer, _threshold = metrics.compute_eer(bonafide_scores, spoof_scores)
    logger.info(f"OVERALL: n={len(clips):,}  aligned-crop EER={eer:.3f}%")


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Re-score partial-spoof clips with a splice-aligned crop window."
    )
    parser.add_argument(
        "--partial-spoof-root", type=str, default="data/partial_spoof_output",
        help="Root of the partial-spoof output tree.",
    )
    parser.add_argument(
        "--bonafide-scores-file", type=str, required=True,
        help="Existing zero-shot DF-Arena score file to pull the bonafide pool from.",
    )
    parser.add_argument(
        "--model-id", type=str, default="Speech-Arena-2025/DF_Arena_1B_V_1",
        help="Hugging Face repository identifier of the published model.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=8,
        help="Clips scored per forward pass.",
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Destination path for the aligned-crop score file.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the aligned-crop re-scoring experiment end to end."""
    args = _parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "No CUDA device visible. Set CUDA_VISIBLE_DEVICES to one free GPU and rerun."
        )
    device = torch.device("cuda:0")

    clips = collect_spoof_clips(args.partial_spoof_root)
    report_coverage_by_tier(clips)
    bonafide_scores = load_bonafide_scores(Path(args.bonafide_scores_file))
    logger.info(f"Loaded {bonafide_scores.size} bonafide scores from {args.bonafide_scores_file}")

    model = DFArenaDetector(model_id=args.model_id, freeze_backbone=True).to(device)
    spoof_scores = score_clips(clips, model, device, args.batch_size)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        for clip, score in zip(clips, spoof_scores):
            handle.write(f"{clip['splice_key']} {clip['system']}-{clip['tier']} spoof {score:.6f}\n")
    logger.info(f"Aligned-crop scores written: {output_path}")

    report_eer_by_tier(clips, spoof_scores, bonafide_scores)


if __name__ == "__main__":
    main()
