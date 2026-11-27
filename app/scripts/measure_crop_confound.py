"""
Measure whether DF-Arena's fixed centre crop can exclude every spliced word.

WHY
---
DF-Arena's published feature extractor truncates every clip to a centre crop
of exactly 64,600 samples (4.0375 s at 16 kHz), or tiles it if shorter. A
partial-spoof clip is only detectable as spoof if at least one spliced word
falls inside that window. If the crop misses every spliced word, the model is
shown genuine speech and asked to call it spoof -- unwinnable by construction,
regardless of detector quality.

The near-chance partial-spoof EER observed in the zero-shot DF-Arena baseline
(18-47% across tiers, on both dev and eval) could be genuine attack difficulty,
a scoring-window artefact, or both. This script measures the artefact
component directly from the partial-spoof pipeline's own splice metadata,
which records bonafide_start_s/bonafide_end_s per spliced word and
total_duration_s per clip.

METHOD
------
For each spliced word, its position within the FINAL spliced file is not
simply bonafide_start_s: the pipeline uses natural-duration splicing with no
time-stretch, so words after the first one shift by the cumulative duration
delta of every earlier splice in the same clip (recorded in
cloned_natural_duration_s vs the original bonafide word duration). This script
reconstructs that cumulative offset from the ordered spoofed_words list.

A centre crop of 64,600 samples is then computed for each clip's
total_duration_s (or the whole clip, if shorter -- the feature extractor tiles
those instead of cropping, so every spliced word survives). A clip is a "crop
miss" when none of its spliced-word intervals overlap the crop window at all.

Only clips longer than the crop window can miss; clips at or under it always
survive since they are tiled rather than truncated.

USAGE
-----
    cd ~/ANTI-SPOOFING-VOICE-LATIN-AMERICA
    source envs/dfarena_env/bin/activate
    python -m app.scripts.measure_crop_confound
    deactivate

No GPU, no torch. Reads only splice metadata JSON.
"""
import argparse
import glob
import json
from pathlib import Path
from typing import Dict, List, Tuple

CROP_SAMPLES = 64600
SAMPLE_RATE = 16000
CROP_SECONDS = CROP_SAMPLES / SAMPLE_RATE


def reconstruct_final_positions(
    spoofed_words: List[dict],
) -> List[Tuple[float, float]]:
    """Reconstruct each spliced word's position within the final spliced file.

    The pipeline splices with natural duration and no time-stretch, so each
    word after the first shifts by the cumulative duration delta of every
    earlier splice in the same clip: a cloned word shorter than the bonafide
    word it replaced pulls everything after it earlier, and vice versa.

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


def centre_crop_window(total_duration_s: float) -> Tuple[float, float]:
    """Compute the centre-crop window DF-Arena's feature extractor applies.

    Args:
        total_duration_s: Duration of the final spliced clip.

    Returns:
        A (start_s, end_s) tuple. When the clip is at or under the crop
        length, the window covers the whole clip, matching the tiling
        behaviour that keeps every sample.
    """
    if total_duration_s <= CROP_SECONDS:
        return 0.0, total_duration_s
    start = (total_duration_s - CROP_SECONDS) / 2.0
    return start, start + CROP_SECONDS


def overlaps(a_start: float, a_end: float, b_start: float, b_end: float) -> bool:
    """Return whether two intervals overlap.

    Args:
        a_start: Start of the first interval.
        a_end: End of the first interval.
        b_start: Start of the second interval.
        b_end: End of the second interval.

    Returns:
        True when the intervals share any positive-length overlap.
    """
    return a_start < b_end and b_start < a_end


def analyze_file(path: Path) -> Dict[str, dict]:
    """Analyze one system/partition's partial_spoof_metadata.json.

    Args:
        path: Path to a LA/partial_spoof_metadata.json file.

    Returns:
        Mapping of tier to its aggregate statistics for this file.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    by_tier: Dict[str, dict] = {}

    for sample in data.values():
        tier = sample.get("tier", "UNKNOWN")
        stats = by_tier.setdefault(
            tier, {"total": 0, "crop_miss": 0, "clip_tiled": 0, "clip_cropped": 0}
        )
        stats["total"] += 1

        total_duration = sample["total_duration_s"]
        window_start, window_end = centre_crop_window(total_duration)
        if total_duration <= CROP_SECONDS:
            stats["clip_tiled"] += 1
        else:
            stats["clip_cropped"] += 1

        positions = reconstruct_final_positions(sample.get("spoofed_words", []))
        any_visible = any(
            overlaps(start, end, window_start, window_end)
            for start, end in positions
        )
        if not any_visible:
            stats["crop_miss"] += 1

    return by_tier


def main() -> None:
    """Scan every system's partial-spoof metadata and report the crop miss rate."""
    parser = argparse.ArgumentParser(
        description="Measure the DF-Arena centre-crop confound on partial spoof."
    )
    parser.add_argument(
        "--root", type=str, default="data/partial_spoof_output",
        help="Root of the partial-spoof output tree.",
    )
    args = parser.parse_args()

    pattern = f"{args.root}/*/*/LA/partial_spoof_metadata.json"
    paths = sorted(glob.glob(pattern))
    if not paths:
        print(f"No files matched: {pattern}")
        return

    grand_total: Dict[str, dict] = {}

    for path_str in paths:
        path = Path(path_str)
        system = path.parts[-4]
        partition = path.parts[-3]
        by_tier = analyze_file(path)

        for tier, stats in by_tier.items():
            key = f"{system}/{partition}/{tier}"
            print(
                f"{key:<40} total={stats['total']:>6,}  "
                f"crop_miss={stats['crop_miss']:>6,} "
                f"({100 * stats['crop_miss'] / stats['total']:5.1f}%)  "
                f"tiled={stats['clip_tiled']:>6,}  cropped={stats['clip_cropped']:>6,}"
            )

            agg = grand_total.setdefault(
                tier, {"total": 0, "crop_miss": 0, "clip_tiled": 0, "clip_cropped": 0}
            )
            for field in stats:
                agg[field] += stats[field]

    print("\n" + "=" * 90)
    print("AGGREGATE BY TIER (all systems, both partitions)")
    print("=" * 90)
    overall_total = 0
    overall_miss = 0
    for tier in sorted(grand_total):
        stats = grand_total[tier]
        overall_total += stats["total"]
        overall_miss += stats["crop_miss"]
        print(
            f"{tier:<10} total={stats['total']:>7,}  "
            f"crop_miss={stats['crop_miss']:>7,} "
            f"({100 * stats['crop_miss'] / stats['total']:5.1f}%)  "
            f"tiled={stats['clip_tiled']:>7,}  cropped={stats['clip_cropped']:>7,}"
        )
    print("-" * 90)
    print(
        f"{'OVERALL':<10} total={overall_total:>7,}  "
        f"crop_miss={overall_miss:>7,} ({100 * overall_miss / overall_total:5.1f}%)"
    )
    print(
        "\ncrop_miss = fraction of clips where NO spliced word overlaps the "
        f"{CROP_SECONDS:.4f}s centre-crop window DF-Arena's feature extractor "
        "applies. These clips cannot be correctly classified as spoof by "
        "construction, independent of detector quality."
    )


if __name__ == "__main__":
    main()
