"""
A/B test the valley-snap splice fix on known-bad samples.

For each (attack, partition, splice_key) triple in PROBE_SAMPLES, this
script:

1. Loads the bonafide and cloned audio plus the alignment metadata for
   the original sample.
2. Reconstructs the selected_indices that produced the existing splice
   from splice_metadata.json.
3. Re-runs ``splice_words`` twice: once with ``valley_search_ms=0`` (legacy
   behaviour, identical to what is currently on disk) and once with
   ``valley_search_ms`` taken from settings (default 50 ms).
4. Writes both outputs to OUTPUT_DIR with ``_OLD`` and ``_NEW`` suffixes so
   the user can audition them side by side.
5. Prints a per-sample table showing how far each boundary was snapped,
   the bonafide RMS at the original cut vs. the snapped cut, and the
   ratio (snapped / original) -- values much less than 1 confirm the cut
   moved into a silent valley.

Run on ml-server03 inside any of the partial_spoof venvs (only needs
librosa, soundfile, numpy, loguru). Listen to the WAV pairs in
``data/ab_valley_snap`` and decide whether to enable VALLEY_SEARCH_MS for
the production sweep.
"""
import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import librosa
import numpy as np
import soundfile as sf
from loguru import logger

from app.pipeline.partial_spoof.settings import settings
from app.pipeline.partial_spoof.utils.splice_engine import splice_words


OUTPUT_DIR_BASE = REPO_ROOT / "data" / "ab_valley_snap"


PROBE_SAMPLES: List[Tuple[str, str, str]] = [
    ("omnivoice", "not_jittered", "arf_00295_01131884458_W1"),
    ("omnivoice", "not_jittered", "arf_00295_01755168214_W1"),
    ("omnivoice", "not_jittered", "arf_00295_01131884458_W2"),
    ("fishgram",  "not_jittered", "pef_00610_01884033555_W2"),
]


def _resolve_selected_indices(splice_entry: dict) -> List[int]:
    """Pull the spliced word indices out of a splice_metadata entry."""
    return sorted(int(w["word_index"]) for w in splice_entry["spoofed_words"])


def _slot_rms(audio: np.ndarray, start_s: float, end_s: float, sr: int) -> float:
    a = int(start_s * sr)
    b = int(end_s * sr)
    a = max(0, min(a, len(audio)))
    b = max(a, min(b, len(audio)))
    if b == a:
        return 0.0
    return float(np.sqrt(np.mean(audio[a:b] ** 2) + 1e-12))


def _resolve_cell_dir(attack: str, partition: str) -> Path:
    return REPO_ROOT / "data" / "partial_spoof_output" / attack / partition


def _run_one(attack: str, partition: str, splice_key: str, output_dir: Path, valley_search_ms: float) -> None:
    cell = _resolve_cell_dir(attack, partition)
    if not cell.exists():
        logger.warning(f"Cell missing: {cell}")
        return

    alignment_path = cell / "alignment_metadata.json"
    splice_meta_path = cell / "splice_metadata.json"
    if not alignment_path.exists() or not splice_meta_path.exists():
        logger.warning(f"Metadata missing for cell {cell}")
        return

    with open(alignment_path, "r", encoding="utf-8") as f:
        alignment = json.load(f)
    with open(splice_meta_path, "r", encoding="utf-8") as f:
        splice_md = json.load(f)

    if splice_key not in splice_md:
        logger.warning(f"splice_key not found in splice_metadata: {splice_key}")
        return

    sample_key = splice_key.rsplit("_", 1)[0]
    if sample_key not in alignment:
        logger.warning(f"sample_key not found in alignment: {sample_key}")
        return

    entry_align = alignment[sample_key]
    entry_splice = splice_md[splice_key]

    bonafide_path = Path(entry_align["bonafide_audio_path"])
    cloned_path = Path(entry_align["cloned_audio_path"])
    if not bonafide_path.exists() or not cloned_path.exists():
        logger.warning(
            f"Audio missing for {splice_key}: bonafide={bonafide_path.exists()} "
            f"cloned={cloned_path.exists()}"
        )
        return

    bonafide_audio, _ = librosa.load(
        str(bonafide_path), sr=settings.SAMPLE_RATE, mono=True
    )
    cloned_audio, _ = librosa.load(
        str(cloned_path), sr=settings.SAMPLE_RATE, mono=True
    )

    selected_indices = _resolve_selected_indices(entry_splice)
    # Match step_05's masking so the A/B run uses the exact same RNG
    # path as production (and never crashes on a negative hash).
    splice_seed = (settings.RANDOM_SEED + hash(splice_key)) & ((1 << 63) - 1)

    common_kwargs = dict(
        bonafide_audio=bonafide_audio,
        cloned_audio=cloned_audio,
        bonafide_words=entry_align["bonafide_words"],
        cloned_words=entry_align["cloned_words"],
        selected_indices=selected_indices,
        sample_rate=settings.SAMPLE_RATE,
        crossfade_min_ms=settings.CROSSFADE_MIN_MS,
        crossfade_max_ms=settings.CROSSFADE_MAX_MS,
        max_silence_steal_ms=settings.MAX_SILENCE_STEAL_MS,
        max_stretch_ratio=settings.MAX_STRETCH_RATIO,
        splice_seed=splice_seed,
        energy_refine_silence_rms=settings.ENERGY_REFINE_SILENCE_RMS,
    )

    spliced_old, details_old = splice_words(
        **common_kwargs, valley_search_ms=0.0, energy_refine_radius_s=0.0,
    )
    spliced_new, details_new = splice_words(
        **common_kwargs,
        valley_search_ms=valley_search_ms,
        energy_refine_radius_s=settings.ENERGY_REFINE_RADIUS_S,
    )

    out_subdir = output_dir / f"{attack}_{partition}_{splice_key}"
    out_subdir.mkdir(parents=True, exist_ok=True)

    sf.write(str(out_subdir / "OLD_no_snap.wav"), spliced_old, settings.SAMPLE_RATE)
    sf.write(str(out_subdir / "NEW_valley_snap.wav"), spliced_new, settings.SAMPLE_RATE)
    sf.write(str(out_subdir / "REF_bonafide.wav"), bonafide_audio, settings.SAMPLE_RATE)
    sf.write(str(out_subdir / "REF_cloned.wav"), cloned_audio, settings.SAMPLE_RATE)

    logger.info("=" * 78)
    logger.info(
        f"{attack:10s}/{partition:12s}  {splice_key}  "
        f"(valley_search={valley_search_ms} ms)"
    )
    logger.info(
        f"  Outputs at: {out_subdir.relative_to(REPO_ROOT)}/{{OLD_no_snap,NEW_valley_snap,REF_*}}.wav"
    )

    for i, idx in enumerate(selected_indices):
        old = details_old[i] if i < len(details_old) else None
        new = details_new[i] if i < len(details_new) else None
        if old is None or new is None:
            logger.info(f"  word_idx={idx}: missing splice detail (old={bool(old)}, new={bool(new)})")
            continue

        old_rms = _slot_rms(
            bonafide_audio, old["bonafide_start_s"], old["bonafide_end_s"], settings.SAMPLE_RATE
        )
        new_start = new.get("bonafide_start_s", old["bonafide_start_s"])
        new_end = new.get("bonafide_end_s", old["bonafide_end_s"])
        new_rms = _slot_rms(bonafide_audio, new_start, new_end, settings.SAMPLE_RATE)
        snap_start = new.get("valley_snap_start_ms", 0.0)
        snap_end = new.get("valley_snap_end_ms", 0.0)

        refine_start_ms = new.get("energy_refine_shift_start_ms", 0.0)
        refine_end_ms = new.get("energy_refine_shift_end_ms", 0.0)
        parakeet_start = new.get("parakeet_start_s", old["bonafide_start_s"])
        parakeet_end = new.get("parakeet_end_s", old["bonafide_end_s"])

        logger.info(
            f"  word='{new['word']}' (idx={idx}) "
            f"| Parakeet=[{parakeet_start:.3f}-{parakeet_end:.3f}] "
            f"OLD slot=[{old['bonafide_start_s']:.3f}-{old['bonafide_end_s']:.3f}] "
            f"RMS_inside={old_rms:.4f}"
        )
        logger.info(
            f"    NEW slot=[{new_start:.3f}-{new_end:.3f}] "
            f"RMS_inside={new_rms:.4f}  "
            f"refine_start={refine_start_ms:+.1f}ms refine_end={refine_end_ms:+.1f}ms  "
            f"snap_start={snap_start:+.1f}ms snap_end={snap_end:+.1f}ms  "
            f"method={new['splice_method']} cf_eff={new['effective_crossfade_ms']}ms"
        )
        if new_rms < old_rms * 0.6:
            logger.info(
                f"    => Snap moved cut into a clearly quieter region "
                f"(RMS dropped to {new_rms/max(old_rms,1e-9):.2%} of original)."
            )
        elif new_rms > old_rms * 1.4:
            logger.warning(
                f"    => Snap moved cut into a LOUDER region "
                f"(RMS rose to {new_rms/max(old_rms,1e-9):.2%}). Consider tightening "
                "VALLEY_SEARCH_MS or revisit Step 4 selection for this sample."
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "A/B test the valley-snap splice fix. Generates OLD (current "
            "production) and NEW (snap-enabled) spliced WAVs side by side."
        )
    )
    parser.add_argument(
        "--valley-search-ms",
        type=float,
        default=None,
        help=(
            "Override settings.VALLEY_SEARCH_MS for the NEW splice. "
            "Default reads from settings (currently "
            f"{settings.VALLEY_SEARCH_MS} ms). Try 80-120 if the default "
            "doesn't move the cut into a clear valley."
        ),
    )
    args = parser.parse_args()

    valley_search_ms = (
        args.valley_search_ms
        if args.valley_search_ms is not None
        else settings.VALLEY_SEARCH_MS
    )

    output_dir = OUTPUT_DIR_BASE / f"search_{int(valley_search_ms)}ms"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        f"Running A/B valley-snap test on {len(PROBE_SAMPLES)} samples. "
        f"valley_search_ms={valley_search_ms}. "
        f"Outputs under {output_dir.relative_to(REPO_ROOT)}/."
    )
    for attack, partition, splice_key in PROBE_SAMPLES:
        try:
            _run_one(attack, partition, splice_key, output_dir, valley_search_ms)
        except Exception as exc:
            logger.exception(f"Failed on {attack}/{partition}/{splice_key}: {exc}")

    logger.info("=" * 78)
    logger.info("Done. Audition each subdir:")
    logger.info(
        "  REF_bonafide.wav  - original utterance (untouched)"
    )
    logger.info(
        "  REF_cloned.wav    - full TTS clone of the same text"
    )
    logger.info(
        "  OLD_no_snap.wav   - splice with current production behaviour (ghost expected)"
    )
    logger.info(
        "  NEW_valley_snap.wav - splice with valley-snap enabled"
    )


if __name__ == "__main__":
    sys.exit(main())
