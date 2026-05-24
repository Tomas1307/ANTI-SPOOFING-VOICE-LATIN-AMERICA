"""
Core word-level audio splicing engine (natural-duration).

Replaces selected word segments in bonafide audio with corresponding
segments from cloned audio. The cloned word is inserted at its native
duration with no time-stretch -- the splice loop concatenates
bonafide_prefix + cloned + bonafide_suffix and lets the result audio's
total length adjust by the duration difference. Pitch is therefore
preserved: spliced clones sound exactly like the standalone TTS
output, without the chipmunk / thickening artefacts that a
duration-preserving linear-interpolation stretch would introduce.

Each splice boundary independently draws a random technique and
overlap duration from a seeded generator, producing a heterogeneous
dataset of splice artefacts. Crossfade happens at the SEAMS where
bonafide meets cloned (not inside a fixed slot); with valley snap and
energy refinement the seams land in inter-word silence so the
crossfade removes the bonafide signal cleanly.

W2/W3 splices accumulate a per-call offset: when splice 1 makes the
audio longer or shorter, splice 2's original-bonafide coordinates are
translated to the current result via that offset before extraction
and replacement.
"""
import librosa
import numpy as np
from loguru import logger
from typing import Dict, List, Tuple

from app.pipeline.partial_spoof.utils.crossfade import (
    _compute_fade_curves,
    draw_splice_method,
    find_nearest_valley,
    find_nearest_zero_crossing,
    normalize_energy,
)
from app.pipeline.partial_spoof.utils.energy_refiner import (
    refine_word_boundary_by_energy,
)


def _silent_run_backward(
    audio: np.ndarray,
    position: int,
    sample_rate: int,
    max_ms: float,
    silence_threshold: float = 0.015,
    window_ms: float = 5.0,
) -> int:
    """Return how many samples before ``position`` are continuous silence.

    Scans short RMS windows walking backward from ``position`` until
    one exceeds ``silence_threshold`` or the maximum search distance
    is reached. Used by the splice engine to size the crossfade so
    the extension never bleeds into the previous cloned word ("leak").

    Args:
        audio: 1-D float waveform.
        position: Sample index from which to walk backward.
        sample_rate: Audio sample rate in Hz.
        max_ms: Maximum distance to consider in milliseconds.
        silence_threshold: RMS at or below which a window counts as silence.
        window_ms: RMS analysis window length in milliseconds.

    Returns:
        Number of consecutive silent samples immediately preceding
        ``position``. Capped at ``max_ms`` and at ``position`` itself.
    """
    max_samples = max(0, int(max_ms * sample_rate / 1000))
    win = max(1, int(window_ms * sample_rate / 1000))
    silent = 0
    while silent + win <= max_samples and (position - silent - win) >= 0:
        start = position - silent - win
        end = position - silent
        segment = audio[start:end]
        if len(segment) == 0:
            break
        rms = float(np.sqrt(np.mean(segment.astype(np.float32) ** 2) + 1e-12))
        if rms > silence_threshold:
            break
        silent += win
    return silent


def _silent_run_forward(
    audio: np.ndarray,
    position: int,
    sample_rate: int,
    max_ms: float,
    silence_threshold: float = 0.015,
    window_ms: float = 5.0,
) -> int:
    """Return how many samples after ``position`` are continuous silence.

    Mirror of :func:`_silent_run_backward` walking forward.
    """
    max_samples = max(0, int(max_ms * sample_rate / 1000))
    win = max(1, int(window_ms * sample_rate / 1000))
    silent = 0
    while silent + win <= max_samples and (position + silent + win) <= len(audio):
        start = position + silent
        end = start + win
        segment = audio[start:end]
        if len(segment) == 0:
            break
        rms = float(np.sqrt(np.mean(segment.astype(np.float32) ** 2) + 1e-12))
        if rms > silence_threshold:
            break
        silent += win
    return silent
from app.pipeline.partial_spoof.utils.splice_method import SpliceMethod


def _normalize_word(word: str) -> str:
    """Normalize a word for text-based matching.

    Strips punctuation, lowercases, and removes diacritical marks
    (accents) to handle differences like 'hotel.' vs 'hotel',
    'Hay' vs 'hay', or 'Como' vs 'Cómo'.

    Args:
        word: Raw word string from ASR timestamps.

    Returns:
        Normalized lowercase word without punctuation or accents.
    """
    import unicodedata

    stripped = word.lower().strip(".,;:!?()[]{}\"'¿¡")
    nfkd = unicodedata.normalize("NFKD", stripped)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def _build_cloned_word_map(cloned_words: List[Dict]) -> Dict[str, List[Tuple[int, Dict]]]:
    """Build a lookup map from normalized word text to (index, entry) pairs.

    Groups by normalized word text so that duplicate words (e.g. two
    occurrences of 'de') can each be consumed once in order.
    Stores the original index for gap calculation.

    Args:
        cloned_words: Word-level timestamps from cloned audio.

    Returns:
        Dict mapping normalized word text to a list of (index, timestamp_dict) tuples.
    """
    word_map: Dict[str, List[Tuple[int, Dict]]] = {}
    for i, cw in enumerate(cloned_words):
        key = _normalize_word(cw["word"])
        if key not in word_map:
            word_map[key] = []
        word_map[key].append((i, cw))
    return word_map


def _time_stretch(segment: np.ndarray, target_length: int) -> np.ndarray:
    """Time-stretch audio segment to exactly target_length samples.

    Uses two strategies depending on how aggressive the stretch is:

    * **Linear interpolation** for tiny stretches (< 5% deviation
      from 1.0x). These produce imperceptible pitch shifts and the
      simpler method is ~50x faster than the phase vocoder.
    * **librosa.effects.time_stretch** (phase vocoder) for anything
      bigger. The phase vocoder preserves pitch: a cloned 'casa'
      compressed from 240 ms to 200 ms keeps the same fundamental
      frequency and formant positions. The previous linear-interp
      implementation behaved like changing a tape speed -- compressing
      raised the pitch, expanding lowered it -- making the spliced
      word sound "chipmunk" or "thick / cartoony" relative to its
      surrounding bonafide. That artefact was perceptually severe
      from ~10% stretch upward (e.g. ratio 0.80 = +20% pitch).

    The phase vocoder may return a slightly off-by-one length, so the
    output is padded with zeros or truncated to land exactly on
    ``target_length``.

    Args:
        segment: Audio segment (1-D float32).
        target_length: Desired number of output samples.

    Returns:
        Time-stretched segment of exactly ``target_length`` samples.
    """
    if target_length <= 0 or len(segment) == 0:
        return segment
    if len(segment) == target_length:
        return segment.copy()

    src_length = len(segment)
    rate = src_length / target_length

    if abs(rate - 1.0) < 0.05:
        indices = np.linspace(0, src_length - 1, target_length)
        return np.interp(indices, np.arange(src_length), segment).astype(np.float32)

    stretched = librosa.effects.time_stretch(
        segment.astype(np.float32), rate=rate
    )

    if len(stretched) >= target_length:
        return stretched[:target_length].astype(np.float32)

    padded = np.zeros(target_length, dtype=np.float32)
    padded[: len(stretched)] = stretched
    return padded


def splice_words(
    bonafide_audio: np.ndarray,
    cloned_audio: np.ndarray,
    bonafide_words: List[Dict],
    cloned_words: List[Dict],
    selected_indices: List[int],
    sample_rate: int,
    crossfade_min_ms: float,
    crossfade_max_ms: float,
    max_silence_steal_ms: float,
    max_stretch_ratio: float,
    splice_seed: int = 42,
    valley_search_ms: float = 0.0,
    energy_refine_radius_s: float = 0.0,
    energy_refine_silence_rms: float = 0.015,
) -> Tuple[np.ndarray, List[Dict]]:
    """Replace selected words in bonafide audio with cloned word segments.

    Natural-duration splice: the cloned word is inserted at its native
    duration with no time-stretch. The total audio length of the result
    therefore differs from the bonafide whenever cloned and bonafide
    word durations diverge -- this is intentional. The previous
    "duration-preserving" design used linear-interpolation time-stretch
    to force the cloned into the bonafide slot length, which raised or
    lowered the cloned word's pitch ("chipmunk" / "thick" voice
    artefacts perceptible from ~10% stretch upward). Since the
    downstream partial-spoof pipeline never compares spliced against
    bonafide and labels spoof regions by their position in the spliced
    audio, preserving the cloned's natural pitch is strictly preferable
    to preserving total duration.

    The crossfade happens at the SEAMS where bonafide meets cloned --
    the last cf samples of bonafide blend into the first cf samples of
    cloned at the start seam, and symmetrically at the end seam. With
    valley snap and energy refinement, the seams land in inter-word
    silence so the crossfade mixes cloned against silence rather than
    against speech, removing the bonafide ghost.

    For each selected word, independently draws:
    - A splice technique (SpliceMethod) sampled from SPLICE_METHOD_WEIGHTS
    - An overlap duration in [crossfade_min_ms, crossfade_max_ms] ms

    Per-word RNG is seeded from (splice_seed, word_index) for reproducibility.

    Word dicts must have keys: 'word' (str), 'start' (float s), 'end' (float s).

    Multi-word splices (W2, W3): each splice in the loop may change the
    result's total duration. Subsequent splices map their original
    bonafide positions onto current-result positions via a cumulative
    offset, so all splices land where Step 4 selected them despite
    intermediate length changes.

    Args:
        bonafide_audio: Full bonafide waveform (1-D float32 array).
        cloned_audio: Full cloned waveform (1-D float32 array).
        bonafide_words: Word-level timestamps for bonafide audio.
        cloned_words: Word-level timestamps for cloned audio.
        selected_indices: Zero-based indices of words to replace.
        sample_rate: Audio sample rate in Hz.
        crossfade_min_ms: Minimum crossfade overlap drawn per splice (ms).
        crossfade_max_ms: Maximum crossfade overlap drawn per splice (ms).
        max_silence_steal_ms: Unused (kept for API compatibility).
        max_stretch_ratio: Retained for API compatibility but no longer
            enforced inside this function -- Step 4 already filters
            words whose cloned/bonafide duration ratio is extreme, so
            re-checking here just rejects samples Step 4 already
            accepted. Without time-stretch the cloned is inserted at
            its natural duration regardless of ratio.
        splice_seed: Base seed for per-word RNG. Seeded as (splice_seed, idx).
        valley_search_ms: Half-width (ms) of the search window used to
            snap each bonafide slot boundary to the nearest energy
            valley. Without this, Parakeet's word boundaries often clip
            inside the acoustic word, so parts of the bonafide word
            survive outside the splice slot and bleed through the
            crossfade -- the listener hears both the cloned word and
            the original word. Setting this to 30-60 ms reliably
            relocates each cut onto a silent inter-word gap; the
            crossfade then mixes cloned speech against bonafide
            silence and only the cloned signal is audible. Set to 0.0
            to disable snapping and preserve the legacy behaviour.
        energy_refine_radius_s: Search radius (seconds) for refining
            each Parakeet word boundary by acoustic energy BEFORE the
            valley snap runs. Parakeet TDT routinely drifts 100-300 ms
            on phrase-merged words ("la casa"); when the drift exceeds
            the valley-snap window, the splice slot lands in the wrong
            region entirely and the bonafide word survives outside it.
            Energy refinement locates the actual word by detecting the
            speech segment closest to Parakeet's centre within +/-
            radius seconds. Set to 0.0 to disable.
        energy_refine_silence_rms: RMS threshold below which audio is
            considered silence in the energy refinement. Tune per
            dataset; ~0.015 works for HABLA at 16 kHz mono.

    Returns:
        Tuple of (spliced_audio, splice_details) where splice_details is a
        list of dicts with per-word splice metadata including splice_method.
    """
    cloned_map = _build_cloned_word_map(cloned_words)

    result = bonafide_audio.copy().astype(np.float32)
    splice_details = []
    # For W2/W3: each splice can change the result's total duration
    # (cloned word duration != bonafide slot duration). Subsequent
    # splices need to map their original-bonafide coordinates onto the
    # current state of ``result``. This running offset tracks the net
    # number of samples added (positive) or removed (negative) by all
    # preceding splices in this call.
    cumulative_offset_samples = 0

    for idx in sorted(selected_indices):
        if idx >= len(bonafide_words):
            logger.warning(
                f"Word index {idx} out of range for bonafide "
                f"({len(bonafide_words)} words). Skipping."
            )
            continue

        bw = bonafide_words[idx]
        target_key = _normalize_word(bw["word"])

        if target_key not in cloned_map or len(cloned_map[target_key]) == 0:
            logger.warning(
                f"Word '{bw['word']}' (index {idx}) not found in cloned audio. Skipping."
            )
            continue

        cw_idx, cw = cloned_map[target_key].pop(0)

        parakeet_start_s = float(bw["start"])
        parakeet_end_s = float(bw["end"])
        parakeet_cloned_start_s = float(cw["start"])
        parakeet_cloned_end_s = float(cw["end"])

        # Energy refinement runs on BOTH sides. Parakeet drifts on
        # natural fast speech (bonafide) and occasionally on TTS
        # output (cloned). Refining only one side leaves a duration
        # mismatch -- the refined bonafide slot can be 150 ms while
        # the cloned source is still 240 ms per Parakeet, blowing
        # past the stretch envelope. Applying the same refinement to
        # the cloned word keeps source and destination acoustically
        # comparable so the resulting splice respects the stretch
        # ratio constraint.
        if energy_refine_radius_s > 0.0:
            refined_start_s, refined_end_s = refine_word_boundary_by_energy(
                bonafide_audio,
                parakeet_start_s,
                parakeet_end_s,
                sample_rate,
                search_radius_s=energy_refine_radius_s,
                silence_threshold_rms=energy_refine_silence_rms,
            )
            refined_cloned_start_s, refined_cloned_end_s = refine_word_boundary_by_energy(
                cloned_audio,
                parakeet_cloned_start_s,
                parakeet_cloned_end_s,
                sample_rate,
                search_radius_s=energy_refine_radius_s,
                silence_threshold_rms=energy_refine_silence_rms,
            )
        else:
            refined_start_s = parakeet_start_s
            refined_end_s = parakeet_end_s
            refined_cloned_start_s = parakeet_cloned_start_s
            refined_cloned_end_s = parakeet_cloned_end_s

        b_start_raw = _clamp(int(refined_start_s * sample_rate), 0, len(bonafide_audio))
        b_end_raw = _clamp(int(refined_end_s * sample_rate), b_start_raw, len(bonafide_audio))
        c_start = _clamp(int(refined_cloned_start_s * sample_rate), 0, len(cloned_audio))
        c_end = _clamp(int(refined_cloned_end_s * sample_rate), c_start, len(cloned_audio))

        cl_natural_len = c_end - c_start

        # Valley snap (asymmetric outward) on ORIGINAL bonafide coords.
        # The snap extends the bonafide slot into the silence flanking
        # the word so the crossfade falls in silence and the bonafide
        # word is fully removed. With duration-preserving stretch
        # removed, the snap no longer needs the stretch_ratio gate --
        # the cloned is inserted at its natural length regardless of
        # slot size, so any slot is acceptable.
        b_start_snapped = b_start_raw
        b_end_snapped = b_end_raw
        if valley_search_ms > 0.0 and cl_natural_len > 0:
            cand_start = find_nearest_valley(
                bonafide_audio,
                b_start_raw,
                sample_rate,
                search_ms=valley_search_ms,
                direction="earlier",
            )
            cand_end = find_nearest_valley(
                bonafide_audio,
                b_end_raw,
                sample_rate,
                search_ms=valley_search_ms,
                direction="later",
            )
            b_start_snapped = _clamp(cand_start, 0, len(bonafide_audio))
            b_end_snapped = _clamp(cand_end, b_start_snapped, len(bonafide_audio))

        slot_len_orig = b_end_snapped - b_start_snapped

        if slot_len_orig == 0 or cl_natural_len == 0:
            logger.warning(f"Zero-length segment for word index {idx}, skipping.")
            continue

        # Map ORIGINAL-bonafide positions onto CURRENT result coordinates.
        # Earlier splices in this loop may have grown or shrunk result,
        # so the position where this word's slot lives in result is
        # original_bonafide_position + cumulative_offset.
        b_start = _clamp(b_start_snapped + cumulative_offset_samples, 0, len(result))
        b_end = _clamp(b_end_snapped + cumulative_offset_samples, b_start, len(result))

        # NumPy's SeedSequence rejects negative integers. Python's
        # hash() can return any signed 64-bit int, so mask the sign
        # bit. XOR-ing idx perturbs the per-word stream so word index
        # variations produce visibly different draws.
        seed_safe = (splice_seed & ((1 << 63) - 1)) ^ (idx & 0xFFFF)
        word_rng = np.random.default_rng([seed_safe, idx])
        method = draw_splice_method(word_rng)
        overlap_ms = float(word_rng.uniform(crossfade_min_ms, crossfade_max_ms))
        cf_target = int(overlap_ms * sample_rate / 1000)

        # KEY DESIGN POINT: extend the cloned source by cf samples on
        # each side so the crossfade falls on the TTS silence padding
        # AROUND the word, not on the word's first/last phonemes.
        # Without this, the fade-in attenuates the word's onset (e.g.
        # the 'l' of 'lugar') and the fade-out attenuates the offset
        # (e.g. the 'r' of 'lugar') -- the listener hears 'Luga-'
        # because the trailing fricative dies under the fade.
        #
        # BUT: the extension must NOT cross into adjacent cloned words.
        # Some TTS clones pack consecutive words tightly with little or
        # no silence between them; blindly extending by cf samples then
        # captures the previous or next word's onset, which the
        # crossfade mixes into the seam ("leak" of the neighbouring
        # word into the spliced region).
        #
        # The extension is therefore bounded by the actual silent run
        # on each side of the cloned word -- we walk outward sample by
        # sample and stop when we hit speech. When the silent run is
        # zero on one side (word starts/ends right at the next word),
        # the crossfade collapses to a hard cut on that side, which is
        # acceptable: with valley snap placing the bonafide seam in
        # silence, a clean cut on the cloned side still avoids clicks.
        cloned_silence_left = _silent_run_backward(
            cloned_audio,
            c_start,
            sample_rate,
            max_ms=float(crossfade_max_ms),
            silence_threshold=energy_refine_silence_rms,
        )
        cloned_silence_right = _silent_run_forward(
            cloned_audio,
            c_end,
            sample_rate,
            max_ms=float(crossfade_max_ms),
            silence_threshold=energy_refine_silence_rms,
        )
        max_left_ext = min(c_start, cf_target, cloned_silence_left)
        max_right_ext = min(
            len(cloned_audio) - c_end, cf_target, cloned_silence_right
        )
        effective_cf = max(
            0,
            min(
                cf_target,
                b_start,
                len(result) - b_end,
                max_left_ext,
                max_right_ext,
            ),
        )

        if method is SpliceMethod.CUT_PASTE or effective_cf == 0:
            cloned_segment = cloned_audio[c_start:c_end].astype(np.float32).copy()
            bonafide_slot = result[b_start:b_end].copy()
            cloned_segment = normalize_energy(cloned_segment, bonafide_slot)
            result = np.concatenate([
                result[:b_start],
                cloned_segment,
                result[b_end:],
            ]).astype(np.float32)
            if method is SpliceMethod.CUT_PASTE:
                effective_cf = 0
        else:
            # Extract cloned WITH silence padding on each side.
            c_start_ext = c_start - effective_cf
            c_end_ext = c_end + effective_cf
            cloned_ext = cloned_audio[c_start_ext:c_end_ext].astype(np.float32).copy()

            # Energy-normalize using ONLY the word portion (not the
            # silence pads, which would skew the RMS towards zero and
            # blow up the scale).
            cloned_word_only = cloned_ext[effective_cf:effective_cf + cl_natural_len]
            bonafide_slot = result[b_start:b_end].copy()
            scaled_word = normalize_energy(cloned_word_only, bonafide_slot)
            # Apply the same gain to the padding regions for continuity
            # so the silence remains silent (multiplied by the same scale)
            # and the seams between word and pads stay smooth.
            if np.sqrt(np.mean(cloned_word_only ** 2) + 1e-12) > 1e-8:
                scale = float(
                    np.sqrt(np.mean(bonafide_slot ** 2) + 1e-12)
                    / np.sqrt(np.mean(cloned_word_only ** 2) + 1e-12)
                )
                cloned_ext = cloned_ext * scale

            t = np.linspace(0.0, 1.0, effective_cf, dtype=np.float32)
            fade_in, fade_out = _compute_fade_curves(t, method)

            # Start seam: bonafide tail (silence after valley snap)
            # blends with cloned PADDING (TTS silence before the word).
            bonafide_tail = result[b_start - effective_cf:b_start].astype(np.float32)
            start_cf_region = (
                bonafide_tail * fade_out
                + cloned_ext[:effective_cf] * fade_in
            )

            # End seam: cloned PADDING (TTS silence after the word)
            # blends with bonafide head (silence after valley snap).
            bonafide_head = result[b_end:b_end + effective_cf].astype(np.float32)
            end_cf_region = (
                cloned_ext[-effective_cf:] * fade_out
                + bonafide_head * fade_in
            )

            # The actual word content sits in the middle of cloned_ext
            # at full amplitude -- no fade touches it.
            word_middle = cloned_ext[effective_cf:effective_cf + cl_natural_len]

            result = np.concatenate([
                result[:b_start - effective_cf],
                start_cf_region.astype(np.float32),
                word_middle,
                end_cf_region.astype(np.float32),
                result[b_end + effective_cf:],
            ]).astype(np.float32)

        # The cloned region in the FINAL result starts at b_start and
        # spans cl_natural_len samples. (Crossfade reshapes the seams
        # but the cloned content still occupies that interval.)
        spliced_start_samples = b_start
        spliced_end_samples = b_start + cl_natural_len

        # Net change in total result length for this splice. Subsequent
        # splices in the same call shift by this much.
        size_diff = cl_natural_len - slot_len_orig
        cumulative_offset_samples += size_diff

        duration_ratio = cl_natural_len / max(1, slot_len_orig)

        splice_details.append({
            "word_index": idx,
            "word": bw["word"],
            # Position of the spoofed region in the SPLICED audio
            # (what Step 6 / Step 7 use to find spoof boundaries).
            "bonafide_start_s": spliced_start_samples / sample_rate,
            "bonafide_end_s": spliced_end_samples / sample_rate,
            # Position of the bonafide region that was removed, in
            # the original bonafide coordinates -- for traceability
            # and debugging only; downstream steps should ignore.
            "bonafide_orig_start_s": b_start_snapped / sample_rate,
            "bonafide_orig_end_s": b_end_snapped / sample_rate,
            "bonafide_orig_start_raw_s": b_start_raw / sample_rate,
            "bonafide_orig_end_raw_s": b_end_raw / sample_rate,
            "parakeet_start_s": parakeet_start_s,
            "parakeet_end_s": parakeet_end_s,
            "energy_refine_shift_start_ms": round(
                (refined_start_s - parakeet_start_s) * 1000, 2
            ),
            "energy_refine_shift_end_ms": round(
                (refined_end_s - parakeet_end_s) * 1000, 2
            ),
            "valley_snap_start_ms": round(
                (b_start_snapped - b_start_raw) * 1000 / sample_rate, 2
            ),
            "valley_snap_end_ms": round(
                (b_end_snapped - b_end_raw) * 1000 / sample_rate, 2
            ),
            "cloned_start_s": c_start / sample_rate,
            "cloned_end_s": c_end / sample_rate,
            "cloned_parakeet_start_s": parakeet_cloned_start_s,
            "cloned_parakeet_end_s": parakeet_cloned_end_s,
            "cloned_refine_shift_start_ms": round(
                (refined_cloned_start_s - parakeet_cloned_start_s) * 1000, 2
            ),
            "cloned_refine_shift_end_ms": round(
                (refined_cloned_end_s - parakeet_cloned_end_s) * 1000, 2
            ),
            "cloned_natural_duration_s": cl_natural_len / sample_rate,
            "slot_duration_s": slot_len_orig / sample_rate,
            "duration_diff_ms": round(size_diff * 1000 / sample_rate, 2),
            "duration_ratio": round(duration_ratio, 4),
            "stretch_ratio": round(duration_ratio, 4),
            "crossfade_ms": round(overlap_ms, 2),
            "effective_crossfade_ms": round(effective_cf * 1000 / sample_rate, 2),
            "splice_method": method.value,
            # FALSE under the new architecture: the spliced audio's
            # total duration shifts by ``size_diff`` per splice. The
            # field is kept for schema compatibility with downstream
            # consumers; readers that care about duration should
            # check ``duration_diff_ms`` instead.
            "slot_preserved": False,
        })

        logger.debug(
            f"Spliced word '{bw['word']}' (idx={idx}) | method={method.value} "
            f"| cloned={cl_natural_len}smp ({round(cl_natural_len*1000/sample_rate,1)}ms) "
            f"| slot_orig={slot_len_orig}smp ({round(slot_len_orig*1000/sample_rate,1)}ms) "
            f"| ratio={duration_ratio:.2f}x | diff={size_diff:+d}smp "
            f"| cf={effective_cf}smp | cum_offset={cumulative_offset_samples:+d}smp"
        )

    splice_details.sort(key=lambda d: d["word_index"])
    return result, splice_details


def _clamp(value: int, low: int, high: int) -> int:
    """Clamp an integer value to [low, high] range.

    Args:
        value: Value to clamp.
        low: Minimum allowed value.
        high: Maximum allowed value.

    Returns:
        Clamped value.
    """
    return max(low, min(value, high))
