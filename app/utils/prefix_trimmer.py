"""
Spurious prefix detection and audio trimming for TTS artifact correction.

Identifies words inserted at the beginning of generated audio that are not
present in the original reference text, and trims the audio to remove them.

This targets a known Qwen3-TTS artifact where the model occasionally
hallucinates one or more words before the intended speech content. The
detection is performed via sequence alignment between the ASR word sequence
and the reference word sequence: leading insertions at hypothesis position 0
are identified as spurious prefix words.

The trim point is derived from the end timestamp of the last spurious word,
as provided by Parakeet TDT word-level timestamps.

This module also provides detection of non-verbal prefix artifacts (such as
reference voice bleed in OmniVoice) where audible energy precedes the first
ASR-transcribed word. Such artifacts are not visible in the transcription
itself, so they require an energy-based detector.

Requires: soundfile, numpy
"""
import difflib
import re
import unicodedata
from pathlib import Path
from typing import List, Tuple

import numpy as np
import soundfile as sf
from loguru import logger

from app.utils.word_timestamp import WordTimestamp


def _normalize_word(word: str) -> str:
    """Normalize a single word for sequence alignment.

    Applies unicode NFKD decomposition, ASCII transliteration, and
    punctuation stripping. Must match the normalization in wer_cer.py
    so that alignment operates in the same token space.

    Args:
        word: Input word string.

    Returns:
        Normalized lowercase ASCII word with punctuation removed.
        Returns empty string if the word contains no alphanumeric characters.
    """
    word = unicodedata.normalize("NFKD", word)
    word = word.encode("ascii", "ignore").decode("ascii")
    word = re.sub(r"[^\w]", "", word).lower()
    return word


def detect_prefix_trim_point(
    word_timestamps: List[WordTimestamp],
    reference_text: str,
) -> float:
    """Detect the trim start time for a spurious prefix in the ASR output.

    Performs sequence alignment between the hypothesis word sequence (from
    ASR) and the reference word sequence (original TTS input text). Leading
    insertions at hypothesis position 0 — words present in the ASR output
    but absent from the reference at that position — are classified as
    spurious prefix words.

    The trim point is the end timestamp of the last spurious word.

    Args:
        word_timestamps: Word-level timestamps from Parakeet TDT, ordered by
            position in the audio. Each entry has word, start, and end fields.
        reference_text: Original text string that was passed to the TTS system.

    Returns:
        Trim start time in seconds. Returns 0.0 if no spurious prefix is
        detected or if the input lists are empty.
    """
    if not word_timestamps or not reference_text.strip():
        return 0.0

    ref_words = [_normalize_word(w) for w in reference_text.split()]
    ref_words = [w for w in ref_words if w]

    hyp_words = [_normalize_word(wt.word) for wt in word_timestamps]
    hyp_words_filtered = [w for w in hyp_words if w]

    if not hyp_words_filtered or not ref_words:
        return 0.0

    matcher = difflib.SequenceMatcher(
        None, ref_words, hyp_words_filtered, autojunk=False
    )
    opcodes = matcher.get_opcodes()

    spurious_hyp_end_idx = 0
    for tag, _i1, _i2, j1, j2 in opcodes:
        if j1 != spurious_hyp_end_idx:
            break
        if tag == "insert":
            spurious_hyp_end_idx = j2
        elif tag in ("equal", "replace"):
            break

    if spurious_hyp_end_idx == 0:
        return 0.0

    spurious_words = [wt.word for wt in word_timestamps[:spurious_hyp_end_idx]]
    trim_time = word_timestamps[spurious_hyp_end_idx - 1].end

    logger.warning(
        f"Spurious prefix detected: {spurious_hyp_end_idx} word(s) "
        f"[{' '.join(spurious_words)}] — trimming audio at {trim_time:.3f}s"
    )
    return trim_time


def trim_audio_prefix(
    audio_path: Path,
    trim_start_seconds: float,
    output_path: Path,
) -> None:
    """Trim leading audio and save the result to output_path.

    Reads the audio file, discards the segment before trim_start_seconds,
    and writes the remainder to output_path. If trim_start_seconds is zero
    or negative, no trimming is performed and no file is written.

    Args:
        audio_path: Path to the source WAV audio file.
        trim_start_seconds: Start time in seconds to keep. Everything before
            this point is discarded.
        output_path: Destination path for the trimmed audio file.

    Raises:
        FileNotFoundError: If audio_path does not exist.
        ValueError: If trim_start_seconds is greater than or equal to the
            total audio duration, which would produce an empty file.
    """
    if trim_start_seconds <= 0.0:
        return

    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    audio, sr = sf.read(str(audio_path))
    total_duration = len(audio) / sr
    trim_sample = int(trim_start_seconds * sr)

    if trim_sample >= len(audio):
        raise ValueError(
            f"Trim point {trim_start_seconds:.3f}s >= audio duration "
            f"{total_duration:.3f}s for {audio_path.name}"
        )

    trimmed = audio[trim_sample:]
    sf.write(str(output_path), trimmed, sr)

    logger.info(
        f"Prefix trimmed: {audio_path.name} "
        f"{total_duration:.2f}s -> {len(trimmed) / sr:.2f}s "
        f"(removed {trim_start_seconds:.3f}s)"
    )


def detect_nonverbal_prefix_artifact(
    audio: np.ndarray,
    sample_rate: int,
    word_timestamps: List[WordTimestamp],
    silence_floor_db: float = -55.0,
) -> Tuple[bool, float]:
    """Detect a non-linguistic prefix artifact via pre-speech RMS energy.

    Targets artifacts that the alignment-based detector cannot see: reference
    voice bleed, breaths, clicks, or any audible non-word content before the
    first ASR-transcribed word. Empirically validated on OmniVoice diffusion
    output where reference voice fragments occasionally precede the prompt
    content. The fragment carries voice-level energy yet is sub-syllabic, so
    Parakeet drops it and a transcription-based detector misses it.

    Detection logic:
        1. Read the start time of the first transcribed word, T_first.
        2. Compute RMS energy in dBFS over the audio interval [0, T_first].
        3. If the RMS exceeds silence_floor_db, the pre-speech window contains
           audible non-linguistic content and the sample is flagged.

    Empirical reference points (OmniVoice validation 2026-05-06, 6 samples):
        Artifact samples : pre_RMS in [-25, -22] dB, T_first in [0.56, 0.64] s
        Clean samples    : pre_RMS = -120 dB (silence floor), T_first <= 0.08 s
    A floor of -55 dB sits 30 dB above the artifact band and 65 dB below the
    silence band, providing wide separation in both directions.

    Args:
        audio: 1D numpy array of audio samples in [-1.0, 1.0] range.
        sample_rate: Sample rate of the waveform in Hz.
        word_timestamps: Word-level timestamps from Parakeet TDT, ordered by
            position in the audio.
        silence_floor_db: dBFS threshold above which pre-speech energy counts
            as audible. Defaults to -55.0.

    Returns:
        Tuple of:
            - is_artifact: True if pre-speech RMS exceeds the floor.
            - pre_rms_db: RMS of the pre-speech window in dBFS, or -120.0
              if the window is empty (T_first == 0 or no timestamps).
    """
    if not word_timestamps:
        return False, -120.0

    t_first = float(word_timestamps[0].start)
    pre_end_idx = int(t_first * sample_rate)

    if pre_end_idx <= 0:
        return False, -120.0

    pre_samples = audio[:pre_end_idx]
    if pre_samples.size == 0:
        return False, -120.0

    rms = float(np.sqrt(np.mean(pre_samples.astype(np.float64) ** 2)))
    if rms < 1e-9:
        return False, -120.0

    pre_rms_db = 20.0 * float(np.log10(rms))
    is_artifact = pre_rms_db > silence_floor_db

    if is_artifact:
        logger.warning(
            f"Non-verbal prefix artifact detected: T_first={t_first:.3f}s, "
            f"pre_RMS={pre_rms_db:.1f}dB > floor {silence_floor_db:.1f}dB"
        )

    return is_artifact, pre_rms_db
