"""
Word Error Rate (WER) and Character Error Rate (CER) computation utilities.

Both metrics are computed after text normalization to ensure that minor
formatting differences (punctuation, accents, casing) do not inflate the
error rate when comparing TTS input text against ASR transcriptions.

Normalization pipeline:
    1. Unicode NFKD decomposition
    2. ASCII transliteration (strips diacritics: e.g. 'o' replaces both
       U+00F3 LATIN SMALL LETTER O WITH ACUTE)
    3. Lowercase
    4. Punctuation removal
    5. Whitespace normalization

This ensures 'decision' and 'decisión' are treated identically for
accent-invariant comparison. Even with Parakeet TDT 0.6b-v3 (which
supports Spanish natively), ASCII normalization is applied to both
reference and hypothesis so that diacritic handling differences between
TTS text input and ASR output do not inflate error rates.

Requires: jiwer >= 3.0
    pip install jiwer
"""
import re
import unicodedata

from jiwer import cer as jiwer_cer
from jiwer import wer as jiwer_wer


def normalize_text(text: str) -> str:
    """Normalize text for WER/CER comparison.

    Applies unicode decomposition, ASCII transliteration, lowercase
    conversion, punctuation removal, and whitespace normalization.
    Both reference and hypothesis must pass through this function
    before metric computation.

    Args:
        text: Input text string (may contain Spanish diacritics or punctuation).

    Returns:
        Normalized ASCII lowercase string with punctuation removed.
    """
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def compute_wer(reference: str, hypothesis: str) -> float:
    """Compute Word Error Rate between reference and ASR hypothesis.

    WER = (Substitutions + Deletions + Insertions) / Total reference words.
    A WER of 0.0 means the ASR transcription matches the reference exactly
    after normalization. The target for TTS validation is 0.0.

    Args:
        reference: Ground truth text that was passed to the TTS system.
        hypothesis: ASR transcription of the generated audio.

    Returns:
        WER as a float in [0.0, inf). Values above 1.0 indicate more
        insertion errors than reference words (possible hallucination).
        Returns 0.0 if both strings are empty after normalization.
    """
    ref_norm = normalize_text(reference)
    hyp_norm = normalize_text(hypothesis)

    if not ref_norm and not hyp_norm:
        return 0.0
    if not ref_norm:
        return 1.0

    return float(jiwer_wer(ref_norm, hyp_norm))


def compute_cer(reference: str, hypothesis: str) -> float:
    """Compute Character Error Rate between reference and ASR hypothesis.

    CER = (Substitutions + Deletions + Insertions at character level)
          / Total reference characters.
    Finer-grained than WER. Useful for detecting partial word errors that
    WER would count as one full word error regardless of severity.

    Args:
        reference: Ground truth text that was passed to the TTS system.
        hypothesis: ASR transcription of the generated audio.

    Returns:
        CER as a float in [0.0, inf). Target for TTS validation is 0.0.
        Returns 0.0 if both strings are empty after normalization.
    """
    ref_norm = normalize_text(reference)
    hyp_norm = normalize_text(hypothesis)

    if not ref_norm and not hyp_norm:
        return 0.0
    if not ref_norm:
        return 1.0

    return float(jiwer_cer(ref_norm, hyp_norm))
