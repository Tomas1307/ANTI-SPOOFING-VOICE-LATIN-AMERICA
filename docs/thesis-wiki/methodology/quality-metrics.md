# Quality Metrics

**Status:** Active
**Last updated:** 2026-04-25
**Source:** app/pipeline/*/steps/step_*_validate*.py, settings.py

---

## Validation Pipeline

Every generated sample passes through automated quality validation before inclusion in the dataset. Samples that fail are rejected with logged reasons.

## Metrics

### WER (Word Error Rate)
- **Tool:** NVIDIA Parakeet TDT 0.6b-v3 (transcribe generated audio, compare with original text)
- **Rejection threshold:** WER > 30%
- **Purpose:** Ensures the TTS output contains the correct words. High WER means the TTS dropped, added, or garbled words.

### CER (Character Error Rate)
- **Tool:** Same Parakeet transcription
- **Rejection threshold:** CER > 20%
- **Purpose:** Finer-grained than WER. Catches partial word errors (e.g. "presidentes" → "presidente").

### NISQA MOS (Mean Opinion Score)
- **Tool:** NISQA v2 (non-intrusive speech quality assessment)
- **Range:** 1.0 - 5.0 (higher = better)
- **No rejection threshold** — informational metric
- **Purpose:** Measures perceived audio quality (noise, distortion, naturalness). Production averages: FishGram 4.57, OpenVoice 4.41, Qwen 4.37.

### Speaker Similarity (ECAPA-TDNN)
- **Tool:** `app/utils/ecapa_similarity.py` — SpeechBrain `spkrec-ecapa-voxceleb`
- **Embedding:** 192-dim, L2-normalized
- **Metric:** Cosine similarity in [-1, 1]
- **Clone gate threshold:** SIM >= 0.60 (between Steps 2 and 3 in partial spoof)
- **No rejection threshold in Step 6** — informational metric
- **Purpose:** Measures how much the generated voice sounds like the original speaker.
- **Production averages:** Qwen 0.720, FishGram 0.602, OpenVoice 0.394

### Valley Score (Partial Spoof Only)
- **Tool:** `app/pipeline/partial_spoof/utils/valley_scorer.py`
- **Formula:** `score = min_rms / avg_rms` in +-100ms window of 5ms frames
- **Range:** 0.0 (perfect silence at boundary) to 1.0 (no energy dip)
- **Selection threshold:** score <= 0.65
- **Purpose:** Ensures words selected for replacement have clean energy valleys at both boundaries in the cloned audio. Prevents bad-sounding splices at fluid speech boundaries.

### Stretch Ratio (Partial Spoof Only)
- **Formula:** `cloned_word_duration / bonafide_word_duration`
- **Acceptable range:** [0.75, 1.25] (configurable via MAX_STRETCH_RATIO)
- **Purpose:** Limits time-stretching distortion. Words requiring stretch outside this range are ineligible for replacement.

## Threshold Summary

| Metric | Threshold | Action |
|--------|-----------|--------|
| WER | > 30% | Reject sample |
| CER | > 20% | Reject sample |
| NISQA MOS | None | Informational |
| Speaker SIM (clone gate) | < 0.60 | Reject clone before alignment |
| Speaker SIM (Step 6) | None | Informational |
| Valley score | > 0.65 | Word ineligible for selection |
| Stretch ratio | outside [0.75, 1.25] | Word ineligible |
| Word duration | < 200ms | Word ineligible |

## Related Pages
- [Partial Spoof Approach](partial-spoof-approach.md) — valley score design and edge cases
- [Production Runs](../experiments/production-runs.md) — actual metric distributions
- [Detection Methods](../state-of-art/detection-methods.md) — how detectors use these features
