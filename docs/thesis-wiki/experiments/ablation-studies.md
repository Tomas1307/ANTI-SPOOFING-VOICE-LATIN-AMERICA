# Ablation Studies

**Status:** Draft — experiments pending
**Last updated:** 2026-04-25
**Source:** Listening tests April 25, 2026

---

## Crossfade Technique Comparison (April 25, 2026)

**Setup:** Sample "Dame unicamente el dato mas relevante" (speaker arf_00295). Word "unicamente" replaced. All 7 crossfade techniques applied at 60ms overlap. Same bonafide + clone pair.

**Finding:** All 7 techniques sounded identical. The crossfade technique does NOT determine audible quality — the word boundary alignment and duration mismatch are the dominant factors.

**Implication:** The 7 techniques provide forensic diversity (different spectral signatures for detector training) but do not meaningfully affect perceptual quality. This is consistent with Negroni et al.'s finding that even at 4096-sample OLA windows, AUC remains 88.99%.

## Valley Score Threshold Tuning (PENDING)

**Planned experiment:** Run partial spoof with VALLEY_SCORE_THRESHOLD values of 0.3, 0.5, 0.65, 0.8, 1.0. Measure:
- Number of eligible words per threshold
- Tier completion rate (W1/W2/W3)
- Audible quality of splices (subjective listening)
- WER/NISQA/SIM of outputs

**Hypothesis:** 0.65 is a reasonable starting point. Lower thresholds reject more words (higher quality but fewer samples). Higher thresholds accept more (more samples but worse quality).

## Clone Similarity Gate Calibration (PENDING)

**Planned experiment:** Run with MIN_CLONE_SIMILARITY values of 0.4, 0.5, 0.6, 0.7. Measure rejection rate per TTS system.

**Known data points:**
- FishGram avg SIM: 0.602 → at 0.6 threshold, ~50% rejected
- Qwen avg SIM: 0.720 → at 0.6 threshold, most pass
- OpenVoice avg SIM: 0.394 → at 0.6 threshold, almost all rejected

## Duration Stretch Ratio Impact (PENDING)

**Planned experiment:** Compare audible quality at stretch ratios 0.75x, 0.85x, 1.0x, 1.15x, 1.25x. At what ratio does time-stretching become audible?

**Known reference:** TSM Subjective Quality Dataset (Roberts 2020) shows subjective MOS falls steeply outside ratio 0.85-1.20.

## Presentation Progress Tracking

**Commit 38514e4 (April 22):** 20 slides
**Current (April 25):** 34 slides (+14)

New slides added this session:
- 06b, 06c: Chatterbox/OuteTTS production progress
- 13a: Visual splicing pipeline diagram
- 13b: Splice boundary challenge
- 13c-13i: 7 technique slides with SVG curves
- 13j: Technique summary + literature comparison table
- 13k: Splice quality problem discovery
- 13l: Valley-score solution

## Related Pages
- [Partial Spoof Approach](../methodology/partial-spoof-approach.md) — methodology
- [Splicing Techniques](../state-of-art/splicing-techniques.md) — literature techniques
- [Decision Log](../decisions/decision-log.md) — why these experiments were chosen
