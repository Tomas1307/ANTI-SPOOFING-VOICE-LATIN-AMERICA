# Validation Results

**Status:** Draft — pending v2 pipeline validation
**Last updated:** 2026-04-25
**Source:** ml-server03 validation runs

---

## Partial Spoof Validation (v1 — pre-valley-score)

**Date:** April 22, 2026
**Speakers:** 3 (arf_00295, arf_00610, arf_01523)
**TTS:** Qwen3-TTS
**Samples:** 7

| Metric | Value |
|--------|-------|
| Pass rate | 7/7 (100%) |
| WER | 3.9% |
| NISQA MOS | 4.72 |
| Speaker SIM | 0.789 |

**Note:** This was before valley-score selection. Listening tests revealed audible artifacts at fluid speech boundaries despite all metrics passing. This led to the v2 pipeline rewrite (valley score + duration preserving + clone gate).

## Partial Spoof Validation (v2 — PENDING)

**Planned speakers:** 5 (arf_00295, arf_00610, arf_01523, arm_00412, arm_00780)
**Minimum audios per speaker:** 10
**Changes to validate:**
1. Valley-score word selection (score <= 0.65 threshold)
2. Duration-preserving splice (output length = bonafide length)
3. Clone similarity gate (ECAPA SIM >= 0.60)

**Expected checks:**
- All spliced WAVs have identical duration to bonafide source
- `word_selection_metadata.json` contains `valley_score` per word
- `clone_similarity_filter.json` exists
- Listen to 10+ samples across speakers for natural rhythm
- Compare WER/NISQA/SIM against v1 baseline

## Full-Synthesis Production Validation

See [Production Runs](production-runs.md) for per-pipeline metrics.

## Related Pages
- [Partial Spoof Approach](../methodology/partial-spoof-approach.md) — methodology
- [Quality Metrics](../methodology/quality-metrics.md) — thresholds
- [Production Runs](production-runs.md) — full-synthesis results
