# Validation Results

**Status:** Draft — pending v2 pipeline validation
**Last updated:** 2026-05-06
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

## OmniVoice Standalone Validation

**Date:** May 6, 2026
**Speakers:** 3 (arf_00295, arf_00610, arf_01523)
**TTS:** OmniVoice (k2-fsa)
**Samples:** 6 (2 per speaker)
**GPU:** 1 (ml-server03)

| Metric | Value | Threshold | Status |
|--------|-------|-----------|--------|
| Pass rate | 6/6 (100%) | -- | PASSED |
| Avg WER | 3.94% | <= 15% | PASSED |
| Avg CER | 1.81% | <= 10% | PASSED |
| Avg NISQA MOS | 4.53 | >= 2.5 (info) | PASSED |
| Avg Speaker SIM | 0.680 | >= 0.70 (info) | INFORMATIONAL MISS |
| Prefix trims | 0 | -- | -- |

**Findings:**
- Pipeline works end-to-end (Steps 1-5 all completed without errors).
- Content quality is the highest of any attack in the suite (NISQA 4.53).
- Speaker similarity 0.680 is below the 0.70 informational floor. OmniVoice is the **weakest cloner** of the 6-attack suite by ECAPA-TDNN cosine similarity. From an anti-spoofing dataset perspective this is fine -- diversity of attack quality is desired -- but it should be flagged in the paper as a weakness of OmniVoice for high-fidelity cloning. For comparison, Qwen avg SIM is 0.720, FishGram 0.602, OpenVoice 0.394.
- All 6 samples landed in the train split because the 3 validation speakers (arf_00295, arf_00610, arf_01523) all live in train per the HABLA canonical partition. Not a bug; production mode will hit all splits.

**Cleared for production run** (`VALIDATION_MODE=False`, `MATCH_BONAFIDE_COUNT=True`).

## Full-Synthesis Production Validation

See [Production Runs](production-runs.md) for per-pipeline metrics.

## Related Pages
- [Partial Spoof Approach](../methodology/partial-spoof-approach.md) — methodology
- [Quality Metrics](../methodology/quality-metrics.md) — thresholds
- [Production Runs](production-runs.md) — full-synthesis results
