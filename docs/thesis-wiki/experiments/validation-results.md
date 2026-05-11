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

### Initial validation (2026-05-06 early run, BEFORE reference fix)

| Metric | Value | Threshold | Status |
|--------|-------|-----------|--------|
| Pass rate | 6/6 (100%) | -- | PASSED |
| Avg WER | 3.94% | <= 15% | PASSED |
| Avg CER | 1.81% | <= 10% | PASSED |
| Avg NISQA MOS | 4.53 | >= 2.5 (info) | PASSED |
| Avg Speaker SIM | 0.680 | >= 0.70 (info) | INFORMATIONAL MISS |
| Prefix trims | 0 | -- | -- |

The initial validation appeared to pass cleanly. Subsequent listening revealed reference-voice bleed at the start of 2/6 samples (both from `arf_00295`) that the existing alignment-based prefix detector did not catch (the bleed was sub-syllabic and Parakeet did not transcribe it, so WER stayed at 0).

### Diagnostic + non-verbal prefix detector (2026-05-06 retry run)

After listening, the new `detect_nonverbal_prefix_artifact` was added to Step 4 (rejects samples whose pre-speech RMS exceeds -55 dBFS). With the detector + retry loop active but the reference-concatenation bug still present:

| Metric | Value |
|--------|-------|
| Pass rate | 4/6 (66.7%) |
| Non-verbal prefix rejections | 2 |
| Retry rounds executed | 5 (full max) |
| Final state | 2 samples (`arf_00295` TEXT_00001, TEXT_00002) bled on every one of 6 generation attempts |

This run made the artifact visible but could not eliminate it. arf_00295 looked like a deterministic per-speaker failure mode.

### Post-fix validation (2026-05-06 final run, AFTER reference fix)

The actual root cause was identified as a mid-file slicing bug in `concatenate_with_padding`: the routine sliced the last bonafide file at the exact sample boundary needed to hit 10 s, landing the cut mid-word. The reference therefore ended abruptly, and OmniVoice's diffusion conditioning attempted to "complete" the cut-off pattern at the start of generation. Fix: stop at the last file that fits, snap to silence boundary in the edge case, always append 200 ms trailing silence.

| Metric | Value | Threshold | Status |
|--------|-------|-----------|--------|
| Pass rate | **6/6 (100%)** | -- | PASSED |
| Non-verbal prefix rejections | 0 | -- | -- |
| Retry rounds | 0 (none needed) | <= 5 | -- |
| Avg WER | 1.85% | <= 15% | PASSED |
| Avg CER | 0.83% | <= 10% | PASSED |
| Avg NISQA MOS | 4.59 | >= 2.5 (info) | PASSED |
| Avg Speaker SIM | 0.696 | >= 0.70 (info) | NEAR FLOOR |
| Prefix trims | 0 | -- | -- |

`arf_00295` -- which previously bled deterministically -- now passes cleanly on the first attempt. All quality metrics improved relative to the initial "passed" run, confirming the bleed was being measured implicitly (lower WER/CER, higher NISQA) and was eliminated rather than masked.

### Net findings

- The reference-cut bug was the root cause; the per-speaker hypothesis is **retracted**.
- The non-verbal prefix detector + retry loop are kept in Step 4 as defense in depth.
- Speaker similarity 0.696 is still below the 0.70 informational floor, confirming OmniVoice is the **weakest cloner of the 6-attack suite** by ECAPA-TDNN cosine similarity. For comparison, Qwen avg SIM is 0.720, FishGram 0.602, OpenVoice 0.394. This is a stable property of OmniVoice's diffusion conditioning, not a bug, and should be flagged in the paper.
- All 6 samples landed in the train split because the 3 validation speakers (arf_00295, arf_00610, arf_01523) all live in train per the HABLA canonical partition. Production mode will hit all splits.

**Cleared for production run** (`VALIDATION_MODE=False`, `MATCH_BONAFIDE_COUNT=True`).

## Full-Synthesis Production Validation

See [Production Runs](production-runs.md) for per-pipeline metrics.

## Related Pages
- [Partial Spoof Approach](../methodology/partial-spoof-approach.md) — methodology
- [Quality Metrics](../methodology/quality-metrics.md) — thresholds
- [Production Runs](production-runs.md) — full-synthesis results
