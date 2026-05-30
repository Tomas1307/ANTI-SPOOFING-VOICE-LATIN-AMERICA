# Validation Results

**Status:** Active — splice rewrite validated 2026-05-24
**Last updated:** 2026-05-24
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

## Partial Spoof Validation (v2 — manifest mode, splice rewrite, 2026-05-24)

**Date:** 2026-05-24 (parallel orchestrator run, 12 cells, 23:47-00:56 UTC)
**Mode:** manifest-driven, parallel launcher with `--max-concurrent 2` on GPU 1
**Validation slice:** small subset (~5 speakers x 6 attacks x 2 partitions = 12 cells)
**Configuration:** `ENABLE_BOUNDARY_JITTER=True/False` per cell, `ENABLE_STEP_6_REJECTION=False` (keep-bad-stuff), clone gate `MIN_CLONE_SIMILARITY=0.60` active.

### Per-attack yield (total spliced samples in samples.csv)

| Attack     | Total samples | not_jittered | jittered |
|------------|---------------|--------------|----------|
| omnivoice  | 169           | --           | --       |
| chatterbox | 62            | --           | --       |
| fishgram   | 30            | --           | --       |
| qwen       | 25            | --           | --       |
| outetts    | 9             | --           | --       |
| openvoice  | 0             | 0            | 0        |
| **Total**  | **295**       |              |          |

OpenVoice consistently yielded 0 across both partitions because its avg ECAPA SIM (0.394 in standalone) sits well below the `MIN_CLONE_SIMILARITY=0.60` clone gate; this is expected behaviour per the 2026-04-25 decision to keep the gate at 0.60. The corpus continues to ship without OpenVoice samples until either (a) the gate is relaxed for that pipeline or (b) OpenVoice itself is improved.

### Structural assertions (post-run grep)

| Assertion | Value | Status |
|---|---|---|
| `Alignment missing` warnings across all 12 logs | 0 | PASSED (Step 3 accumulate fix working) |
| `expected non-negative integer` crashes across all 12 logs | 0 | PASSED (seed-mask fix working) |
| Total runtime (parallel launcher) | 1h 8min | PASSED (vs. 2-3h pre-rewrite from retry overhead) |
| Failed cells | 0/12 | PASSED |

### Yield comparison vs. previous validation run (pre-splice-rewrite, 2026-05-23)

| Attack     | Pre-rewrite | Post-rewrite | Factor |
|------------|-------------|--------------|--------|
| omnivoice  | 81          | 169          | 2.1x   |
| qwen       | 6           | 25           | 4.2x   |
| fishgram   | 19          | 30           | 1.6x   |
| chatterbox | 41          | 62           | 1.5x   |
| outetts    | 4           | 9            | 2.3x   |
| openvoice  | 0           | 0            | --     |
| **Total**  | **151**     | **295**      | **1.95x** |

The Qwen 4.2x improvement is largest because the negative-seed bug (`hash(splice_key)` returning a signed int64 then crashing `np.random.default_rng`) disproportionately affected Qwen's W1/W3 splices that were silently rejected. Fixing the seed mask unblocks those splices; combined with the other architectural changes (energy refinement, valley snap, natural duration, silent-run extension gate), the entire splice pipeline now completes per sample instead of crashing mid-loop.

### Audit auditiva (listening test)

Master Tomas downloaded spot-check WAVs (4 problematic samples + sample-of-sample from each attack/tier) and confirmed:
- Spliced word no longer audible twice (cloned + bonafide residual).
- Cloned word no longer "cartoony/grueso" pitch-shifted.
- Word body intact end-to-end ("lugar" sounds like "lugar", not "Luga-").
- No leak of adjacent cloned words at the seams.

Pass auditiva: **YES** ("Si me parecio bien").

### Cleared for production

- Pipeline architectural changes from this session locked.
- Production sweep on full 1,567-speaker corpus is the next step.
- Expected production yield: ~scale by manifest dispatch (manifest has ~35,927 bonafide files target * tier multiplicities).

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
