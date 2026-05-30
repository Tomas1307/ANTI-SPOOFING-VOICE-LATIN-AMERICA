# Partial Spoof Approach

**Status:** Active
**Last updated:** 2026-05-24
**Source:** Session discoveries from listening tests + implementation

---

## Overview

The partial spoof pipeline replaces 1-3 individual words in bonafide HABLA utterances with voice-cloned versions from TTS systems. Unlike full-synthesis attacks (where the entire utterance is fake), partial spoof makes detection significantly harder because 90%+ of the audio is genuine.

## Key Design Decisions

### Full-sentence cloning (not word-by-word)
We clone the **full sentence** with TTS, then extract individual words via forced alignment. Words generated in isolation have flat, citation-form prosody that is trivially detectable. In-context words carry natural prosody — making the splice harder to detect.

### Valley-score word selection (not random)

**Discovery (April 25):** Listening tests showed all 7 crossfade techniques sounded identical. The audible problem was blind word selection — randomly picking words at fluid speech boundaries (where the TTS generates continuous speech with no energy dip between words).

**Solution:** Score each word boundary by energy valley depth:

```
score = min_rms / avg_rms    (in +-100ms window, 5ms frames)
combined = max(left_score, right_score)
```

- Score 0.0 = perfect silence at boundary (ideal cut point)
- Score 1.0 = no energy dip (impossible to cut cleanly)
- Threshold: 0.65 (configurable). Words above are ineligible.

**Selection algorithm:** Greedy best-first with non-adjacency constraint. Not random.

### Natural-duration splice (changed 2026-05-24)

**Earlier design (April 25 - May 23):** Time-stretched the cloned word to fit the exact bonafide slot duration, overwriting in place so total audio length never changed. Stretch ratio constrained to `[1/MAX_STRETCH_RATIO, MAX_STRETCH_RATIO]` (`MAX_STRETCH_RATIO=1.25`).

**Problem found in May 24 audit:** Linear-interpolation time-stretch (`np.interp`) is mathematically equivalent to changing playback speed: stretching by 1.20x raises pitch ~20% ("chipmunk"); compressing by 0.80x lowers pitch ~20% ("thick/cartoony"). Master Tomas's listening tests on the validation run flagged this as audibly bad — spliced words obviously different in pitch from their surroundings, especially at stretch ratios near the 0.80x and 1.25x extremes.

**Current design (May 24):** Remove time-stretch entirely. Insert the cloned word at its natural duration. Total audio length varies per sample. For W2/W3 splices, a `cumulative_offset_samples` counter maps original-bonafide positions onto current-result positions so later splices in the same call land at the right place despite earlier splices having grown or shrunk the audio. The cumulative_offset is reset per utterance.

**Why we can drop duration preservation:** The corpus is consumed as `partial_spoof_audio + per-word boundary labels` (no pairing with the original bonafide for frame-level comparison). Word-level spoof labels work regardless of total duration. Position fields in `splice_metadata.json` (`bonafide_start_s`, `bonafide_end_s`) now refer to positions in the SPLICED audio (where the spoofed region lives in the final WAV), not in the original bonafide. Step 6 and Step 7 already use them this way for boundary metrics and label tables.

**Crossfade at the seams (not inside the slot):** Without a fixed slot length, the crossfade can no longer happen "inside" the slot. Instead, the cloned source is extended by `cf` samples on each side (capturing TTS silence padding around the word). At each seam:

```
result = bonafide[:b_start - cf]
       + crossfade(bonafide_tail, cloned_padding_left, cf)
       + cloned_word_body
       + crossfade(cloned_padding_right, bonafide_head, cf)
       + bonafide[b_end + cf:]
```

The crossfade falls in silence on both sides (bonafide_tail = post-valley-snap silence, cloned_padding = TTS silence around the word), so the bonafide ghost disappears and the cloned word body is never attenuated. Cloned-side extension is bounded by `_silent_run_backward/_forward(cloned_audio, ...)` to prevent the extension from capturing a neighbouring cloned word, which would leak into the seam.

`MAX_STRETCH_RATIO=1.25` is kept as a Step 4 word-selection filter (only select words where cloned/bonafide durations are within the envelope) but no longer enforced in Step 5; the natural-duration splice accepts whatever duration the TTS produced. `_time_stretch` remains in the module as dead code for API compatibility.

### Clone similarity gate

ECAPA-TDNN cosine similarity between bonafide and clone must be >= 0.60. Bad clones are rejected before alignment/splicing (between Steps 2 and 3), saving compute.

## Edge Cases

| Case | Handling |
|------|----------|
| All words have bad valley scores | Reject sample for tier |
| Not enough good words for W2/W3 | Skip higher tiers |
| Word < 200ms | Ineligible (too short to be meaningful) |
| First/last word | Only internal boundary scored (external = silence, artificially good) |
| Stretch ratio outside [0.75, 1.25] | Word ineligible at Step 4 selection time (no longer enforced in Step 5 splice) |
| All clones for a speaker fail similarity | Speaker produces no partial spoofs |
| Parakeet word boundary drifts >100ms from acoustic position | Energy refiner (Step 5) corrects it when Parakeet centre falls in silence; otherwise trusts Parakeet |
| Cloned word adjacent to next/previous cloned word with no silence gap | Crossfade extension on that side collapses to 0 -> CUT_PASTE seam (no leak of neighbour into splice) |
| Negative `hash(splice_key)` | Masked with `& ((1<<63)-1)` at both call site (Step 5) and inside `splice_engine` (defensive) so `np.random.default_rng` accepts the seed |

## Splice Engine Architecture (post-2026-05-24)

The Step 5 splice engine assembles each spliced WAV by concatenation, not in-place overwrite. The per-word loop:

```
1. Resolve bonafide and cloned word boundaries (Parakeet timestamps)
2. ENERGY REFINEMENT (utils/energy_refiner.py)
     - Conservative gate: only refine if Parakeet centre is in silence
     - Detect speech segments in +- ENERGY_REFINE_RADIUS_S window
     - Merge adjacent segments separated by < merge_gap_ms (catches phoneme stop closures)
     - Filter segments by duration (>= min_segment_dur_ratio * parakeet_duration)
     - Pick the longest qualifying segment, tiebreak by closeness to Parakeet centre
     - Applied to BOTH bonafide and cloned word boundaries independently
3. VALLEY SNAP (utils/crossfade.find_nearest_valley)
     - Asymmetric outward: direction="earlier" for start, "later" for end
     - Extends bonafide slot into adjacent silence so seams fall in low energy
     - Never shrinks the slot inward
4. SLOT EXTRACTION
     - bonafide slot = [b_start_snapped, b_end_snapped] in original-bonafide coords
     - cloned source = [c_start, c_end] in cloned coords
     - Both ranges are AFTER refinement + (bonafide-only) valley snap
5. CUMULATIVE OFFSET MAPPING (W2/W3 only)
     - Map original-bonafide positions to current-result positions
     - b_start_in_result = b_start_snapped + cumulative_offset_samples
6. ENERGY NORMALIZATION
     - Scale cloned amplitude so RMS matches bonafide slot RMS
     - Prevents loudness discontinuity at seams
7. CROSSFADE ASSEMBLY
     - seed_safe = (splice_seed & ((1<<63)-1)) ^ (idx & 0xFFFF)  # mask sign bit
     - method = draw_splice_method(seeded_rng)
     - overlap_ms = uniform(CROSSFADE_MIN_MS, CROSSFADE_MAX_MS)
     - cf_target = overlap_ms * sample_rate / 1000
     - Bound cf by:
        - cloned_silence_left = _silent_run_backward(cloned, c_start, max_ms=cf)
        - cloned_silence_right = _silent_run_forward(cloned, c_end, max_ms=cf)
        - bonafide context available before/after the cut
     - Extract cloned_ext = cloned[c_start - cf : c_end + cf]  (word + silence padding)
     - At start seam: bonafide_tail * fade_out + cloned_ext[:cf] * fade_in
     - At end seam: cloned_ext[-cf:] * fade_out + bonafide_head * fade_in
     - Middle: cloned_ext[cf:cl_len+cf] (word body, untouched by fades)
     - Concat: result[:b_start-cf] + start_seam + middle + end_seam + result[b_end+cf:]
8. UPDATE CUMULATIVE OFFSET
     - size_diff = cl_natural_len - slot_len_orig
     - cumulative_offset_samples += size_diff
9. RECORD splice_details
     - bonafide_start_s, bonafide_end_s in SPLICED audio coords (positional labels for Step 7)
     - bonafide_orig_start_s, bonafide_orig_end_s in original-bonafide coords (traceability)
     - parakeet_*, energy_refine_shift_*, valley_snap_*, cloned_refine_shift_* for diagnostics
```

Critical invariants of the design:
- The full cloned word body always survives untouched (no fade across the word interior).
- Seams always fall in silence on both sides (after refinement + snap + silent-run gate).
- Per-utterance audio duration varies by `sum(cl_natural_len_i - slot_len_orig_i)` across the splices in that utterance.
- For W2/W3, all splices in the same utterance share a single `cumulative_offset_samples` counter; the counter resets per utterance.

## Quality Metrics

From validation run (3 speakers, 7 samples, pre-v2 pipeline):
- WER: 3.9%
- NISQA MOS: 4.72
- Speaker similarity: 0.789
- Pass rate: 7/7 (100%)

From 2026-05-24 splice-rewrite validation (manifest mode, 12 cells, 295 spliced samples total across 6 attacks):
- See `experiments/validation-results.md` for per-cell metrics.
- omnivoice 169, qwen 25, fishgram 30, chatterbox 62, outetts 9, openvoice 0.
- Pipeline runtime: 1h 8min (vs. 2-3h pre-rewrite from intermediate-crash overhead).

## 7 Crossfade Techniques

Still applied per splice for forensic diversity. The technique affects the fade curve at slot boundaries (first/last 15-80ms). Drawn per-splice from weighted distribution:

| Technique | Proportion | Energy behavior |
|-----------|-----------|----------------|
| Direct cut-paste | 10% | No blend |
| OLA Hanning | 20% | Equal-gain (S-curve) |
| Linear | 15% | Equal-gain (amplitude dip) |
| Cosine | 20% | Equal-power |
| Square-root | 15% | Equal-power |
| Logarithmic | 10% | Neither (energy dip) |
| Inv. parabola | 10% | Equal-gain |

## Boundary Jitter Variant (Step 5b)

**Status:** Implemented 2026-05-01. Pilot pending on Qwen.

### Two parallel datasets per attack

The boundary-jitter feature produces a **second, separate dataset** per attack system, alongside the existing main partial spoof. Both folders are independent — no shared audio, no shared sentences, no shared audio_id range. Two folders coexist for the same TTS so a detector can be trained/evaluated on either or both.

| Variant | Output folder | Sentences | Processing | System ID | Audio ID range |
|---|---|---|---|---|---|
| **Main** (existing) | `data/<attack>_partial_spoof/` | First half of bonafide pool (per-speaker shuffle, seed 42) | Splice only — replace 1/2/3 words with cloned audio | `<ATTACK>_PSW{1,2,3}` | 12M / 13M / 14M |
| **Jitter** (new) | `data/<attack>_partial_spoof_jitter/` | Second half of bonafide pool (disjoint from main) | Splice + per-boundary structural jitter (truncate / overlap / bleed) | `<ATTACK>_PSW{1,2,3}J` | 16M / 17M / 18M |

The two are produced by running the same `PartialSpoofPipeline` twice with different config:

```python
# Main dataset (no jitter, sentence half A)
PartialSpoofPipelineConfig(
    attack_system="qwen",
    enable_boundary_jitter_override=False,
    bonafide_file_partition_override="main",
)

# Jitter dataset (jitter on, sentence half B - disjoint)
PartialSpoofPipelineConfig(
    attack_system="qwen",
    enable_boundary_jitter_override=True,
    bonafide_file_partition_override="jitter",
)
```

The `pipeline_facade` automatically appends `_jitter` to the output directory name when `ENABLE_BOUNDARY_JITTER=True`, and `step_07_format_output` automatically uses the W1J/W2J/W3J audio range and `_PSW{N}J` system_id suffix. **No manual override of paths or IDs is needed**; the two flags are sufficient.

### Why two datasets and not one combined

Each dataset isolates a different research question:
- **Main** measures detector performance on the canonical partial spoof distribution (clean bonafide segments + spliced cloned word).
- **Jitter** measures detector performance under the homogenized-boundary adversarial threat model.
- Training/evaluating on each separately reveals whether the detector learned the boundary-anomaly shortcut (high EER on jitter only) or genuine TTS-artifact features (similar EER on both).
- A combined training pool of both folders also lets the detector learn the shortcut-resistant feature set in one shot.

### Motivation

The current splice produces detectable artifacts only at the 2 boundaries surrounding each cloned word; the other N-3 internal boundaries (bonafide-bonafide) remain clean. A detector can exploit "find the noisy boundary" as a shortcut. Negroni et al. (2024) achieves 6.16% EER on PartialSpoof with **zero training**, purely by analyzing dynamic-range discontinuity at the splice join. Muller (2024) confirms that detectors learn dataset-specific shortcuts (silence length, spectral cleanness) rather than synthesis properties.

**Approach.** After splice, every internal word boundary independently undergoes a coin flip with `JITTER_PROBABILITY = 0.5`. Heads -> apply one of three structural manipulations chosen uniformly at random; tails -> leave natural. Spoof boundaries receive the same coin flip on top of the splice, so the detector cannot identify the spoof boundary as "the only one with a manipulation".

**Random selection algorithm.** For each internal boundary i in an utterance, independently (executed right-to-left so each manipulation only affects later sample positions):

```python
if rng.uniform(0, 1) < JITTER_PROBABILITY:                # 1) coin flip, p = 0.5
    op = rng.choice([truncate, overlap, bleed])           # 2) uniform pick of operation
    if op == truncate:
        ms = rng.uniform(10, 40)                          # 3a) uniform magnitude
        side = rng.choice([left_tail, right_head])        # 4a) uniform side
    elif op == overlap:
        ms = rng.uniform(30, 80)                          # 3b) uniform magnitude
        # Hanning fade always applied (deterministic)
    else:  # bleed
        ms = rng.uniform(20, 60)                          # 3c) uniform magnitude
        direction = rng.choice([right_to_left, left_to_right])  # 4b) uniform direction
    apply(audio, op, magnitude=ms, ...)
else:
    pass  # leave the boundary natural
```

The RNG is seeded as `JITTER_SEED + stable_hash(splice_key)` where `stable_hash = int(sha256(splice_key)[:4])` (Python's built-in `hash()` is randomized per process and would break reproducibility). The same utterance receives the same jitter plan across runs. Coin flips, operation picks, magnitude draws, and side/direction picks are independent per boundary — there is no smoothing between adjacent boundaries.

**Why p = 0.5 specifically.** Bernoulli(0.5) is the **maximum-entropy** choice on a binary indicator. Concretely, before jitter the manipulation-count signal is deterministic — bonafide-bonafide = 0, bonafide-spoof = 1 — and a detector can trivially classify on "manipulation present". After jitter with p = 0.5:

| Boundary type | Distribution over manipulation count |
|---|---|
| Bonafide-bonafide | 0 (50%) or 1 (50%) |
| Bonafide-spoof | 1 (50%, just splice) or 2 (50%, splice + jitter) |

Both distributions have equal mass at count=1, making count=1 maximally ambiguous. Other choices skew the leakage:
- p = 0.3 -> B-B is 0 (70%) / 1 (30%), B-S is 1 (70%) / 2 (30%): signal preserved through "absence of manipulation".
- p = 0.7 -> B-B is 0 (30%) / 1 (70%), B-S is 1 (30%) / 2 (70%): signal preserved through "double manipulation".
- p = 1.0 -> all boundaries always manipulated; B-S always has exactly 2 manipulations, becoming the new shortcut.
- p = 0.5 balances both sources of leakage.

**Pending ablation:** sweep p in {0.3, 0.5, 0.7, 1.0} once the initial pilot validates that jitter changes detector behavior at all.

**Why uniform among the three operations.** Each operation produces a distinct artifact type (truncate -> onset/offset abruptness, overlap -> energy summation/dip, bleed -> foreign-content pre/post-echo). A detector may be unequally sensitive to each, but we have **no a priori knowledge** of which artifact type is the detector's blind spot. Uniform sampling is the least-informative prior and matches the realistic threat model where the attacker does not know the detector. A weighted pick that targets the detector's weakness would be more effective, but constructing such weights requires running the detector first — circular.

**Why uniform magnitude within each range.** LlamaPartialSpoof (Luong et al. 2024) uses uniform random within 30-80 ms for crossfade overlap — the only published parameter-distribution convention in the partial-spoof literature. No paper biases toward smaller or larger magnitudes; no evidence that one direction is more "natural" or more detector-resistant. We adopt uniform-within-range for all three operations to remain consistent with this convention and avoid introducing a hand-tuned prior.

**Why uniform side / direction.** Truncate has two valid sides (`left_tail` cuts the left word's tail, `right_head` cuts the right word's onset), and bleed has two valid directions (`right_to_left` appends the right word's onset to the left word's tail, creating pre-echo; `left_to_right` does the reverse, creating post-echo). Each side/direction produces a distinct acoustic signature. Uniform random selection prevents the detector from learning a fixed pattern (e.g. "all truncates are left-tail cuts").

**The three operations.**

| Operation | Magnitude (uniform random) | Acoustic effect | Mimics |
|-----------|---------------------------|------------------|--------|
| **Truncate** | 10-40 ms | Cut left tail or right head -> abrupt onset/offset | Hard cut/paste splice (no crossfade) |
| **Overlap**  | 30-80 ms | Sum left tail with right head, Hanning fade | OLA-Hanning crossfade splice |
| **Bleed**    | 20-60 ms | Insert fragment of one word into the other -> pre/post-echo | Tail bleed of crossfade splice |

**Magnitude grounding.** Overlap matches the LlamaPartialSpoof crossfade range (Luong et al. 2024) exactly, anchoring our temporal scale to published partial-spoof literature. Truncate stays below Spanish syllable nucleus duration (~50-90 ms) so intelligibility is preserved; the 10 ms minimum is at the threshold of audibility (160 samples at 16 kHz). Bleed covers Spanish VOT for voiceless stops (4-29 ms) plus consonant-vowel transition, ensuring the inserted fragment sounds like a partial phoneme rather than a full second phoneme.

**Algorithm details.** Boundaries are processed right-to-left so each manipulation only affects later (un-touched) audio; the absolute sample indices of earlier boundaries remain valid. Total length drift is bounded (truncate and overlap shrink, bleed grows) and recorded per utterance. Per-utterance jitter plans are saved to `boundary_jitter_metadata.json` for reproducibility and traceability.

**Word-interior invariant.** All three operations act exclusively at the **boundary** between two words: truncate cuts a tail or head adjacent to the boundary, overlap blends the tail of the left word with the head of the right word, bleed inserts a fragment of one word into the other across the boundary. None of the operations modifies the middle of a word. Consequently, with `JITTER_PROBABILITY = 0.5` per boundary, the probability that both flanking boundaries of a given word flip natural is `0.5 * 0.5 = 0.25`. By construction, **on average ~25 % of non-spoof, non-edge words in any jittered utterance are bit-identical to the bonafide source** -- their audio is unmodified passthrough. The boundary anomaly distribution is homogenized while the linguistic content of most words is preserved verbatim.

This is intentional, not a deficiency. Manipulating the interior of a word (e.g. cutting 30 ms from the middle of a vowel) would damage intelligibility and ASR transcription, defeating the WER/CER quality gate in Step 6. Manipulating boundaries only is the maximum perturbation that preserves the linguistic information of each word -- the detector loses the "noisy boundary" shortcut without losing the prompt content.

**Validation observation (2026-05-09, n=2 utterances, speaker `arf_00295`).** Direct measurement on the small Qwen jitter pilot run confirmed the predicted distribution. For the 12-word utterance "Necesito que me des informacion sobre el mural que hicieron Frida Kahlo y Diego Rivera antes de morir" (W1 tier, "Necesito" cloned): 5 of 17 non-spoof words clean (29 %). For the 12-word "Segun la television y los medios, hubieron bastantes muertos por el terremoto" (W1 tier, "muertos" cloned): 3 of 11 clean (27 %). For the same utterance under W2 (both "hubieron" and "muertos" cloned): 5 of 10 clean (50 %). Aggregate ~35 % across n=38 boundaries -- within sampling variance of the 25 % theoretical mean. Drift values were -46 / +24 / -67 ms, all within the +/-100 ms acceptance criterion.

**Disjoint utterance pool.** The jitter dataset uses sentences DISJOINT from main partial_spoof. Setting `BONAFIDE_FILE_PARTITION` to `"main"` or `"jitter"` shuffles each speaker's bonafide files with a deterministic seed (`BONAFIDE_PARTITION_SEED + sha256(speaker_id)`) and takes the first or second half. No speakers are discarded; speakers with one file contribute to whichever partition the shuffle assigned. This means the combined training pool is roughly 2x in size with no input duplication.

**System ID and audio range.** Outputs use system_id `<ATTACK>_PSW{N}J` (e.g., `QWEN3TTS_PSW1J`) and audio IDs in 16M-18M range (W1J=16M, W2J=17M, W3J=18M), disjoint from main partial_spoof at 12M-14M.

**Pilot scope.** First run: Qwen only (already validated, ECAPA SIM 0.720). If detector EER changes meaningfully vs main Qwen partial spoof, replicate to Chatterbox, OpenVoice, OuteTTS, FishGram. OmniVoice joins after standalone validation.

## Corpus Composition Plan (2026-05-20)

The HABLA-Spoof corpus is constructed as a single dispatch over the six TTS attacks and two partitions, weighted so the strongest attacks dominate the corpus and the weaker attacks still contribute a meaningful minority.

### Target distribution

| Attack    | Share | Notes                                        |
|-----------|-------|----------------------------------------------|
| OmniVoice | 40 %  | Strongest end-to-end (NISQA 4.59, RTF 0.025) |
| Qwen      | 20 %  | Highest ECAPA SIM (0.720)                    |
| FishGram  | 10 %  | Reference pipeline                           |
| OpenVoice | 10 %  | Weakest cloner (avg SIM 0.394); kept for diversity |
| Chatterbox| 10 %  | High-quality but slowest                     |
| OuteTTS   | 10 %  | High-quality, second-slowest                 |

Sum = 1.0. Each attack's share is split 50/50 across the two partitions (`not_jittered`, `jittered`). The partitions use disjoint bonafide utterance pools per speaker so there is no phrase duplication across the corpus.

### Per-speaker probabilistic assignment

For each speaker, a deterministic RNG seeded with `ATTACK_ASSIGNMENT_SEED + sha256(speaker_id)[:4]` draws one attack from `Multinomial(p=ATTACK_WEIGHTS)` per bonafide file in each partition. The corpus-wide marginal converges to the target weights under the law of large numbers; speakers with too few files may not see all six attacks but every speaker contributes to at least one.

Largest-Remainder per-speaker was rejected because the six-way 40/20/10/10/10/10 distribution applied to small speakers (six files or fewer) forces at least one 10 % attack to round to zero, which would systematically over-represent the 40 % attack at the corpus level.

### Tier eligibility (opportunistic)

Each bonafide file's eligible tiers are pre-computed from its Parakeet word count:

| Word count | planned_tiers     |
|------------|-------------------|
| 4-7        | `[W1]`            |
| 8-11       | `[W1, W2]`        |
| >= 12      | `[W1, W2, W3]`    |
| < 4        | excluded entirely |

No padding, no tier rebalancing. The HABLA v2 sentence-length distribution determines the W1 / W2 / W3 yield naturally. Expected corpus output: ~17,963 files per partition x avg 2.3 tiers/file = ~41,000 outputs per partition, ~82,000 total spliced WAVs across the corpus.

### Manifest CSV (dispatch authority)

A single pre-flight script (`app/scripts/generate_partial_spoof_manifest.py`) writes `data/manifests/partial_spoof_plan.csv` with one row per eligible bonafide file: `(sample_key, speaker_id, audio_path, split, partition, attack, planned_tiers, word_count, bonafide_transcript)`. The manifest is the single source of truth consumed by all 12 per-attack pipeline runs.

Companion files:
- `partial_spoof_plan_summary.json` -- target vs actual marginals, speaker coverage, tier potential counts. The paper cites these numbers directly.
- `bonafide_transcripts_full.json` -- cached Parakeet output so per-attack runs skip re-transcription.

### Keep-bad-stuff principle

Step 6 (`SpliceQualityValidator`) computes WER, CER, NISQA, ECAPA SIM, and boundary metrics for every spliced sample but does NOT filter on quality (`ENABLE_STEP_6_REJECTION = False`). Each sample receives a `quality_flag` label ('high' / 'medium' / 'low'); downstream detector training stratifies on the flag instead of pre-filtering. Only STRUCTURAL failures (zero spoofed words, missing audio, audio load errors) are rejected because those are not actual partial spoofs. The upstream ECAPA clone gate (>= 0.60) stays enabled to filter clones that don't even resemble the target speaker -- those are not attacks, they are noise.

### Per-pipeline + corpus CSVs

Step 7 emits two flat CSVs per `(attack, partition)` cell:
- `samples.csv` -- one row per spliced WAV with all paths, metrics, and `quality_flag`.
- `spoofed_words.csv` -- one row per spoofed word with bonafide/cloned boundary timestamps and splice method (the frame-level boundary-label table).

The orchestrator (`app/runner/partial_spoof_orchestrator.py --mode aggregate`) concatenates the twelve per-pipeline CSVs into `corpus_samples.csv` and `corpus_spoofed_words.csv` at the partial spoof output root, plus a `corpus_summary.json` with realised marginals.

## Related Pages
- [Splicing Techniques](../state-of-art/splicing-techniques.md) — literature review
- [Quality Metrics](quality-metrics.md) — thresholds and formulas
- [Decision Log](../decisions/decision-log.md) — chronological decisions
