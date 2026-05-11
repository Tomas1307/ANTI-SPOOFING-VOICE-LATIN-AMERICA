# Partial Spoof Approach

**Status:** Active
**Last updated:** 2026-04-25
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

### Duration-preserving splice (not variable-length insert)

**Discovery (April 25):** Inserting a 480ms cloned word into a 640ms bonafide slot shifted all subsequent audio by 160ms, destroying speech rhythm.

**Solution:** Time-stretch the cloned word to fit the exact bonafide slot duration. Overwrite in place. Total audio length never changes. Stretch ratio limited to [0.75, 1.25].

### Clone similarity gate

ECAPA-TDNN cosine similarity between bonafide and clone must be >= 0.60. Bad clones are rejected before alignment/splicing (between Steps 2 and 3), saving compute.

## Edge Cases

| Case | Handling |
|------|----------|
| All words have bad valley scores | Reject sample for tier |
| Not enough good words for W2/W3 | Skip higher tiers |
| Word < 200ms | Ineligible (too short to be meaningful) |
| First/last word | Only internal boundary scored (external = silence, artificially good) |
| Stretch ratio outside [0.75, 1.25] | Word ineligible |
| All clones for a speaker fail similarity | Speaker produces no partial spoofs |

## Quality Metrics

From validation run (3 speakers, 7 samples, pre-v2 pipeline):
- WER: 3.9%
- NISQA MOS: 4.72
- Speaker similarity: 0.789
- Pass rate: 7/7 (100%)

**Pending:** v2 validation with 5 speakers, 10+ audios each, including valley-score selection.

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

## Related Pages
- [Splicing Techniques](../state-of-art/splicing-techniques.md) — literature review
- [Quality Metrics](quality-metrics.md) — thresholds and formulas
- [Decision Log](../decisions/decision-log.md) — chronological decisions
