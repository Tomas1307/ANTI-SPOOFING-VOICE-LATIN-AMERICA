# Splicing Techniques for Partial Spoof Construction

**Status:** Active
**Last updated:** 2026-05-01
**Source:** investigation.md Sections 8.3-8.5

---

## Overview

This page documents all waveform-level splicing techniques found in the partial-spoof literature, plus the 7-technique implementation designed for the HABLA 2.0 dataset. The techniques range from naive hard concatenation (most detectable) to sophisticated overlap-add with spectral alignment (least detectable but most complex).

**Key finding from the literature:** No pipeline releases its splicer source code. Only LlamaPartialSpoof names concrete waveform parameters (30-80 ms overlap, 5 fade shapes). No paper ablates crossfade duration against detection EER.

---

## Techniques from the Literature

### 1. Direct Cut-Paste (No Smoothing)

**Source:** LlamaPartialSpoof (baseline condition), PartialSpoof

**Method:** Hard concatenation at splice point. No overlap, no fading. The bonafide segment ends at sample N; the cloned segment begins at sample N+1.

**Artifacts produced:** Audible clicks, energy discontinuities, phase jumps at the splice boundary. Most detectable of all techniques.

**Detection performance:** Negroni et al. achieves 6.16% EER on PartialSpoof with zero training -- purely spectral-dynamic-range analysis of the join. This sets the detectability baseline.

**Use case:** Baseline condition representing the worst-case (most naive) attacker. Essential to include for detector calibration.

---

### 2. Overlap-Add with Hanning Window (OLA-Hanning)

**Source:** Negroni et al. (arXiv:2408.13784, 2024) -- applied to their own experimental splicing. **Critical correction: this technique is NOT from HAD.** HAD uses pydub (simple cut/paste). The OLA-Hanning attribution to HAD found in some secondary sources is incorrect.

**Method:** Apply a half Hanning window to the end of the first segment (fade-out) and a half Hanning window to the beginning of the second segment (fade-in). Sum the two windowed segments in the overlap region.

```
Segment A: [..., a_n-3, a_n-2, a_n-1, a_n] * hanning_fadeout
Segment B: [b_0, b_1, b_2, b_3, ...] * hanning_fadein
Overlap:   sum of windowed regions
```

**Parameters tested by Negroni et al.:**

| Window Size (samples) | Duration (at 16 kHz) | AUC (detectability) | Assessment |
|----------------------|---------------------|---------------------|------------|
| 256 | 16 ms | 98.04% | Near-original artifact levels |
| 512 | 32 ms | ~95% (interpolated) | Marginal improvement |
| 1024 | 64 ms | ~92% (interpolated) | Minimum for effective hiding |
| 2048 | 128 ms | ~90% (interpolated) | Better mitigation |
| 4096 | 256 ms | 88.99% | Best tested, still detectable |

**Key findings:**
- Minimum 1024 samples (64 ms) needed to effectively hide artifacts
- Even at 4096 samples (256 ms), minimum AUC is still 88.99% -- artifacts persist
- The Hanning window shape provides smooth energy transitions but does not address spectral discontinuities

---

### 3. Crossfade with 5 Fading Functions

**Source:** LlamaPartialSpoof (arXiv:2409.14743, ICASSP 2025)

**Method:** Overlap region where one of five fading functions controls the blend between bonafide and cloned segments. Function randomly assigned per splice to prevent detector learning a fixed pattern.

**Parameters:**
- Overlap duration: uniform random 30-80 ms per splice
- Pre-processing: loudness normalization, downsample to 16 kHz
- Post-processing: random peak level -0.01 to -10 dBFS

**The five functions:**

| Function | Fade-out curve (bonafide) | Fade-in curve (cloned) | Character |
|----------|--------------------------|------------------------|-----------|
| **Linear** | 1 -> 0 linearly | 0 -> 1 linearly | Constant energy rate of change |
| **Quarter sine** | cos(t * pi/2) | sin(t * pi/2) | Slow start, fast end |
| **Half sine** | sin(t * pi) | 1 - sin(t * pi) | Smooth S-curve |
| **Logarithmic** | log decay | log rise | Fast initial change, slow tail |
| **Inverted parabola** | 1 - t^2 | t^2 | Quadratic (accelerating) curve |

where t ranges from 0 to 1 over the overlap duration.

**Key finding:** Table V(b) in LlamaPartialSpoof ablates crossfade vs cut-paste vs OLA -- **the only published insertion-technique-vs-EER comparison in the literature**. Diverse methods improve robustness. This table is the primary empirical anchor for technique selection.

---

### 4. OLA + Cosine + Spectral Pre-emphasis

**Source:** HQ-MPSD (arXiv:2512.13012, Dec 2025)

**Method:** Fixed 30 ms cosine overlap-add with acoustic pre-processing pipeline.

**Parameters:**
- Overlap: fixed 30 ms, cosine window
- RMS-based loudness alignment
- "Spectral-characteristic alignment" (algorithmic detail not specified in arXiv v1 -- cite with caution)
- Cut placement: midpoints between aligned word pairs (Montreal Forced Aligner)
- Post-processing: room impulse responses + noise at 15 dB SNR

**Key finding:** Word-midpoint cuts significantly reduce prosodic discontinuities compared to word-boundary cuts. By cutting between words rather than at word boundaries, the splice point falls in a natural pause or transition where discontinuities are less perceptible.

**Caveat:** The "spectral-characteristic alignment" claim should be cited carefully. Only the phrase "spectral-characteristic alignment" appears in the arXiv v1 text -- the more specific "adaptive pre-emphasis" sometimes attributed is not verified verbatim.

---

### 5. Cross-Correlation Best-Join

**Source:** PartialSpoof (IEEE/ACM TASLP 2023, Section III-B step 3)

**Method:** Within the VAD-detected silent margin around each segment, use time-domain cross-correlation to find the optimal concatenation point that minimizes discontinuity. Then apply OLA within the silence region.

**How it works:**
1. VAD identifies speech/silence boundaries
2. Within the silent margin (typically 50-200 ms), slide a window across possible join positions
3. Compute cross-correlation between the end of segment A and the beginning of segment B at each position
4. Select the position with maximum correlation (smoothest transition)
5. Apply OLA at the selected point

**Key insight:** Most sophisticated boundary-selection technique in the literature. Elegant when a silent margin exists. **Undefined when segments abut directly** (no margin available for the sliding window). This limits applicability to VAD-boundary splicing -- not suitable for word-level splicing where segments may be adjacent.

---

### 6. Zero-Crossing Alignment

**Source:** Universal audio editing practice (not specific to any partial-spoof paper).

**Method:** Snap splice points to the nearest zero-crossing within a +/-2 ms search window. A zero-crossing is a point where the waveform amplitude passes through zero, which minimizes the click artifact produced by a discontinuity.

**Optional enhancement:** GCI (Glottal Closure Instant) alignment using the DYPSA algorithm (Naylor et al., 2007, 95.7% identification rate) when both sides of the splice are voiced speech. GCI alignment produces joins that are pitch-synchronous, further reducing audible artifacts.

**Literature gap:** No anti-spoofing paper measures the EER impact of GCI alignment vs zero-crossing alignment vs arbitrary cuts. This is a tractable publishable experiment.

---

## Our 7-Technique Implementation for MARSA

Seven techniques in varied proportions, informed by the literature review.
**Code-verified against `app/pipeline/partial_spoof/utils/splice_method.py`
(`SPLICE_METHOD_WEIGHTS`) and `utils/crossfade.py` (`_compute_fade_curves`),
2026-07-09.** The fade-curve column gives the exact fade-in envelope for
`t in [0, 1]`; fade-out = fade-in(1 - t).

| # | Enum member | Proportion | Fade-in curve | Class |
|---|-------------|-----------|---------------|-------|
| 1 | `CUT_PASTE` | 10% | none (hard concat) | maximum discontinuity |
| 2 | `OLA_HANNING` | 20% | `0.5(1 - cos pi*t)` | equal-gain S-curve |
| 3 | `LINEAR` | 15% | `t` | equal-gain |
| 4 | `COSINE` | 20% | `sin(pi*t/2)` | equal-power |
| 5 | `HALF_SINE` | 15% | `sqrt(t)` | equal-power (square-root law) |
| 6 | `LOGARITHMIC` | 10% | `log(1 + 9t)/log 10` | aggressive initial rise |
| 7 | `PARABOLA` | 10% | `1 - (1 - t)^2` | equal-gain concave |

Weights sum to 1.00. A single overlap duration is drawn per splice, uniformly
from `[CROSSFADE_MIN_MS, CROSSFADE_MAX_MS]` (one range for ALL methods, not
per-method ranges as an earlier draft of this table implied). The effective
crossfade is then bounded by the available silent run on each side of the
cloned word so the fade never bleeds a neighbouring word into the seam.

**NAMING CAVEAT FOR THE PAPER.** The enum member `HALF_SINE` is a misnomer:
its curve is the square-root law `sqrt(t)` (equal-power), NOT a half-sine.
In paper prose, describe method 5 by its mathematical form (`sqrt(t)`,
equal-power) — never copy the enum name, or a reviewer checking the code
will flag the inconsistency.

### Design rationale

- **10% direct cut-paste:** Baseline representing naive attacker. Essential for detector calibration.
- **20% OLA Hanning + 20% cosine crossfade:** Together 40% of samples use the two most common techniques from the literature (Negroni et al. and HQ-MPSD).
- **15% linear + 15% half-sine:** LlamaPartialSpoof fade functions for diversity.
- **10% logarithmic + 10% inverted parabola:** Least common fade shapes, testing detector robustness to unusual curves.
- **Random overlap duration per splice:** Prevents detector learning a fixed window size.

### Applied universally across all techniques

1. **Zero-crossing alignment** at splice points (within +/-2 ms search window)
2. **RMS energy normalization** of cloned segment to match bonafide region
3. **Random overlap duration** per splice (within technique-specific ranges)
4. **Full natural duration** of cloned word (no compression/truncation -- per design decision)
5. **Adjacent cloned words:** cluster them, apply crossfade only at cluster-bonafide boundaries (per PartialSpoof's margin-based approach)

### Synthesized best-practice procedure

From the deep research audit (Section D of partial_spoof_inv_1.md):

1. Cluster adjacent replacements into contiguous runs
2. Per-cluster duration policy: accept mismatch if ratio in [0.90, 1.10], global-shift if [0.80, 1.25], reject outside
3. Intra-cluster: concatenate at zero-crossings with no crossfade (words from same TTS run)
4. Cluster-boundary: cross-correlation best-join within available bonafide margin (a la PartialSpoof)
5. Loudness match (RMS normalization)
6. Apply chosen crossfade technique at cluster boundaries
7. Post-process: random peak normalization (-0.01 to -10 dBFS)

---

## Technique Comparison: Detectability vs Complexity

| Technique | Detectability | Complexity | Literature Support |
|-----------|--------------|------------|-------------------|
| Direct cut-paste | Highest (6.16% EER with no training) | Lowest | PartialSpoof, LlamaPartialSpoof |
| OLA Hanning | High (88.99% AUC at 256 ms) | Low | Negroni et al. (2024) |
| Crossfade (5 functions) | Medium-High | Low | LlamaPartialSpoof (only EER ablation) |
| OLA + cosine + spectral | Medium | Medium | HQ-MPSD |
| Cross-correlation best-join | Lower (within margins) | High | PartialSpoof |
| Zero-crossing alignment | Enhancement (reduces clicks) | Very Low | Universal practice |

---

## What the Literature Does NOT Cover

These gaps represent potential thesis contributions (see [Partial Spoof Literature](partial-spoof-literature.md) for full gap analysis):

1. **Crossfade duration vs EER:** No paper maps overlap length to detection performance. LlamaPartialSpoof's 30-80 ms is the de facto anchor but was never ablated.
2. **F0 smoothing at splice boundaries:** Toolkit exists (TD-PSOLA, HNM) but is undocumented for partial-spoof construction.
3. **GCI vs zero-crossing vs arbitrary cuts:** No EER comparison exists.
4. **Spectral envelope interpolation:** No pipeline performs LPC or MFCC smoothing at joins.
5. **Adjacent word handling:** No pipeline specifies behavior when two cloned segments abut with <30 ms of bonafide margin.

---

## Boundary Jitter: Per-Boundary Structural Manipulations (2026-05-01)

A novel proposal in this project, motivated by the Negroni 6.16%-EER-no-training result and the Muller 2024 detector-shortcut analysis. Instead of (or in addition to) crossfading the splice boundary, we apply **the same kind of structural artifacts** to ALL internal word boundaries (including the spoof boundary), making them statistically indistinguishable.

### Three operations

Each defined per-boundary, with magnitude drawn from a literature-grounded or phonetics-informed range:

**1. Truncate.** Cut a fragment from the left word's tail or the right word's head. Mimics direct cut-paste. Magnitude 10-40 ms uniform; below this the cut is sub-audible (<160 samples at 16 kHz), above this it begins to chop into Spanish syllable nuclei (typically 50-90 ms vowel duration).

**2. Overlap.** Shift the right word backward in time so its onset overlaps with the left word's tail; sum the two with a Hanning fade. Identical in principle and parameter range to OLA-Hanning crossfade. Magnitude 30-80 ms uniform — **literature-grounded**, matches LlamaPartialSpoof crossfade range (Luong et al. 2024) exactly.

**3. Bleed.** Insert a fragment of one adjacent word into the other (right_to_left appends right word's onset to left word's tail, creating pre-echo; left_to_right does the reverse, creating post-echo). Mimics tail bleed of crossfade splices. Magnitude 20-60 ms uniform; lower bound covers Spanish VOT (4-29 ms) plus brief consonant transition, upper bound covers consonant onset plus vowel attack without inserting a complete second phoneme.

### Per-boundary application

For every internal word boundary in a partial-spoofed utterance:
- Coin flip with `JITTER_PROBABILITY = 0.5`.
- Heads -> uniformly pick one operation, draw magnitude, apply.
- Tails -> leave natural.

The spoof boundary receives the same coin flip on top of the splice already applied in Step 5. The detector cannot identify the spoof boundary as "the only one with a manipulation"; the manipulation distribution is 0/1 (bonafide-bonafide) vs 1/2 (bonafide-spoof) instead of the original 0 vs 1.

### Why this is novel

The 4 canonical partial-spoof corpora (PartialSpoof, LlamaPartialSpoof, HAD, HQ-MPSD) all keep the bonafide segments **clean**. None of them apply structural artifacts to non-spoof boundaries. The detector trained on these corpora can use boundary anomaly as a near-perfect shortcut. Boundary jitter is, to our knowledge, the first proposal to systematically homogenize boundary artifacts across the entire utterance.

### Magnitude grounding summary

| Operation | Range | Grounding |
|---|---|---|
| Truncate | 10-40 ms | Phonetics: < Spanish syllable nucleus (50-90 ms); > audibility floor |
| Overlap | 30-80 ms | **Literature**: LlamaPartialSpoof (Luong 2024), exact match |
| Bleed | 20-60 ms | Phonetics: VOT (4-29 ms) + consonant-vowel transition |

### Implementation

`app/pipeline/partial_spoof/steps/step_05b_apply_boundary_jitter.py` runs after Step 5 (splice) and before Step 6 (validate). Per-utterance boundaries processed right-to-left so manipulations don't invalidate earlier sample indices. Total length drift bounded; recorded per utterance for analysis.

System ID: `<ATTACK>_PSW{N}J`. Audio ID range: 16M-18M (jitter W1/W2/W3) disjoint from main 12M-14M.

---

## Cross-references

- [Partial Spoof Literature](partial-spoof-literature.md) -- full literature review, 6 unsolved gaps
- [Anti-Spoofing Datasets](anti-spoofing-datasets.md) -- datasets where these techniques are used
- [Detection Methods](detection-methods.md) -- how detectors exploit splice artifacts
- [TTS Systems](tts-systems.md) -- systems generating the cloned segments
- [Partial Spoof Approach](../methodology/partial-spoof-approach.md) -- our implementation of valley-score selection, duration-preserving splice, and boundary jitter
