# Partial Spoof Literature Review

**Status:** Active
**Last updated:** 2026-04-25
**Source:** investigation.md Section 8 (Comprehensive Literature Review)

---

## Overview

This page synthesizes the comprehensive literature review of partial-spoof speech -- utterances where only some segments are replaced with synthetic or converted speech while the rest remains bonafide. The review audited 4 target pipelines, 2 third-party analyses, and classical concatenative synthesis literature.

**Central finding:** Four canonical partial-spoof corpora (PartialSpoof, LlamaPartialSpoof, HAD, HQ-MPSD) collectively publish far less waveform-level splicing detail than their prominence would suggest. Only LlamaPartialSpoof names concrete waveform parameters. No public pipeline documents time-stretching, F0 smoothing, formant matching, or any ablation of overlap duration versus detector EER. **No pipeline releases its splicer source code.**

---

## The 4 Pipelines x 7 Critical Questions

| Question | PartialSpoof (Zhang 2023) | LlamaPartialSpoof (Luong 2024) | HAD (Yi 2021) | HQ-MPSD (Li 2025) |
|----------|--------------------------|-------------------------------|---------------|-------------------|
| Zero-gap handling | N/A (VAD margins) | Not specified | N/A (1 replacement/utt) | Not specified |
| Duration mismatch | Selects similar-duration segments | Not specified | Not specified | Not specified |
| F0 discontinuity | Not specified | Not specified | Not specified | Not specified |
| Spectral envelope | ITU-T SV56 (-26 dBov) | Loudness norm only | Volume norm only | "Loudness + spectral alignment" (no detail) |
| Cut placement | VAD boundaries | MFA word boundaries | Character-level timestamps | Word midpoints |
| Code released | No (listed "TBA") | No (metadata only) | No (audio only) | No |
| Perceptual metric | None | EER per concat method (Table V) | EER only | DNSMOS 3.58 |

### What this table reveals

- **F0 discontinuity** is universally unaddressed. No partial-spoof paper reports pitch discontinuity at splice boundaries or applies any smoothing.
- **Spectral envelope continuity** ranges from nothing (HAD) to vague claims (HQ-MPSD). No pipeline performs LPC-coefficient interpolation, MFCC-trajectory smoothing, or formant matching.
- **Code availability** is zero across all four pipelines. This is the single largest reproducibility gap in the field.
- **Cut placement** varies significantly: VAD boundaries (PartialSpoof), MFA word boundaries (LlamaPartialSpoof), character timestamps (HAD), word midpoints (HQ-MPSD).

---

## Third-Party Analyses (Critical Evidence)

### Negroni et al. (2024) -- No-Training Spectral Analysis

**Paper:** "Analyzing the Impact of Splicing Artifacts in Partially Fake Speech Signals," ASVspoof Workshop 2024. arXiv:2408.13784.

**Key result:** Achieves **6.16% EER on PartialSpoof** and **7.36% EER on HAD** with zero training -- purely hand-coded spectral-dynamic-range analysis of the splice join.

**Method:** Analyzes spectral discontinuities at splice boundaries without any machine learning. Tests OLA-Hanning windows of varying sizes (256, 512, 1024, 2048, 4096 samples) to measure artifact mitigation.

**Key findings on OLA-Hanning window sizes:**
- 256 samples (16 ms at 16 kHz): near-original artifact levels (AUC=98.04%)
- 1024+ samples (64+ ms): better mitigation but still detectable (minimum AUC=88.99%)
- Minimum 1024 samples (64 ms) needed to effectively hide artifacts
- Even at 4096 samples (256 ms), minimum AUC is still 88.99% -- artifacts persist

**Critical correction:** The "OLA-Hanning" sometimes attributed to HAD in secondary sources is actually from Negroni et al.'s own experimental splicing on ASVspoof data, NOT the HAD paper itself. HAD uses pydub (simple cut/paste wrapper).

**Implication for our work:** A ~6% EER floor exists for any detector analyzing splice boundaries, even with no training. This sets the baseline for what simple spectral analysis can achieve. Multi-technique splicing with varied crossfade durations is necessary to push beyond this floor.

### Huang et al. (SLT 2024) -- Perceptual-vs-Detection Gap

**Paper:** "Detecting the Undetectable: Assessing Efficacy of Spoof Detection Against Seamless Speech Edits," SLT 2024. arXiv:2501.03805.

**Key result:** Neural infilling fools humans but NOT SSL-based detectors. Establishes the **perceptual-vs-detection asymmetry** -- edits that are imperceptible to human listeners can still be detected by machine learning systems, and vice versa.

**Implication:** The goal of partial-spoof construction is not necessarily to fool humans (that is achievable) but to create diverse artifacts that challenge automated detectors. This reinforces the multi-technique approach.

**Together these two papers bound the gap:** Negroni et al. shows simple splicing is detectable even without training. Huang et al. shows sophisticated neural edits fool humans but not SSL detectors. The detection problem lies in the space between these bounds.

---

## Historical Context: Three Eras of Splicing Detection

The detection problem has inverted in 30 years. Each era of synthesis produces a characteristic forensic trace, and each subsequent generation eliminates the previous trace:

| Era | Synthesis Method | Detectable Trace | Detection Method | Typical EER |
|-----|-----------------|------------------|------------------|-------------|
| Pre-2010 | Splice of real recordings | ENF phase, room impulse | Phase analysis, filtering | ~6% (clean) |
| Concatenative TTS (2000s) | Unit selection from database | Cepstral/F0/power discontinuity at joins | Join-cost analysis, MFCC variance | Audible seams |
| Neural TTS (2020+) | End-to-end generation | Statistical artifacts of generative models | Raw-waveform CNN/Transformer, SSL | 0.1-5% lab, 30%+ in-the-wild |

**Key insight:** Every detection regime exploits a signature that the next generation of synthesis eliminates. ENF disappears when recording goes off-mains. Concatenative joins disappear when synthesis goes neural. The current SSL-based detectors will degrade as synthesis models improve.

**The Hunt-Black to In-the-Wild arc:** From Hunt & Black (1996) establishing unit selection concatenative synthesis, through Moulines & Charpentier (1990, TD-PSOLA) and Stylianou (2001, HNM) for pitch-synchronous manipulation, to the modern neural TTS era where the artifacts are statistical rather than acoustic.

---

## The Generalization Crisis

**Muller et al. (Interspeech 2022):** "Does Audio Deepfake Detection Generalize?"

Demonstrated that detectors trained on ASVspoof 2019 degrade by **200-1000% EER** on in-the-wild audio. The "difference" term (distributional mismatch between training and test conditions) dominates over "hardness" (intrinsic difficulty of detecting the fake).

**Expanded in Muller (2024):** "Harder or Different? Understanding Generalization of Audio Deepfake Detection." Confirmed that models learn dataset-specific artifacts (e.g., "silence shortcut" -- length of leading silence correlates with bonafide/spoof class) rather than genuine synthesis properties.

**Implication for dataset construction:** Diverse artifacts that force detectors to learn generalizable features, not dataset-specific shortcuts, are essential. This directly motivates the multi-technique splicing approach with varied crossfade durations and randomized parameters.

---

## Six Unsolved Literature Gaps

Based on the deep research audit, these problems are **completely unaddressed** in the partial-spoof literature:

### 1. Zero-Gap Boundary Policy (the adjacency problem)

No pipeline specifies behavior when two cloned segments abut with <30 ms of bonafide margin. Five candidate strategies -- butt-splice at zero-crossing, silence insertion, micro-shift, multi-word TTS regeneration, cluster-external overlap only -- are all plausible but none is attested in any paper or repository. **This is the single largest gap. Defensible thesis contribution on its own.**

### 2. F0 Discontinuity

No partial-spoof paper reports |DF0| distribution at splice boundaries. No paper applies F0 smoothing. The toolkit exists (TD-PSOLA, Moulines & Charpentier 1990; HNM, Stylianou 2001) but is undocumented for partial-spoof construction. Publishing DF0 histograms for Qwen/FishGram at MFA word boundaries would be a novel contribution.

### 3. Duration Mismatch Absorption

No pipeline discriminates between global-shift, silence-compression, and accept-mismatch strategies. Each has different implications for prosody and detection. TSM Subjective Quality Dataset (Roberts 2020) shows subjective MOS falls steeply outside ratio 0.85-1.20, but no anti-spoofing-specific threshold exists.

### 4. Spectral Envelope Continuity

No pipeline performs LPC-coefficient interpolation, MFCC-trajectory smoothing, or formant matching at joins. Negroni et al. strongly suggests the loudness-only policy leaves a ~6% EER floor for any detector.

### 5. Crossfade Duration vs Detection EER

LlamaPartialSpoof's 30-80 ms range is the de facto anchor but was never ablated against EER. No paper maps overlap length to detection performance. This is a tractable publishable experiment.

### 6. Perceptual-vs-Detection Pareto Frontier

No paper plots MOS-vs-EER scatter for word-level edits with MFA boundaries and zero-shot voice cloning. Huang et al. established the phenomenon for sentence-level infilling; the word-level case is open.

---

## Key Premise Corrections

Two corrections from the deep research audit:

1. **HAD does NOT use OLA-Hanning.** That attribution belongs to Negroni et al. (2024), who applied OLA-Hanning in their own experimental splicing on ASVspoof data. HAD uses pydub (simple cut/paste).

2. **HQ-MPSD "adaptive pre-emphasis" is not verified verbatim** in the arXiv v1 text. Only "spectral-characteristic alignment" appears. The citation should be softened to reflect this.

---

## References

**Target pipelines:**
- Zhang et al., "The PartialSpoof Database and Countermeasures," IEEE/ACM TASLP 31:813-825, 2023. arXiv:2204.05177
- Luong et al., "LlamaPartialSpoof: An LLM-Driven Fake Speech Dataset," ICASSP 2025. arXiv:2409.14743
- Yi et al., "Half-Truth: A Partially Fake Audio Detection Dataset," Interspeech 2021. arXiv:2104.03617
- Li et al., "HQ-MPSD: A Multilingual Artifact-Controlled Benchmark," arXiv:2512.13012, Dec 2025

**Third-party analyses:**
- Negroni et al., "Analyzing the Impact of Splicing Artifacts in Partially Fake Speech Signals," ASVspoof Workshop 2024. arXiv:2408.13784
- Huang et al., "Detecting the Undetectable: Assessing Efficacy of Spoof Detection Against Seamless Speech Edits," SLT 2024. arXiv:2501.03805

**Generalization and benchmarks:**
- Muller et al., "Does Audio Deepfake Detection Generalize?" Interspeech 2022
- Muller, "Harder or Different? Understanding Generalization of Audio Deepfake Detection," 2024
- ML-ITW benchmark, 2026 (14 languages, 7 platforms, 28.4 hours)

**Foundations (concatenative synthesis):**
- Hunt & Black, "Unit Selection in a Concatenative Speech Synthesis System," ICASSP 1996
- Moulines & Charpentier, "Pitch-synchronous waveform processing techniques," Speech Communication 9(5-6), 1990 (TD-PSOLA)
- Stylianou, "Applying the harmonic plus noise model in concatenative speech synthesis," IEEE TSAP 9(1), 2001
- Naylor et al., "Estimation of Glottal Closure Instants (DYPSA)," IEEE TASLP 15(1), 2007

**Time-scale modification:**
- Roberts, "A Time-Scale Modification Dataset with Subjective Quality Labels," IEEE DataPort, 2020. arXiv:2006.00848

---

## Cross-references

- [Anti-Spoofing Datasets](anti-spoofing-datasets.md) -- dataset details and sizes
- [Splicing Techniques](splicing-techniques.md) -- the 7 techniques and implementation parameters
- [Detection Methods](detection-methods.md) -- countermeasure architectures
- [TTS Systems](tts-systems.md) -- TTS systems used to generate the attack audio
