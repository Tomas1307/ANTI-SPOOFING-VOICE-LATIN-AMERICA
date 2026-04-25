# Anti-Spoofing Datasets

**Status:** Active
**Last updated:** 2026-04-25
**Source:** investigation.md Section 8.2, 8.8; general literature

---

## Overview

This page catalogs the primary anti-spoofing and partial-spoof datasets referenced in the literature and relevant to this thesis. The four canonical partial-spoof corpora (PartialSpoof, LlamaPartialSpoof, HAD, HQ-MPSD) are documented in detail alongside the foundational ASVspoof challenges. Related multilingual and Spanish-specific datasets are noted for completeness.

A critical finding from the literature review: **no public pipeline releases its splicer source code.** All four partial-spoof datasets provide audio and metadata but not the waveform-level splicing implementation. See [Partial Spoof Literature](partial-spoof-literature.md) for the full comparison.

---

## Foundational Datasets

### ASVspoof 2019 LA (Logical Access)

| Attribute | Value |
|-----------|-------|
| **Language** | English |
| **Size** | ~130,000 utterances (training + development + evaluation) |
| **Attack types** | 19 spoofing systems: 6 known (A01-A06) + 13 unknown (A07-A19). Neural waveform models, vocoders, VC systems |
| **Splicing method** | N/A (full-utterance spoofing, not partial) |
| **Code released** | Evaluation toolkit released; spoofing system code varies by contributor |
| **Notes** | De facto benchmark for anti-spoofing research. Logical Access track focuses on synthesized/converted speech. Muller et al. (Interspeech 2022) showed detectors trained on this dataset degrade by 200-1000% EER on in-the-wild audio. |
| **Reference** | Todisco et al., "ASVspoof 2019," 2019 |

### ASVspoof 2021 LA

| Attribute | Value |
|-----------|-------|
| **Language** | English |
| **Size** | Extends 2019 with telephony and codec conditions |
| **Attack types** | Same spoofing systems as 2019 + transmission channel effects |
| **Splicing method** | N/A (full-utterance spoofing) |
| **Code released** | Evaluation toolkit released |
| **Notes** | Added real-world transmission conditions (VoIP, PSTN) to test robustness. Codec-based attacks showed 41.4% performance degradation for detectors trained only on GAN-vocoder artifacts. |
| **Reference** | Yamagishi et al., "ASVspoof 2021," 2021 |

---

## Partial-Spoof Datasets (Canonical Four)

### PartialSpoof (Zhang et al., 2023)

| Attribute | Value |
|-----------|-------|
| **Language** | English |
| **Size** | Based on ASVspoof 2019 LA utterances with segment-level labels |
| **Attack types** | Partial replacement of bonafide utterances with TTS/VC segments at varying substitution ratios |
| **Splicing method** | VAD-boundary cuts. Cross-correlation best-join within silent margins to find optimal concatenation point, then OLA within the silence. ITU-T SV56 (-26 dBov) loudness normalization. Selects similar-duration segments for replacement. |
| **Cut placement** | VAD boundaries |
| **Code released** | No (listed as "TBA" -- never published) |
| **Perceptual metric** | None reported |
| **Reference** | Zhang et al., "The PartialSpoof Database and Countermeasures," IEEE/ACM TASLP 31:813-825, 2023. arXiv:2204.05177 |
| **Notes** | Most sophisticated boundary-selection technique in the literature (cross-correlation best-join), but only works when a silent margin exists between segments. Undefined behavior when segments abut directly. |

### LlamaPartialSpoof (Luong et al., 2024)

| Attribute | Value |
|-----------|-------|
| **Language** | English |
| **Size** | LLM-driven dataset generation |
| **Attack types** | Word-level replacement using LLM-selected semantically plausible substitutions |
| **Splicing method** | Crossfade with 5 fading functions randomly assigned per splice. Overlap: uniform random 30-80 ms. Functions: linear, quarter sine, half sine, logarithmic, inverted parabola. Pre-processing: loudness normalization, downsample to 16 kHz. Post-processing: random peak level -0.01 to -10 dBFS. Also includes direct cut-paste (no smoothing) as a baseline condition. |
| **Cut placement** | MFA (Montreal Forced Aligner) word boundaries |
| **Code released** | No (metadata only) |
| **Perceptual metric** | EER per concatenation method (Table V) -- the only published insertion-technique-vs-EER comparison in the literature |
| **Reference** | Luong et al., "LlamaPartialSpoof: An LLM-Driven Fake Speech Dataset," ICASSP 2025. arXiv:2409.14743 |
| **Notes** | Only dataset that names concrete waveform parameters (30-80 ms overlap, 5 fade shapes). Table V(b) ablates crossfade vs cut-paste vs OLA -- critical for technique selection. |

### HAD / Half-Truth (Yi et al., 2021)

| Attribute | Value |
|-----------|-------|
| **Language** | Chinese |
| **Size** | ~100,000 utterances |
| **Attack types** | Single character-level replacement per utterance (one fake segment per sample) |
| **Splicing method** | pydub (simple cut/paste wrapper). Volume normalization only. **Critical correction: the "OLA-Hanning" sometimes attributed to HAD in secondary sources is actually from Negroni et al.'s external analysis, NOT the HAD paper itself.** |
| **Cut placement** | Character-level timestamps |
| **Code released** | No (audio only) |
| **Perceptual metric** | EER only |
| **Reference** | Yi et al., "Half-Truth: A Partially Fake Audio Detection Dataset," Interspeech 2021. arXiv:2104.03617 |
| **Notes** | Chinese language limits direct applicability to this thesis but establishes the partial-spoof paradigm. Single-replacement design is simpler than multi-word approaches. |

### HQ-MPSD (Li et al., 2025)

| Attribute | Value |
|-----------|-------|
| **Language** | Multilingual (multiple languages) |
| **Size** | Not specified in available sources |
| **Attack types** | High-quality multi-point splicing with acoustic pre-processing |
| **Splicing method** | Fixed 30 ms cosine overlap-add with RMS-based loudness alignment and "spectral-characteristic alignment" (algorithmic detail not specified in arXiv v1). Cuts at midpoints between aligned word pairs using Montreal Forced Aligner. Post-processing: room impulse responses + noise at 15 dB SNR. |
| **Cut placement** | Word midpoints (between aligned word pairs) |
| **Code released** | No |
| **Perceptual metric** | DNSMOS 3.58 |
| **Reference** | Li et al., "HQ-MPSD: A Multilingual Artifact-Controlled Benchmark," arXiv:2512.13012, Dec 2025 |
| **Notes** | Word-midpoint cuts significantly reduce prosodic discontinuities compared to word-boundary cuts. "Spectral-characteristic alignment" claim should be cited with caution -- only "spectral-characteristic alignment" appears in arXiv v1, not the more specific "adaptive pre-emphasis" sometimes attributed. |

---

## Comparison: 4 Pipelines x 7 Critical Questions

| Question | PartialSpoof | LlamaPartialSpoof | HAD | HQ-MPSD |
|----------|-------------|-------------------|-----|---------|
| Zero-gap handling | N/A (VAD margins) | Not specified | N/A (1 replacement/utt) | Not specified |
| Duration mismatch | Selects similar-duration segments | Not specified | Not specified | Not specified |
| F0 discontinuity | Not specified | Not specified | Not specified | Not specified |
| Spectral envelope | ITU-T SV56 (-26 dBov) | Loudness norm only | Volume norm only | "Loudness + spectral alignment" (no detail) |
| Cut placement | VAD boundaries | MFA word boundaries | Character-level timestamps | Word midpoints |
| Code released | No (listed "TBA") | No (metadata only) | No (audio only) | No |
| Perceptual metric | None | EER per concat method (Table V) | EER only | DNSMOS 3.58 |

---

## Related Datasets (Multilingual and Spanish)

### LRLSpoof

Low-resource language spoofing dataset. Targets languages underrepresented in ASVspoof. Relevant as a reference for methodology when building datasets for non-English languages.

### SpeechFake-MD

Multi-domain speech deepfake detection dataset. Covers multiple recording conditions and synthesis methods. Relevant for the generalization analysis.

### HISPASpoof

Spanish-language anti-spoofing dataset. Directly relevant to this thesis as one of the few datasets targeting Spanish voice anti-spoofing.

### ML-ITW (Multilingual In-The-Wild)

| Attribute | Value |
|-----------|-------|
| **Languages** | 14 languages |
| **Platforms** | 7 platforms |
| **Duration** | 28.4 hours |
| **Notes** | In-the-wild benchmark (2026). Relevant for evaluating detector generalization across languages and conditions. |

### Adjacent Datasets

- **LAV-DF / AV-Deepfake1M** (Cai et al., DICTA 2022 / ACM MM 2024) -- audio-visual deepfake detection
- **Psynd** (Zhang & Sim, ICPR 2022) -- localizing fake segments in speech
- **ADD 2022/2023** (Yi et al., ICASSP 2022/2023) -- audio deepfake detection challenges

---

## Key Observations

1. **No dataset releases splicer code.** All four canonical partial-spoof datasets provide audio and/or metadata but not the waveform-level splicing implementation. This is the single largest reproducibility gap in the field.

2. **English dominance.** Three of four canonical datasets are English-only. HAD is Chinese. HQ-MPSD is multilingual but details are limited. No canonical partial-spoof dataset targets Latin American Spanish.

3. **Sparse waveform detail.** Only LlamaPartialSpoof names concrete waveform parameters (30-80 ms overlap, 5 fade shapes). The others describe their method at a high level without implementation-grade specifics.

4. **No F0 or spectral smoothing.** No partial-spoof dataset reports F0 discontinuity handling, formant matching, or spectral envelope interpolation at splice boundaries.

5. **Generalization crisis.** Muller et al. (Interspeech 2022) showed that detectors trained on ASVspoof 2019 degrade by 200-1000% EER on in-the-wild audio. Dataset-specific shortcuts (e.g., leading silence length correlating with class) dominate over intrinsic synthesis difficulty.

---

## Cross-references

- [Partial Spoof Literature](partial-spoof-literature.md) -- full literature review of the 4 pipelines
- [Splicing Techniques](splicing-techniques.md) -- the 7 techniques and their parameters
- [Detection Methods](detection-methods.md) -- countermeasure architectures evaluated on these datasets
- [TTS Systems](tts-systems.md) -- the TTS systems used to generate attack audio
