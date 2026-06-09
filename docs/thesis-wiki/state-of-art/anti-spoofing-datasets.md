# Anti-Spoofing Datasets

**Status:** Active
**Last updated:** 2026-05-25
**Source:** investigation.md Section 8.2, 8.8; general literature; verified bibliography pass 2026-05-25 (IEEE/references.bib)

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

### HABLA (Tamayo-Florez et al., 2023) -- bonafide foundation of this thesis

| Attribute | Value |
|-----------|-------|
| **Language** | Latin American Spanish |
| **Size (v1, published)** | ~22,000 bonafide samples (5 nations) + ~58,000 spoof samples from 6 synthesis methods |
| **Size (v2, used here)** | 1,567 speakers across 7 accents spanning TWO continents; ~35,927 bonafide utterances (16 kHz). Verified on ml-server03 2026-06-01. |
| **Code released** | Dataset on Zenodo (10.5281/...7370805); GitHub Ruframapi/HABLA |
| **Reference** | Tamayo-Florez, Manrique, Pereira Nunes, "HABLA: A Dataset of Latin American Spanish Accents for Voice Anti-Spoofing," Interspeech 2023, pp. 1963-1967. DOI:10.21437/Interspeech.2023-2272 |
| **Notes** | **Advisor Ruben Manrique is a co-author.** The published v1 paper describes 5 LatAm nations / ~22k bonafide. HABLA-Spoof (this work) is a deliberate EXTENSION of v1: the bonafide pool is expanded to 1,567 speakers / 7 accents ("v2"), and we add the full-synthesis attacks, the multi-system partial-spoof corpus, and boundary jitter. Cite v1 (`habla`) as prior work; position v2 + attacks as the novel extension. |

**HABLA v2 authoritative accent inventory** (verified on ml-server03, `data/bonafide_dataset_by_speaker_v2`, 2026-06-01):

| Code | Accent | Speakers | (m / f) | Share |
|------|--------|---------:|---------|------:|
| `es` | **Spain (Peninsular / European)** | **528** | 297 / 231 | 33.7% |
| `mx` | Mexico | 406 | 267 / 139 | 25.9% |
| `co` | Colombia | 357 | 270 / 87 | 22.8% |
| `cl` | Chile | 173 | 126 / 47 | 11.0% |
| `ar` | Argentina | 42 | 12 / 30 | 2.7% |
| `pe` | Peru | 38 | 20 / 18 | 2.4% |
| `ve` | Venezuela | 23 | 12 / 11 | 1.5% |
| **Total** | **7 accents, 2 continents** | **1,567** | | 100% |

**CRITICAL CORRECTIONS (2026-06-01):**
1. **HABLA v2 is CROSS-CONTINENTAL, not Latin-American-only.** Peninsular Spanish (`es`) is the LARGEST accent (33.7%) -- 6 LatAm accents + European Spanish. The paper's scope is cross-continental Spanish; European Spanish is a CONTRIBUTION, not future work.
2. **Prior records were WRONG.** Auto-memory said "7 Latin American accents (ar, co, mx, pe, cl, ve, cu)"; presentation slides (`00b/00c/12_hispaspoof`) say "7 LatAm accents" and "we include Venezuelan and Puerto Rican; HISPASpoof includes Peninsular." All false: there is NO Cuba (`cu`) or Puerto Rico (`pr`) code, and HABLA DOES include Peninsular (more than HISPASpoof). Slides need correcting before any external use.
3. **`transcriber.py` `country_map` is incomplete** (maps only ar/cl/co/pe/ve -> sends `es`/`mx` to "unknown"). Latent bug.
4. **Severe accent imbalance.** es+mx+co = 1,291 / 1,567 (82%); ar+pe+ve = 103 (6.6%). Per-accent EER for AR/PE/VE will be statistically fragile -- must be declared in Limitations. The `382:1 Mexico:Caribbean` ratio in `app/helpers/generate_cv_graphs.py` suggests HABLA v2 was sourced from Mozilla Common Voice (would explain the Spain plurality) -- confirm provenance.

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

### HISPASpoof (2025)

First large-scale Spanish dataset for synthetic-speech detection and attribution. Real speech across 6 accents + synthetic speech from 6 zero-shot TTS systems. Key finding directly supporting this thesis: **detectors trained on English fail to generalize to Spanish; training on HISPASpoof substantially improves detection.** The closest published competitor to HABLA-Spoof, but it targets full-synthesis (not partial spoof) and does not release a reproducible attack pipeline. Purdue VIPER Lab, CC-BY-SA 4.0 on HuggingFace.
**Reference:** "HISPASpoof: A New Dataset for Spanish Speech Forensics," arXiv:2509.09155, 2025. [author list TBD]

### SpeechFake / SpeechFake-MD (Huang, Gu et al., 2025)

Large-scale multilingual speech deepfake dataset: 3M+ fake samples, 3,000+ hours, 30 generation models (TTS/VC/NV) across 46 languages. Split into a Bilingual Dataset (BD: en/zh) and a Multilingual Dataset (**MD**, 46 languages) -- the MD subset is the "SpeechFake-MD" referenced for the generalization analysis. Spanish is one of the 46 but without Latin American sub-dialect targeting.
**Reference:** "SpeechFake: A Large-Scale Multilingual Speech Deepfake Dataset Incorporating Cutting-Edge Generation Methods," ACL 2025, arXiv:2507.21463.

### LRLSpoof (2026)

Large-scale multilingual synthetic-speech corpus for cross-lingual spoof detection: 2,732 hours, 24 open-source TTS systems, 66 languages (45 low-resource). Benchmarks 11 public countermeasures under language mismatch via threshold transfer; finds language is an independent source of domain shift -- another data point for the cross-corpus benchmarking pillar.
**Reference:** "When Spoof Detectors Travel: Evaluation Across 66 Languages in the Low-Resource Language Spoofing Corpus," arXiv:2603.02364, 2026. [author list TBD]

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
