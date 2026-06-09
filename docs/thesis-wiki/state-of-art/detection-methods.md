# Detection Methods for Voice Anti-Spoofing

**Status:** Active
**Last updated:** 2026-04-25
**Source:** investigation.md Sections 8.1, 8.3, 8.6; general literature references

---

## Overview

This page documents the detection and countermeasure methods referenced in the investigation, ranging from no-training spectral analysis to SSL-based neural detectors. The central tension in the field is the **generalization crisis**: detectors that achieve near-zero EER in controlled benchmarks (ASVspoof 2019 LA) degrade by 200-1000% on in-the-wild audio.

---

## No-Training Spectral Analysis

### Negroni et al. (2024) -- Splice Artifact Detection Without Training

**Paper:** "Analyzing the Impact of Splicing Artifacts in Partially Fake Speech Signals," ASVspoof Workshop 2024. arXiv:2408.13784.

**Method:** Purely hand-coded spectral-dynamic-range analysis of the splice join. No machine learning, no training data, no neural network. Analyzes spectral discontinuities at the boundary between bonafide and cloned segments.

**Results:**
- **6.16% EER on PartialSpoof** (English)
- **7.36% EER on HAD** (Chinese)

**Significance:** Demonstrates that a ~6% EER floor exists for detectors analyzing splice boundaries, achievable with zero training. This is the baseline that any splice concealment technique must overcome. The result also suggests that current partial-spoof datasets have artifacts detectable by simple signal analysis, raising questions about whether trained detectors are learning genuinely deep features or merely exploiting these surface-level discontinuities.

**OLA-Hanning experiments:** Negroni et al. tested OLA-Hanning windows of 256-4096 samples (16-256 ms at 16 kHz) to measure artifact mitigation:
- 256 samples: AUC = 98.04% (near-original detectability)
- 4096 samples: AUC = 88.99% (best mitigation tested, still highly detectable)
- Minimum 1024 samples (64 ms) needed for meaningful artifact reduction

See [Splicing Techniques](splicing-techniques.md) for full parameter details.

---

## SSL-Based Detectors (Self-Supervised Learning)

### wav2vec 2.0

**Architecture:** Self-supervised speech representation model (Facebook/Meta AI). Pre-trained on large unlabeled speech corpora to learn general-purpose speech features. Fine-tuned downstream for anti-spoofing classification.

**Relevance:** Used as a feature extractor in multiple anti-spoofing systems. The pre-trained representations capture phonetic, speaker, and acoustic features that can distinguish bonafide from spoofed speech without hand-crafted features.

### WavLM

**Architecture:** Self-supervised speech representation model (Microsoft). Similar to wav2vec 2.0 but trained with a denoising objective, making it more robust to real-world noise and channel conditions.

**Relevance:** Huang et al. (SLT 2024) showed that SSL-based detectors (including WavLM-based systems) can detect neural infilling edits that fool human listeners. This establishes the **perceptual-vs-detection asymmetry**: what sounds natural to humans may still carry statistical artifacts detectable by SSL models.

### The SSL Detection Paradigm

SSL-based detectors follow a common pattern:
1. Pre-trained SSL model (wav2vec 2.0, WavLM, HuBERT) extracts frame-level features
2. A lightweight classification head (linear layer, attention pooling, or small transformer) maps features to bonafide/spoof prediction
3. Fine-tuning on labeled anti-spoofing data

**Advantages:** Leverage representations learned from massive unlabeled speech corpora. Capture subtle statistical patterns that hand-crafted features miss. State-of-the-art on ASVspoof benchmarks.

**Limitations:** Generalization crisis (see below). Models may learn dataset-specific shortcuts rather than genuine synthesis artifacts.

---

## Standard Countermeasure: AASIST

**Full name:** Audio Anti-Spoofing using Integrated Spectro-Temporal features.

**Architecture:** Graph-based neural network operating on spectro-temporal representations. Uses graph attention layers to model both spectral patterns (within a frame) and temporal dynamics (across frames). Integrates raw waveform and spectral features.

**Status in the field:** De facto standard countermeasure architecture for anti-spoofing evaluation. Used as a benchmark system in ASVspoof challenges. When new partial-spoof datasets or attacks are introduced, AASIST is typically the first detector evaluated against them.

**Relevance to this thesis:** AASIST will likely be one of the primary countermeasure systems evaluated against the HABLA 2.0 attack dataset.

---

## Speaker Verification: ECAPA-TDNN

**Full name:** Emphasized Channel Attention, Propagation and Aggregation in Time Delay Neural Networks.

**Architecture:** TDNN (Time Delay Neural Network) with squeeze-and-excitation blocks, multi-layer feature aggregation, and channel-dependent attention. Produces fixed-dimensional speaker embeddings from variable-length utterances.

**Role in this thesis:** Not a spoofing detector per se, but used for **speaker similarity (SIM) scoring** in the attack pipeline. After generating a cloned voice, ECAPA-TDNN cosine similarity between the original speaker embedding and the cloned audio embedding measures how well the TTS preserved speaker identity.

**Production metrics (as of 2026-04-22):**
- FishGram SIM: 0.602
- Qwen3-TTS SIM: 0.720
- OpenVoice SIM: 0.394

**Threshold:** SIM score used as a quality gate in the pipeline. Low SIM indicates the TTS failed to capture the target speaker's voice characteristics.

---

## The Generalization Crisis

### Muller et al. (Interspeech 2022)

**Paper:** "Does Audio Deepfake Detection Generalize?"

**Key finding:** Detectors trained on ASVspoof 2019 degrade by **200-1000% EER** on in-the-wild audio. This is not a marginal degradation -- it represents a fundamental failure of generalization.

**Root cause analysis (Muller 2024):** "Harder or Different? Understanding Generalization of Audio Deepfake Detection." The "difference" term (distributional mismatch between training and test conditions) dominates over "hardness" (intrinsic difficulty of detecting the fake). Models learn dataset-specific artifacts rather than synthesis properties.

**Dataset shortcuts identified:**
- **Silence shortcut:** Length of leading silence correlates with bonafide/spoof class in ASVspoof 2019
- **Channel artifacts:** Recording conditions differ systematically between bonafide and spoof partitions
- **Duration bias:** Spoofed utterances may have systematically different durations than bonafide

**Implication for HABLA 2.0 construction:** The multi-technique splicing approach with varied crossfade durations, randomized parameters, and diverse TTS systems is directly motivated by this finding. If the dataset uses a single splicing technique with fixed parameters, detectors will learn to detect that specific technique rather than partial spoofing in general.

### Perceptual-vs-Detection Gap (Huang et al., SLT 2024)

**Paper:** "Detecting the Undetectable: Assessing Efficacy of Spoof Detection Against Seamless Speech Edits." arXiv:2501.03805.

**Key finding:** Neural infilling edits that fool human listeners are still detected by SSL-based systems. Conversely, some artifacts that machines miss are audible to humans. The perceptual frontier (MOS) and the detection frontier (EER) are not aligned.

**Implication:** The goal of partial-spoof dataset construction is not to create perfectly imperceptible edits (humans can already be fooled) but to create diverse artifacts that challenge automated detectors across multiple dimensions.

---

## Detection Method Summary

| Method | Type | Training Required | Key Result | Reference |
|--------|------|-------------------|------------|-----------|
| Spectral-dynamic-range analysis | Hand-coded | None | 6.16% EER on PartialSpoof | Negroni et al. 2024 |
| SSL + classification head | Neural (fine-tuned) | Yes (labeled data) | SOTA on ASVspoof benchmarks | Various |
| AASIST | Neural (graph-based) | Yes (labeled data) | Standard countermeasure | Jung et al. |
| ECAPA-TDNN | Neural (speaker verification) | Pre-trained | SIM scoring for quality gates | Desplanques et al. |

---

## Open Questions for This Thesis

1. **How do different splicing techniques affect detector EER?** No published ablation exists for the 7-technique set we propose. See [Splicing Techniques](splicing-techniques.md).

2. **Does multi-TTS diversity improve detector generalization?** Training on FishGram + Qwen + OpenVoice artifacts vs training on a single TTS. See [TTS Systems](tts-systems.md).

3. **Does the Chatterbox watermark act as a confounding variable?** Detector may learn to detect Perth watermark rather than TTS artifacts. See [TTS Systems -- Chatterbox section](tts-systems.md).

4. **What is the in-the-wild degradation for a detector trained on HABLA 2.0?** Given Muller et al.'s findings, how does multi-technique, multi-TTS training compare to single-technique approaches on out-of-distribution test sets?

---

## Cross-references

- [Splicing Techniques](splicing-techniques.md) -- the artifacts these detectors must find
- [Partial Spoof Literature](partial-spoof-literature.md) -- literature review of detection results
- [Anti-Spoofing Datasets](anti-spoofing-datasets.md) -- datasets used for detector evaluation
- [TTS Systems](tts-systems.md) -- attack systems that generate the spoofed audio
