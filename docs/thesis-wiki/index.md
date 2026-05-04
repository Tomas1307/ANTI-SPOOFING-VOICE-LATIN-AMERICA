# Thesis Wiki Index

**Project:** HABLA 2.0 — Voice Anti-Spoofing for Latin American Spanish
**Author:** Tomas Acosta
**Status:** Active

---

## State of the Art

| Page | Summary |
|------|---------|
| [TTS Systems](state-of-art/tts-systems.md) | 6 TTS systems evaluated: Fish Speech, Qwen3-TTS, OpenVoice, Chatterbox, OuteTTS, CosyVoice. Selection criteria, Spanish quality, trade-offs |
| [Anti-Spoofing Datasets](state-of-art/anti-spoofing-datasets.md) | ASVspoof 2019/2021, PartialSpoof, LlamaPartialSpoof, HAD, HQ-MPSD, LRLSpoof, SpeechFake-MD, HISPASpoof |
| [Detection Methods](state-of-art/detection-methods.md) | AASIST, RawNet3, SSL-based (wav2vec2, WavLM), ECAPA-TDNN for speaker verification |
| [Partial Spoof Literature](state-of-art/partial-spoof-literature.md) | 4 pipelines x 7 questions comparison, Negroni et al. spectral analysis, Huang et al. perceptual gap |
| [Splicing Techniques](state-of-art/splicing-techniques.md) | 7 crossfade methods, OLA-Hanning, energy valley analysis, literature gaps |

## Methodology

| Page | Summary |
|------|---------|
| [Pipeline Architecture](methodology/pipeline-architecture.md) | 7-step pipeline: Transcribe, Clone, Align, Select, Splice, Validate, Format. Facade + Strategy patterns |
| [Attack Systems](methodology/attack-systems.md) | Per-TTS configuration, venv setup, production run parameters |
| [Partial Spoof Approach](methodology/partial-spoof-approach.md) | Valley-score selection, duration-preserving splice, clone similarity gate, edge cases |
| [Dataset Design](methodology/dataset-design.md) | HABLA v2 (1,567 speakers, 7 accents), W1/W2/W3 tiers, bonafide ratio |
| [Quality Metrics](methodology/quality-metrics.md) | WER/CER thresholds, NISQA MOS, ECAPA SIM, valley score formula |

## Experiments

| Page | Summary |
|------|---------|
| [Production Runs](experiments/production-runs.md) | Per-pipeline status + **Operational Runbook** with exact ml-server03 commands, success criteria, and pending-work checklist (OmniVoice validation + boundary jitter pilot) — **read this first when continuing on a new machine** |
| [Validation Results](experiments/validation-results.md) | Partial spoof validation: 5 speakers, metrics comparison |
| [Ablation Studies](experiments/ablation-studies.md) | Crossfade technique comparison, valley score threshold tuning |

## Decisions

| Page | Summary |
|------|---------|
| [Decision Log](decisions/decision-log.md) | Chronological record of all architectural and research decisions |
