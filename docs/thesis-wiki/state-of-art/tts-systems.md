# TTS Systems for Voice Anti-Spoofing Attack Generation

**Status:** Active
**Last updated:** 2026-05-01
**Source:** investigation.md Sections 1-7, Comparative Analysis, Final Recommendations

---

## Overview

Seven modern TTS systems were evaluated for their suitability in generating synthetic Latin American Spanish voice attacks for anti-spoofing detection research. The evaluation focused on three criteria: (1) Spanish language quality (mandatory), (2) implementation feasibility, and (3) research usefulness (attack sophistication, codec diversity). Hardware target: ml-server03 with 4x NVIDIA A40 GPUs (46 GB VRAM each, CUDA 12.6).

Five systems are validated and in production; OmniVoice (k2-fsa) was added 2026-05-01 as a 6th attack pipeline pending standalone validation; CosyVoice 3.0 was dropped (generates Chinese for Spanish input); Nari Dia 1.6B was not implemented (English-only, dialogue-focused, no single-speaker TTS).

---

## 1. Fish Speech (OpenAudio-S1) -- PRIMARY

**Architecture:** Dual-AR + Firefly-GAN vocoder. Two autoregressive models (semantic + acoustic) plus GAN-based vocoder. 4 billion parameters. Trained with RLHF for human-perceived naturalness.

**Spanish support:** 20,000 hours of Spanish training data. Spanish listed in "high training data" tier alongside English and Chinese. Emotion control and fine-grained prosody in Spanish. Cross-lingual voice cloning from 10-30 seconds reference audio.

**VRAM:** 12 GB minimum (26% of A40 capacity).

**License:** CC-BY-NC-SA-4.0 (academic research permitted; non-commercial only).

**Key strengths:**
- Best validated Spanish support among all evaluated systems
- Most sophisticated attack modeling (RLHF, 4B parameters)
- Docker deployment available; 24,906 GitHub stars; published paper (arXiv:2411.01156)
- Batch generation: ~2 s per 10 s audio on A40; ~33 min for 1,000 samples

**Key problems:**
- 12 GB VRAM footprint (irrelevant on A40 but worth noting)
- Voice cloning inconsistency on unusual accents (potentially beneficial -- creates challenging training data)
- No specific Latin American sub-dialect benchmarks published
- Streaming latency issues (irrelevant for batch generation)

**Risk level:** LOW
**Work hours:** 50-80 h | Calendar: 10-15 weeks at 6 hrs/week

---

## 2. Qwen3-TTS (Alibaba) -- SECONDARY

**Architecture:** Dual-Track design processing semantic content and acoustic features in parallel, then merging. Two tokenizers (25 Hz high-quality, 12 Hz streaming). Three variants: Base (0.6B/1.7B), CustomVoice, VoiceDesign. 1.7 billion parameters (recommended variant).

**Spanish support:** Spanish is explicitly "second-tier" in the research paper. Not listed in the top-performing language group (Chinese, English, Italian, French, Korean, Russian). Paper states "competitive" -- diplomatic for "good but not best." User reports of slight Asian accent in English suggest similar issues for Spanish.

**VRAM:** 4-8 GB (1.7B model).

**License:** Apache 2.0 (unrestricted).

**Key strengths:**
- Fastest inference: ~0.8 s per 10 s audio; ~20-30 min for 1,000 samples
- Easy installation (pip install qwen-tts)
- Different codec architecture from Fish Speech -- improves detector generalization
- FlashAttention 2 support (30-40% speedup)

**Key problems:**
- Fine-tuning is broken: progressive speech speedup, checkpoint corruption, speaker encoder deletion
- Audio artifacts: truncated outputs, silent failures, stray Chinese characters in metadata
- Hard pin on transformers==4.57.3 (dependency conflicts)
- 0.6B model: embedding dimension mismatch bug (2048 vs 1024) blocks fine-tuning
- Spanish quality mediocre -- suitable for diversity, not as sole primary TTS

**Risk level:** MEDIUM
**Work hours:** 30-50 h | Calendar: 7-10 weeks at 6 hrs/week

---

## 3. OpenVoice V2 (MyShell AI) -- THIRD PIPELINE

**Architecture:** Decoupled two-component design. Base TTS (MeloTTS) generates speech in target language with a base speaker profile. Separate Tone Color Converter (2D CNN on mel-spectrogram) transplants reference speaker's timbre. Vocoder: HiFi-GAN (GAN-based mel-spectrogram vocoder). Total parameters: ~150-200M. Published paper: arXiv:2312.01479.

**Spanish support:** MeloTTS provides native Spanish support (dedicated language model, not cross-lingual transfer). However, Tone Color Converter training data is ~60% English, ~20% Chinese, ~20% Japanese -- zero Latin American Spanish speakers. Systematic accent flattening confirmed: regional Latin American accents converge toward neutral output after tone color conversion.

**VRAM:** 4-8 GB.

**License:** MIT (fully open, no restrictions).

**Key strengths:**
- Third distinct vocoder architecture (HiFi-GAN) -- maximizes codec diversity alongside VQGAN (FishGram) and Dual-Track codec (Qwen)
- Minimal VRAM; local Python inference; no server required
- Fast inference: ~12x real-time on A10G
- Active community (myshell-ai/OpenVoice)

**Key problems:**
- Systematic Latin American accent flattening (must disclose in thesis methodology)
- Older architecture (VITS + HiFi-GAN, 2021-era) -- lower naturalness ceiling
- No pip install: requires git clone + editable install + separate MeloTTS install
- Zero published Spanish benchmarks
- Online vs open-source quality gap (proprietary post-processing not in open-source release)

**Risk level:** LOW-MEDIUM
**Work hours:** 20-35 h | Calendar: 4-7 weeks at 6 hrs/week

---

## 4. Chatterbox (Resemble.ai) -- TIER 4

**Architecture:** 350 million parameters. Single-step mel-spectrogram decoder. Perth neural watermarking mandatory on all outputs (cannot be disabled). Developed by Resemble.ai as open-source version of commercial product.

**Spanish support:** Spanish listed among 23 supported languages. No benchmarks, no audio samples, no user reviews of Spanish quality. Documentation warns "May inherit accent from reference clip."

**VRAM:** 8 GB.

**License:** MIT (fully open).

**Key strengths:**
- Easiest installation of all systems (pip install chatterbox-tts)
- Outperformed ElevenLabs Turbo in blind tests (63.75% preference)
- Paralinguistic features: [laugh], [sigh], [cough], [hesitation]
- Shortest voice cloning reference: 5 seconds
- Smallest VRAM footprint -- could run 5+ instances on single A40

**Key problems:**
- Mandatory Perth watermarking: potential confounding variable (detector could learn watermark pattern instead of TTS artifacts). Must disclose in thesis and isolate as separate experimental group
- Latency fraud: claims <200 ms, actual 300-600 ms (irrelevant for batch generation but indicates documentation culture)
- CPU inference broken despite being documented as supported
- 224 open GitHub issues vs 34 commits (6.6 issues/commit -- high bug density)
- No Latin American Spanish validation

**Risk level:** MEDIUM
**Work hours:** 20-35 h | Calendar: 5-8 weeks at 6 hrs/week

---

## 5. OuteTTS -- TIER 4 (BACKGROUND)

**Architecture:** LLM-based TTS built on Qwen3 0.6B or Llama 3.2-1B. Treats speech synthesis as language modeling (next audio token prediction). DAC (Descript Audio Codec) for audio encoding (2 codebooks). Speaker profiles stored as JSON metadata.

**Spanish support:** 20,000 hours (0.6B) / 60,000 hours (1B) -- listed in "high training data" tier. However, performance disaster negates any quality advantage.

**VRAM:** 6-12 GB.

**License:** Apache 2.0 (0.6B); CC-BY-NC-SA-4.0 (1B).

**Key strengths:**
- llama.cpp compatibility enables CPU inference (slow but hardware-flexible)
- Speaker profiles as JSON (easy to version-control)
- Unique LLM-based architecture adds maximum codec diversity
- Hosted API available ($0.0006/second)

**Key problems:**
- Performance catastrophe: 3 minutes to generate 14 seconds on RTX 4090. Estimated 1.5-2.5 days for 1,000 samples on A40 -- 76x slower than Fish Speech
- CPU inference fraud: claims "real-time on CPUs," actual 2.5-7.5x slower than real-time
- DAC codec quality issues: lossy reconstruction, sensitive to input audio quality
- Sampling configuration fragility: repetition penalty must apply to exactly 64-token window -- misconfiguration produces garbled output silently
- Audio truncation (Issue #45); attention mask warnings (Issue #3)
- Context window limit: 8,192 tokens (~32 seconds effective with speaker reference)

**Risk level:** HIGH for large-scale; LOW for small supplementary experiments
**Work hours:** 15-30 h | Calendar: 4-7 weeks + 1-2 days server time per batch

---

## 6. CosyVoice 3.0 (Alibaba) -- DROPPED

**Architecture:** Conditional Flow Matching (CFM) generative model. Chunk-aware processing. Supervised semantic tokens from Whisper ASR. Matcha-TTS integration required. Trained on 1 million hours of speech data.

**Spanish support:** COMPLETELY UNKNOWN. Zero Spanish benchmarks published. All metrics focus on Chinese and English. Spanish listed as "supported" with no further detail.

**VRAM:** 8-16 GB.

**License:** Apache 2.0.

**Why dropped:** During testing, CosyVoice generated Chinese-language audio when given Spanish text input. Additionally:
- vLLM dependency hell: only compatible with 0.9.0 OR 0.11.x+ (NOT 0.10.x)
- Quality regression reports: community consensus that CosyVoice 2 sounded better than v3
- Python 3.10 required; Matcha-TTS submodule must be initialized separately
- 1-2 day deployment time (experienced engineer)
- No Spanish benchmarks to justify the investment

**Risk level:** HIGH
**Status:** Dropped from implementation

---

## 7. Nari Dia 1.6B -- NOT IMPLEMENTED

**Architecture:** 1.6 billion parameters. Dialogue-focused training. Speaker tags [S1]/[S2] for multi-speaker. Optimized for 5-20 second conversational chunks.

**Spanish support:** NONE. English language generation only. Official statement: "Dia currently supports English language generation only." Next language targets are Asian languages, not Spanish. No timeline for Spanish support.

**VRAM:** ~10 GB.

**License:** Apache 2.0.

**Why not implemented:** Fundamental capability gap -- cannot produce Spanish speech. Additionally, dialogue-focused design ([S1]/[S2] format) is architecturally mismatched for single-speaker batch augmentation. Included in evaluation for due diligence only.

**Possible future role:** English-language baseline experiments for cross-lingual detector comparisons, if thesis scope expands.

**Status:** Not implemented (English only)

---

## Comparative Summary Matrix

| System | Spanish Quality | VRAM | License | Perf (per 10s) | Key Constraint | Status |
|--------|----------------|------|---------|----------------|----------------|--------|
| **Fish Speech** | Good (20k hrs) | 12 GB | CC-BY-NC-SA-4.0 | ~2 s | No LatAm dialect benchmarks | PRIMARY |
| **Qwen3-TTS** | Mediocre | 4-8 GB | Apache 2.0 | ~0.8 s | Fine-tuning broken | SECONDARY |
| **OpenVoice V2** | Moderate (accent flattening) | 4-8 GB | MIT | ~0.5-1 s | Accent flattening -- must disclose | THIRD PIPELINE |
| **Chatterbox** | Unvalidated | 8 GB | MIT | ~4-6 s | Mandatory watermark -- must disclose | TIER 4 |
| **OuteTTS** | Adequate (on paper) | 6-12 GB | Apache 2.0 | **2-4 min** | 76x slower than Fish Speech | BACKGROUND |
| **CosyVoice** | Unknown | 8-16 GB | Apache 2.0 | ~1.5 s | Generates Chinese for Spanish input | DROPPED |
| **Nari Dia** | None (English only) | 10 GB | Apache 2.0 | N/A | English only, dialogue-only | NOT IMPLEMENTED |

## Batch Generation Performance (1,000 samples, 10 seconds each)

| System | Time Required | Cost | Reliability |
|--------|--------------|------|-------------|
| **Fish Speech** | ~33 minutes | Free | High |
| **Qwen3-TTS** | ~20-30 minutes | Free | Medium (artifacts) |
| **Chatterbox** | ~1.5-2 hours | Free | Medium (224 issues) |
| **OuteTTS (GPU)** | ~1.5-2.5 days | Free | Low |
| **OuteTTS (API)** | 10 days (100/day limit) | $360 | Low (early access) |

## Spanish Language Support Ranking

| Rank | System | Evidence | Confidence |
|------|--------|----------|------------|
| 1st | Fish Speech | 20k hours, high training tier, explicit support | HIGH |
| 2nd | Qwen3-TTS | Paper admits "not top-tier," user accent reports | MEDIUM |
| 3rd | OpenVoice V2 | MeloTTS native Spanish; accent flattening for LatAm | MEDIUM |
| 4th | OuteTTS | 20k-60k hours, high training tier (performance negates) | MEDIUM |
| 5th | Chatterbox | Listed in 23 languages, zero validation | LOW |
| 6th | CosyVoice | No benchmarks, no samples, generates Chinese | VERY LOW |
| 7th | Nari Dia | English only | N/A |

## Production Results (as of 2026-04-22)

| Pipeline | Samples Passed | Pass Rate | WER | NISQA MOS | ECAPA SIM |
|----------|---------------|-----------|-----|-----------|-----------|
| FishGram | 34,197 / 35,927 | 95.2% | 2.17% | 4.57 | 0.602 |
| Qwen3-TTS | 31,568 / 35,927 | 87.9% | 1.46% | 4.37 | 0.720 |
| OpenVoice | 29,626 / 35,544 | 83.4% | 1.50% | 4.41 | 0.394 |
| Chatterbox | Running | ~41% | -- | -- | -- |
| OuteTTS | Running | ~66% | -- | -- | -- |
| OmniVoice | Pending first run | -- | -- | -- | -- |

---

## 7. OmniVoice (k2-fsa) -- ADDED 2026-05-01

**Architecture:** Diffusion language model TTS, zero-shot voice cloning. State-of-the-art massively multilingual model from the k2-fsa group.

**Spanish support:** **27,559 hours** of Spanish training data (per k2-fsa/OmniVoice languages.md), one of the largest Spanish coverage among open zero-shot TTS models. ISO 639-1 code `es`, ISO 639-3 `spa`.

**Architecture details:** Diffusion-language-model with `num_step` (default 32 for higher quality, 16 for speed) and `speed` (default 1.0) generation parameters. Native sample rate **24 kHz** (resampled to 16 kHz on FLAC write in our pipeline). Reference duration: 3-10 seconds recommended (longer degrades cloning quality, per upstream docs). Inference: very fast, RTF approximately 0.025 per upstream benchmarks.

**License:** Open release on HuggingFace at `k2-fsa/OmniVoice`. Disclaimer prohibits unauthorized voice cloning, fraud, and impersonation; defensive anti-spoofing research is the legitimate use case.

**Key papers:** Zhu et al. 2026 "OmniVoice: Towards Omnilingual Zero-Shot Text-to-Speech with Diffusion Language Models" (arXiv:2604.00688).

**VRAM:** Float16 inference fits comfortably on a single A40.

**Production status:** Standalone pipeline written 2026-04-30. Validation run pending on ml-server03. Not yet included in boundary jitter pilot until quality is confirmed.

---

## Cross-references

- [Anti-Spoofing Datasets](anti-spoofing-datasets.md) -- datasets used to evaluate detection
- [Detection Methods](detection-methods.md) -- countermeasure architectures
- [Partial Spoof Literature](partial-spoof-literature.md) -- Section 8 literature review
- [Splicing Techniques](splicing-techniques.md) -- waveform-level splice methods
