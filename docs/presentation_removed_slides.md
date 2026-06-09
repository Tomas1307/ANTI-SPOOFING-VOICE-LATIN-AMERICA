# Presentation - Archived Slides Content

> **Note:** These slides were removed from `presentation.html` during a restructuring pass to
> streamline the deck. The technical details they contained are preserved here for reference,
> reproducibility documentation, and future use.

---

## Table of Contents

1. [Five Pipelines Overview](#1-five-pipelines-overview)
2. [Step 2a: Reference Embeddings (ECAPA-TDNN)](#2-step-2a-reference-embeddings)
3. [Step 2b: Speaker Validation (Cosine Similarity Filtering)](#3-step-2b-speaker-validation)
4. [Step 3: NISQA MOS (Non-Intrusive Speech Quality)](#4-step-3-nisqa-mos)
5. [Step 4: Speaker Similarity (SIM) - ECAPA-TDNN Architecture](#5-step-4-speaker-similarity-sim)
6. [Step 5: WER/CER Validation](#6-step-5-wercer-validation)
7. [FishGram Pipeline (5 Steps)](#7-fishgram-pipeline)
8. [Qwen3-TTS Pipeline (5 Steps)](#8-qwen3-tts-pipeline)
9. [Chatterbox Pipeline (5 Steps)](#9-chatterbox-pipeline)
10. [OpenVoice Pipeline (5 Steps)](#10-openvoice-pipeline)
11. [Demo: Chatterbox](#11-demo-chatterbox)
12. [Demo: OpenVoice](#12-demo-openvoice)

---

## 1. Five Pipelines Overview

The HABLA Anti-Spoofing project generates synthetic voice attacks through **five distinct pipelines**, each contributing a different TTS architecture to ensure vocoder diversity in the spoofed dataset:

| # | Pipeline | TTS System | Architecture Type | Parameters | Audio ID Range |
|---|----------|------------|-------------------|------------|----------------|
| 0 | **Bonafide Expansion** | N/A (real speech) | Mozilla Common Voice augmentation | -- | Original IDs |
| 1 | **FishGram** | Fish Speech 4B | VQGAN + Dual AR (codec-based) | 4B | 9,000,000+ |
| 2 | **Qwen3-TTS** | Qwen3-TTS 1.7B | Dual-Track (LLM + subtalker codec) | 1.7B | 8,000,000+ |
| 3 | **Chatterbox** | Chatterbox Multilingual 500M | Flow Matching + VoiceCraft | 500M | 6,000,000+ |
| 4 | **OpenVoice** | OpenVoice V2 (MeloTTS + ToneColorConverter) | VITS + HiFi-GAN tone conversion | ~100M | 7,000,000+ |

**Design rationale:** Four distinct vocoder architectures (VQGAN, Dual-Track LLM, Flow Matching, VITS+HiFi-GAN) ensure the anti-spoofing detector is trained against diverse synthesis artifacts rather than overfitting to a single TTS paradigm.

Each attack pipeline follows the same 5-step canonical structure (Facade + Strategy pattern):

1. **Prepare References** -- Build speaker reference audio clips from bonafide training samples
2. **Prepare Texts** -- Assign Spanish text prompts from Mozilla Common Voice
3. **Generate Speech** -- Synthesize voice-cloned audio using the target TTS system
4. **Validate Quality** -- Automated quality filtering (WER/CER, duration, artifacts)
5. **Format Output** -- Convert to ASVspoof2019 LA protocol format

---

## 2. Step 2a: Reference Embeddings

### ECAPA-TDNN 192-Dimensional Speaker Embeddings

For each of the 162 HABLA speakers, a fixed-length **192-dimensional speaker embedding** is extracted using the ECAPA-TDNN model (SpeechBrain pretrained on VoxCeleb1+2).

**Process:**

1. Load the speaker's training audio files (sorted alphabetically for determinism)
2. Concatenate up to 5 samples with 0.1s silence padding between clips
3. Trim/pad to the target reference duration (10s for Chatterbox, 15s for others)
4. Extract the 192-dim embedding vector via ECAPA-TDNN forward pass
5. Store as `{speaker_id}_ref.wav` alongside the embedding metadata

**Key details:**

- **Model:** ECAPA-TDNN (Emphasized Channel Attention, Propagation and Aggregation in TDNN)
- **Output dimensionality:** 192 floats (L2-normalized)
- **Training data:** VoxCeleb1 + VoxCeleb2 (7,000+ speakers)
- **Speakers processed:** 162 (HABLA bonafide set across 7 Latin American accents)

---

## 3. Step 2b: Speaker Validation

### Cosine Similarity Candidate Filtering

Before generating synthetic attacks, the pipeline validates that the reference embedding is usable for voice cloning by measuring how well the reference audio captures the speaker's identity.

**Metric:** Cosine similarity between the reference embedding and embeddings extracted from individual bonafide utterances of the same speaker.

**Formula:**

```
SIM(a, b) = (a . b) / (||a|| * ||b||)
```

Where `a` is the reference embedding (concatenated clip) and `b` is each individual utterance embedding.

**Filtering criteria:**

- If the mean intra-speaker similarity falls below a threshold, the speaker is flagged for review
- This catches cases where the reference clip is corrupted, too short, or captures cross-talk
- Candidates passing the filter proceed to TTS generation

---

## 4. Step 3: NISQA MOS

### Non-Intrusive Speech Quality Assessment

**NISQA** (Non-Intrusive Speech Quality Assessment) is a deep-learning-based model that predicts **Mean Opinion Score (MOS)** from a speech signal without requiring a clean reference. This makes it suitable for evaluating synthetic speech where no paired clean reference exists.

**How it works:**

1. The input audio waveform is transformed into a mel-spectrogram representation
2. A CNN-based feature extractor processes the spectrogram into frame-level features
3. An attention-pooling mechanism aggregates frame features into a single utterance-level representation
4. A regression head predicts the MOS on a 1-5 scale

**MOS Scale Interpretation:**

| Score | Quality | Description |
|-------|---------|-------------|
| 4.5 - 5.0 | Excellent | Indistinguishable from natural speech |
| 3.5 - 4.5 | Good | Minor artifacts, still natural-sounding |
| 2.5 - 3.5 | Fair | Noticeable artifacts but intelligible |
| 1.5 - 2.5 | Poor | Significant distortion, hard to understand |
| 1.0 - 1.5 | Bad | Unintelligible or severely corrupted |

**Usage in the pipeline:**

- NISQA MOS is computed as an **informational metric** for each generated sample
- Minimum acceptable threshold: 2.5 (configured via `NISQA_MIN_ACCEPTABLE` in each pipeline's settings)
- Currently used for logging and analysis, not as a hard rejection gate
- Enables comparison of perceptual quality across different TTS systems

---

## 5. Step 4: Speaker Similarity (SIM)

### ECAPA-TDNN Architecture Details

The speaker similarity metric measures how well the synthetic voice preserves the target speaker's identity. It uses the same ECAPA-TDNN model used for reference embedding extraction.

#### Architecture Overview

ECAPA-TDNN (Emphasized Channel Attention, Propagation and Aggregation in Time Delay Neural Network) is the state-of-the-art speaker verification architecture, extending the standard TDNN/x-vector framework with three key innovations:

**1. SE-Res2Net Blocks (Squeeze-and-Excitation with Res2Net)**

- **Res2Net** splits each residual block's channels into multiple groups processed at different scales, capturing multi-scale temporal features within a single layer
- **Squeeze-and-Excitation (SE)** adds a channel recalibration mechanism:
  1. Global average pooling squeezes temporal information into a channel descriptor
  2. Two FC layers learn channel-wise attention weights
  3. These weights rescale channel features, emphasizing the most discriminative ones
- Each SE-Res2Net block thus captures both multi-resolution temporal patterns and channel-wise importance

**2. Multi-Layer Feature Aggregation (MFA)**

- Traditional x-vector systems only use the output of the final frame-level layer
- ECAPA-TDNN aggregates features from **all** SE-Res2Net blocks (not just the last one)
- Features from different depths are concatenated before the pooling layer
- This captures both low-level acoustic details (early layers) and high-level speaker characteristics (deep layers)

**3. Attentive Statistical Pooling (ASP)**

- Standard statistical pooling computes a fixed mean and standard deviation across all frames
- Attentive statistical pooling learns a **frame-level attention mechanism** that weights each frame's contribution:
  1. A small attention network assigns an importance score to each frame
  2. Weighted mean and weighted standard deviation are computed using these scores
  3. This allows the model to focus on the most speaker-discriminative frames and suppress noise/silence
- The resulting utterance-level representation is a **192-dimensional embedding**

#### Similarity Computation

```
SIM(ref, syn) = cosine_similarity(ECAPA(ref_audio), ECAPA(syn_audio))
```

| SIM Range | Interpretation |
|-----------|---------------|
| 0.80 - 1.00 | Very high similarity (near-identical speaker identity) |
| 0.65 - 0.80 | Good clone quality (same speaker perceived) |
| 0.50 - 0.65 | Moderate similarity (some speaker drift) |
| < 0.50 | Poor clone (different speaker perceived) |

**Usage in the pipeline:**

- Informational metric (configured via `SPEAKER_SIM_MIN_ACCEPTABLE`, default 0.70)
- Logged per sample and averaged per speaker for analysis
- Enables ranking of TTS systems by voice cloning fidelity

---

## 6. Step 5: WER/CER Validation

### Word Error Rate and Character Error Rate Measurement

The final validation step transcribes each synthetic audio sample using an ASR model and compares the transcription against the original text prompt to detect intelligibility failures.

**ASR Model:** NVIDIA Parakeet TDT 0.6B v3

- 600M parameters, Token-and-Duration Transducer architecture
- Supports 25 languages including Spanish
- Spanish benchmarks: 3.45% WER (FLEURS), 4.39% WER (MLS), 3.41% WER (CoVoST2)
- Caveat: benchmarks measured on European Spanish; Latin American accents may show higher WER

**Word Error Rate (WER):**

```
WER = (Substitutions + Insertions + Deletions) / Total Reference Words
```

Measures the proportion of words that were incorrectly transcribed relative to the original text prompt.

**Character Error Rate (CER):**

```
CER = (Substitutions + Insertions + Deletions) / Total Reference Characters
```

Finer-grained metric that operates at the character level, more robust for agglutinative or morphologically rich languages.

**Rejection Thresholds:**

| Metric | Threshold | Action |
|--------|-----------|--------|
| WER | > 15% | Sample rejected |
| CER | > 10% | Sample rejected |
| Duration | < 0.5s or > 30s | Sample rejected |

**Spurious Prefix Trimming:**

Some TTS systems prepend metadata-like text (e.g., speaker tags, language codes) to their output. The validator detects and trims these prefixes before computing WER/CER, preventing false rejections.

**Expected pass rates:** 85-95% depending on the TTS system and text complexity.

---

## 7. FishGram Pipeline

### Overview

- **TTS System:** Fish Speech 4B (VQGAN + Dual Autoregressive codec)
- **Inference:** External HTTP API server on ml-server03 (decoupled from pipeline process)
- **System ID:** `FISHGRAM`
- **Audio ID Range:** 9,000,000+

### Detailed Steps

| Step | Class | Description |
|------|-------|-------------|
| 1 | `ReferenceAudioPreparator` | Concatenate up to 5 training samples per speaker with 0.1s silence padding, trim to 15.0s, save as `{speaker_id}_ref.wav`. Alphabetical file sort for determinism. |
| 2 | `TextPromptPreparator` | Load Mozilla CV Spanish transcripts (15,000+ unique). Filter by length (5-100 words). Seeded random sampling (N texts per speaker). Output: `text_prompts.json`. |
| 3 | `SpeechGenerator` | Health-check Fish Speech server (`GET /`). For each speaker-text pair: read reference audio, base64-encode, `POST /v1/tts`. Save returned audio as WAV. Track Real-Time Factor (RTF). |
| 4 | `QualityValidator` | Transcribe with Parakeet TDT 0.6B v3. Compute WER/CER with spurious prefix trimming. Reject samples exceeding 15% WER or 10% CER. Duration bounds: 0.5-30s. Compute NISQA MOS and speaker SIM (informational). |
| 5 | `OutputFormatter` | Convert WAV to FLAC (16kHz, PCM_16). Generate audio IDs (`LA_T_9000001`, etc.). Write ASVspoof2019 LA protocol files. Organize into train/dev/eval splits. |

**Fish Speech API Integration:**

- Health check: `GET /` (verifies server is running)
- Generation: `POST /v1/tts` (text + base64-encoded reference audio)
- Parameters: `top_p=0.8`, `temperature=0.8`, `repetition_penalty=1.1`
- Server persists across multiple pipeline runs, avoiding GPU memory allocation overhead

**Performance (NVIDIA A40):**

| Mode | Speakers | Samples | Estimated Time |
|------|----------|---------|----------------|
| Validation | 3 | 6 | ~4 minutes |
| Production | 162 | 810 | ~61 minutes |

---

## 8. Qwen3-TTS Pipeline

### Overview

- **TTS System:** Qwen3-TTS 1.7B (Dual-Track: LLM talker + subtalker codec decoder)
- **Inference:** Local model (loaded directly into Python, no HTTP server)
- **System ID:** `QWEN3TTS`
- **Audio ID Range:** 8,000,000+

### Detailed Steps

| Step | Class | Description |
|------|-------|-------------|
| 1 | `ReferenceAudioPreparator` | Same concatenation as FishGram (15s clips), **plus** transcribes reference audio with faster-whisper (large-v3) to provide `ref_text` for full voice cloning mode (not x-vector-only). Whisper model loaded as lazy singleton. |
| 2 | `TextPromptPreparator` | Same Mozilla CV loading, but with **stricter text length filtering: 5-40 words** (conservative ceiling to prevent Qwen3-TTS silent truncation on long texts). |
| 3 | `SpeechGenerator` | Load Qwen3-TTS 1.7B model to GPU. Per speaker: call `create_voice_clone_prompt()` once (reuse for all N utterances). Per text: `generate_voice_clone()` with full sampling params (top_k=50, top_p=1.0, temp=0.9). Release model after completion (`torch.cuda.empty_cache()`). |
| 4 | `QualityValidator` | Standard Parakeet WER/CER validation, **plus Qwen-specific artifact detection**: (a) Duration anomaly: reject < 0.5s or > 30s; (b) Low energy: reject if RMS < 0.001 (near-silent outputs); (c) Truncation detection: reject if audio suspiciously short for text length (< 1.5 words/second speaking rate). |
| 5 | `OutputFormatter` | Convert to FLAC, generate IDs from 8,000,000+, write protocol files with `QWEN3TTS` system ID. |

**Key Architectural Decisions:**

- **Local model** (not HTTP): Model loaded in Step 3, GPU memory freed after generation
- **Speaker prompt reuse**: `create_voice_clone_prompt()` pre-computes speaker features once, reused for all N utterances per speaker
- **STT transcription in Step 1**: faster-whisper transcribes reference audio for `ref_text` parameter, enabling full voice cloning mode (significantly higher quality than embedding-only mode)
- **Requires** `transformers==4.57.3` (hard pin)

---

## 9. Chatterbox Pipeline

### Overview

- **TTS System:** Chatterbox Multilingual TTS 500M (Flow Matching + VoiceCraft architecture)
- **Inference:** Fully local, no HTTP server. Model auto-downloaded from HuggingFace Hub.
- **System ID:** `CHATTERBOX`
- **Audio ID Range:** 6,000,000+

### Detailed Steps

| Step | Class | Description |
|------|-------|-------------|
| 1 | `ReferenceAudioPreparator` | Concatenate training samples, trim to **10.0s** (Chatterbox internally clips to 10s). Save as `{speaker_id}_ref.wav`. |
| 2 | `TextPromptPreparator` | Load Mozilla CV transcripts. Filter 5-100 words. Seeded random sampling. When `MATCH_BONAFIDE_COUNT=True`, generate as many samples as bonafide files per speaker. |
| 3 | `SpeechGenerator` | Load ChatterboxMultilingualTTS to GPU. Generate with `language_id='es'`, `exaggeration=0.5`, `CFG_weight=0.5`, `temperature=0.8`, `repetition_penalty=2.0`. **Perth watermark bypassed** via `perth_patcher.py` NoOpWatermarker (must be imported BEFORE `chatterbox.mtl_tts`). **Trailing noise artifact trimmed** via `speech_trimmer.py` gap-detection algorithm. |
| 4 | `QualityValidator` | Parakeet TDT transcription + WER/CER validation. Spurious prefix trimming. Duration bounds 0.5-30s. NISQA MOS and speaker SIM computed as informational metrics. |
| 5 | `OutputFormatter` | Convert to FLAC, IDs from 6,000,000+, protocol files with `CHATTERBOX` system ID. |

**Chatterbox-Specific Technical Details:**

- **Perth watermark bypass:** The native Perth watermark binary is broken; `perth_patcher.py` injects a NoOpWatermarker before Chatterbox imports, preventing watermark embedding for research validity
- **Trailing noise artifact:** Chatterbox generates loud noise (15-40% of peak amplitude) after speech ends. Fixed with a gap-detection trimmer that:
  1. Computes smoothed RMS energy envelope (25ms frames, 100ms rolling average)
  2. Finds all silence gaps (< 3% of peak, > 150ms duration)
  3. Selects the longest gap in the second half of the audio
  4. Verifies mean energy after gap < before gap (confirms noise, not speech)
  5. Trims at gap start + 150ms margin

---

## 10. OpenVoice Pipeline

### Overview

- **TTS System:** OpenVoice V2 (MeloTTS base synthesis + ToneColorConverter for voice cloning)
- **Inference:** Fully local, two-stage process (base TTS + tone conversion)
- **System ID:** `OPENVOICE`
- **Audio ID Range:** 7,000,000+

### Detailed Steps

| Step | Class | Description |
|------|-------|-------------|
| 1 | `ReferenceAudioPreparator` | Concatenate training samples, trim to **15.0s**. Save reference clips. MeloTTS requires `gruut-lang-es` + `espeak-ng` for Spanish phonemization (espeak-ng built from source on ml-server03). |
| 2 | `TextPromptPreparator` | Load Mozilla CV transcripts. Filter 5-100 words. Seeded random sampling. `MATCH_BONAFIDE_COUNT=True` by default. |
| 3 | `SpeechGenerator` | **Two-stage generation:** (1) MeloTTS generates base Spanish speech from text (`language='ES'`, `speed=1.0`); (2) ToneColorConverter transfers the target speaker's tone color onto the base audio (`tau=0.3`). Both models loaded to GPU during step, released after. |
| 4 | `QualityValidator` | Parakeet TDT transcription + WER/CER validation. Spurious prefix trimming. Duration bounds 0.5-30s. NISQA MOS and speaker SIM as informational metrics. |
| 5 | `OutputFormatter` | Convert to FLAC, IDs from 7,000,000+, protocol files with `OPENVOICE` system ID. |

**OpenVoice-Specific Technical Details:**

- **Two-stage architecture:** Unlike end-to-end TTS systems, OpenVoice separates content generation (MeloTTS) from speaker identity transfer (ToneColorConverter)
- **Tone Color Conversion (`tau=0.3`):** Controls the intensity of speaker identity transfer; 0.0 = no conversion, 1.0 = full conversion
- **Checkpoint path:** Uses `es.pth` (not `es-default.pth` as some docs suggest)
- **espeak-ng dependency:** Built from source in `~/.local/bin` on ml-server03 (required for Spanish phonemization via gruut)

---

## 11. Demo: Chatterbox

### Bonafide vs. Chatterbox Audio Comparison

Comparison of a bonafide HABLA sample against its Chatterbox-generated synthetic clone for the same speaker.

**Quality Metrics:**

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Speaker Similarity (SIM)** | 0.679 | Good clone quality -- same speaker identity perceived |
| **NISQA MOS** | 4.85 | Excellent perceptual quality -- near-natural speech |
| **WER** | 2.22% | Very high intelligibility -- near-perfect transcription |

**Analysis:**

- The 0.679 SIM score indicates the Chatterbox clone preserves the target speaker's identity well, falling in the "good clone quality" range (0.65-0.80)
- The 4.85 NISQA MOS is exceptionally high, placing the synthetic speech in the "excellent" tier, nearly indistinguishable from natural speech perceptually
- The 2.22% WER demonstrates near-perfect text intelligibility, suggesting the content was faithfully reproduced
- Chatterbox produces the highest perceptual quality (NISQA) among the four attack systems tested

---

## 12. Demo: OpenVoice

### Bonafide vs. OpenVoice Audio Comparison

Comparison of a bonafide HABLA sample against its OpenVoice-generated synthetic clone for the same speaker.

**Quality Metrics:**

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Speaker Similarity (SIM)** | 0.381 | Poor clone quality -- different speaker perceived |
| **NISQA MOS** | 4.06 | Good perceptual quality -- minor artifacts present |
| **WER** | 0.00% | Perfect intelligibility -- flawless transcription |

**Analysis:**

- The 0.381 SIM score is significantly below the 0.50 threshold, indicating the OpenVoice clone does not preserve the target speaker's identity well -- listeners would likely perceive a different speaker
- The 4.06 NISQA MOS is still in the "good" range, indicating the audio sounds natural even if the speaker identity is not preserved
- The 0.00% WER is perfect, meaning every word was correctly transcribed -- OpenVoice excels at text intelligibility
- The low SIM / high intelligibility pattern is characteristic of the two-stage architecture: MeloTTS produces clean, intelligible speech, but the ToneColorConverter with `tau=0.3` does not fully transfer the target speaker's characteristics

---

## Summary: Pipeline Comparison Matrix

| Metric | FishGram | Qwen3-TTS | Chatterbox | OpenVoice |
|--------|----------|-----------|------------|-----------|
| **Parameters** | 4B | 1.7B | 500M | ~100M |
| **Architecture** | VQGAN + Dual AR | Dual-Track LLM | Flow Matching | VITS + HiFi-GAN |
| **Inference** | HTTP API | Local model | Local model | Local model |
| **Reference Duration** | 15s | 15s | 10s | 15s |
| **Text Length** | 5-100 words | 5-40 words | 5-100 words | 5-100 words |
| **System ID** | FISHGRAM | QWEN3TTS | CHATTERBOX | OPENVOICE |
| **ID Range** | 9M+ | 8M+ | 6M+ | 7M+ |
| **Validation** | WER/CER + NISQA + SIM | WER/CER + artifact detection + NISQA + SIM | WER/CER + NISQA + SIM | WER/CER + NISQA + SIM |

---

*Last updated: 2026-04-12*
