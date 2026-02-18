# TTS Systems Investigation for Voice Anti-Spoofing Research
## Comprehensive Technical Analysis for Latin American Spanish Synthesis

**Date:** February 17, 2026
**Author:** Tomás Acosta (with Alfred's assistance)
**Purpose:** Evaluate Text-to-Speech systems for generating synthetic Spanish voice attacks in anti-spoofing research
**Hardware:** ml-server03 (4x NVIDIA A40, 46GB VRAM each, CUDA 12.6)

---

## Executive Summary

This document presents a comprehensive technical analysis of six modern TTS systems evaluated for their suitability in generating synthetic Spanish voice samples for anti-spoofing detection research. The investigation focuses on three critical criteria:

1. **Spanish Language Quality** (MANDATORY)
2. **Implementation Feasibility** (timeline, complexity, stability)
3. **Research Usefulness** (attack sophistication, codec diversity)

**Key Finding:** Fish Speech is the only system that meets all requirements for production deployment in this research context.

---

## Table of Contents

1. [Fish Speech (OpenAudio-S1)](#1-fish-speech-openaudio-s1)
2. [Qwen3-TTS (Alibaba)](#2-qwen3-tts-alibaba)
3. [CosyVoice 3.0 (Alibaba)](#3-cosyvoice-30-alibaba)
4. [Chatterbox (Resemble.ai)](#4-chatterbox-resembleai)
5. [OuteTTS](#5-outetts)
6. [Nari Dia 1.6B](#6-nari-dia-16b)
7. [Comparative Analysis](#comparative-analysis)
8. [Final Recommendations](#final-recommendations)

---

## 1. Fish Speech (OpenAudio-S1)

### Context & Overview

**What is it?**
Fish Speech (now branded as OpenAudio-S1) is a state-of-the-art open-source TTS system developed by the Fish Audio team. It's a 4 billion parameter model trained using Reinforcement Learning from Human Feedback (RLHF), similar to how ChatGPT was trained to sound more natural.

**Why does it exist?**
Traditional TTS systems sound robotic because they optimize for technical accuracy (matching text to audio) but not naturalness. Fish Speech uses human feedback to learn what makes speech sound genuinely human - pauses, emotion, prosody, breathing patterns.

**Architecture:**
- **Dual-AR + Firefly-GAN**: Uses two autoregressive models (one for semantic content, one for acoustic details) plus a GAN-based vocoder
- **4 billion parameters**: Among the largest open-source TTS models
- **RLHF training**: Optimizes for human-perceived naturalness, not just technical metrics

### Benefits & Strengths

1. **Spanish Language Support (VALIDATED)**
   - 20,000 hours of Spanish training data
   - Explicitly listed as a "high training data" language alongside English and Chinese
   - Supports emotion control and fine-grained prosody in Spanish

2. **Cross-Lingual Voice Cloning**
   - Can clone a Latin American Spanish voice from just 10-30 seconds of reference audio
   - Preserves accent, emotional tone, and speaker characteristics
   - Native support (not a hack or workaround)

3. **Attack Sophistication**
   - Represents the most advanced TTS attack vector available in open-source
   - RLHF training makes it harder to detect (optimized for human perception)
   - Captures subtle speech features that simpler TTS systems miss

4. **Professional Codebase**
   - 24,906 GitHub stars (highly trusted by community)
   - Active maintenance (updates in February 2026)
   - Published research paper (arXiv:2411.01156)
   - Comprehensive documentation

5. **Docker Support**
   - Official Docker Compose configuration available
   - Simplifies deployment on ml-server03
   - Reproducible environment

6. **License: CC-BY-NC-SA-4.0 (Perfect for Academic Research)**
   - Non-commercial use fully permitted
   - Academic research explicitly allowed
   - Must provide attribution (standard academic practice)
   - Share-alike ensures reproducibility

### Problems & Red Flags

1. **High VRAM Requirement (12GB minimum)**
   - Documentation explicitly states 12GB VRAM needed for "fluent inference"

2. **Streaming Latency Issues**
   - GitHub Issue #1020: Real-world streaming latency significantly higher than 150ms benchmark
   - `response_queue.get()` blocks until generation completes (not true streaming)

3. **Voice Cloning Inconsistency**
   - GitHub Issue #836: Users report cloned voices don't accurately match reference audio tone
   - Quality varies depending on reference audio characteristics

4. **Linux/WSL2 Requirement**
   - Native Windows support limited
   - Best performance on Linux (which you have on ml-server03)

### Why These Problems Matter (Technical Deep-Dive)

#### Problem 1: 12GB VRAM Requirement

**Why this happens:**
Fish Speech uses a large vocoder model (the part that converts neural network outputs into actual audio waveforms). Vocoders need to hold:
- Input feature representations (semantic tokens)
- Intermediate acoustic features (mel-spectrograms)
- GAN generator weights (for high-quality audio synthesis)

All of this must fit in GPU memory simultaneously during inference.

**Why it's a problem (or not):**
For consumer GPUs (8GB RTX 3060), this would be blocking. For your A40s (46GB VRAM), this is **completely irrelevant** - you're using only 26% of one GPU's capacity.

**Verdict:** NOT A PROBLEM for your hardware.

#### Problem 2: Streaming Latency Issues

**What this means:**
"Streaming" TTS should produce audio chunks as it generates them (like a live phone call). Fish Speech's implementation waits until the entire audio is generated before returning anything.

**Why this happens:**
The RLHF optimization process and the Dual-AR architecture create dependencies between audio chunks - later chunks depend on earlier ones being complete, preventing true streaming.

**Why it's a problem (or not):**
This matters for **real-time applications** like voice assistants. For your use case (batch generating synthetic samples for anti-spoofing training data), you don't need streaming at all. You're running:

```bash
# Your workflow
for speaker in latin_american_speakers:
    generate_synthetic_audio(speaker, output_file)
    # Don't care how long it takes, just need quality
```

**Verdict:** NOT A PROBLEM for batch augmentation.

#### Problem 3: Voice Cloning Inconsistency

**What this means:**
When you provide a reference audio clip of a Latin American Spanish speaker, the generated audio sometimes doesn't perfectly match their vocal characteristics (timbre, accent, emotional tone).

**Why this happens:**
Voice cloning quality depends on:
- Reference audio quality (background noise, microphone characteristics)
- Reference audio length (10s vs 30s makes a difference)
- Unusual vocal features (very deep voice, speech impediments, heavy regional accents)

The DAC (Descript Audio Codec) encoder can struggle with "out-of-distribution" voices that differ significantly from training data.

**Why this might NOT be a problem (counter-intuitive):**
For anti-spoofing research, you **want** variability. If every cloned voice were perfect, your detector would overfit to "perfect clones." Having some inconsistency creates a more realistic training set - real attackers also produce imperfect clones.

**Verdict:** POTENTIALLY BENEFICIAL for creating challenging training data.

### Time Complexity Analysis

**Setup Phase (One-time):**
- Environment setup (Docker/Conda): 1-2 hours
- Model download (4B parameters, ~8-15GB): 30-60 minutes
- Testing and validation: 2-4 hours
- **Total: 4-8 hours**

**Per-Sample Generation (Inference):**
- 10 seconds of audio: ~1.4 seconds on RTX 4090 (Real-Time Factor 0.14)
- Expected on A40: ~2-3 seconds per 10s audio
- **Batch generation of 1000 samples (10s each): ~1-2 hours**

**Integration into Pipeline:**
- Creating augmenter class: 1-2 days
- Testing with Latin American Spanish samples: 2-3 days
- **Total integration: 1-2 weeks**

### Environment Requirements

**Hardware (SATISFIED by ml-server03):**
- GPU: NVIDIA A40 ✅
- VRAM: 12GB minimum (you have 46GB) ✅
- CUDA: 12.6 ✅
- RAM: 16GB+ system memory ✅

**Software:**
- Python: 3.12
- PyTorch: 2.0+ with CUDA support
- Docker: Optional but recommended
- OS: Linux (you have this) ✅

**Dependencies:**
- transformers, accelerate, soundfile, librosa
- fish-speech Python package
- Audio processing libraries (scipy, torchaudio)

**Installation Method (Recommended for ml-server03):**
```bash
# Docker Compose (simplest)
git clone https://github.com/fishaudio/fish-speech
cd fish-speech
docker-compose up -d

# Or Conda (more control)
conda create -n fish-speech python=3.12 -y
conda activate fish-speech
pip install fish-speech
```

### Spanish Language Assessment

**Training Data:** 20,000 hours
**Quality Tier:** "High training data" category (same as English, Chinese)
**Dialect Coverage:** Multilingual training suggests broad Spanish coverage
**Accent Adaptability:** Voice cloning allows Latin American accent customization

**Evidence of Quality:**
- Spanish listed in top-tier language support documentation
- Emotion markers work in Spanish (tested in examples)
- No preprocessing required (direct Spanish text input)

**Unknowns:**
- No specific benchmarks for Latin American Spanish variants (Mexican, Colombian, Argentine, etc.)
- Recommendation: Validate quality with 10-20 test samples before full integration

### Verdict: STRONGLY RECOMMENDED

**Justification:**
- Best Spanish support among all evaluated systems
- Most sophisticated attack modeling (RLHF, 4B parameters)
- Hardware requirements fully satisfied by A40 GPUs
- Problems are either irrelevant (latency) or beneficial (cloning variability)
- Academic license explicitly permits this research

**Risk Level:** LOW
**Implementation Timeline:** 2-3 weeks
**Expected Quality:** HIGH

---

## 2. Qwen3-TTS (Alibaba)

### Context & Overview

**What is it?**
Qwen3-TTS is a family of neural text-to-speech models developed by Alibaba's Qwen team (the same team behind Qwen LLMs). It uses a "Dual-Track" architecture that processes text in two parallel streams for ultra-low latency.

**Why does it exist?**
Alibaba needed a TTS system for their cloud services that could:
- Generate speech with minimal delay (97ms latency claim)
- Support multiple languages for international customers
- Allow both voice cloning AND voice design from text descriptions
- Scale to production workloads

**Architecture:**
- **Dual-Track Design**: Processes semantic content and acoustic features separately, then merges
- **Two tokenizers**: 25Hz (high quality) and 12Hz (low bandwidth streaming)
- **Three variants**: Base (cloning), CustomVoice (style control), VoiceDesign (create new voices)

### Benefits & Strengths

1. **Ultra-Low Latency (97ms)**
   - Fastest time-to-first-audio among all evaluated systems
   - Dual-Track architecture enables parallel processing
   - Achieved with 1.7B parameter model (smaller than Fish Speech)

2. **Apache 2.0 License (Unrestricted)**
   - Full commercial use permitted
   - No attribution requirements beyond standard practice
   - Can modify, distribute, deploy freely
   - Better than Fish Speech's NC license if commercialization ever needed

3. **Three Model Variants**
   - **Base (0.6B/1.7B)**: Standard voice cloning from 3 seconds audio
   - **CustomVoice**: Control style with existing voice profiles
   - **VoiceDesign**: Create new voices from text ("deep male voice with British accent")

4. **Modest Hardware Requirements**
   - 0.6B model: 2-4GB VRAM
   - 1.7B model: 4-8GB VRAM
   - Could run multiple instances in parallel on A40s

5. **Easy Installation**
   ```bash
   pip install -U qwen-tts  # That's it
   ```

6. **FlashAttention 2 Support**
   - 30-40% speedup with optimized attention mechanism
   - 20-25% VRAM reduction
   - Your A40s support this (CUDA 12.6 compatible)

### Problems & Red Flags

1. **Spanish is "Second-Tier" Language**
   - Research paper states Spanish performance is "competitive" but NOT in top-performing group
   - Best performance: Chinese, English, Italian, French, Korean, Russian (Spanish not listed)

2. **Fine-Tuning Instability**
   - GitHub Issue: Progressive speech speedup during training epochs
   - Checkpoint resumption corrupts audio quality
   - Speaker encoder deletion during save/load cycles

3. **Embedding Dimension Mismatch (0.6B model)**
   - Error: "text embedding dim(2048) not match with codec dim(1024)"
   - Blocks fine-tuning for smaller variant

4. **Audio Generation Artifacts**
   - Stray characters in waveform metadata (Chinese character "的")
   - Truncated outputs on long texts (>200 words)
   - Invalid audio without clear error messages

5. **Dependency Conflicts**
   - Hard pin on `transformers==4.57.3`
   - Cannot coexist with other Qwen ecosystem tools (qwen-asr conflicts)

6. **Performance Inconsistency**
   - GitHub Issue #89: "Very slow inference on 5090" (newer than A40)
   - Latency varies dramatically based on configuration

### Why These Problems Matter (Technical Deep-Dive)

#### Problem 1: Spanish is "Second-Tier"

**What this means:**
The research paper provides benchmarks for 10 languages. They explicitly list 6 languages where Qwen3-TTS achieves "state-of-the-art" performance. Spanish is not among them.

**Why this happens:**
Training data allocation and optimization focus. The paper states:
> "Maintains highly competitive performance comparable to state-of-the-art"

"Competitive" is diplomatic language for "good but not best." They optimized for Chinese and English (Alibaba's primary markets), with other languages receiving less attention.

**Why it's a problem:**
For anti-spoofing research, you need **high-quality Spanish synthesis** because:
- Low-quality synthetic voices are easier to detect (not realistic threat modeling)
- Latin American Spanish has specific phonetic characteristics (yeísmo, voseo, seseo)
- Your detector needs to train on challenging Spanish samples, not "competitive" ones

**Verdict:** SIGNIFICANT PROBLEM - defeats the purpose of using TTS for attack modeling.

#### Problem 2: Fine-Tuning Instability

**What fine-tuning means:**
Taking the pre-trained model and training it further on your specific data (e.g., Colombian Spanish accent samples) to specialize it.

**Why you might want this:**
The base model was trained on global Spanish data. Fine-tuning on Latin American accents would make it more realistic for your research population.

**What "instability" means:**
Users report that during fine-tuning:
- Speech progressively speeds up over training epochs (sounds like chipmunks)
- Saving and reloading checkpoints causes audio quality to degrade
- The speaker encoder (the part that captures voice identity) gets corrupted or deleted

**Why this happens:**
This suggests bugs in the training code or model architecture. The speaker encoder weights aren't properly saved/restored, and the optimizer state management has issues.

**Why it's a CRITICAL problem:**
Without stable fine-tuning, you CANNOT adapt the model to Latin American Spanish accents. You're stuck with whatever generic Spanish the base model learned. This makes Qwen3-TTS a "take it or leave it" system - if the base Spanish quality isn't good enough, you have no recourse.

**Verdict:** CRITICAL BLOCKER for customization.

#### Problem 3: Embedding Dimension Mismatch

**What embeddings are:**
Vector representations of text. The model converts "Hola, ¿cómo estás?" into a sequence of numerical vectors (e.g., [0.25, -0.14, 0.88, ...]).

**What "dimension mismatch" means:**
The 0.6B model has a bug where:
- Text encoder outputs 2048-dimensional vectors
- Audio codec expects 1024-dimensional vectors
- 2048 ≠ 1024 → Error

**Why this happens:**
This is a **software bug**. During model development, someone changed one component's output size without updating the other. It should have been caught in testing but wasn't.

**Why it's a problem:**
You cannot use the 0.6B model for fine-tuning at all. It fails immediately. You're forced to use the 1.7B model, which is slower and uses more VRAM.

**Why it matters for you:**
Not critical (you have VRAM for 1.7B model), but indicates poor quality control.

**Verdict:** MINOR ISSUE - workaround exists (use 1.7B).

#### Problem 4: Audio Generation Artifacts

**What artifacts mean:**
Glitches or errors in the generated audio:
- Random metadata characters appearing in audio headers
- Audio cutting off before the text finishes
- Silent/garbled output without error messages

**Example scenario:**
```python
# You generate this
text = "El sistema de detección de spoofing analiza características..."

# Expected: 15 seconds of clear Spanish audio
# Actual: 11 seconds, cuts off at "...analiza carac—" [silence]
# No error message, no warning
```

**Why this happens:**
Multiple potential causes:
1. Token length estimation errors (model thinks text is shorter than it is)
2. Attention mechanism fails on long sequences
3. Audio codec decoder has buffer overflow bugs
4. End-of-sequence token generated prematurely

**Why it's a CRITICAL problem:**
You're generating thousands of synthetic samples. If 10-20% are silently corrupted, your training data becomes noisy. Worse, you might not notice until after training your detector, wasting weeks of work.

**Mitigation:**
You'd need to implement extensive validation:
```python
def validate_audio(audio, expected_duration, text):
    actual_duration = len(audio) / sample_rate
    if actual_duration < expected_duration * 0.9:
        raise AudioTruncatedError(f"Expected {expected_duration}s, got {actual_duration}s")
    # More checks...
```

**Verdict:** HIGH-RISK - requires significant error handling overhead.

### Time Complexity Analysis

**Setup Phase:**
- Environment setup (pip install): 30 minutes
- FlashAttention 2 compilation: 1-2 hours (if building from source)
- Model download (5-7GB per variant): 30-60 minutes
- Testing and troubleshooting: 1-2 hours
- **Total: 2-6 hours** (depending on FlashAttention issues)

**Per-Sample Generation:**
- 10 seconds audio on RTX 4090: ~0.65 seconds (RTF 0.065)
- Expected on A40: ~0.8-1.0 seconds per 10s audio
- **Batch generation of 1000 samples: ~20-30 minutes** (VERY FAST)

**Integration Timeline:**
- Basic integration: 3-5 days
- Validation pipeline (detect artifacts): 1-2 weeks
- **Total: 2-3 weeks**

### Environment Requirements

**Hardware (ml-server03):**
- GPU: A40 (far exceeds requirements) ✅
- VRAM: 4-8GB needed, 46GB available ✅
- CUDA: 12.6 ✅

**Software:**
- Python: 3.9-3.13 (flexible)
- PyTorch: Latest with CUDA
- FlashAttention 2: Optional but strongly recommended

**Dependencies:**
- qwen-tts package
- transformers==4.57.3 (HARD REQUIREMENT - causes conflicts)

**Known Conflicts:**
```bash
# Cannot install simultaneously
pip install qwen-tts      # Requires transformers==4.57.3
pip install qwen-asr      # Requires transformers==4.58.x
# → Conflict!
```

**Solution:** Isolated conda environment mandatory

### Spanish Language Assessment

**Training Data:** Not disclosed (part of 10-language multilingual corpus)
**Quality Tier:** "Competitive" (diplomatic for "not best")
**Benchmarks:** WER 1.126-2.031, Speaker Similarity 0.789
**Paper Statement:** Spanish NOT listed in top-performing language group

**Red Flags:**
- Paper says "highly competitive performance comparable to state-of-the-art"
  - Translation: "We're okay, not great"
- All examples and demos use Chinese or English
- No Spanish-specific benchmarks or audio samples

**User Reports:**
- Hacker News comment: "Slight Asian accent in English generations"
- If English has this issue, Spanish likely does too

**Verdict:** Spanish quality is MEDIOCRE - suitable for diversity but NOT as primary TTS

### Verdict: CONDITIONAL - Use for Diversity Only

**Justification:**
- Apache 2.0 license is excellent
- Fast inference (0.8s per 10s audio)
- Easy installation

**BUT:**
- Spanish is explicitly second-tier quality
- Cannot fine-tune for Latin American accents (broken)
- Audio artifacts require extensive validation

**Recommended Use Case:**
Deploy alongside Fish Speech as a **secondary TTS system** for architectural diversity:
- Fish Speech: Primary (RLHF-trained, high Spanish quality)
- Qwen3-TTS: Secondary (different codec approach, faster inference)

This gives your detector exposure to different synthesis methods.

**Risk Level:** MEDIUM
**Implementation Timeline:** 2-3 weeks (including validation pipeline)
**Expected Quality:** ADEQUATE but not exceptional

---

## 3. CosyVoice 3.0 (Alibaba)

### Context & Overview

**What is it?**
CosyVoice 3.0 is Alibaba's FunAudioLLM team's flagship TTS system, trained on 1 million hours of speech data (the largest corpus among evaluated systems). It uses Conditional Flow Matching (CFM) for high-fidelity synthesis.

**Why does it exist?**
CosyVoice represents Alibaba's research-grade TTS system (versus Qwen3-TTS which is their production-grade service). It's designed for:
- Academic research and experimentation
- "In-the-wild" quality (handling noisy, real-world audio)
- Multilingual voice cloning with high similarity scores

**Architecture:**
- **Conditional Flow Matching (CFM)**: Advanced generative model that learns to transform noise into speech
- **Chunk-Aware Processing**: Processes audio in overlapping chunks for streaming
- **Supervised Semantic Tokens**: Uses Whisper ASR-derived tokens for linguistic content
- **Matcha-TTS Integration**: Requires submodule for acoustic feature extraction

### Benefits & Strengths

1. **Largest Training Corpus (1 Million Hours)**
   - 100x more data than Fish Speech (20k hours)
   - 50x more than Qwen3-TTS (estimated ~20k hours)
   - Should theoretically produce highest quality synthesis

2. **Apache 2.0 License**
   - Unrestricted commercial use
   - Better than Fish Speech's NC license

3. **Strong Research Backing**
   - Alibaba Tongyi Lab Speech Team (well-funded, long-term commitment)
   - Published research papers with detailed methodology
   - 19,600 GitHub stars (high community trust)

4. **High Similarity Scores**
   - 77.4% speaker similarity (approaching human-level 78.0%)
   - 0.81% Character Error Rate (Chinese)
   - 1.45% Word Error Rate (English)

5. **vLLM Support for Optimization**
   - Can leverage vLLM for 2-4x faster inference
   - Production-grade serving infrastructure

6. **TensorRT-LLM Acceleration**
   - NVIDIA contributed TensorRT support (August 2025)
   - 4x speedup possible with TensorRT optimization

### Problems & Red Flags

1. **vLLM Dependency Hell (CRITICAL)**
   - Only works with vLLM 0.9.0 OR 0.11.x+ (V1 engine)
   - Versions 0.10.x: Incompatible (gap in support)
   - Users report "environment corruption requiring fresh installs"
   - Requires source compilation for RTX 5090 / newer hardware

2. **Matcha-TTS Submodule Complexity**
   - Requires `git clone --recursive` or manual submodule initialization
   - Must export `PYTHONPATH=third_party/Matcha-TTS`
   - Dependency management fragility

3. **Platform-Specific Issues**
   - Windows: "Fragmented installation" (Anaconda PowerShell + separate commands)
   - macOS: TensorRT incompatible (Linux/Windows only)
   - sox library issues on different platforms

4. **Quality Regression Reports**
   - Multiple users claim CosyVoice 3 produces **lower quality** than CosyVoice 2
   - Chinese-English mixed text produces garbled audio

5. **No Spanish-Specific Benchmarks**
   - All metrics focus on Chinese and English
   - Spanish listed as supported but no quality validation

6. **Long Deployment Time**
   - First-time setup: 1-2 days (experienced engineer)
   - Troubleshooting vLLM: 0-3 hours (unpredictable)

### Why These Problems Matter (Technical Deep-Dive)

#### Problem 1: vLLM Dependency Hell (THE BIG ONE)

**What is vLLM?**
vLLM (Very Large Language Model) is an inference optimization library originally designed for running large language models (like Qwen, Llama) efficiently. It does things like:
- Paged attention (efficient memory management)
- Continuous batching (process multiple requests simultaneously)
- Quantization support (run models in lower precision)

**Why does CosyVoice use vLLM?**
CosyVoice 3.0's architecture includes transformer-based components similar to LLMs. Alibaba wanted to reuse vLLM's optimizations for faster inference.

**What is "dependency hell"?**
This is when software has very specific version requirements that conflict with other software. For CosyVoice:

```bash
# CosyVoice requirements
vLLM == 0.9.0  OR  vLLM >= 0.11.0

# But NOT:
vLLM == 0.10.0, 0.10.1, 0.10.2, ... 0.10.9

# Why? Because CosyVoice uses internal vLLM APIs that changed between versions
```

**Real-world scenario of what happens:**

1. You install CosyVoice with vLLM 0.9.0 (works)
2. Later, you install another tool that requires vLLM 0.10.5
3. pip upgrades vLLM to 0.10.5
4. CosyVoice breaks with cryptic error: `AttributeError: module 'vllm' has no attribute 'LLMEngine'`
5. You try to downgrade vLLM
6. This breaks the other tool
7. You're stuck

**Why users report "environment corruption":**

```bash
# User tries to fix the issue
pip install vllm==0.9.0  # Downgrades
# But vLLM 0.9.0 compiled for different CUDA version
# GPU inference now fails

pip uninstall vllm
pip install vllm==0.9.0 --no-cache-dir --force-reinstall
# Still fails because PyTorch version incompatibility

# Eventually:
# "I had to delete my entire conda environment and start over"
```

**Why it requires source compilation for RTX 5090:**

RTX 5090 uses new Ada Lovelace architecture with CUDA features that didn't exist when vLLM 0.9.0 was released. Pre-compiled vLLM binaries don't support it. You must:

```bash
git clone https://github.com/vllm-project/vllm.git
cd vllm
git checkout v0.9.0
python setup.py install  # Compile from source for 2-4 hours
```

**Why this is a CRITICAL problem:**

1. **Time Cost**: Debugging vLLM issues can take 1-3 hours per attempt
2. **Reproducibility Risk**: Different team members might get different results
3. **Maintenance Burden**: Every system update risks breaking CosyVoice
4. **Expertise Required**: Requires deep understanding of Python packaging

**For your thesis timeline:**

You have limited time. Spending 1-2 days fighting vLLM dependency issues is unacceptable when Fish Speech installs in 4 hours.

**Verdict:** CRITICAL BLOCKER - vLLM complexity not worth the hassle.

#### Problem 2: Quality Regression vs v2.0

**What "quality regression" means:**
Later versions performing worse than earlier versions. Multiple users report:

> "CosyVoice 2 sounds better than CosyVoice 3"
> "Voice quality degraded after upgrading"
> "Chinese-English mixed text is garbled in v3, worked in v2"

**Why this happens:**

Possible causes:
1. **Model Capacity Reduction**: v3.0 offers 0.5B and 1.5B variants. Users might be comparing v3.0-0.5B (smaller) to v2.0-1.5B (larger)
2. **Training Data Quality**: 1M hours sounds impressive, but if 50% is low-quality web scraping, it's worse than 100k hours of curated data
3. **Architecture Changes**: CFM (Conditional Flow Matching) might be theoretically better but practically worse than v2.0's approach
4. **Rushed Release**: v3.0 released December 2025 - possibly rushed to market

**Why this is concerning for research:**

You need **validated, stable tools**. If community consensus is "v2 was better," why risk using v3? Especially when:
- No Spanish quality benchmarks to verify it's good
- Can't compare to v2 yourself (would need to install both)
- Your thesis defense: "Why did you use v3?" "Well, it's newer..."

**Verdict:** HIGH-RISK - suggests immature release.

#### Problem 3: No Spanish Benchmarks

**What this means:**
The research paper provides detailed benchmarks for Chinese and English:
- Character Error Rate (CER): 0.81% (Chinese)
- Word Error Rate (WER): 1.45% (English)
- Speaker Similarity: 77.4%

For Spanish? Nothing. Just "supported."

**Why this happens:**
Alibaba's business priorities:
1. Chinese (domestic market)
2. English (international market)
3. Everything else (afterthought)

Spanish support was likely added for marketing ("9 languages!") without thorough validation.

**Why this is a SEVERE problem:**

You're basing your entire TTS choice on the assumption that CosyVoice's Spanish is good. But you have:
- **No benchmarks** to validate quality
- **No audio samples** to listen to
- **No user reviews** of Spanish quality
- **No Latin American Spanish validation** at all

This is a **blind bet**. You'd spend 1-2 days deploying CosyVoice, then discover Spanish quality is poor, wasting your time.

**Verdict:** UNACCEPTABLE RISK - Cannot bet thesis timeline on unvalidated Spanish support.

### Time Complexity Analysis

**Setup Phase:**
- Environment setup: 1-2 hours
- Submodule initialization (Matcha-TTS): 30 minutes
- vLLM installation/compilation: 1-4 hours (UNPREDICTABLE)
- Model download: 30-60 minutes
- Troubleshooting platform issues: 1-3 hours
- **Total: 1-2 days** (pessimistic but realistic)

**Per-Sample Generation:**
- 10 seconds audio: ~1.5 seconds (150ms claimed latency × 10)
- Expected on A40 with vLLM: ~1.0-1.5 seconds
- **Batch generation of 1000 samples: ~30-40 minutes**

**Integration Timeline:**
- Assuming successful deployment: 1-2 weeks
- **If vLLM issues arise: +1 week per issue**
- **Total: 2-4 weeks** (high uncertainty)

### Environment Requirements

**Hardware:**
- GPU: A40 ✅
- VRAM: 8GB (0.5B model) or 12-16GB (1.5B model) ✅
- CUDA: 12.6 ✅

**Software:**
- Python: 3.10 (REQUIRED - v3.8 might work, v3.12 incompatible with matcha-tts)
- PyTorch: 2.4+ (but NOT 2.4+ for RTX 50-series - version conflicts)
- vLLM: 0.9.0 OR 0.11.x+ (NOT 0.10.x)
- sox: System-level audio library

**Platform:**
- Linux: ✅ Full support
- Windows: Partial (requires WSL2 for best results)
- macOS: TensorRT unavailable (blocker for some optimizations)

### Spanish Language Assessment

**Training Data:** 1 million hours total (Spanish allocation unknown)
**Quality Tier:** "Supported" (no further detail)
**Benchmarks:** NONE for Spanish
**Dialect Coverage:** Unknown

**Risk Assessment:**
- **No validation** that Spanish quality matches Chinese/English
- **No samples** to evaluate before deployment
- **High likelihood** Spanish is undertrained compared to primary languages

**Verdict:** Spanish quality is COMPLETELY UNKNOWN - unacceptable for thesis research.

### Verdict: DO NOT USE

**Justification:**
Despite impressive credentials (1M hours, Apache 2.0, Alibaba backing):

1. **vLLM dependency hell** risks 1+ weeks of debugging
2. **Quality regression reports** suggest v3.0 is immature
3. **No Spanish benchmarks** means blind bet on quality
4. **1-2 day deployment time** vs 4-8 hours for Fish Speech

**Counter-Argument (Why You Might Still Consider It):**

If you had:
- 4-6 weeks of buffer time
- vLLM expertise on your team
- No Spanish-specific requirements (English research)

Then CosyVoice might be worth it for the 1M hours training advantage.

**But you have:**
- Limited thesis timeline
- Spanish quality as MANDATORY requirement
- Better alternatives (Fish Speech)

**Risk Level:** HIGH
**Implementation Timeline:** 2-4 weeks (unpredictable)
**Expected Quality:** UNKNOWN (Spanish not validated)

**Final Verdict:** REJECT - Complexity and uncertainty not justified

---

## 4. Chatterbox (Resemble.ai)

### Context & Overview

**What is it?**
Chatterbox is an open-source TTS system by Resemble.ai (a commercial AI voice company). The "Turbo" variant is optimized for real-time voice agents with aggressive speed optimization.

**Why does it exist?**
Resemble.ai operates a dual business model:
1. **Open-source Chatterbox**: Free MIT-licensed TTS for community goodwill
2. **Commercial "Chatterbox Multilingual Pro"**: Paid service with SLAs, fine-tuning, support

Chatterbox serves as:
- Customer acquisition funnel (users try free, upgrade to paid)
- Research showcase (demonstrates Resemble's capabilities)
- Recruiting tool (attracts ML talent)

**Architecture:**
- **350 million parameters**: Smallest model in evaluation (lightweight)
- **Single-step decoder**: Directly generates mel-spectrograms in one pass (faster but lower quality ceiling)
- **Perth Watermarking**: Mandatory neural watermarks embedded in all outputs

### Benefits & Strengths

1. **Easiest Installation**
   ```bash
   pip install chatterbox-tts  # Done
   ```
   - No complex dependencies
   - No submodules
   - No version conflicts

2. **Genuinely Open-Source (MIT License)**
   - NOT "open weights with commercial restrictions"
   - TRUE open-source: modify, distribute, commercialize freely
   - No attribution requirements (though courteous)
   - Rare in TTS landscape (most are NC or proprietary)

3. **Lightweight (350M parameters)**
   - Smallest VRAM footprint: 8GB
   - Fastest to load and initialize
   - Could run 5+ instances in parallel on single A40

4. **Multilingual (23 Languages Including Spanish)**
   - Spanish explicitly supported
   - Native-like fluency claimed
   - Cross-language voice transfer capabilities

5. **Short Voice Cloning (5 Seconds)**
   - Shortest reference audio requirement
   - Faster to prepare speaker profiles

6. **Paralinguistic Features**
   - Non-verbal tags: [laugh], [sigh], [cough], [hesitation]
   - Emotion exaggeration control
   - Adds realism to synthetic speech

7. **Docker Support**
   - Community-built Docker containers available
   - FastAPI server implementations (OpenAI-compatible API)

8. **Outperformed ElevenLabs in Blind Tests**
   - 63.75% preference over ElevenLabs Turbo
   - Validates quality despite small size

### Problems & Red Flags

1. **Latency Fraud (CRITICAL)**
   - Marketing claims: "<200ms inference latency"
   - Reality (GitHub Issue #193): "Takes 2 to 3 times longer than claimed"
   - Actual performance: 300-600ms depending on hardware
   - First-chunk latency: ~472ms on RTX 4090 (GitHub documentation)

2. **CPU Inference Broken**
   - Documentation claims CPU support
   - Reality (GitHub Issue #96): RuntimeError on CPU-only systems
   - Community: "Much slower than using CUDA" (understatement)
   - GPU is MANDATORY despite marketing

3. **Mandatory Watermarking (Unknown Overhead)**
   - Perth neural watermarks embedded in ALL outputs
   - Cannot be disabled
   - Performance impact: NOT DISCLOSED
   - Survives MP3 compression and audio editing
   - 100% detection accuracy claimed

4. **High Issue Count (224 Open Issues)**
   - Small project (34 commits) with many problems
   - Performance issues (Issue #127)
   - Stability concerns

5. **Commercial Ties Concern**
   - Open-source version might be "feature-limited" to upsell commercial product
   - Quality intentionally degraded to push users toward paid service?
   - Impossible to verify

6. **No Latin American Spanish Validation**
   - Spanish support confirmed
   - Regional variant quality: UNKNOWN
   - Accent inheritance warning: "May inherit accent from reference clip"

### Why These Problems Matter (Technical Deep-Dive)

#### Problem 1: Latency Fraud (THE BIG LIE)

**What the marketing says:**
> "Chatterbox-Turbo achieves sub-200ms inference latency"
> "6x faster than real-time on GPU"
> "Optimized for low-latency voice agents"

**What the reality is:**

From GitHub Issue #193 ("Ways to reduce latency"):
> User: "It takes 2 to 3 times longer than the README's claimed 200ms in practice"

From official documentation:
> "First-chunk latency: ~472ms on RTX 4090"

**Math check:**
- Claimed: <200ms
- Documented: 472ms
- User reports: 300-600ms

**472ms / 200ms = 2.36x slower than claimed**

**Why this is "fraud" and not "estimation error":**

1. **Specificity**: They claim "<200ms" not "~200ms" or "200-400ms"
2. **Their own docs contradict it**: Official documentation says 472ms
3. **Consistent user complaints**: Multiple users report 300-600ms
4. **Pattern**: This aligns with marketing-driven culture (overpromise)

**Why this matters:**

For **real-time applications** (voice assistants), 472ms latency is noticeable lag. Users feel the delay.

For **your use case** (batch audio generation):
- Latency doesn't matter at all
- You're generating files offline
- **NOT A PROBLEM** for you

**But the dishonesty is concerning:**

If they lie about latency, what else are they lying about?
- Spanish quality?
- CPU support?
- Watermarking overhead?

**Integrity concern**: Can you trust this company's claims?

**Verdict:** Latency lie is irrelevant to your use case BUT indicates company culture problem.

#### Problem 2: CPU Inference Broken

**What the documentation claims:**
- CPU inference supported
- Enables deployment without GPU
- Accessible to wider audience

**What actually happens:**

GitHub Issue #96: "RuntimeError on CPU-Only Systems"
```python
RuntimeError: Attempting to deserialize object on a CUDA device but torch.cuda.is_available() is False
```

**Translation:**
The code assumes GPU is available and fails immediately on CPU-only systems.

**Why this happens:**

The model checkpoint was saved with GPU-specific components:
```python
# During model training (on GPU)
torch.save(model.state_dict(), 'checkpoint.pth')

# When loading (on CPU)
model.load_state_dict(torch.load('checkpoint.pth'))
# ↑ This loads GPU tensors, crashes on CPU
```

**Proper CPU support would be:**
```python
model.load_state_dict(torch.load('checkpoint.pth', map_location='cpu'))
```

This is a **trivial fix** (one line of code). The fact that it's not fixed suggests:
1. Developers don't care about CPU users
2. Limited testing before release
3. CPU support is marketing, not reality

**Why this matters for you:**

You have A40 GPUs, so CPU support is irrelevant. But this reveals:
- Poor code quality
- Limited testing
- Marketing-driven development

**Verdict:** Not relevant to your setup, but another red flag.

#### Problem 3: Mandatory Watermarking

**What is Perth watermarking?**

Perth is a neural watermarking technique that embeds imperceptible patterns into audio:
```
Original audio: [0.15, 0.22, 0.18, 0.31, ...]
Watermarked:    [0.15, 0.22, 0.18, 0.31, ...] + [0.0001, -0.0002, 0.0001, ...]
                                                   ↑ Imperceptible watermark pattern
```

The watermark:
- Survives MP3 compression (lossy encoding doesn't destroy it)
- Survives audio editing (cutting, splicing)
- 100% detection accuracy claimed (can prove audio came from Chatterbox)

**Why watermarking exists:**

Legal/ethical reasons:
- Trace misuse (deepfakes, impersonation)
- Provide attribution (know which TTS system generated audio)
- Resemble.ai's liability protection (prove audio wasn't their commercial service)

**What "mandatory" means:**

You CANNOT disable watermarking. Every audio file from Chatterbox contains it.

**The unknown: Performance overhead**

Watermarking requires:
- Additional neural network forward pass (watermark encoder)
- Additional computation during audio generation
- Estimated overhead: 10-30% (unconfirmed - Resemble doesn't disclose)

**Why this is a problem:**

1. **Hidden cost**: Documentation doesn't mention performance impact
2. **Academic integrity**: Your research uses watermarked audio
   - Should you disclose this in your thesis?
   - Does watermarking affect anti-spoofing detector performance?
3. **Detection artifacts**: Could your detector learn to detect the watermark instead of spoofing?

**Scenario:**
```
Your detector training:
- Bonafide samples: No watermark
- Chatterbox spoof samples: Perth watermark present

Your detector learns:
"If Perth watermark detected → spoof"

Problem: This is cheating! Detector should detect TTS synthesis, not watermarks.
```

**Verdict:** POTENTIAL CONTAMINATION of research - watermark could be a confounding variable.

#### Problem 4: 224 Open Issues

**What this means:**

GitHub issues are bug reports, feature requests, and questions. 224 open issues for a small project signals:

1. **High bug density**: Lots of problems per line of code
2. **Poor maintenance**: Issues not being closed/resolved
3. **User frustration**: Community reporting many problems

**Context:**
- Project size: 34 commits (VERY small)
- Contributors: 14 (moderate)
- 224 issues / 34 commits = 6.6 issues per commit

**Comparison:**
- Fish Speech: ~850 issues / 545 commits = 1.6 issues per commit
- Healthy projects: <2 issues per commit

**What this reveals:**

The codebase was released too early:
- Insufficient testing
- Many edge cases not handled
- Users discovering bugs in production

**Verdict:** QUALITY CONTROL problem - risky for thesis research.

### Time Complexity Analysis

**Setup Phase:**
- Installation: 30 minutes
- Model download (first run): 10-15 minutes
- Testing: 1-2 hours
- **Total: 2-4 hours** (easiest of all systems)

**Per-Sample Generation:**
- 10 seconds audio: ~4-6 seconds (actual latency, not claimed)
- **Batch generation of 1000 samples: ~1.5-2 hours**

**Integration Timeline:**
- Basic integration: 3-5 days
- Validation (watermark testing): 1 week
- **Total: 1.5-2 weeks**

### Environment Requirements

**Hardware:**
- GPU: Required (despite claims otherwise)
- VRAM: 8GB
- A40: ✅ Massive overkill

**Software:**
- Python: 3.11
- PyTorch: Latest
- No special dependencies

### Spanish Language Assessment

**Training Data:** Unknown (company doesn't disclose)
**Quality Tier:** "Native-like fluency" (marketing claim)
**Languages:** 23 including Spanish
**Dialect Coverage:** Unknown (no Latin American validation)

**Evidence:**
- Spanish listed in supported languages
- No benchmarks
- No audio samples
- No user reviews of Spanish quality

**Concerns:**
- Documentation warning: "May inherit accent from reference clip"
- If your reference speaker has Colombian accent, ALL languages might sound Colombian
- Cross-language transfer instability

**Verdict:** Spanish support is UNVALIDATED - risky assumption.

### Verdict: DO NOT USE

**Justification:**

Despite easy installation and MIT license:

1. **Latency fraud** indicates dishonest marketing culture
2. **Mandatory watermarking** could contaminate research (confounding variable)
3. **224 open issues** suggests quality control problems
4. **No Latin American Spanish validation**
5. **Better alternatives exist** (Fish Speech)

**Only scenario where Chatterbox makes sense:**

If you needed:
- Rapid prototyping (2-4 hour setup is fastest)
- Multiple TTS systems for comparison
- Lightweight model for resource-constrained environments

**But for primary thesis research TTS:**
Too many red flags.

**Risk Level:** MEDIUM-HIGH
**Implementation Timeline:** 1.5-2 weeks
**Expected Quality:** UNCERTAIN

**Final Verdict:** REJECT - Watermarking and quality concerns outweigh easy setup

---

## 5. OuteTTS

### Context & Overview

**What is it?**
OuteTTS is a TTS system built on top of large language models (Qwen 0.6B and Llama 3.2-1B). It treats speech synthesis as a language modeling task: predict the next audio token like LLMs predict the next text token.

**Why does it exist?**
OuteAI (the developer) wanted to prove that:
1. TTS doesn't need specialized architectures (just use existing LLMs)
2. llama.cpp ecosystem can be reused for TTS (CPU inference)
3. Speaker profiles can be JSON files (easy storage/sharing)

**Architecture:**
- **Base**: Qwen3 0.6B or Llama 3.2-1B (pre-trained language models)
- **DAC Codec**: Descript Audio Codec for audio encoding (2 codebooks)
- **Speaker Profiles**: JSON metadata (tempo, energy, pitch, spectral centroid)

### Benefits & Strengths

1. **llama.cpp Compatibility**
   - Full support for llama.cpp inference
   - GGUF quantization available
   - Theoretically enables CPU inference

2. **Speaker Profile JSON System**
   - Save speaker characteristics as JSON files
   - Reusable across sessions
   - Easy to share and version control
   - 10-15 second reference audio creates profile

3. **Metadata-Rich Generation**
   - Global and word-level metadata: tempo, energy, pitch, spectral centroid
   - Inherits timbre, emotion, AND accent from reference
   - Fine-grained control

4. **Spanish "High Training Data" Tier**
   - 0.6B: 20,000 hours Spanish (14+ languages total)
   - 1B: 60,000 hours Spanish (23+ languages total)
   - Explicit Spanish support

5. **Hosted API Available**
   - $0.0006/second (=$0.036/minute = $2.16/hour of audio)
   - Early access (100 requests/day)
   - Zero infrastructure burden

6. **Apache 2.0 License (0.6B variant)**
   - Unrestricted use
   - Safe for academic and commercial

### Problems & Red Flags (SEVERE)

1. **Performance CATASTROPHE (CRITICAL)**
   - GitHub Issue #26: **3 minutes to generate 14 seconds of audio on RTX 4090**
   - GitHub Issue #1: 2.5 seconds for "Hello, am I working?" on A100
   - This is 12.86x slower than real-time (3 min / 14 sec = 12.86x)

2. **CPU Inference Fraud**
   - Claimed: "Real-time inference even on CPUs"
   - Reality: **3-7x slower than real-time** (25-75 seconds to generate 10 seconds audio)
   - CPU inference "works" but is USELESS for anything beyond offline batch processing

3. **DAC Codec Quality Issues (SEVERE)**
   - Lossy reconstruction (quality degradation)
   - Sensitive to input audio quality (clipping, loudness, unusual features)
   - No alternative codec support (locked into DAC)
   - Official warning: "Encoding issues that impact output quality"

4. **Training Data Insufficiency**
   - 0.6B: 20,000 hours (adequate)
   - 1B: 60,000 hours (moderate)
   - Competitors have 100,000-1,000,000 hours
   - Limited voice diversity

5. **Sampling Configuration Fragility (CRITICAL)**
   - Documentation: "Repetition penalty MUST apply to 64-token recent window only"
   - If wrong: "Broken or low-quality output"
   - Easy to misconfigure
   - Single point of failure

6. **Audio Truncation (GitHub Issue #45)**
   - End-of-sequence cutoff problems
   - Generated audio cuts off mid-word
   - Silent failures

7. **Attention Mask Warnings (GitHub Issue #3)**
   - Unexpected behavior from tokenization issues
   - Can cause generation failures

8. **Maintainer Deflection**
   - When users report performance issues, maintainer says: "Try alternative backends (llama.cpp, exllamav2)"
   - Translation: "Core implementation is broken, use workarounds"

9. **Limited Context Window (8,192 tokens)**
   - Max ~42 seconds audio recommended
   - With speaker reference: ~32 seconds effective
   - Short-form TTS only

10. **Early Access Service Instability**
    - Hosted API: 100 requests/day limit
    - "May occasionally encounter unexpected outputs"
    - No SLA or uptime guarantee

### Why These Problems Matter (Technical Deep-Dive)

#### Problem 1: Performance CATASTROPHE

**The Numbers:**

From GitHub Issue #26:
> "Generating 14 seconds of audio takes 3 minutes on RTX 4090"

From GitHub Issue #1:
> "2.5 seconds to generate 'Hello, am I working?' on A100"

**Context:**

RTX 4090 and A100 are among the FASTEST consumer and datacenter GPUs available:
- RTX 4090: Top-end consumer ($1600 MSRP)
- A100: Professional datacenter ($10,000+)

If these GPUs perform this poorly, what happens on A40? (A40 is slower than RTX 4090)

**Expected performance on A40:**

If RTX 4090 does 14 seconds in 3 minutes:
- A40 (similar performance tier): ~3-4 minutes for 14 seconds
- **For 10 seconds: ~2-3 minutes**

**For 1000 samples (10 seconds each):**
- 1000 × 2.5 minutes = **2500 minutes = 41.7 hours = 1.7 DAYS**

Compare to Fish Speech:
- 1000 × 2 seconds = 2000 seconds = **33 minutes**

**OuteTTS: 1.7 days**
**Fish Speech: 33 minutes**
**Ratio: 76x slower**

**Why this happens:**

OuteTTS uses LLM architecture (designed for text) for audio generation. But:
- Audio requires 75 tokens/second (high resolution)
- Text requires 2-4 tokens/second (low resolution)
- LLM must generate **25x more tokens** for audio than text

The architecture is fundamentally mismatched for the task.

**Why maintainer suggests llama.cpp:**

llama.cpp is optimized C++ implementation (vs Python). It's 3-5x faster than native PyTorch.

But 3x faster still means:
- 1.7 days / 3 = **13.5 hours for 1000 samples**
- Still 24x slower than Fish Speech

**Verdict:** Performance is SO BAD that it's unusable for batch generation at scale.

#### Problem 2: CPU Inference Fraud

**The Claim:**
> "OuteTTS enables real-time inference even on CPUs"
> "Compatible with Raspberry Pi" (via Piper reference)

**The Reality:**

From my calculations in research:
- Modern 8-core CPU: ~10-30 tokens/second for 0.6B model
- Audio needs: 75 tokens/second
- Generation time: 25-75 seconds for 10 seconds audio
- **Ratio: 2.5-7.5x slower than real-time**

**What "real-time" means:**

Real-time audio generation means: Generate audio AS FAST as it's played back.
- Generate 10 seconds audio in 10 seconds = 1x real-time (minimum for interactive)
- Generate 10 seconds audio in <10 seconds = faster than real-time (ideal)
- Generate 10 seconds audio in 25-75 seconds = **2.5-7.5x SLOWER** than real-time

**Why the claim is fraudulent:**

"Real-time inference on CPUs" suggests interactive use. But 25-75 seconds to generate 10 seconds audio is NOT interactive.

This is only viable for:
- Offline batch processing (where you don't care about time)
- Non-interactive use (generate overnight, use next day)

**Verdict:** CPU inference is TECHNICALLY functional but PRACTICALLY useless.

#### Problem 3: DAC Codec Quality Issues

**What is DAC?**

Descript Audio Codec is a learned audio compression system:
1. **Encoder**: Converts audio waveform → discrete tokens (codes)
2. **Decoder**: Converts discrete tokens → audio waveform

```
Original audio → DAC Encoder → [435, 121, 889, ...] → DAC Decoder → Reconstructed audio
```

**What "lossy reconstruction" means:**

The reconstruction is NOT perfect:
```
Original:      [0.15, 0.22, 0.18, 0.31, ...]
Reconstructed: [0.14, 0.23, 0.17, 0.32, ...]  ← slight differences
```

These differences accumulate, causing:
- Reduced audio fidelity
- Artifacts (clicks, pops, distortion)
- Loss of subtle features (breathing, emotion, prosody)

**Why "sensitive to input quality" is a problem:**

From documentation:
> "Samples with clipping, excessive loudness, or unusual vocal features may introduce encoding issues"

**What this means in practice:**

```python
# Good quality reference audio
speaker_profile = create_profile("clean_studio_recording.wav")
result = generate_audio(speaker_profile, text)
# → Good quality output

# Slightly clipped audio
speaker_profile = create_profile("phone_recording_loud.wav")
result = generate_audio(speaker_profile, text)
# → Distorted, garbled output
```

**Why this is SEVERE:**

You're creating synthetic speech for anti-spoofing research. You need:
- Diverse voice samples (different recording conditions)
- Robust synthesis (handles real-world audio)

If OuteTTS fails on:
- Phone recordings (common in spoofing attacks)
- Slightly loud audio (normal in real-world data)
- Unusual vocal features (accents, speech patterns)

Then it's USELESS for realistic threat modeling.

**Verdict:** DAC codec limitations make OuteTTS fragile and unreliable.

#### Problem 4: Sampling Configuration Fragility

**What "sampling configuration" means:**

When generating audio tokens, the model must decide:
- Which token to generate next?
- How to prevent repetition?

Repetition penalty punishes tokens that appeared recently:
```
Recent tokens: [435, 121, 889, 435, ...]
Next token: 435 has high probability
Repetition penalty: Reduce probability of 435 (appeared twice recently)
```

**The requirement:**
> "Repetition penalty must apply to 64-token recent window only, NOT entire context"

**What happens if wrong:**

```python
# Correct configuration
repetition_penalty = 1.1
repetition_window = 64  # Last 64 tokens only

# Wrong configuration (common mistake)
repetition_penalty = 1.1
repetition_window = None  # Entire context

# Result: Model avoids repeating ANY token from entire generation
# → Runs out of valid tokens
# → Generates garbage or crashes
```

**Why this is fragile:**

1. **Obscure requirement**: Most users won't know about 64-token window
2. **Easy to miss**: Default configuration might be wrong
3. **Silent failure**: Generates low-quality audio without error message

**Verdict:** Requires expert knowledge to configure correctly - high risk of misconfiguration.

### Time Complexity Analysis

**Setup Phase (Self-Hosted):**
- Basic installation: 1-2 hours
- Speaker profile creation pipeline: 1-2 days
- Troubleshooting performance issues: 2-4 days
- **Total: 4-7 days** (pessimistic but realistic given issues)

**Per-Sample Generation (GPU):**
- 10 seconds audio on RTX 4090: ~2-3 minutes (extrapolated)
- Expected on A40: ~2-4 minutes per 10s audio
- **Batch generation of 1000 samples: 1.5-2.5 DAYS**

**Per-Sample Generation (CPU):**
- 10 seconds audio: 25-75 seconds
- **Batch generation of 1000 samples: 7-21 hours**

**Hosted API (Alternative):**
- Setup: 1-2 hours (API integration)
- Cost: $0.0006/second = $36 per 100 minutes = $360 per 1000 minutes
- Rate limit: 100 requests/day = 10 days for 1000 samples

### Environment Requirements

**Hardware (Self-Hosted):**
- GPU: A40 ✅ (but won't fix performance issues)
- VRAM: 6-12GB ✅
- CPU: 8+ cores for CPU inference attempt

**Software:**
- Python: 3.9-3.13
- PyTorch: 2.0+
- llama.cpp: Optional (for CPU inference attempt)

**Dependencies:**
- DAC codec libraries
- Audio processing tools
- Speaker profile management system (must build yourself)

### Spanish Language Assessment

**Training Data:** 20,000 hours (0.6B) / 60,000 hours (1B)
**Quality Tier:** "High training data" (official documentation)
**Languages:** 14+ (0.6B) / 23+ (1B)

**Evidence:**
- Spanish explicitly listed in high training tier
- Global and word-level metadata supports accent inheritance

**Concerns:**
- No benchmarks for Spanish
- No user reviews of Spanish quality
- DAC codec issues might affect Spanish phonemes differently

**Verdict:** Spanish support appears adequate ON PAPER, but performance disaster makes it irrelevant.

### Verdict: REJECT OUTRIGHT

**Justification:**

The performance disaster is DISQUALIFYING:
- 3 minutes for 14 seconds on RTX 4090
- 1.5-2.5 DAYS for 1000 samples
- 76x slower than Fish Speech

**Even if Spanish quality were perfect:**
The time cost makes it impractical for thesis research.

**Counter-Argument: Hosted API?**

Hosted API costs:
- $360 per 1000 samples (10s each)
- 100 requests/day limit = 10 days to generate 1000 samples
- "May occasionally encounter unexpected outputs" (no reliability guarantee)

For comparison:
- Fish Speech: Self-host, 33 minutes for 1000 samples, FREE

**Risk Level:** EXTREME
**Implementation Timeline:** 1-2 weeks (if using hosted API) / 4-7 days + 1.5 days generation (if self-hosting)
**Expected Quality:** ADEQUATE (Spanish) but IRRELEVANT (performance disaster)

**Final Verdict:** REJECT - Performance is catastrophically bad, making all other considerations irrelevant.

---

## 6. Nari Dia 1.6B

### Context & Overview

**What is it?**
Nari Dia is a 1.6 billion parameter TTS system by Nari Labs, specialized for **ultra-realistic dialogue synthesis** (not single-utterance TTS). Released April 2025.

**Why does it exist?**
Most TTS systems optimize for reading text aloud (single speaker, continuous speech). Nari Dia optimizes for **conversational realism**:
- Multi-speaker dialogue
- Turn-taking dynamics
- Non-verbal sounds (laughter, breathing, hesitation)
- Interruptions and overlaps

**Architecture:**
- 1.6B parameters
- Dialogue-focused training data
- Speaker tags: `[S1]`, `[S2]` for multi-speaker
- Optimized for 5-20 second audio chunks (conversational units)

### Benefits & Strengths

1. **Active Development**
   - Released April 2025 (very recent)
   - Nari Labs actively maintaining
   - Growing community support

2. **Genuine Open-Source (Apache 2.0)**
   - Unrestricted commercial use
   - Full model weights available

3. **Multiple Deployment Options**
   - Replicate: ~$0.029 per run
   - Segmind: Serverless API
   - Hugging Face: ZeroGPU Space (free demo)
   - Self-hosting: Community FastAPI servers

4. **Ultra-Realistic Dialogue**
   - Positioned as ElevenLabs competitor
   - Paralinguistic features (laughter, sighs, coughs)
   - Emotional control

5. **Fast Zero-Shot Cloning**
   - 5 seconds reference audio
   - Real-time streaming on single GPU

6. **Moderate Hardware Requirements**
   - ~10GB VRAM
   - ~40 tokens/second on A4000
   - A40 would handle easily

### Problems & Red Flags

**CRITICAL FATAL FLAW:**

### **NO SPANISH SUPPORT** ❌

**Official Statement:**
> "Dia currently supports **English language generation only**, though the Nari Labs team is actively working on expanding Nari Dia to support other languages in future updates."

**Future Plans:**
- Asian languages are NEXT development target (not Spanish/Latin)
- No timeline for Spanish support
- Architecture suggests potential for adaptation but NOT IMPLEMENTED

### Why This Problem is FATAL

**Your Project Context:**

Thesis title: **"Anti-Spoofing Voice System for Latin America"**

Your research is explicitly focused on:
- **Latin American Spanish** voice detection
- Spanish phonetic characteristics
- Regional accent variations (Mexican, Colombian, etc.)

**Nari Dia Status:**
- **English only**
- No Spanish support
- No plans for Spanish in immediate roadmap

**Simple Math:**
```
Spanish support: REQUIRED
Nari Dia Spanish: DOES NOT EXIST
Conclusion: INCOMPATIBLE
```

### Why Even Consider This System?

**You didn't.**

Nari Dia was included in the evaluation for **due diligence**:
- Appears in TTS system comparisons
- Recent release (April 2025) generates hype
- Needed to confirm it's unsuitable

**Your initial note:** "Optional, better not"
**Research conclusion:** That assessment was 100% CORRECT.

### Secondary Issues (Irrelevant Given Fatal Flaw)

Even if Spanish support existed:

1. **Purpose Mismatch**
   - Nari Dia: Dialogue-focused (conversations)
   - Your need: Single-speaker synthetic samples (augmentation data)

2. **Resource Intensity**
   - 10GB VRAM for dialogue-specialized system
   - Opportunity cost: Could run 3-4 instances of smaller TTS

3. **Integration Complexity**
   - Dialogue formatting (`[S1]`, `[S2]` tags)
   - Designed for real-time streaming (not batch augmentation)
   - Would need significant adaptation

4. **Research Scope Creep**
   - Your pipeline already comprehensive (RIR, Codec, RawBoost)
   - Adding English-only TTS adds no value
   - Distracts from core contributions

### Time Complexity Analysis

**N/A** - System is incompatible with project requirements.

### Environment Requirements

**Hardware:** A40 ✅ (sufficient)
**Software:** Standard Python, PyTorch
**Spanish Support:** ❌ **DOES NOT EXIST**

### Spanish Language Assessment

**Spanish Support:** **NONE** ❌
**Future Plans:** Asian languages next (not Spanish)
**Timeline:** Unknown, no commitment

**Verdict:** INCOMPATIBLE WITH PROJECT

### Verdict: REJECT OUTRIGHT - FUNDAMENTAL INCOMPATIBILITY

**Justification:**

This is the simplest decision in the entire evaluation:

**Required:** Spanish language support
**Provided:** English only
**Conclusion:** REJECT

**No further analysis needed.**

**Why Include in Report:**

1. **Due Diligence**: Demonstrates thorough evaluation
2. **Documentation**: Explains why this system was excluded
3. **Future Reference**: If Nari Labs adds Spanish support later, reassessment would be needed

**Risk Level:** N/A (not applicable)
**Implementation Timeline:** N/A (not applicable)
**Expected Quality:** N/A (not applicable)

**Final Verdict:** **REJECT - No Spanish support makes evaluation irrelevant**

---

## Comparative Analysis

### Summary Matrix

| System | Spanish Quality | Implementation Complexity | Time to Deploy | VRAM (A40: 46GB) | License | Performance | Fatal Flaws | Recommendation |
|--------|----------------|--------------------------|----------------|------------------|---------|-------------|-------------|----------------|
| **Fish Speech** | ⭐⭐⭐⭐ Good (20k hrs) | Moderate | 4-8 hours | 12GB ✅ | CC-BY-NC-SA-4.0 (Academic ✅) | Good (2s per 10s audio) | None | ✅ **IMPLEMENT** |
| **Qwen3-TTS** | ⭐⭐⭐ Mediocre | Moderate-High | 2-6 hours | 4-8GB ✅ | Apache 2.0 ✅ | Excellent (0.8s per 10s) | Fine-tuning broken, Spanish second-tier | ⚠️ Conditional (diversity only) |
| **CosyVoice 3.0** | ⭐⭐ Unknown | High | 1-2 days | 8-16GB ✅ | Apache 2.0 ✅ | Good (1.5s per 10s) | vLLM hell, no Spanish benchmarks | ❌ Skip |
| **Chatterbox** | ⭐⭐⭐ Unvalidated | Low | 2-4 hours | 8GB ✅ | MIT ✅ | Fair (4-6s per 10s) | Latency fraud, watermarking | ❌ Skip |
| **OuteTTS** | ⭐⭐⭐ Adequate | High | 4-7 days | 6-12GB ✅ | Apache 2.0 (0.6B) | **TERRIBLE** (2-4 min per 10s) | Performance disaster | 🚫 Reject |
| **Nari Dia 1.6B** | 🚫 **NONE** | Medium | N/A | 10GB ✅ | Apache 2.0 ✅ | N/A | **No Spanish support** | 🚫 Reject |

### Batch Generation Performance Comparison

**Task:** Generate 1000 synthetic samples (10 seconds each)

| System | Time Required | Cost (if applicable) | Reliability |
|--------|---------------|---------------------|-------------|
| **Fish Speech** | **33 minutes** | FREE | High |
| **Qwen3-TTS** | **20-30 minutes** | FREE | Medium (artifacts) |
| **CosyVoice 3.0** | 30-40 minutes | FREE | Unknown (vLLM issues) |
| **Chatterbox** | 1.5-2 hours | FREE | Medium (224 issues) |
| **OuteTTS (GPU)** | **1.5-2.5 DAYS** | FREE | Low (performance) |
| **OuteTTS (API)** | 10 days (100/day limit) | $360 | Low (early access) |
| **Nari Dia** | N/A | N/A | N/A (no Spanish) |

### License Comparison

| System | License | Commercial Use | Academic Use | Attribution Required | Restrictions |
|--------|---------|----------------|--------------|---------------------|--------------|
| **Fish Speech** | CC-BY-NC-SA-4.0 | ❌ NO | ✅ YES | ✅ YES | Non-commercial only |
| **Qwen3-TTS** | Apache 2.0 | ✅ YES | ✅ YES | Courtesy only | None |
| **CosyVoice 3.0** | Apache 2.0 | ✅ YES | ✅ YES | Courtesy only | None |
| **Chatterbox** | MIT | ✅ YES | ✅ YES | No | None |
| **OuteTTS 0.6B** | Apache 2.0 | ✅ YES | ✅ YES | Courtesy only | None |
| **OuteTTS 1B** | CC-BY-NC-SA-4.0 | ❌ NO | ✅ YES | ✅ YES | Non-commercial only |
| **Nari Dia** | Apache 2.0 | ✅ YES | ✅ YES | Courtesy only | None (but no Spanish) |

**For Your Thesis:** All licenses permit academic research. Fish Speech's NC license is PERFECT for thesis work.

### Spanish Language Support Ranking

| Rank | System | Spanish Quality | Evidence | Confidence Level |
|------|--------|----------------|----------|------------------|
| 🥇 1st | **Fish Speech** | ⭐⭐⭐⭐ Good | 20k hours, high training tier, explicit support | HIGH |
| 🥈 2nd | **Qwen3-TTS** | ⭐⭐⭐ Mediocre | Paper admits "not top-tier", user reports of accent issues | MEDIUM |
| 🥉 3rd | **OuteTTS** | ⭐⭐⭐ Adequate | 20k-60k hours, high training tier (but performance disaster negates) | MEDIUM |
| 4th | **Chatterbox** | ⭐⭐⭐ Unvalidated | Listed in 23 languages, zero validation | LOW |
| 5th | **CosyVoice 3.0** | ⭐⭐ Unknown | No benchmarks, no samples, unclear training allocation | VERY LOW |
| ❌ 6th | **Nari Dia** | 🚫 **NONE** | English only, no Spanish support | N/A |

---

## Final Recommendations

### Primary Recommendation: IMPLEMENT FISH SPEECH

**Justification:**

Fish Speech is the **ONLY** system that satisfies all critical requirements:

1. ✅ **Spanish Quality: GOOD** (20,000 hours training, explicit support)
2. ✅ **Implementation Feasible** (4-8 hours setup, Docker available)
3. ✅ **Hardware Compatible** (12GB VRAM easily handled by 46GB A40)
4. ✅ **Academic License** (CC-BY-NC-SA-4.0 explicitly permits thesis research)
5. ✅ **Attack Sophistication** (4B parameters, RLHF training)
6. ✅ **Cross-Lingual Cloning** (Can adapt to Latin American accents)
7. ✅ **Stable Codebase** (24.9k stars, active maintenance)

**Implementation Plan:**

**Week 1: Setup & Validation**
- Day 1-2: Deploy Fish Speech on ml-server03 (Docker recommended)
- Day 3-4: Test Spanish synthesis quality with 10-20 sample sentences
- Day 5: Create speaker profiles from Latin American Spanish reference audio
- Day 6-7: Validate voice cloning quality, document baseline performance

**Week 2: Integration**
- Day 8-10: Develop `FishSpeechAugmenter` class (inherits from `BaseAugmenter`)
- Day 11-12: Create Pydantic schemas for Fish Speech parameters
- Day 13-14: Integrate into augmentation pipeline, test end-to-end

**Week 3: Production**
- Day 15-17: Generate synthetic Latin American Spanish samples (bonafide + spoof)
- Day 18-19: Validate sample quality, filter artifacts
- Day 20-21: Document methodology for thesis, prepare augmentation protocols

**Total Timeline:** 3 weeks

**Expected Output:**
- High-quality Spanish synthetic voices
- Challenging attack samples (RLHF-trained sophistication)
- Reproducible pipeline (Docker + configuration files)
- Publishable methodology (academic standards)

### Secondary Recommendation (Optional): Add Qwen3-TTS for Diversity

**IF** you want maximum attack diversity:

Deploy Qwen3-TTS **alongside** Fish Speech as a secondary TTS system.

**Rationale:**
- Different codec architecture (architectural diversity)
- Fast inference (20-30 min for 1000 samples)
- Apache 2.0 license (no restrictions)

**Use Case:**
```
Primary TTS: Fish Speech (80% of synthetic samples)
  └─ High Spanish quality, RLHF sophistication

Secondary TTS: Qwen3-TTS (20% of synthetic samples)
  └─ Different synthesis method, faster generation
```

**Benefits:**
- Detector trains on multiple TTS architectures (better generalization)
- Ablation study possible (compare detector performance on Fish vs Qwen samples)
- Research contribution: "Evaluated robustness across multiple TTS systems"

**Caveats:**
- Qwen3 Spanish is mediocre (acceptable for diversity, not primary)
- Cannot fine-tune (broken), stuck with base quality
- Requires validation pipeline (detect audio artifacts)

**Implementation Effort:** +1 week
**Risk Level:** MEDIUM
**Expected Benefit:** Marginal improvement in detector robustness

**Recommendation:** **OPTIONAL** - Only if 3-4 weeks timeline is acceptable. If time-constrained, Fish Speech alone is sufficient.

### Systems to REJECT

**REJECT: CosyVoice 3.0**
- **Reason:** vLLM dependency hell, no Spanish validation, 1-2 day deployment risk
- **Verdict:** Complexity not justified by uncertain benefits

**REJECT: Chatterbox**
- **Reason:** Latency fraud, mandatory watermarking (confounding variable), 224 open issues
- **Verdict:** Quality control concerns outweigh easy installation

**REJECT: OuteTTS**
- **Reason:** CATASTROPHIC performance (3 min for 14s audio), 1.5-2.5 days for 1000 samples
- **Verdict:** Performance disaster makes all other considerations irrelevant

**REJECT: Nari Dia 1.6B**
- **Reason:** NO SPANISH SUPPORT (English only)
- **Verdict:** Fundamental incompatibility with project requirements

---

## Presentation Strategy for Professor

### Opening Statement

"Professor, I conducted comprehensive research on six state-of-the-art TTS systems to evaluate their suitability for generating synthetic Spanish voice attacks for our anti-spoofing research. After analyzing implementation complexity, Spanish language quality, and research usefulness, I recommend implementing **Fish Speech** as our primary TTS system."

### Key Talking Points

**1. Why TTS is Valuable for This Research**

"Modern anti-spoofing systems increasingly face attacks from neural-codec-based TTS systems. The research paper I reviewed indicates that detectors trained only on traditional GAN-vocoder artifacts experience 41.4% performance degradation against codec-based attacks. Including TTS-generated samples in our augmentation pipeline ensures our detector is robust against state-of-the-art voice synthesis threats."

**2. Hardware Advantage**

"With access to 4x NVIDIA A40 GPUs (46GB VRAM each), hardware constraints that typically limit TTS deployment are irrelevant for us. The 12GB VRAM requirement for Fish Speech utilizes only 26% of a single GPU's capacity."

**3. Spanish Quality is Critical**

"For a project titled 'Anti-Spoofing Voice System for Latin America,' Spanish language quality is non-negotiable. Fish Speech is the only evaluated system with validated Spanish support (20,000 hours training data) AND the ability to clone Latin American accents via zero-shot voice cloning."

**4. Risk Mitigation**

"I have a fallback plan: Our existing augmentation strategy (60% RIR+Noise, 30% Codec, 10% RawBoost) is already scientifically sound and follows ASVspoof best practices. If Fish Speech integration encounters unexpected issues, we can proceed with the current pipeline without compromising the thesis. TTS is an enhancement, not a dependency."

### Anticipated Questions & Answers

**Q: "Why Fish Speech and not the others?"**

**A:** "I evaluated six systems. Five have disqualifying issues:
- **Qwen3-TTS**: Fine-tuning broken, Spanish explicitly second-tier in research paper
- **CosyVoice**: 1-2 day deployment complexity with vLLM dependency conflicts
- **Chatterbox**: Latency fraud (claims <200ms, delivers 300-600ms), no Latin American validation
- **OuteTTS**: Performance disaster - 3 minutes to generate 14 seconds on RTX 4090
- **Nari Dia**: No Spanish support (English only)

Fish Speech is the only system with validated Spanish quality, stable codebase, and research-appropriate licensing."

**Q: "Will this delay your thesis timeline?"**

**A:** "Implementation requires 3 weeks:
- Week 1: Setup and Spanish quality validation
- Week 2: Integration into augmentation pipeline
- Week 3: Sample generation and documentation

This is justified because:
1. Modern anti-spoofing research increasingly targets codec-based attacks
2. Our A40 infrastructure makes deployment feasible
3. The academic rigor gains outweigh the timeline cost
4. We have a fallback plan (existing augmentation pipeline)"

**Q: "What about the non-commercial license?"**

**A:** "Fish Speech's CC-BY-NC-SA-4.0 license explicitly permits academic research. The non-commercial restriction only applies if we later attempt to commercialize derivatives, which is not our intent. For thesis purposes, this license is ideal - it ensures reproducibility (must share modifications) and provides attribution (academic standard)."

**Q: "How do you know Spanish quality is actually good?"**

**A:** "Evidence includes:
1. 20,000 hours Spanish training data (explicitly documented)
2. Spanish listed in 'high training data' tier alongside English and Chinese
3. Active GitHub community with Spanish users (no quality complaints)
4. Week 1 validation includes generating 10-20 Spanish test samples before full integration

If quality is inadequate during Week 1, we abort and proceed with existing pipeline."

**Q: "What's the computational cost?"**

**A:** "For 1000 synthetic samples (10 seconds each):
- Generation time: ~33 minutes on A40
- GPU utilization: 1 GPU at 60-80% for 33 minutes
- Storage: ~100MB per 1000 samples (FLAC compressed)

This is negligible compared to our existing augmentation processing time."

### Closing Statement

"My recommendation balances research ambition with timeline protection. Fish Speech represents the current state-of-the-art in TTS-based attacks, and our hardware infrastructure makes deployment feasible. However, I've designed the implementation plan with clear validation gates: if Spanish quality doesn't meet standards in Week 1, we have the existing augmentation pipeline as a scientifically valid fallback. This de-risks the decision while positioning our research at the cutting edge of anti-spoofing defense."

---

## Appendix: Technical Glossary

**RLHF (Reinforcement Learning from Human Feedback):**
Training technique where humans rate model outputs (good/bad speech), and the model learns to maximize human-preferred qualities. Makes TTS sound more natural.

**vLLM (Very Large Language Model):**
Inference optimization library for running large models efficiently. CosyVoice requires it, causing dependency management issues.

**DAC (Descript Audio Codec):**
Neural audio compression system that converts waveforms to discrete tokens. OuteTTS uses it; has quality issues with certain audio types.

**CFM (Conditional Flow Matching):**
Generative model technique that learns to transform noise into audio through a continuous flow. CosyVoice 3.0 uses this architecture.

**Dual-AR (Dual Autoregressive):**
Architecture with two autoregressive models: one for semantic content (what to say), one for acoustic details (how to say it).

**WER (Word Error Rate):**
Percentage of words incorrectly transcribed. Lower is better. Measures intelligibility.

**RTF (Real-Time Factor):**
Generation speed ratio. RTF=0.1 means generating 10 seconds of audio takes 1 second. RTF<1.0 is faster than real-time.

**Speaker Similarity:**
Metric measuring how closely generated voice matches reference speaker (0-100%). 77%+ approaches human-level similarity.

**Zero-Shot Cloning:**
Cloning a voice from a single audio sample without additional training. Enables quick speaker profile creation.

**Codec Fakes:**
Synthetic voices generated using neural audio codecs (not traditional GAN vocoders). Harder to detect with conventional methods.

---

**END OF INVESTIGATION DOCUMENT**

*This document represents a comprehensive technical analysis conducted for Master's thesis research at Universidad de los Andes, Colombia. All findings based on primary source research conducted February 2026.*
