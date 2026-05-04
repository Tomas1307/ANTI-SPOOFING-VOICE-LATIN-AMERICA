# Attack Systems

**Status:** Active
**Last updated:** 2026-05-01
**Source:** app/pipeline/*/strategies/, settings.py files

---

## Overview

Each TTS system runs as an isolated attack pipeline with its own virtual environment on ml-server03. All follow the Strategy pattern — the Facade calls `strategy.generate()` without knowing which TTS backend is running.

## Per-System Configuration

### FishGram (Fish Speech / OpenAudio-S1)
- **Venv:** `envs/fishgram_env/`
- **Model:** 4B params, Dual-AR + Firefly-GAN
- **Deployment:** HTTP API server at `localhost:8080`. Runs separately from the pipeline.
- **Reference:** 15s concatenated reference audio per speaker
- **Key params:** `--checkpoint-path ~/fish-speech/checkpoints/s1-mini/`
- **Spanish data:** 20,000 hours training
- **Production:** DONE, 34,197 passed (95.2%)

### Qwen3-TTS
- **Venv:** `envs/qwen_env/`
- **Model:** Qwen3-TTS (Alibaba), local inference
- **Key params:** `x_vector_only_mode=True`, `do_sample=True`, `temperature`, `top_k`, `top_p`, `repetition_penalty`
- **Critical bug fixed:** Missing sampling params caused garbage output. `x_vector_only_mode=True` required because concatenated 15s reference mismatches ref_text.
- **Production:** DONE, 31,568 passed (87.9%)

### OpenVoice V2
- **Venv:** `envs/openvoice_env/`
- **Model:** MeloTTS base + ToneColorConverter (tau=0.3)
- **Fastest pipeline:** RTF 0.07-0.10x (10-14x real-time)
- **Weakness:** Lowest speaker similarity (0.394) — base voice bleeds through
- **Production:** DONE, 29,626 passed (83.4%)

### Chatterbox (Resemble.ai)
- **Venv:** `envs/chatterbox_env/`
- **Model:** GPT-style autoregressive + CFG + EnCodec + Vocos vocoder
- **Slowest pipeline:** RTF 31-45x
- **Production:** RUNNING, 14,818/35,927 (41%), ETA ~May 13

### OuteTTS
- **Venv:** `envs/outetts_env/`
- **Model:** Llama 3.1 500M + WavTokenizer discrete codes
- **Key params:** `RepetitionPenaltyLogitsProcessorPatch` with `penalty_last_n=64`
- **Known issue:** PyLoudNorm clipping warnings
- **Production:** RUNNING, 23,561/35,927 (66%), ETA ~May 1

### CosyVoice 3.0 (DROPPED)
- Generates Chinese output for Spanish input text. No actual Spanish support despite multilingual claims.

### OmniVoice (k2-fsa) — ADDED 2026-05-01
- **Venv:** `envs/omnivoice_env/`
- **Model:** `k2-fsa/OmniVoice` (HuggingFace), diffusion language model TTS
- **Deployment:** In-process Python API (`OmniVoice.from_pretrained`); no separate server
- **Reference:** 10s concatenated reference audio per speaker (OmniVoice docs warn against >10s)
- **Sample rate:** 24 kHz native, resampled to 16 kHz on FLAC write
- **Spanish data:** 27,559 hours training (one of the largest Spanish coverage in any open zero-shot TTS)
- **Reference text:** Pre-computed with Parakeet TDT (consistent with project STT stack), passed as `ref_text` to `model.generate()`
- **Torch pin:** 2.8.0+cu126 (matches ml-server03 driver 560.35.03; cu128 not used despite upstream recommendation, to avoid driver mismatch)
- **System ID:** `OMNIVOICE`. Audio ID range: 15M-15.99M.
- **Production:** Pipeline written 2026-04-30; first validation run pending. NOT yet in boundary jitter pilot (Qwen first).

## Virtual Environment Paths

All venvs are INSIDE the project: `~/ANTI-SPOOFING-VOICE-LATIN-AMERICA/envs/<name>/`

NEVER pip install without activating a venv first. Use `python3` not `python` on ml-server03.

## Related Pages
- [TTS Systems](../state-of-art/tts-systems.md) — technical evaluation
- [Production Runs](../experiments/production-runs.md) — metrics and progress
- [Pipeline Architecture](pipeline-architecture.md) — design patterns
