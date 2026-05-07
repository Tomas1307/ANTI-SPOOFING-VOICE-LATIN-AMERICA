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
- **Validation:** PASSED 2026-05-06 (6/6, avg WER 3.94%, NISQA 4.53, ECAPA SIM 0.680). Production run pending.
- **Custom post-processing:** Two-layer prefix detection plus retry loop. See "OmniVoice prefix-bleed handling" below and `methodology/quality-metrics.md` for the detector.

#### OmniVoice prefix-bleed handling (post-processing)
- **Failure mode observed in validation (2026-05-06):** Diffusion sampling occasionally emits a 200-600 ms fragment of the reference voice before the prompt content begins. The fragment is sub-syllabic and Parakeet TDT does not transcribe it, so WER stays at 0.0 even though the audio is clearly contaminated. The existing `detect_prefix_trim_point` in `app/utils/prefix_trimmer.py` (Qwen-style hallucinated-word alignment) is blind to this case.
- **Detection:** New function `detect_nonverbal_prefix_artifact(audio, sample_rate, word_timestamps, silence_floor_db)` in `app/utils/prefix_trimmer.py`. Computes RMS dBFS of the audio interval `[0, word_timestamps[0].start]` (the gap before the first transcribed word). If pre-RMS exceeds `NONVERBAL_PREFIX_RMS_FLOOR_DB`, the sample is rejected.
- **Threshold:** `NONVERBAL_PREFIX_RMS_FLOOR_DB = -55.0` dBFS. Empirically, OmniVoice artifacts fall in [-25, -22] dBFS and clean samples sit at -120 dBFS (silence floor). The threshold sits 30 dB above the artifact band and 65 dB below the silence band.
- **Reject, do not trim:** Detected samples are added to `rejected_samples` with `reason="Non-verbal prefix artifact: pre_RMS X dB > floor -55 dB"` instead of being trimmed in place. Trimming risks cutting the natural Spanish vowel onset (e.g., the `/e/` of "Eurídice"). Rejection lets the retry loop produce a fresh clean sample without surgery.
- **Retry loop:** OmniVoice is registered in `app/runner/production_runner.py` (key `"3"`) with a new `_execute_omnivoice` method that mirrors the Qwen and FishGram pattern. Rejected samples have their WAVs deleted and Step 3 + Step 4 re-run with `skip_existing=True` up to `MAX_GENERATION_RETRIES = 5` rounds.
- **Counter:** `nonverbal_prefix_rejection_count` is added to `ValidationResult` for visibility in logs and metrics.
- **Validation outcome (6 samples, 5 retry rounds):** Detector achieved 100 % recall (12/12 bleed instances flagged across all attempts). Two samples from speaker `arf_00295` bled on every one of 6 generation attempts; the retry mechanism could not produce a clean version. This indicates a deterministic failure tied to either the reference clip or a property of OmniVoice's diffusion conditioning for that speaker. Reference-selection follow-up needed before production.
- **Known limitation:** When Parakeet absorbs the bleed into the first word's start time (`word_timestamps[0].start ≈ 0`), the pre-speech window is empty and the detector returns False. This case did not appear in the 12 validation trials but cannot be ruled out at production scale. Forced phoneme alignment (Wav2Vec2 CTC or MFA) would close the gap; deferred until production data shows it is necessary.

## Virtual Environment Paths

All venvs are INSIDE the project: `~/ANTI-SPOOFING-VOICE-LATIN-AMERICA/envs/<name>/`

NEVER pip install without activating a venv first. Use `python3` not `python` on ml-server03.

## Related Pages
- [TTS Systems](../state-of-art/tts-systems.md) — technical evaluation
- [Production Runs](../experiments/production-runs.md) — metrics and progress
- [Pipeline Architecture](pipeline-architecture.md) — design patterns
