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

**Root cause (identified 2026-05-06).** OmniVoice generation occasionally emitted a 200-600 ms fragment of the reference voice before the prompt content. Initial hypothesis was a deterministic per-speaker failure mode, but the actual root cause is in **Step 1 reference preparation**: `app/pipeline/omnivoice_attack/utils/audio_concatenation.py` was slicing the last bonafide file at the exact sample boundary needed to hit a 10 s target duration, landing the cut mid-word. The reference therefore ended abruptly, and OmniVoice's diffusion conditioning -- which is trained to produce continuous speech -- attempted to "complete" the cut-off pattern at the start of generation, manifesting as a fragment of the reference voice in the first frames.

Why this hits OmniVoice and not the other attacks: diffusion-LM is uniquely sensitive to abrupt-end conditioning because it samples from a continuous distribution and treats the end of the reference as a contextual anchor. Autoregressive (Qwen, OuteTTS), VQGAN-based (FishGram), and flow-matching (Chatterbox) backends do not have the same "complete the pattern" failure mode -- they tokenize or quantize the reference and don't try to extrapolate from its trailing edge.

**The fix.** [audio_concatenation.py](../../../app/pipeline/omnivoice_attack/utils/audio_concatenation.py) was rewritten to (1) stop at the last bonafide file that fits without overflowing the 10 s target instead of slicing mid-file, (2) snap to the nearest silent frame within +/- 1 s of target if a single file alone exceeds 10 s (edge case), and (3) always append 200 ms trailing silence so the reference ends on a clean silence boundary regardless of where the last file ends. References are now 3-10 s with guaranteed silence trailers.

**Detection layer (kept as a backstop).** Even with the reference fix, a non-verbal-prefix detector remains in Step 4 to catch any residual or future artifacts:
- New function `detect_nonverbal_prefix_artifact(audio, sample_rate, word_timestamps, silence_floor_db)` in `app/utils/prefix_trimmer.py`. Computes RMS dBFS of the audio interval `[0, word_timestamps[0].start]` (the gap before the first transcribed word). If pre-RMS exceeds `NONVERBAL_PREFIX_RMS_FLOOR_DB = -55.0`, the sample is rejected.
- Threshold rationale: empirically (pre-fix data), OmniVoice artifacts fell in [-25, -22] dBFS and clean samples sat at -120 dBFS silence floor. The threshold sits 30 dB above the artifact band and 65 dB below the silence band.
- **Reject, do not trim.** Trimming risks cutting natural Spanish vowel onsets (e.g., `/e/` of "Eurídice"). Rejection lets the retry loop produce a fresh clean sample.
- Retry loop: OmniVoice is registered in `app/runner/production_runner.py` (key `"3"`) with a new `_execute_omnivoice` method that mirrors the Qwen and FishGram pattern. Rejected samples have their WAVs deleted and Step 3 + Step 4 re-run with `skip_existing=True` up to `MAX_GENERATION_RETRIES = 5` rounds.
- Counter: `nonverbal_prefix_rejection_count` is added to `ValidationResult` for visibility in logs and metrics.

**Validation outcome (2026-05-06, post-fix, 6 samples).** 6/6 passed on the first attempt with **zero non-verbal-prefix rejections** and **zero retries needed**. arf_00295 -- which had bled on all 6 generation attempts pre-fix -- now passes cleanly. Comparison:

| Metric | Pre-fix (with detector + retry) | Post-fix (clean refs) |
|---|---|---|
| Pass rate | 4/6 (66.7%) | **6/6 (100%)** |
| Non-verbal prefix rejections | 2 | **0** |
| Avg WER | 0.0394 | 0.0185 |
| Avg CER | 0.0181 | 0.0083 |
| Avg NISQA MOS | 4.53 | 4.59 |
| Avg Speaker Sim | 0.680 | 0.696 |

**Known limitation.** When Parakeet absorbs the bleed into the first word's start time (`word_timestamps[0].start ≈ 0`), the pre-speech window is empty and the detector returns False. This case did not appear in any of the 12 pre-fix validation trials nor in the 6 post-fix trials. Forced phoneme alignment (Wav2Vec2 CTC or MFA) would close the gap; deferred until production data shows it is necessary.

#### Future improvement: retrofit the audio-concatenation fix to other pipelines

The same `audio[:needed_samples]` mid-file slicing bug exists in **all 7 other attack pipelines**: FishGram, Qwen, OpenVoice, Chatterbox, OuteTTS, CosyVoice, and partial_spoof. References for those pipelines therefore also end mid-phrase. We have not directly observed prefix bleed on those pipelines (their non-diffusion architectures are likely tolerant of the abrupt end), but it is a silent quality issue that could be subtly degrading reference fidelity.

**Decision (2026-05-06): not retrofitting now.** FishGram, Qwen, and OpenVoice production runs are complete; retrofitting would either invalidate ~100k existing samples or be a no-op. Chatterbox and OuteTTS are mid-run (41% and 66% per 2026-04-25 wiki state); interrupting them risks part-old / part-new reference distributions. **Track this as future work**: if any of those pipelines is ever re-run, port the audio_concatenation fix first. Code change is the same ~50-line rewrite already done for OmniVoice.

## Virtual Environment Paths

All venvs are INSIDE the project: `~/ANTI-SPOOFING-VOICE-LATIN-AMERICA/envs/<name>/`

NEVER pip install without activating a venv first. Use `python3` not `python` on ml-server03.

## Related Pages
- [TTS Systems](../state-of-art/tts-systems.md) — technical evaluation
- [Production Runs](../experiments/production-runs.md) — metrics and progress
- [Pipeline Architecture](pipeline-architecture.md) — design patterns
