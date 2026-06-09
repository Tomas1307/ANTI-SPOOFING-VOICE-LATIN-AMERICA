# Tech Debt: Spanish ASR Validation (Step 4) — RESOLVED

**Status**: RESOLVED (2026-03-21)
**Solution**: Replaced `nvidia/parakeet-tdt-1.1b` (English-only) with `nvidia/parakeet-tdt-0.6b-v3` (25 languages including Spanish, 3.45% WER on FLEURS).

---

## Original Problem

The quality validation step (Step 4) across all attack pipelines uses NVIDIA Parakeet TDT 1.1B
(`nvidia/parakeet-tdt-1.1b`) for STT transcription to compute WER/CER metrics. This model is
**English-only** and cannot transcribe Spanish audio accurately.

When given Spanish speech, Parakeet produces English approximations (e.g., "euridi segio perofue
capura da jeccia pricionera" instead of "euridice huyo pero fue capturada y hecha prisionera"),
resulting in WER close to 1.0 and all samples being rejected.

## Evidence

Tested on OpenVoice-generated audio (2026-03-21):
- Input text: "Euridice huyo, pero fue capturada y hecha prisionera."
- Audio duration: 4.0s (proper Spanish speech, confirmed by human listening)
- Parakeet transcript: "euridi segio perofue capura da jeccia pricionera"
- WER: ~1.0 (all 6 validation samples rejected)

## Affected Pipelines

All four attack pipelines use the same Parakeet-based validation in Step 4:
- `app/pipeline/fishgram_attack/steps/step_04_validate_quality.py`
- `app/pipeline/qwen_attack/steps/step_04_validate_quality.py`
- `app/pipeline/openvoice_attack/steps/step_04_validate_quality.py`
- `app/pipeline/chatterbox_attack/steps/step_04_validate_quality.py`

The shared transcriber singleton is in `app/utils/parakeet_transcriber.py`.

## Attempted Alternatives

1. **stt_multilingual_fastconformer_hybrid_large_pc** (NVIDIA NeMo): Supports Spanish (es) but
   requires language ID passed through an AggregateTokenizer. The `transcribe()` API does not
   expose a `language_id` parameter, making it non-trivial to use for simple file transcription.
   Needs manifest files with `lang` fields or internal configuration changes.

2. **Whisper** (OpenAI): Already installed in openvoice_env (`openai-whisper`). Supports Spanish
   natively with simple API. Viable drop-in replacement but was not the advisor's recommendation.

## Resolution Options

1. **Replace Parakeet with Whisper**: Swap `parakeet_transcriber.py` for a Whisper-based
   transcriber. Minimal code changes (same interface). Confirmed working for Spanish.

2. **Fix NeMo multilingual model usage**: Investigate proper API for
   `stt_multilingual_fastconformer_hybrid_large_pc` with language specification. May require
   creating NeMo manifest JSON files instead of passing file paths directly.

3. **Consult advisor**: Clarify which specific model was intended. The advisor recommended
   "Parakeet" which is English-only. They may have meant the NeMo multilingual family or may
   not have been aware of the language limitation.

## Resolution Applied

**Model**: `nvidia/parakeet-tdt-0.6b-v3` — 0.6B parameter FastConformer TDT with unified
SentencePiece tokenizer (8,192 tokens) supporting 25 European languages.

**Spanish benchmarks**:
- FLEURS: 3.45% WER
- MLS: 4.39% WER
- CoVoST2: 3.41% WER

**Changes made**:
1. Updated `PARAKEET_MODEL_ID` in all 4 pipeline settings files to `nvidia/parakeet-tdt-0.6b-v3`
2. Updated `app/utils/parakeet_transcriber.py` docstrings and default parameter
3. Updated `app/utils/wer_cer.py` docstring rationale (ASCII normalization kept for accent-invariant comparison)
4. Step 4 logic unchanged (model ID loaded from settings at runtime)

**Caveats**:
- Benchmarks are on European/neutral Spanish. HABLA dataset has Latin American accents — WER may be higher.
- No language forcing: Parakeet 0.6b-v3 auto-detects language (no `language="es"` parameter like Whisper).
- VRAM: ~3-4 GB on GPU, trivial on A40 (46GB).

## Previous Workaround (no longer needed)

~~Run pipelines with Steps 1-3 and Step 5 only (skip Step 4 validation).~~
Step 4 can now run normally with the default `run_step_4=True`.
