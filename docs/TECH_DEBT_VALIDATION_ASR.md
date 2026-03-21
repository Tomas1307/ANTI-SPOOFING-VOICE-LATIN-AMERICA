# Tech Debt: Spanish ASR Validation (Step 4)

## Problem

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

## Interim Workaround

Run pipelines with Steps 1-3 and Step 5 only (skip Step 4 validation). The generated audio
has been verified as intelligible Spanish by human listening. Validation can be re-run once
the ASR model issue is resolved.

## Settings to Change

Each pipeline's `settings.py` has:
```python
PARAKEET_MODEL_ID: str = "nvidia/parakeet-tdt-1.1b"
```

This needs to be swapped to a Spanish-capable model ID once the resolution is chosen.
