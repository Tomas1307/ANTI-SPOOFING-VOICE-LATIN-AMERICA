# Qwen Attack Pipeline - Technical Architecture

## Design Patterns

| Pattern | Location | Purpose |
|---------|----------|---------|
| **Facade** | `pipeline_facade.py` | Single entry point orchestrating all 5 steps |
| **Strategy** | `steps/step_XX_*.py` | Interchangeable, independently testable implementations |
| **Dependency Injection** | Step constructors | All parameters optional with settings defaults |
| **Singleton** | `settings.py` module-level | Single shared configuration instance |

## Data Flow

```
bonafide_dataset_by_speaker/
    |
    v
[Step 1: Prepare References]
    - Concatenate training audio -> 15s clips
    - Transcribe with faster-whisper (Spanish)
    - Output: reference_metadata.json (with reference_text)
    |
    v
[Step 2: Prepare Texts]
    - Load Mozilla CV transcripts
    - Filter: 5-40 words (Qwen truncation prevention)
    - Assign N texts per speaker (seeded RNG)
    - Output: text_prompts.json
    |
    v
[Step 3: Generate Speech]
    - Load Qwen3-TTS 1.7B model to GPU
    - Per speaker: build voice_clone_prompt once (reuse)
    - Per text: generate_voice_clone() with full params
    - Release model after completion
    - Output: generation_metadata.json + WAV files
    |
    v
[Step 4: Validate Quality]
    - Qwen artifact checks: truncation, low energy, duration
    - DNSMOS perceptual quality (placeholder)
    - Speaker similarity via ECAPA-TDNN (placeholder)
    - Output: validated_samples.json
    |
    v
[Step 5: Format Output]
    - Convert WAV -> FLAC (16kHz, PCM_16)
    - Generate audio IDs (8000000+ range)
    - Write ASVspoof2019 LA protocol files
    - Output: LA/ directory structure
```

## Key Architecture Decisions

### Local Model vs HTTP API

FishGram uses an external HTTP API server for Fish Speech. Qwen3-TTS runs as a
local model loaded directly into Python via `qwen_tts.Qwen3TTSModel`. This means:

- No separate server process to manage
- Model loaded in Step 3, released after generation completes
- GPU memory freed via `torch.cuda.empty_cache()` after step

### Speaker Prompt Reuse

`create_voice_clone_prompt()` pre-computes speaker features from reference audio.
This is called once per speaker and reused for all N utterances, avoiding
redundant feature extraction. Critical for production mode (162 speakers x 5 samples).

### STT Transcription in Step 1

Reference audio is transcribed using faster-whisper to provide `ref_text` for
full voice cloning mode (`x_vector_only_mode=False`). This produces significantly
higher quality clones than embedding-only mode. The Whisper model is loaded lazily
as a singleton and shared across all speakers.

### Conservative Text Length Filtering

Qwen3-TTS silently truncates audio on long texts (>200 words) without raising
errors. We filter to max 40 words in Step 2 as a conservative safety margin.
This also prevents wasted compute on corrupted samples.

### Artifact Detection Layer

Step 4 adds Qwen-specific checks beyond standard DNSMOS/similarity:
- **Duration anomaly**: Reject audio outside 0.5-30s bounds
- **Low energy**: Reject near-silent outputs (RMS < 0.001)
- **Truncation**: Reject audio suspiciously short for text length

## Configuration

All settings in `settings.py` as Pydantic BaseModel singleton. Key groups:

- **Model**: `QWEN_MODEL_ID`, `QWEN_LANGUAGE`, `QWEN_ATTN_IMPLEMENTATION`
- **Sampling**: `TOP_K`, `TOP_P`, `TEMPERATURE`, `REPETITION_PENALTY`
- **Subtalker**: `SUBTALKER_TOP_K`, `SUBTALKER_TOP_P`, `SUBTALKER_TEMPERATURE`
- **Validation**: `DNSMOS_THRESHOLD_OVRL`, `SPEAKER_SIM_THRESHOLD`
- **Artifacts**: `MIN_AUDIO_DURATION`, `MAX_AUDIO_DURATION`, `LOW_ENERGY_THRESHOLD`
- **Output**: `QWEN_SYSTEM_ID` ("QWEN3TTS"), `AUDIO_ID_START_*` (8000000)

## File Dependencies

```
settings.py          <- All steps import settings
schemas/*            <- Steps return typed Pydantic results
utils/audio_concat   <- Re-exported from fishgram_attack (DRY)
utils/quality_metrics <- Re-exported from fishgram_attack (DRY)
utils/reference_transcriber <- NEW: faster-whisper STT
utils/artifact_detector     <- NEW: Qwen-specific checks
```
