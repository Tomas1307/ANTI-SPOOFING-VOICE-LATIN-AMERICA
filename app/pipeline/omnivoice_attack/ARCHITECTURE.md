# OmniVoice Attack Pipeline - Architecture

## 1. High-Level Flow

```
+-----------------------------+      +------------------------------+
| HABLA bonafide speakers     |----->| Step 1: Prepare References   |
| data/bonafide_dataset_*/    |      | - Concat 10s ref audio       |
+-----------------------------+      | - Parakeet TDT ref_text      |
                                     +------------------------------+
                                                     |
                                                     v
+-----------------------------+      +------------------------------+
| Mozilla CV validated.tsv    |----->| Step 2: Prepare Texts        |
| (Spanish transcripts)       |      | - Filter 5-100 words         |
+-----------------------------+      | - Sample N texts per speaker |
                                     +------------------------------+
                                                     |
                                                     v
                                     +------------------------------+
                                     | Step 3: Generate Speech      |
                                     | - OmniVoice.from_pretrained  |
                                     | - generate(text,ref,ref_text)|
                                     | - 24 kHz native WAV          |
                                     +------------------------------+
                                                     |
                                                     v
                                     +------------------------------+
                                     | Step 4: Validate Quality     |
                                     | - Parakeet TDT STT           |
                                     | - WER/CER (hard reject)      |
                                     | - NISQA MOS (informational)  |
                                     | - ECAPA-TDNN sim (info)      |
                                     +------------------------------+
                                                     |
                                                     v
                                     +------------------------------+
                                     | Step 5: Format Output        |
                                     | - Resample 24 -> 16 kHz      |
                                     | - FLAC + protocol files      |
                                     | - LA_T/D/E_12NNNNNN.flac     |
                                     +------------------------------+
                                                     |
                                                     v
                                     data/omnivoice_output/LA/
```

## 2. Design Patterns

| Pattern | Where | Purpose |
|---|---|---|
| **Facade** | [`pipeline_facade.py`](./pipeline_facade.py) | Single entry point, orchestrates 5 steps |
| **Strategy** | [`steps/step_XX_*.py`](./steps/) | Interchangeable, independently testable steps |
| **Singleton** | [`settings.py`](./settings.py), `ParakeetTranscriber` | Single shared configuration / ASR model instance |
| **Dependency Injection** | Step constructors | Override paths and parameters without touching settings |
| **Template Method** | Step `__init__` + `execute()` contract | Uniform step interface across all attack pipelines |

## 3. Step Implementations

### Step 1 - `ReferenceAudioPreparator`

- Iterates speaker directories in `BONAFIDE_DIR` (filtered by `VALIDATION_SPEAKERS` in validation mode).
- Concatenates up to 5 train-split audio files with 0.1s silence padding to reach `REFERENCE_DURATION_TARGET = 10.0s`.
- Writes WAV at `SAMPLE_RATE = 16000` Hz to `references/<speaker>_ref.wav`.
- Transcribes the reference using the project-wide `ParakeetTranscriber` singleton (model: `nvidia/parakeet-tdt-0.6b-v3`).
- Writes `reference_metadata.json` with `{speaker_id, reference_path, reference_text, duration_seconds, split, source_files, bonafide_count}`.

### Step 2 - `TextPromptPreparator`

- Loads Mozilla Common Voice `validated.tsv`, deduplicates `sentence` column.
- Filters by `TEXT_LENGTH_RANGE = (5, 100)` words.
- For each speaker:
  - If `MATCH_BONAFIDE_COUNT`, samples `bonafide_count` texts.
  - Else, samples `SAMPLES_PER_SPEAKER` texts.
  - Sampling without replacement when possible, with replacement otherwise.
- Writes `text_prompts.json` with `{speaker_id: [{text_id, text, length_words, source}]}`.
- Reproducibility: `np.random.seed(RANDOM_SEED)`.

### Step 3 - `SpeechGenerator`

- Loads `OmniVoice.from_pretrained(OMNIVOICE_MODEL_ID, device_map=DEVICE, dtype=DTYPE)`.
- For each `(speaker, text)` pair, calls:
  ```python
  audios = model.generate(
      text=text,
      ref_audio=str(ref_path),
      ref_text=ref_text,            # Parakeet output from Step 1
      num_step=OMNIVOICE_NUM_STEP,  # 32 diffusion steps
      speed=OMNIVOICE_SPEED,        # 1.0
  )
  ```
- Writes WAV at `OMNIVOICE_NATIVE_SAMPLE_RATE = 24000` Hz to `generated/OMNIVOICE_<speaker>_<text_id>.wav`.
- Tracks per-sample generation time and RTF.
- Releases model + `torch.cuda.empty_cache()` after the loop.
- `skip_existing=True` enables resume after interruption.

### Step 4 - `QualityValidator`

Sequence per sample:
1. **Existence check**: skip if audio file missing.
2. **Duration check**: reject if outside `[MIN_AUDIO_DURATION, MAX_AUDIO_DURATION]`.
3. **Silence check**: reject if any consecutive silent region >= 1.0s.
4. **Parakeet STT** with word-level timestamps.
5. **Spurious prefix trim**: detect via `detect_prefix_trim_point`, trim with `trim_audio_prefix`, re-transcribe.
6. **WER/CER** vs original prompt. Reject if `WER > WER_MAX_ACCEPTABLE` or `CER > CER_MAX_ACCEPTABLE`.
7. **NISQA MOS** (informational, not used for rejection).
8. **ECAPA-TDNN cosine similarity** vs reference embedding (informational).

Loads Parakeet via `ParakeetTranscriber()` singleton (same instance reused from Step 1 if pipeline runs in the same process).

Writes:
- `validated_samples.json` (passed samples only).
- Per-sample fields: `wer, cer, transcription, nisqa_mos, speaker_similarity`.
- CSV report via `MetricsWriter.write_validation_csv` for paper plots.

### Step 5 - `OutputFormatter`

- Loads `validated_samples.json` (or falls back to `generation_metadata.json` if validation skipped).
- Creates `LA/ASVspoof2019_LA_{train,dev,eval}/flac/` directories.
- For each sample:
  - Maps `val -> dev` (per ASVspoof2019 convention).
  - Increments per-split counter starting from `AUDIO_ID_START_* = 15_000_000`.
  - Loads audio via `librosa.load(sr=SAMPLE_RATE)` -> implicit resample 24 kHz -> 16 kHz.
  - Writes FLAC `PCM_16` at 16 kHz.
  - Appends protocol entry: `{speaker_id} {audio_id} OMNIVOICE spoof`.
- Writes `ASVspoof2019.LA.cm.{train,dev,eval}.trl.txt` per split.

## 4. Configuration Parameters

All parameters live in [`settings.py`](./settings.py) as `OmniVoiceAttackSettings` (Pydantic `BaseModel`). The module exposes a singleton `settings` instance. See the docstring in `settings.py` for the full attribute description.

Key parameters:

| Parameter | Default | Notes |
|---|---|---|
| `VALIDATION_MODE` | `True` | Toggle validation vs production |
| `VALIDATION_SPEAKERS` | `["arf_00295", "arf_00610", "arf_01523"]` | Same 3 ARF speakers as FishGram for cross-pipeline comparability |
| `OMNIVOICE_MODEL_ID` | `"k2-fsa/OmniVoice"` | HuggingFace identifier |
| `REFERENCE_DURATION_TARGET` | `10.0` | OmniVoice docs: 3-10s recommended |
| `OMNIVOICE_NATIVE_SAMPLE_RATE` | `24000` | Native model output rate |
| `SAMPLE_RATE` | `16000` | Target rate for FLAC output and Parakeet input |
| `OMNIVOICE_NUM_STEP` | `32` | Diffusion steps (16=faster, 32=higher quality) |
| `OMNIVOICE_SPEED` | `1.0` | Speed factor |
| `DTYPE` | `"float16"` | Recommended by OmniVoice docs for NVIDIA GPUs |
| `WER_MAX_ACCEPTABLE` | `0.15` | Hard rejection ceiling |
| `CER_MAX_ACCEPTABLE` | `0.10` | Hard rejection ceiling |
| `OMNIVOICE_SYSTEM_ID` | `"OMNIVOICE"` | Protocol file system ID |
| `AUDIO_ID_START_*` | `15000000` | Avoids collision with partial_spoof main W1/W2/W3 (12-14M) |

## 5. File Structure

```
app/pipeline/omnivoice_attack/
├── __init__.py                            # Public API exports
├── pipeline_facade.py                     # OmniVoiceAttackPipeline (Facade)
├── settings.py                            # OmniVoiceAttackSettings + singleton
├── README.md                              # User-facing docs
├── ARCHITECTURE.md                        # This file
├── schemas/
│   ├── __init__.py
│   ├── pipeline_config.py                 # OmniVoicePipelineConfig
│   ├── reference_result.py                # ReferenceResult
│   ├── text_prompts_result.py             # TextPromptsResult
│   ├── generation_result.py               # GenerationResult
│   ├── validation_result.py               # ValidationResult
│   └── formatting_result.py               # FormattingResult
├── steps/
│   ├── __init__.py
│   ├── step_01_prepare_references.py      # ReferenceAudioPreparator
│   ├── step_02_prepare_texts.py           # TextPromptPreparator
│   ├── step_03_generate_speech.py         # SpeechGenerator
│   ├── step_04_validate_quality.py        # QualityValidator
│   └── step_05_format_output.py           # OutputFormatter
└── utils/
    ├── __init__.py
    ├── audio_concatenation.py             # concatenate_with_padding
    ├── quality_metrics.py                 # detect_silence
    └── reference_transcriber.py           # transcribe_audio (Parakeet wrapper)
```

## 6. External Dependencies

### Project-Wide Utilities (under `app/utils/`)

- `parakeet_transcriber.ParakeetTranscriber` - Singleton wrapper for NVIDIA Parakeet TDT.
- `nisqa_scorer.NisqaScorer` - NISQA MOS prediction.
- `ecapa_similarity.EcapaSimilarity` - ECAPA-TDNN speaker similarity.
- `prefix_trimmer.detect_prefix_trim_point`, `trim_audio_prefix` - Spurious prefix detection.
- `wer_cer.compute_wer`, `compute_cer` - Transcription error metrics.
- `metrics_writer.MetricsWriter` - Validation CSV writer.

### Third-Party

- `omnivoice` - OmniVoice Python API (in `envs/omnivoice_env/` only).
- `nemo_toolkit[asr]` - For Parakeet TDT.
- `torch >= 2.8.0+cu126`, `torchaudio` - GPU inference.
- `librosa`, `soundfile` - Audio I/O.
- `pydantic >= 2.x`, `loguru`, `tqdm` - Standard project deps.

## 7. Differences from Other Attack Pipelines

| Aspect | Other Pipelines | OmniVoice |
|---|---|---|
| Native sample rate | 16/22 kHz | **24 kHz** (resampled in Step 5) |
| Reference duration | 15 s | **10 s** (per upstream advice) |
| RTF | 0.1 - 1.0 typical | **~0.025** (40x realtime) |
| Inference mode | HTTP API (FishGram) or in-process | **In-process Python API** |
| ref_text source | Whisper (Qwen) or Parakeet (CosyVoice) | **Parakeet** (DRY with Step 4) |
| Audio ID range | 6M-11M (TTS), 12M-14M (partial_spoof) | **15M+** |

## 8. Known Risks and Mitigations

| Risk | Mitigation |
|---|---|
| `cu128` torch wheels may not match driver 560.35.03 | Pin to `cu126` in requirements |
| OmniVoice's internal Whisper auto-transcription is slow per-call | Pre-compute `ref_text` with Parakeet in Step 1 |
| 24 kHz vs 16 kHz mismatch in downstream tools | `librosa.load(sr=16000)` resamples on read in Steps 4 and 5 |
| Numeric prompts ("123") may degrade quality | Monitor WER on numeric-heavy texts; consider WeTextProcessing if needed |
| Voice design mode (`instruct=...`) accidentally engaged | Pipeline never passes `instruct`; only `text + ref_audio + ref_text` |
