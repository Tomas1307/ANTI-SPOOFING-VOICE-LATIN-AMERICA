# Production Attack Runner

Interactive console for running attack pipelines at production scale on HABLA v2 dataset (1,567 speakers, ~35,927 bonafide samples).

## Quick Start

```bash
# On ml-server03
export CUDA_VISIBLE_DEVICES=1

# For Qwen3-TTS:
source ~/ANTI-SPOOFING-VOICE-LATIN-AMERICA/envs/qwen_env/bin/activate
python3 app/scripts/run_attack.py

# For FishGram (requires Fish Speech server running on another GPU):
source ~/ANTI-SPOOFING-VOICE-LATIN-AMERICA/envs/fishgram_env/bin/activate
python3 app/scripts/run_attack.py
```

## Menu Flow

```
================================================================
HABLA ANTI-SPOOFING - PRODUCTION ATTACK RUNNER
================================================================

Select pipeline:
  [1] Qwen3-TTS   (local model, 1.7B params)
  [2] FishGram     (Fish Speech HTTP API, 4B params)
  [q] Quit
```

After selection, the runner detects existing output and offers resume:

```
--- Existing output detected ---
  reference_metadata.json  : FOUND
  text_prompts.json        : FOUND (35,927 prompts)
  generated/ WAV files     : 12,450/35,927

  [r] Resume Step 3 from 12,450/35,927, then Steps 4-5
  [f] Fresh start (re-run everything)
  [q] Quit
```

## Resume Behavior

The runner uses existing JSON metadata files as natural checkpoints -- no separate checkpoint system needed:

| State | Resume Action |
|-------|--------------|
| Nothing exists | Fresh start from Step 1 |
| References + prompts exist, generation incomplete | Skip Steps 1-2, resume Step 3 (skip_existing=True), run Steps 4-5 |
| Generation complete, no validation | Skip Steps 1-3, run Steps 4-5 |
| Everything complete | Offer re-run of Steps 4-5 or fresh start |

**How skip_existing works**: Step 3 checks if each output WAV file already exists before generating. Existing files are loaded for metadata but not regenerated. This makes resume seamless after interruptions.

## Retry Mechanism

After Step 3 (generation) + Step 4 (validation), the runner checks for rejected samples (WER/CER too high, duration anomalies, etc.). If any exist:

1. Delete rejected WAV files from `generated/`
2. Re-run Step 3 with `skip_existing=True` (only regenerates deleted WAVs)
3. Re-run Step 4 (re-validates all samples including newly regenerated ones)
4. Re-run Step 5 (reformats output)
5. Repeat until no rejections or `MAX_GENERATION_RETRIES` reached (default: 5)

Samples still rejected after all retries are kept in the validation CSV but excluded from the final LA/ output.

## Dynamic Sample Count

When `MATCH_BONAFIDE_COUNT=True` (default), each speaker gets as many synthetic samples as they have bonafide audio files. This ensures 1:1 bonafide-to-spoof ratio per speaker.

Step 1 counts bonafide files per speaker and stores `bonafide_count` in `reference_metadata.json`. Step 2 reads this count and assigns that many text prompts per speaker.

## FishGram Server Pre-Check

Before running FishGram Step 3, the runner tests the Fish Speech HTTP API server health. If unreachable, it prints the startup command and exits.

```bash
# Start Fish Speech server (separate terminal, separate GPU):
cd ~/fish-speech
source ~/ANTI-SPOOFING-VOICE-LATIN-AMERICA/envs/fishgram_env/bin/activate
CUDA_VISIBLE_DEVICES=3 python3 -m tools.api_server \
    --llama-checkpoint-path checkpoints/s1-mini \
    --decoder-checkpoint-path checkpoints/s1-mini/codec.pth \
    --device cuda --half --listen 0.0.0.0:8080
```

## Output Directory Structure

```
data/qwen_output/           # or data/fishgram_output/
  references/               # 15s reference clips per speaker
  generated/                # Synthetic WAV files (QWEN3TTS_*.wav or FISHGRAM_*.wav)
  reference_metadata.json   # Speaker references with bonafide_count
  text_prompts.json         # Text assignments (variable per speaker)
  generation_metadata.json  # Generation stats per sample
  validated_samples.json    # Samples that passed validation
  metrics/                  # NISQA + ECAPA-TDNN + WER/CER CSV
  LA/                       # ASVspoof2019 format output
```

## Configuration

Key settings in each pipeline's `settings.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `VALIDATION_MODE` | `True` | `True`=3 speakers, `False`=all speakers |
| `MATCH_BONAFIDE_COUNT` | `True` | Dynamic samples per speaker matching bonafide count |
| `SAMPLES_PER_SPEAKER` | `2` | Fallback when MATCH_BONAFIDE_COUNT=False |
| `MAX_GENERATION_RETRIES` | `5` | Max retry rounds for rejected samples |
| `BONAFIDE_DIR` | `data/bonafide_dataset_by_speaker_v2` | HABLA v2 dataset path |
| `WER_MAX_ACCEPTABLE` | `0.15` | Hard WER rejection ceiling |
| `CER_MAX_ACCEPTABLE` | `0.10` | Hard CER rejection ceiling |

## Adding a New Attack Pipeline

To add a new TTS attack (e.g., StyleTTS, XTTS) to the production runner:

1. **Create the pipeline** following `app/pipeline/ARCHITECTURE.md`:
   - 5-step Facade pattern (references, texts, generation, validation, format)
   - Pipeline-scoped `settings.py` with `MATCH_BONAFIDE_COUNT`, `MAX_GENERATION_RETRIES`
   - Pydantic schemas in `schemas/`
   - `skip_existing_step_3` in PipelineConfig, wired to SpeechGenerator

2. **Add bonafide_count tracking** to Step 1:
   ```python
   bonafide_count = sum(
       len(list(speaker_dir.rglob(f"*.{ext}")))
       for ext in ("wav", "flac")
   )
   ref_data["bonafide_count"] = bonafide_count
   ```

3. **Add dynamic sample count** to Step 2:
   ```python
   if settings.MATCH_BONAFIDE_COUNT:
       n_samples = references[speaker_id].get("bonafide_count", settings.SAMPLES_PER_SPEAKER)
   ```

4. **Add skip_existing** to Step 3's SpeechGenerator:
   ```python
   if self.skip_existing and output_path.exists():
       # Load existing file metadata, skip generation
       continue
   ```

5. **Register in production_runner.py**:
   ```python
   # In PIPELINES dict at top of file:
   "3": {
       "name": "new_attack",
       "display": "NewAttack TTS (description)",
       "system_id": "NEWATTACK",
       "output_dir_setting": "data/new_attack_output",
   },
   ```

6. **Add execution method** in ProductionRunner:
   ```python
   def _execute_new_attack(self, run_mode: Dict):
       from app.pipeline.new_attack import NewAttackPipeline, NewAttackConfig, settings
       # Same pattern as _execute_qwen / _execute_fishgram
   ```

7. **Wire it** in `_execute_pipeline()`:
   ```python
   elif self.pipeline_name == "new_attack":
       self._execute_new_attack(run_mode)
   ```
