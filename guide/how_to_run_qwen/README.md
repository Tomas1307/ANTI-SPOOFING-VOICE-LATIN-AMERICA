# How to Run Qwen Attack Pipeline

## Overview

The Qwen pipeline generates synthetic Spanish voice cloning attacks using **Qwen3-TTS** (1.7B parameters) for anti-spoofing dataset augmentation. It is the **secondary** attack pipeline (20% of synthetic samples), providing codec architecture diversity alongside the primary FishGram pipeline (80%).

Unlike FishGram (which requires a separate HTTP API server), Qwen3-TTS runs as a **local model** loaded directly into GPU memory. This means a single terminal is sufficient — no server management needed.

## Prerequisites

- SSH access to ml-server03
- Virtual environment: `envs/qwen_env/` (Python 3.10+, `qwen-tts` package)
- HABLA bonafide speakers at: `data/bonafide_dataset_by_speaker/`
- Mozilla Common Voice metadata at: `data/cv-corpus-24.0-2025-12-05/es/validated.tsv`

## Step-by-Step Guide

### 1. Check GPU Availability

Before doing anything, verify which GPUs are free:

```bash
nvidia-smi
```

Pick a free GPU (prefer GPU 1 or GPU 3 to avoid interfering with other researchers).

### 2. Run the Pipeline

Unlike FishGram, Qwen only needs **one terminal** (no separate server):

```bash
# SSH into ml-server03
ssh tacosta@ml-server03

# Navigate to the thesis repo
cd ~/ANTI-SPOOFING-VOICE-LATIN-AMERICA

# Pull latest code
git pull

# Activate Qwen virtual environment
source envs/qwen_env/bin/activate

# Set GPU (use a FREE one from nvidia-smi)
export CUDA_VISIBLE_DEVICES=1

# Set SoX library path (required for audio processing)
export PATH="$HOME/local/sox-extract/usr/bin:$PATH"
export LD_LIBRARY_PATH="$HOME/local/sox-extract/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"

# Run validation mode (3 speakers, 6 samples, ~5 minutes)
python -B -c "
from app.pipeline.qwen_attack import QwenAttackPipeline
from app.pipeline.qwen_attack.settings import settings
settings.VALIDATION_MODE = True
pipeline = QwenAttackPipeline()
pipeline.run()
"
```

### 3. Run Production Mode

```bash
# Same setup as above, then:
python -B -c "
from app.pipeline.qwen_attack import QwenAttackPipeline
from app.pipeline.qwen_attack.settings import settings
settings.VALIDATION_MODE = False
settings.SAMPLES_PER_SPEAKER = 5
pipeline = QwenAttackPipeline()
pipeline.run()
"
```

Production generates 162 speakers x 5 samples = 810 synthetic samples.

### 4. Using tmux (Recommended)

tmux keeps processes alive even if your SSH connection drops. Since Qwen loads a 1.7B model and generates many samples, sessions can run for a long time.

```bash
# Start a tmux session for the pipeline
tmux new -s qwen

# (run the pipeline as shown above)
# Then detach: press Ctrl+B, then D
```

Useful tmux commands:
- `tmux ls` - List all sessions
- `tmux attach -t qwen` - Reattach to pipeline session
- `Ctrl+B, D` - Detach from current session (keeps it running)
- `Ctrl+B, [` - Scroll mode (navigate with arrow keys, press `q` to exit)

## Pipeline Steps

The pipeline runs 5 steps automatically:

| Step | Name | Description |
|------|------|-------------|
| 1 | Prepare References | Concatenates training audio to 15s clips, transcribes with faster-whisper |
| 2 | Prepare Texts | Samples Spanish text prompts from Mozilla CV (max 40 words) |
| 3 | Generate Speech | Loads Qwen3-TTS 1.7B locally, generates with speaker prompt reuse |
| 4 | Validate Quality | DNSMOS + speaker similarity + Qwen artifact detection |
| 5 | Format Output | Converts to ASVspoof2019 LA format with QWEN3TTS system ID |

### Step Details

**Step 1 - Prepare References:**
For each HABLA bonafide speaker, concatenates their training audio files into a single ~15s reference clip. Then transcribes the clip using `faster-whisper` (large-v3) to produce a reference transcript. This transcript enables full voice cloning mode (higher quality than embedding-only mode).

**Step 2 - Prepare Texts:**
Loads Spanish transcripts from Mozilla Common Voice and filters to 5-40 words. The 40-word ceiling is critical because Qwen3-TTS silently truncates audio on long texts without raising errors. Assigns N texts per speaker using a seeded RNG for reproducibility.

**Step 3 - Generate Speech:**
Loads the Qwen3-TTS 1.7B model into GPU memory. For each speaker, builds a voice clone prompt once using `create_voice_clone_prompt()`, then reuses it for all N utterances. This avoids redundant feature extraction. After all generation completes, the model is released and GPU memory is freed.

**Step 4 - Validate Quality:**
Applies five checks per sample:
1. Duration anomaly detection (reject if < 0.5s or > 30s)
2. Low energy detection (reject near-silent/garbled outputs, RMS < 0.001)
3. Truncation detection (reject audio suspiciously short for text length)
4. DNSMOS perceptual quality score (minimum 3.5)
5. Speaker similarity via ECAPA-TDNN (minimum 0.65)

**Step 5 - Format Output:**
Converts validated WAV files to FLAC (16kHz, PCM_16) and generates ASVspoof2019 LA protocol files. Uses `QWEN3TTS` as the system identifier and audio IDs in the 8000000+ range (FishGram uses 9000000+ to avoid collisions).

## Configuration

All settings are in `app/pipeline/qwen_attack/settings.py`:

### Pipeline Mode

| Setting | Default | Description |
|---------|---------|-------------|
| `VALIDATION_MODE` | `True` | True = 3 speakers (testing), False = all 162 speakers |
| `VALIDATION_SPEAKERS` | `["arf_00295", "arf_00610", "arf_01523"]` | Speakers for validation mode |
| `SAMPLES_PER_SPEAKER` | `2` | Texts per speaker (2 for validation, 5 for production) |

### Model Configuration

| Setting | Default | Description |
|---------|---------|-------------|
| `QWEN_MODEL_ID` | `Qwen/Qwen3-TTS-12Hz-1.7B-Base` | HuggingFace model ID |
| `QWEN_LANGUAGE` | `Spanish` | Language tag for generation |
| `QWEN_ATTN_IMPLEMENTATION` | `sdpa` | Attention backend (`sdpa` or `flash_attention_2`) |
| `X_VECTOR_ONLY_MODE` | `False` | False = full cloning with transcript (higher quality) |
| `DEVICE` | `cuda:0` | GPU device |
| `DTYPE` | `bfloat16` | Model precision |

### Generation Parameters

| Setting | Default | Description |
|---------|---------|-------------|
| `TOP_K` | `50` | Top-k sampling for main talker |
| `TOP_P` | `1.0` | Nucleus sampling for main talker |
| `TEMPERATURE` | `0.9` | Temperature for main talker |
| `REPETITION_PENALTY` | `1.05` | Repetition penalty |
| `SUBTALKER_TOP_K` | `50` | Top-k for subtalker (secondary codebook decoder) |
| `SUBTALKER_TOP_P` | `1.0` | Nucleus sampling for subtalker |
| `SUBTALKER_TEMPERATURE` | `0.9` | Temperature for subtalker |
| `MAX_NEW_TOKENS` | `2048` | Maximum tokens per generation |

### Whisper STT Configuration

| Setting | Default | Description |
|---------|---------|-------------|
| `WHISPER_MODEL_SIZE` | `large-v3` | Whisper model for reference transcription |
| `WHISPER_COMPUTE_TYPE` | `float16` | Whisper compute precision |

### Quality Validation

| Setting | Default | Description |
|---------|---------|-------------|
| `DNSMOS_THRESHOLD_OVRL` | `3.5` | Minimum audio quality score |
| `SPEAKER_SIM_THRESHOLD` | `0.65` | Minimum speaker similarity score |
| `MIN_AUDIO_DURATION` | `0.5` | Minimum duration in seconds |
| `MAX_AUDIO_DURATION` | `30.0` | Maximum duration in seconds |
| `LOW_ENERGY_THRESHOLD` | `0.001` | RMS threshold for silence detection |
| `MIN_WORDS_PER_SECOND` | `1.5` | Minimum speaking rate for truncation detection |

## Output

Generated files are saved to `data/qwen_output/`:

```
data/qwen_output/
    references/                     # Step 1: 15s reference clips per speaker
    reference_metadata.json         # Step 1: speaker metadata + STT transcripts
    text_prompts.json               # Step 2: assigned text prompts per speaker
    generated/                      # Step 3: raw WAV files
        QWEN3TTS_arf_00295_QWEN_TEXT_00001.wav
        QWEN3TTS_arf_00295_QWEN_TEXT_00002.wav
        ...
    generation_metadata.json        # Step 3: generation results and RTF
    validated_samples.json          # Step 4: quality-filtered samples
    LA/                             # Step 5: ASVspoof2019 format
        ASVspoof2019_LA_train/
            flac/                   # FLAC audio files (LA_T_8000000.flac, ...)
            ASVspoof2019.LA.cm.train.trl.txt  # Protocol file
        ASVspoof2019_LA_dev/
            flac/
            ASVspoof2019.LA.cm.dev.trl.txt
        ASVspoof2019_LA_eval/
            flac/
            ASVspoof2019.LA.cm.eval.trl.txt
```

## Key Differences from FishGram

| Aspect | FishGram | Qwen |
|--------|----------|------|
| Model | Fish Speech 0.5B (HTTP API) | Qwen3-TTS 1.7B (local) |
| Terminals needed | 2 (server + pipeline) | 1 (pipeline only) |
| Virtual environment | `envs/fishgram_env/` | `envs/qwen_env/` |
| Reference audio | Audio only | Audio + STT transcript (Whisper) |
| Text limit | 5-100 words | 5-40 words (truncation prevention) |
| Artifact checks | Standard | Enhanced (truncation, silence, duration) |
| System ID | `FISHGRAM` | `QWEN3TTS` |
| Audio ID range | 9000000+ | 8000000+ |
| Spanish quality | High (fine-tuned) | Adequate (base model, Spanish is second-tier) |

## Troubleshooting

### Model download takes a long time
First run downloads the Qwen3-TTS 1.7B model from HuggingFace (~3.4 GB). This is cached after the first download. Be patient on the first run.

### `pad_token_id` warning during generation
```
The `pad_token_id` is not set for the text model...
```
This is harmless. Qwen3-TTS does not use padding during generation. Safe to ignore.

### CUDA out of memory
- Qwen3-TTS 1.7B needs ~4-5 GB VRAM in bfloat16. Should fit on any A40 (46 GB).
- If still failing, check if another process is using the GPU: `nvidia-smi`
- Ensure `CUDA_VISIBLE_DEVICES` is set to a free GPU

### SoX library not found
If you see `sox` or `libsox` errors:
```bash
export PATH="$HOME/local/sox-extract/usr/bin:$PATH"
export LD_LIBRARY_PATH="$HOME/local/sox-extract/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"
```
Add these to your `~/.bashrc` for persistence.

### `ModuleNotFoundError: No module named 'qwen_tts'`
Ensure the Qwen virtual environment is activated:
```bash
source ~/ANTI-SPOOFING-VOICE-LATIN-AMERICA/envs/qwen_env/bin/activate
```

### Python loading old code
- Always use `python -B` flag to skip bytecode cache
- Or delete cache: `find . -type d -name __pycache__ -exec rm -rf {} +`

### Flash Attention 2 (optional, not required)
Flash Attention 2 provides 30-40% speedup but takes over an hour to compile. The pipeline works fine with `sdpa` (the default). If you want to try it:
```bash
pip install -U flash-attn --no-build-isolation
```
Then change in settings or at runtime:
```python
settings.QWEN_ATTN_IMPLEMENTATION = "flash_attention_2"
```

### Silence or garbled audio output
Qwen3-TTS occasionally produces near-silent or truncated outputs. Step 4 automatically rejects these via artifact detection. If many samples are rejected:
- Check reference audio quality (should be clear speech, ~15s)
- Try reducing `TEMPERATURE` (e.g., 0.7) for more conservative generation
- Check that `QWEN_LANGUAGE` is set to `"Spanish"` (not `"es"`)

## First-Time Setup (One-Time Only)

If setting up from scratch on a new machine. The installation order matters because
`qwen-tts` depends on the `sox` Python package, which requires both the SoX system
binary and `numpy` to be present before it can compile.

### Step 1: Create Virtual Environment

```bash
nvidia-smi  # Check which GPU is free

cd ~/ANTI-SPOOFING-VOICE-LATIN-AMERICA
python3 -m venv envs/qwen_env
source envs/qwen_env/bin/activate
pip install --upgrade pip wheel
```

### Step 2: Install SoX System Binary

The `sox` Python package (a dependency of `qwen-tts`) requires the SoX command-line
binary and its shared libraries. On a shared server without sudo, extract the `.deb`
packages locally:

```bash
# Create local extraction directory
mkdir -p ~/local/sox-extract
cd ~/local/sox-extract

# Download .deb packages (Ubuntu 22.04 Jammy)
apt-get download sox
apt-get download libsox3
apt-get download libsox-fmt-all

# Extract without sudo
dpkg -x sox_*.deb ~/local/sox-extract/
dpkg -x libsox3_*.deb ~/local/sox-extract/
dpkg -x libsox-fmt-all_*.deb ~/local/sox-extract/

# Add to PATH and library path
export PATH="$HOME/local/sox-extract/usr/bin:$PATH"
export LD_LIBRARY_PATH="$HOME/local/sox-extract/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"

# Verify SoX works
sox --version
# Expected: sox: SoX v14.4.2

# Go back to repo
cd ~/ANTI-SPOOFING-VOICE-LATIN-AMERICA
```

If you have sudo access, this is simpler:
```bash
sudo apt-get install sox libsox-fmt-all
```

### Step 3: Install Build Dependencies

These must be installed **before** `qwen-tts` because `sox` (Python) needs them
at compile time:

```bash
source envs/qwen_env/bin/activate
pip install numpy typing_extensions
```

### Step 4: Install qwen-tts

Now the `sox` Python package can compile successfully:

```bash
pip install -U qwen-tts
```

This automatically pins `transformers==4.57.3`. The install takes a few minutes
as it compiles the `sox` C extension.

### Step 5: Install Pipeline Dependencies

```bash
pip install loguru pydantic tqdm pandas librosa soundfile faster-whisper
```

### Step 6: (Optional) Flash Attention 2

Provides 30-40% generation speedup but takes **over 1 hour** to compile from source.
The pipeline works perfectly fine with `sdpa` (the default). Skip this unless you
have time and want the extra speed.

```bash
# WARNING: Compilation takes 1+ hour. Kill with Ctrl+C if you don't want to wait.
pip install -U flash-attn --no-build-isolation
```

If you install it, update the setting at runtime:
```python
settings.QWEN_ATTN_IMPLEMENTATION = "flash_attention_2"
```

### Step 7: Persist PATH Exports

Add the SoX paths to your shell config so they survive new sessions:

```bash
echo '' >> ~/.bashrc
echo '# SoX for Qwen TTS pipeline' >> ~/.bashrc
echo 'export PATH="$HOME/local/sox-extract/usr/bin:$PATH"' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH="$HOME/local/sox-extract/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"' >> ~/.bashrc
source ~/.bashrc
```

### Step 8: Verify Installation

```bash
source envs/qwen_env/bin/activate
python -c "from qwen_tts import Qwen3TTSModel; print('qwen-tts OK')"
python -c "from faster_whisper import WhisperModel; print('faster-whisper OK')"
python -c "import sox; print('sox OK')"
python -c "import loguru; print('loguru OK')"
sox --version
```

All should print OK without errors.

### Installation Order Summary

The dependency chain that must be respected:

```
1. pip, wheel             (build tools)
2. SoX binary + libsox3   (system library, needed by sox Python package)
3. numpy, typing_extensions (build deps for sox Python package)
4. qwen-tts               (installs sox, transformers==4.57.3, torch, etc.)
5. loguru, faster-whisper  (pipeline deps, no special order)
6. flash-attn              (optional, 1hr+ compile)
```

### Environment Isolation

The Qwen venv (`envs/qwen_env/`) is **separate** from the FishGram venv (`envs/fishgram_env/`). This is mandatory because:
- `qwen-tts` pins `transformers==4.57.3` which conflicts with Fish Speech requirements
- Each pipeline has its own dependency tree to avoid version conflicts
- Never install cross-pipeline dependencies in the wrong venv

## Running Individual Steps

You can run steps selectively using the pipeline config:

```python
from app.pipeline.qwen_attack import QwenAttackPipeline
from app.pipeline.qwen_attack.schemas.pipeline_config import QwenPipelineConfig
from app.pipeline.qwen_attack.settings import settings

settings.VALIDATION_MODE = True

# Skip steps 1-2 if references and texts are already prepared
config = QwenPipelineConfig(
    run_step_1=False,
    run_step_2=False,
    run_step_3=True,
    run_step_4=True,
    run_step_5=True,
)
pipeline = QwenAttackPipeline(config=config)
pipeline.run()
```

## Expected Validation Mode Output

When running in validation mode (3 speakers, 2 samples each), expect:

```
Step 1: 3 reference clips prepared (with STT transcripts)
Step 2: 6 text prompts assigned
Step 3: 6/6 samples generated (avg RTF ~1.7)
Step 4: 6/6 passed validation
Step 5: 6 FLAC files in ASVspoof2019 LA format
Total time: ~5 minutes (first run longer due to model download)
```
