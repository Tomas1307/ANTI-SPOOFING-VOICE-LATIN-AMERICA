# How to Run FishGram Attack Pipeline

## Overview

The FishGram pipeline generates synthetic Spanish voice cloning attacks using Fish Speech (OpenAudio S1-mini, 0.5B parameters) for anti-spoofing dataset augmentation. It runs on ml-server03 using a two-terminal setup: one for the Fish Speech API server and one for the pipeline.

## Prerequisites

- SSH access to ml-server03
- Virtual environment: `envs/fishgram_env/` (Python 3.10, PyTorch >= 2.4 with CUDA 12.4)
- Fish Speech repo cloned at: `~/fish-speech/`
- Model weights downloaded at: `~/fish-speech/checkpoints/s1-mini/`
- HABLA bonafide speakers at: `data/bonafide_dataset_by_speaker/`
- Mozilla Common Voice metadata at: `data/cv-corpus-24.0-2025-12-05/es/validated.tsv`

## Step-by-Step Guide

### 1. Check GPU Availability

Before doing anything, verify which GPUs are free:

```bash
nvidia-smi
```

Pick a free GPU (prefer GPU 1 or GPU 3 to avoid interfering with other researchers).

### 2. Terminal 1: Start Fish Speech API Server

```bash
# SSH into ml-server03
ssh tacosta@ml-server03

# Navigate to Fish Speech repo
cd ~/fish-speech

# Activate fishgram virtual environment
source ~/ANTI-SPOOFING-VOICE-LATIN-AMERICA/envs/fishgram_env/bin/activate

# Set GPU (use a FREE one from nvidia-smi)
export CUDA_VISIBLE_DEVICES=1

# Start API server
python -m tools.api_server \
    --listen 0.0.0.0:8080 \
    --llama-checkpoint-path "checkpoints/s1-mini" \
    --decoder-checkpoint-path "checkpoints/s1-mini/codec.pth" \
    --decoder-config-name modded_dac_vq
```

Wait until you see:
```
INFO:     Uvicorn running on http://0.0.0.0:8080 (Press CTRL+C to quit)
```

**Leave this terminal running.** Do not close it.

### 3. Terminal 2: Run the Pipeline

Open a new SSH session to ml-server03:

```bash
# SSH into ml-server03 (new terminal)
ssh tacosta@ml-server03

# Navigate to the thesis repo
cd ~/ANTI-SPOOFING-VOICE-LATIN-AMERICA

# Pull latest code
git pull

# Activate fishgram virtual environment
source envs/fishgram_env/bin/activate

# Run pipeline (with -B to skip bytecode cache)
python -B test_fishgram_pipeline.py
```

### 4. Using tmux (Recommended)

tmux keeps processes alive even if your SSH connection drops.

```bash
# Start a tmux session for the Fish Speech server
tmux new -s fishspeech

# (start the server as shown in Step 2)
# Then detach: press Ctrl+B, then D

# Start a tmux session for the pipeline
tmux new -s pipeline

# (run the pipeline as shown in Step 3)
```

Useful tmux commands:
- `tmux ls` - List all sessions
- `tmux attach -t fishspeech` - Reattach to server session
- `tmux attach -t pipeline` - Reattach to pipeline session
- `Ctrl+B, D` - Detach from current session (keeps it running)
- `Ctrl+B, [` - Scroll mode (navigate with arrow keys, press `q` to exit)

## Pipeline Steps

The pipeline runs 6 steps automatically:

| Step | Name | Description |
|------|------|-------------|
| 1 | Load Model | Placeholder (model runs as external API server) |
| 2 | Prepare References | Extracts 10-30s reference clips from HABLA bonafide speakers |
| 3 | Prepare Texts | Samples Spanish text prompts from Mozilla Common Voice |
| 4 | Generate Speech | Sends HTTP requests to Fish Speech API for voice cloning |
| 5 | Validate Quality | Checks DNSMOS and speaker similarity thresholds |
| 6 | Format Output | Converts to ASVspoof2019 LA format with protocol files |

## Configuration

All settings are in `app/pipeline/fishgram_attack/settings.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `VALIDATION_MODE` | `True` | True = 3 speakers (testing), False = all 162 speakers |
| `VALIDATION_SPEAKERS` | `["arf_00295", "arf_00610", "arf_01523"]` | Speakers for validation mode |
| `SAMPLES_PER_SPEAKER` | `2` | Number of text prompts per speaker |
| `FISH_SPEECH_API_URL` | `http://localhost:8080` | Fish Speech server URL |
| `FISH_SPEECH_TOP_P` | `0.8` | Nucleus sampling threshold |
| `FISH_SPEECH_TEMPERATURE` | `0.8` | Generation temperature |
| `DNSMOS_THRESHOLD_OVRL` | `3.5` | Minimum audio quality score |
| `SPEAKER_SIM_THRESHOLD` | `0.65` | Minimum speaker similarity score |

## Output

Generated files are saved to `data/fishgram_output/`:

```
data/fishgram_output/
    reference_metadata.json     # Step 2: speaker reference clips info
    text_prompts.json           # Step 3: assigned text prompts
    generation_metadata.json    # Step 4: generation results and RTF
    validated_samples.json      # Step 5: quality-filtered samples
    generated/                  # Step 4: raw WAV files
        FISHGRAM_arf_00295_FISHGRAM_TEXT_00001.wav
        FISHGRAM_arf_00295_FISHGRAM_TEXT_00002.wav
        ...
    formatted/                  # Step 6: ASVspoof2019 LA format
        train/
        dev/
        eval/
        protocols/
```

## Troubleshooting

### Fish Speech server not starting
- Check GPU is free: `nvidia-smi`
- Verify PyTorch version: `python -c "import torch; print(torch.__version__)"` (must be >= 2.4)
- Check model weights exist: `ls ~/fish-speech/checkpoints/s1-mini/`

### Pipeline cannot connect to server
- Verify server is running in Terminal 1 (should show "Uvicorn running on http://0.0.0.0:8080")
- Test manually: `curl http://localhost:8080/`

### Generation failures
- Check server logs in Terminal 1 for error details
- Reference audio must be valid WAV/FLAC files
- Text should be 5-100 words in Spanish

### Python loading old code
- Always use `python -B` flag to skip bytecode cache
- Or delete cache: `find . -type d -name __pycache__ -exec rm -rf {} +`

## First-Time Setup (One-Time Only)

If setting up from scratch on a new machine:

### Option A: Install from requirements.txt (Recommended)

The frozen requirements file captures the exact working environment:

```bash
# 1. Create virtual environment
cd ~/ANTI-SPOOFING-VOICE-LATIN-AMERICA
python3.10 -m venv envs/fishgram_env
source envs/fishgram_env/bin/activate

# 2. Install system dependency (needed for PyAudio)
sudo apt-get install portaudio19-dev

# 3. Clone Fish Speech repo (must exist before pip install)
cd ~
git clone https://github.com/fishaudio/fish-speech.git
cd fish-speech

# 4. Install all dependencies from frozen requirements
cd ~/ANTI-SPOOFING-VOICE-LATIN-AMERICA
pip install -r envs/fishgram_requirements.txt

# 5. Force-install correct PyTorch for CUDA 12.4
pip install --upgrade --force-reinstall torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu124

# 6. Download model weights (requires HuggingFace login)
huggingface-cli login
# Accept terms at: https://huggingface.co/fishaudio/s1-mini
cd ~/fish-speech
huggingface-cli download fishaudio/s1-mini --local-dir checkpoints/s1-mini
```

### Option B: Install from Scratch

If the requirements file is outdated or you need a fresh setup:

```bash
# 1. Create virtual environment
cd ~/ANTI-SPOOFING-VOICE-LATIN-AMERICA
python3.10 -m venv envs/fishgram_env
source envs/fishgram_env/bin/activate

# 2. Install system dependency
sudo apt-get install portaudio19-dev

# 3. Clone Fish Speech repo and install
cd ~
git clone https://github.com/fishaudio/fish-speech.git
cd fish-speech
pip install -e .

# 4. Install PyTorch with CUDA 12.4 (Fish Speech needs >= 2.4)
pip install --upgrade --force-reinstall torch torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install --upgrade torchvision --index-url https://download.pytorch.org/whl/cu124

# 5. Download model weights (requires HuggingFace login)
huggingface-cli login
# Accept terms at: https://huggingface.co/fishaudio/s1-mini
huggingface-cli download fishaudio/s1-mini --local-dir checkpoints/s1-mini

# 6. (Optional) Freeze requirements for future use
cd ~/ANTI-SPOOFING-VOICE-LATIN-AMERICA
pip freeze > envs/fishgram_requirements.txt
```

**Note:** The `fishgram_requirements.txt` includes a `-e git+...` line for Fish Speech. This means the Fish Speech repo must be cloned at `~/fish-speech` before running `pip install -r`.
