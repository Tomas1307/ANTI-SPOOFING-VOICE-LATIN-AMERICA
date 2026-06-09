# FishGram Attack Pipeline

Voice cloning attack generation using **Fish Speech** (4B parameters) for anti-spoofing dataset augmentation.

## Overview

This pipeline generates synthetic Spanish voice cloning attacks to augment the HABLA anti-spoofing dataset. It represents modern neural TTS attack vectors that current augmentation pipelines (RIR, Codec, RawBoost) do not cover.

**Key Features**:
- Zero-shot voice cloning from 15-second reference audio
- Spanish language support (20,000 hours training data)
- Quality validation via DNSMOS and speaker verification
- ASVspoof2019 LA format output
- Validation mode (6 samples) vs Production mode (810 samples)

## Installation

### 1. Create Separate Virtual Environment

```bash
cd /path/to/ANTI-SPOOFING-VOICE-LATIN-AMERICA

# Create venv
python -m venv envs/fishgram_env

# Activate
source envs/fishgram_env/bin/activate  # Linux/Mac
# OR
envs\fishgram_env\Scripts\activate     # Windows
```

### 2. Install Dependencies

```bash
# Install PyTorch with CUDA 12.1
pip install torch==2.1.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
pip install -r envs/fishgram_requirements.txt
```

### 3. Install Fish Speech

```bash
# TODO: Add Fish Speech installation instructions
# (Fish Speech may require manual installation from GitHub)
```

### 4. Download Models

The pipeline uses two pre-trained models:
- **Fish Speech (4B)**: For TTS generation (manual download required)
- **ECAPA-TDNN**: For speaker verification (auto-downloaded by SpeechBrain)

## Usage

### Validation Mode (Quick Test)

Generate 6 samples from 3 speakers to validate Spanish quality:

```python
from app.pipeline.fishgram_attack import FishGramAttackPipeline, settings

# Configure validation mode
settings.VALIDATION_MODE = True
settings.SAMPLES_PER_SPEAKER = 2
settings.VALIDATION_SPEAKERS = ["arf_00295", "cof_12345", "clm_67890"]

# Run pipeline
pipeline = FishGramAttackPipeline()
output_dir = pipeline.run()

print(f"Output: {output_dir}")
# Output: data/fishgram_output/LA/
```

**Expected Runtime**: ~5 minutes on NVIDIA A40

### Production Mode (Full Dataset)

Generate 810 samples from all 162 speakers:

```python
from app.pipeline.fishgram_attack import FishGramAttackPipeline, settings

# Configure production mode
settings.VALIDATION_MODE = False
settings.SAMPLES_PER_SPEAKER = 5

# Run pipeline
pipeline = FishGramAttackPipeline()
output_dir = pipeline.run()

print(f"Output: {output_dir}")
# Output: data/fishgram_output/LA/
```

**Expected Runtime**: ~60 minutes on NVIDIA A40

### Command Line Execution

```bash
# Activate venv
source envs/fishgram_env/bin/activate

# Run pipeline
python -c "
from app.pipeline.fishgram_attack import FishGramAttackPipeline, settings
settings.VALIDATION_MODE = True  # or False for production
pipeline = FishGramAttackPipeline()
pipeline.run()
"
```

## Output Structure

```
data/fishgram_output/
├── references/                          # 15-second reference audio per speaker
│   ├── arf_00295_ref.wav
│   ├── cof_12345_ref.wav
│   └── ...
├── generated/                           # Raw generated samples
│   ├── FISHGRAM_arf_00295_TEXT_00001.wav
│   └── ...
├── reference_metadata.json              # Reference audio metadata
├── text_prompts.json                    # Text prompt assignments
├── generation_metadata.json             # Generation results
├── validated_samples.json               # Quality-validated samples
└── LA/                                  # ASVspoof2019 LA format (final output)
    ├── ASVspoof2019_LA_train/
    │   ├── flac/
    │   │   ├── LA_T_9000001.flac
    │   │   └── ...
    │   └── ASVspoof2019.LA.cm.train.trl.txt
    ├── ASVspoof2019_LA_dev/
    │   ├── flac/
    │   └── ASVspoof2019.LA.cm.dev.trl.txt
    └── ASVspoof2019_LA_eval/
        ├── flac/
        └── ASVspoof2019.LA.cm.eval.trl.txt
```

## Pipeline Steps

### Step 1: Load Fish Speech Model
- Initializes Fish Speech 4B TTS model
- Validates VRAM usage (~12GB)
- Performs warmup inference

### Step 2: Prepare Reference Audio
- Concatenates 3-4 training samples per speaker
- Creates 15-second reference clips
- Preserves speaker split (train/val/test)

### Step 3: Prepare Text Prompts
- Loads Spanish transcripts from Mozilla Common Voice
- Assigns N texts per speaker (N=2 validation, N=5 production)
- Seeded random sampling for reproducibility

### Step 4: Generate Synthetic Speech
- TTS generation using Fish Speech
- Voice cloning from reference audio
- Tracks generation time and RTF (Real-Time Factor)

### Step 5: Validate Quality
- **DNSMOS**: Perceptual quality (threshold ≥3.5)
- **Speaker Similarity**: Cosine similarity (threshold ≥0.65)
- **Silence Detection**: Rejects samples with excessive silence
- Expected pass rate: 85-95%

### Step 6: Format Output
- Converts to FLAC (16kHz, 16-bit)
- Generates ASVspoof2019 LA protocol files
- Audio IDs: 9000000-9999999 range

## Configuration

All settings in `app/pipeline/fishgram_attack/settings.py`:

```python
# Validation mode toggle
VALIDATION_MODE = True  # False for production

# Generation parameters
SAMPLES_PER_SPEAKER = 2  # 2 for validation, 5 for production
RANDOM_SEED = 42
REFERENCE_DURATION_TARGET = 15.0  # seconds

# Quality thresholds
DNSMOS_THRESHOLD_OVRL = 3.5  # Perceptual quality
SPEAKER_SIM_THRESHOLD = 0.65  # Voice cloning accuracy

# Paths
BONAFIDE_DIR = Path("data/bonafide_dataset_by_speaker")
OUTPUT_DIR = Path("data/fishgram_output")
CV_METADATA_PATH = Path("data/cv-corpus-24.0-2025-12-05/es/validated.tsv")
```

## Quality Metrics

### DNSMOS (Perceptual Quality)
- **3.5+**: Good quality (acceptable)
- **3.8+**: Excellent quality
- **<3.5**: Rejected (unnatural, distorted)

### Speaker Similarity (Voice Cloning)
- **0.65+**: Successful clone (acceptable)
- **0.75+**: High confidence clone
- **<0.65**: Rejected (doesn't match reference speaker)

## Troubleshooting

### VRAM Out of Memory
- **Issue**: 12GB model doesn't fit on GPU
- **Solution**: Use CPU offloading or float16 precision
```python
settings.DTYPE = "float16"  # instead of bfloat16
```

### Low Quality Pass Rate (<80%)
- **Issue**: Too many samples rejected
- **Solution**: Lower quality thresholds
```python
settings.DNSMOS_THRESHOLD_OVRL = 3.2  # instead of 3.5
settings.SPEAKER_SIM_THRESHOLD = 0.60  # instead of 0.65
```

### Fish Speech Not Installed
- **Issue**: `ModuleNotFoundError: No module named 'fish_speech'`
- **Solution**: Install Fish Speech manually from GitHub
```bash
pip install git+https://github.com/fishaudio/fish-speech.git
```

## Integration with Augmentation Pipeline

After generating FishGram attacks, merge with existing augmentation dataset:

```bash
# Merge protocol files
cat data/fishgram_output/LA/ASVspoof2019_LA_train/ASVspoof2019.LA.cm.train.trl.txt \
    >> data/augmented/LA/ASVspoof2019_LA_train/protocol.txt

# Copy FLAC files
cp data/fishgram_output/LA/ASVspoof2019_LA_train/flac/*.flac \
   data/augmented/LA/ASVspoof2019_LA_train/flac/
```

## Reproducibility

- **Random Seed**: Set via `settings.RANDOM_SEED = 42`
- **Deterministic Steps**: Steps 1-3 and 6 are deterministic
- **Non-Deterministic**: Step 4 (TTS generation) may vary slightly
- **Validation**: Step 5 uses fixed thresholds

## References

- **Fish Speech**: [GitHub](https://github.com/fishaudio/fish-speech)
- **Paper**: "Fish Speech: Leveraging Large Language Models for Advanced Text-to-Speech" (arXiv:2411.01156)
- **SpeechBrain ECAPA-TDNN**: [HuggingFace](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb)
- **ASVspoof2019**: [Official Website](https://www.asvspoof.org/)

## License

This pipeline is part of the HABLA anti-spoofing research project. Fish Speech is licensed under CC-BY-NC-SA-4.0 (academic research permitted).
