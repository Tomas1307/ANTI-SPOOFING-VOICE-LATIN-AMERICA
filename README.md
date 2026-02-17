# Anti-Spoofing Voice System for Latin America

A comprehensive voice anti-spoofing research project implementing state-of-the-art data augmentation techniques for detecting spoofed (synthetic) voice samples in Latin American Spanish.

## Project Overview

This project is part of a Master's thesis research focused on voice anti-spoofing detection. It implements a full pipeline for:

1. **Dataset Partitioning** - Speaker-independent train/validation/test splits
2. **Data Augmentation** - Multiple augmentation strategies to increase dataset robustness
3. **Sample Preparation** - Tools for extracting and analyzing augmented samples

## Key Features

### 🎯 Speaker-Independent Splits
- Train/Val/Test use **different speakers** (no overlap)
- Prevents overfitting to specific speakers
- Ensures true generalization capability

### 🔊 Three Augmentation Types

1. **RIR + Noise (60%)** - Room acoustics and background noise
   - Room sizes: Small, Medium, Large
   - Noise sources: Environmental noise, Speech babble, Music
   - SNR range: 0-35 dB

2. **Codec Degradation (30%)** - Telecommunication artifacts
   - Sample rates: 8 kHz, 16 kHz
   - Packet loss: 0-5%
   - Bandpass filtering: 300-3400 Hz
   - Quantization: 8-bit, 12-bit

3. **RawBoost (10%)** - Signal-dependent distortions
   - Linear FIR filtering
   - Nonlinear tanh distortion
   - Signal-dependent additive noise
   - Gain variation (AGC effects)
   - Hard clipping

### 📊 Balanced Mode
- Automatically calculates different augmentation factors for bonafide vs spoof
- Achieves target ratios (e.g., 50/50, 60/40)
- Maintains 25-35% clean data in training set
- Val/Test splits remain 100% clean

## Directory Structure

```
ANTI-SPOOFING-VOICE-LATIN-AMERICA/
├── app/
│   ├── augmenter/              # Augmentation implementations
│   │   ├── base_augmenter.py
│   │   ├── rir_augmenter.py    # RIR + Noise augmentation
│   │   ├── codec_augmenter.py  # Codec degradation
│   │   └── rawboost_augmenter.py # RawBoost distortions
│   │
│   ├── config/                 # Configuration management
│   │   └── augmentation_config.py
│   │
│   ├── scripts/                # Main execution scripts
│   │   ├── run_augmentation.py        # Single augmentation run
│   │   ├── run_augmentation_batch.py  # Batch runs (2x, 3x, 5x, 10x)
│   │   └── augmentation_pipeline.py   # Core pipeline logic
│   │
│   ├── utils/                  # Utility functions
│   │   ├── augmentation_calculator.py # Factor calculations
│   │   └── utils.py            # Audio I/O, processing
│   │
│   ├── tests/                  # Unit tests
│   └── schema.py               # Data classes and enums
│
├── data/
│   ├── partition_dataset_by_speaker/  # Original partitioned data
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   │
│   ├── noise_dataset/          # Augmentation resources
│   │   ├── musan/              # MUSAN noise corpus
│   │   └── RIR/                # Room impulse responses
│   │
│   └── augmented/              # Output augmented datasets
│       ├── augmented_2x_balanced_5050/
│       ├── augmented_3x_balanced_5050/
│       ├── augmented_5x_balanced_5050/
│       └── augmented_10x_balanced_5050/
│
├── prepare_samples_for_martin.py  # Extract samples for analysis
├── extract_samples_putty.sh       # Bash version for server
├── verify_pipeline.py             # Dataset verification
└── README.md                      # This file
```

## Installation

### Prerequisites
- Python 3.8+
- Virtual environment (recommended)

### Setup
```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Required Data
1. **Partitioned dataset**: `data/partition_dataset_by_speaker/`
   - Speaker-independent splits (train/val/test)
   - Files named: `bonafide_SPEAKER_NNNN.wav` or `spoof_TYPE_SPEAKER_NNNN.wav`

2. **MUSAN corpus**: `data/noise_dataset/musan/`
   - Download from: https://www.openslr.org/17/

3. **RIR files**: `data/noise_dataset/RIR/`
   - Room impulse response files for reverb simulation

## Usage

### 1. Single Augmentation Run

```bash
# Default: 50/50 ratio with 3x minimum augmentation
python -m app.scripts.run_augmentation --target_ratio 0.50 --min_factor 3x

# Custom: 60/40 ratio (more bonafide) with 5x minimum
python -m app.scripts.run_augmentation --target_ratio 0.60 --min_factor 5x

# With custom paths
python -m app.scripts.run_augmentation \
    --target_ratio 0.50 \
    --min_factor 3x \
    --voices data/partition_dataset_by_speaker \
    --musan data/noise_dataset/musan \
    --rir data/noise_dataset/RIR \
    --output data/augmented \
    --seed 42
```

**Output:**
- Augmented dataset: `data/augmented/augmented_3x_balanced_5050/LA/`
- Log file: `data/augmented/augmented_3x_balanced_5050/3x.txt`

### 2. Batch Augmentation (2x, 3x, 5x, 10x)

```bash
# Run all factors sequentially
python -m app.scripts.run_augmentation_batch

# Custom factors
python -m app.scripts.run_augmentation_batch --factors 2x 5x 10x

# Different ratio
python -m app.scripts.run_augmentation_batch --target_ratio 0.60
```

Each run creates its own directory and log file.

### 3. Extract Samples for Analysis

**On Windows (with Python):**
```bash
# Extract 10 speakers from 10x augmentation
python prepare_samples_for_martin.py

# Extract 15 speakers
python prepare_samples_for_martin.py --n_speakers 15

# Use different augmentation factor
python prepare_samples_for_martin.py --factor 5x
```

**On Linux/Server (with Bash):**
```bash
# Extract 10 speakers (default)
bash extract_samples_putty.sh

# Extract 15 speakers
bash extract_samples_putty.sh 15
```

**Output:** `datos_martin/` with FLAC files, protocol, and README

### 4. Verify Dataset

```bash
python verify_pipeline.py
```

Checks:
- Speaker independence across splits
- Clean data availability
- Augmentation factor correctness
- File naming conventions

## Output Format (ASVspoof2019 LA)

### Directory Structure
```
augmented_3x_balanced_5050/
└── LA/
    ├── ASVspoof2019_LA_train/
    │   ├── flac/
    │   │   ├── LA_T_0000001.flac
    │   │   ├── LA_T_0000002.flac
    │   │   └── ...
    │   └── ASVspoof2019.LA.cm.train.trn.txt
    │
    ├── ASVspoof2019_LA_dev/
    │   ├── flac/
    │   └── ASVspoof2019.LA.cm.dev.trl.txt
    │
    └── ASVspoof2019_LA_eval/
        ├── flac/
        └── ASVspoof2019.LA.cm.eval.trl.txt
```

### Protocol File Format
```
SPEAKER_ID AUDIO_ID SYSTEM_ID KEY
```

**Example:**
```
clm_07049 LA_T_0000001 - bonafide
clm_07049 LA_T_0000002 CODEC_8K_LOSS2PCT_BP bonafide
clm_07049 LA_T_0000003 RIR_SMALL_NOI_SNR12 bonafide
clm_07049 LA_T_0000004 RAWBOOST_LF_AN spoof
```

**Fields:**
- `SPEAKER_ID`: Speaker identifier (target speaker)
- `AUDIO_ID`: Unique sample identifier (LA_T_* for train)
- `SYSTEM_ID`: Augmentation label (`-` = clean/original)
- `KEY`: Ground truth (`bonafide` or `spoof`)

## Augmentation Labels

### RIR + Noise
**Format:** `RIR_{ROOM}_{NOISE}_SNR{DB}`

| Component | Values | Example |
|-----------|--------|---------|
| ROOM | SMALL, MEDIUM, LARGE | SMALL |
| NOISE | NOI, SPE, MUS | NOI |
| SNR | Integer dB (0-35) | 12 |

**Example:** `RIR_SMALL_NOI_SNR12`
- Small room reverb
- Environmental noise
- 12 dB signal-to-noise ratio

### Codec Degradation
**Format:** `CODEC_{SR}K_LOSS{PCT}PCT[_BP][_Q{BITS}]`

| Component | Values | Example |
|-----------|--------|---------|
| SR | 8, 16 (kHz) | 8 |
| LOSS | 0-5 (%) | 2 |
| _BP | (optional) | _BP |
| _Q | 8, 12 (optional) | _Q8 |

**Example:** `CODEC_8K_LOSS2PCT_BP_Q8`
- 8 kHz codec simulation
- 2% packet loss
- 300-3400 Hz bandpass filter applied
- 8-bit quantization

### RawBoost
**Format:** `RAWBOOST_{OP1}[_{OP2}...]`

| Abbreviation | Operation |
|--------------|-----------|
| LF | Linear FIR filtering |
| NL | Nonlinear tanh distortion |
| AN | Additive noise |
| GV | Gain variation |
| CL | Hard clipping |

**Example:** `RAWBOOST_LF_AN_GV`
- Linear filter applied
- Additive noise
- Gain variation

## Configuration

### Augmentation Factors
Pre-configured strategies: **3x**, **5x**, **10x**

Custom factors supported: Any integer (e.g., `2x`, `7x`, `15x`)

### Target Ratios
- **0.50** → 50/50 (bonafide/spoof)
- **0.60** → 60/40 (more bonafide)
- **0.70** → 70/30 (more bonafide)

### Distribution
- RIR + Noise: 60%
- Codec: 30%
- RawBoost: 10%

## Example Calculation

**Input (train split):**
- 80 bonafide originals
- 176 spoof originals
- Total: 256 originals

**Target:** 50/50 ratio with 3x minimum augmentation

**Steps:**
1. Total needed: 256 × 3 = 768 files
2. Bonafide target: 768 × 0.50 = 384 files
3. Spoof target: 768 × 0.50 = 384 files
4. Bonafide factor: 384 / 80 = 4.8 → **5x**
5. Spoof factor: 384 / 176 = 2.2 → **2x**

**Result:**
- Bonafide: 80 × 5 = 400 files (53.2%)
- Spoof: 176 × 2 = 352 files (46.8%)
- Total: 752 files (2.9x actual)
- Clean ratio: 256/752 = 34.0% ✓

## Log Files

Each augmentation run creates a log file: `{min_factor}.txt`

**Contents:**
1. Run configuration (paths, ratio, seed)
2. Dataset statistics (original counts)
3. Calculation summary (factors, projections)
4. Per-split processing (file counts)
5. Protocol file output
6. Final report (achieved ratios, statistics)

**Example:** `data/augmented/augmented_3x_balanced_5050/3x.txt`

## Testing

```bash
# Run all tests
python -m pytest app/tests/

# Run specific test
python -m app.tests.test_augmentation_batch
python -m app.tests.test_codec_augmenter

# Run verification
python verify_pipeline.py
```

## Research References

This implementation is based on:

1. **RIR + Noise Augmentation**
   - Sánchez et al. (2024) - Room impulse response augmentation for robust speech recognition

2. **Codec Degradation**
   - ASVspoof 2019 Challenge - Channel degradation strategies for anti-spoofing

3. **RawBoost**
   - Tak, H., et al. (2022) - "RawBoost: A Raw Data Boosting and Augmentation Method applied to Automatic Speaker Verification Anti-Spoofing"

## Common Issues

### Missing MUSAN dataset
```bash
# Download MUSAN corpus
wget https://www.openslr.org/resources/17/musan.tar.gz
tar -xzf musan.tar.gz -C data/noise_dataset/
```

### Permission denied on Linux
```bash
chmod +x extract_samples_putty.sh
bash extract_samples_putty.sh
```

### Out of memory during augmentation
- Reduce `n_speakers` when extracting samples
- Process in smaller batches
- Use lower augmentation factors (2x, 3x instead of 10x)

## File Naming Conventions

**Original files:**
- Bonafide: `bonafide_SPEAKER_NNNN.wav`
- Spoof: `spoof_TYPE_SPEAKER_NNNN.wav`

**Augmented files:**
- Train: `LA_T_NNNNNNN.flac` (7-digit counter)
- Dev: `LA_D_NNNNNNN.flac`
- Eval: `LA_E_NNNNNNN.flac`

**Speakers:**
- Format: `prefix_NNNNN` (e.g., `clm_07049`, `arf_00295`)
- Prefix indicates dialect/origin

## Contributors

Master's Thesis Research Project
Universidad de los Andes, Colombia

## License

Research use only. Please cite appropriately if using this code or methodology in your research.

---

**Last Updated:** February 2026
**Python Version:** 3.8+
**Status:** Active Development
