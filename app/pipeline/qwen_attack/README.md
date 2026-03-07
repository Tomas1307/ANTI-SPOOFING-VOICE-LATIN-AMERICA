# Qwen Attack Pipeline

Generates synthetic voice cloning attacks using **Qwen3-TTS** (1.7B parameters) for anti-spoofing dataset augmentation. This is the **secondary** attack pipeline, providing codec architecture diversity alongside the primary FishGram (Fish Speech) pipeline.

## Purpose

Qwen3-TTS uses a fundamentally different Dual-Track architecture with a different audio codec than Fish Speech. By training anti-spoofing detectors on synthetic samples from both TTS systems, we force the detector to generalize across codec types rather than overfitting to a single one.

- **FishGram**: 80% of synthetic samples (high Spanish quality)
- **Qwen**: 20% of synthetic samples (different codec, adequate Spanish quality)

## Pipeline Steps

| Step | Class | Description |
|------|-------|-------------|
| 1 | `ReferenceAudioPreparator` | Concatenate training audio to 15s reference clips, transcribe with faster-whisper |
| 2 | `TextPromptPreparator` | Assign Spanish texts from Mozilla CV (max 40 words to prevent truncation) |
| 3 | `SpeechGenerator` | Generate speech via local Qwen3-TTS 1.7B model with speaker prompt reuse |
| 4 | `QualityValidator` | DNSMOS + speaker similarity + Qwen artifact detection (truncation, silence, duration) |
| 5 | `OutputFormatter` | Convert to ASVspoof2019 LA format with QWEN3TTS system ID |

## Quick Start

### 1. Environment Setup (on ml-server03)

```bash
# Check GPU availability
nvidia-smi

# Create isolated venv
python3 -m venv envs/qwen_env
source envs/qwen_env/bin/activate

# Install dependencies
pip install -U qwen-tts
pip install -U flash-attn --no-build-isolation
pip install loguru pydantic tqdm pandas librosa soundfile numpy faster-whisper

# Verify
python -c "from qwen_tts import Qwen3TTSModel; print('OK')"
```

### 2. Run Pipeline

```bash
source envs/qwen_env/bin/activate
export CUDA_VISIBLE_DEVICES=1  # Check nvidia-smi first!

# Validation mode (3 speakers, 6 samples)
python -c "
from app.pipeline.qwen_attack import QwenAttackPipeline
from app.pipeline.qwen_attack.settings import settings
settings.VALIDATION_MODE = True
pipeline = QwenAttackPipeline()
pipeline.run()
"
```

### 3. Production Mode

```python
from app.pipeline.qwen_attack import QwenAttackPipeline
from app.pipeline.qwen_attack.settings import settings

settings.VALIDATION_MODE = False
settings.SAMPLES_PER_SPEAKER = 5
pipeline = QwenAttackPipeline()
pipeline.run()
```

## Output

```
data/qwen_output/
├── references/               # 15s reference clips per speaker
├── reference_metadata.json   # Speaker metadata with STT transcripts
├── text_prompts.json         # Text assignments per speaker
├── generated/                # Raw generated WAV files
├── generation_metadata.json  # Generation statistics
├── validated_samples.json    # Quality-validated samples
└── LA/                       # ASVspoof2019 format
    ├── ASVspoof2019_LA_train/
    ├── ASVspoof2019_LA_dev/
    └── ASVspoof2019_LA_eval/
```

## Key Differences from FishGram

| Aspect | FishGram | Qwen |
|--------|----------|------|
| Model | Fish Speech 4B (HTTP API) | Qwen3-TTS 1.7B (local) |
| Server | External API server required | No server needed |
| Reference | Audio only | Audio + STT transcript (Whisper) |
| Text limit | 5-100 words | 5-40 words (truncation prevention) |
| Artifact checks | Standard | Enhanced (truncation, silence, duration) |
| System ID | FISHGRAM | QWEN3TTS |
| Audio ID range | 9000000+ | 8000000+ |

## Known Limitations

1. **Spanish is second-tier**: Qwen optimized for Chinese/English. Spanish quality is adequate but not exceptional.
2. **Fine-tuning broken**: Cannot adapt to Latin American accents. Using base model as-is.
3. **Silent truncation**: Long texts (>40 words) may truncate without error. Pipeline filters aggressively.
4. **transformers pin**: Requires transformers==4.57.3. Isolated venv mandatory.

## Dependencies

- `qwen-tts>=0.1.1` (includes transformers==4.57.3)
- `flash-attn` (optional, 30-40% speedup)
- `faster-whisper` (reference audio transcription)
- `loguru`, `pydantic`, `tqdm`, `pandas`, `librosa`, `soundfile`, `numpy`
