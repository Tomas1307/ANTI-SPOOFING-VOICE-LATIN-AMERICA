# OmniVoice Attack Pipeline

Voice cloning attack generation using **OmniVoice** (k2-fsa) for anti-spoofing dataset augmentation.

## Overview

OmniVoice is a state-of-the-art massively multilingual zero-shot TTS model supporting **646 languages**, built on a novel diffusion language model architecture. Spanish is among its best-supported languages with **27,559 hours** of training data, making it well suited for Latin American Spanish synthesis.

This pipeline generates synthetic Spanish voice cloning attacks to augment the HABLA anti-spoofing dataset.

**Key Features**:
- Zero-shot voice cloning from 3-10 second reference audio
- Spanish language support (27,559 hours training data)
- Native 24 kHz output, resampled to 16 kHz for ASVspoof2019 LA
- Quality validation via Parakeet TDT WER/CER, NISQA MOS, and ECAPA-TDNN similarity
- ASVspoof2019 LA format output
- Validation mode (3 speakers) vs Production mode (all speakers)
- Very fast inference: RTF ~0.025 (40x faster than real-time per upstream docs)

## Installation

### 1. Create Isolated Virtual Environment (on ml-server03)

```bash
cd ~/ANTI-SPOOFING-VOICE-LATIN-AMERICA

python -m venv envs/omnivoice_env
source envs/omnivoice_env/bin/activate
```

### 2. Install PyTorch (matched to driver 560.35.03 / CUDA 12.6)

```bash
pip install torch==2.8.0 torchaudio==2.8.0 --extra-index-url https://download.pytorch.org/whl/cu126
```

### 3. Install OmniVoice and Pipeline Dependencies

```bash
pip install -r envs/omnivoice_requirements.txt
```

### 4. Verify Installation

```bash
python -c "from omnivoice import OmniVoice; print('OmniVoice OK')"
python -c "import nemo.collections.asr; print('NeMo ASR (Parakeet) OK')"
```

The OmniVoice checkpoint (`k2-fsa/OmniVoice`) and the Parakeet TDT 0.6b-v3 model are downloaded automatically from HuggingFace on first run.

## Usage

All execution must occur on **ml-server03** with a single GPU (do not use the local Windows machine).

### Validation Mode (Quick Test)

```bash
export CUDA_VISIBLE_DEVICES=1
source envs/omnivoice_env/bin/activate

python -c "
from app.pipeline.omnivoice_attack import OmniVoiceAttackPipeline, settings

settings.VALIDATION_MODE = True
settings.SAMPLES_PER_SPEAKER = 2
settings.MATCH_BONAFIDE_COUNT = False

pipeline = OmniVoiceAttackPipeline()
output_dir = pipeline.run()
print(f'Output: {output_dir}')
"
```

### Production Mode (Full Dataset)

```bash
export CUDA_VISIBLE_DEVICES=1
source envs/omnivoice_env/bin/activate

python -c "
from app.pipeline.omnivoice_attack import OmniVoiceAttackPipeline, settings

settings.VALIDATION_MODE = False
settings.MATCH_BONAFIDE_COUNT = True

pipeline = OmniVoiceAttackPipeline()
output_dir = pipeline.run()
print(f'Output: {output_dir}')
"
```

## Pipeline Steps

| Step | Class | Output |
|---|---|---|
| 1 | `ReferenceAudioPreparator` | `references/<speaker>_ref.wav` (10s) + `reference_metadata.json` (with Parakeet transcript) |
| 2 | `TextPromptPreparator` | `text_prompts.json` (Mozilla CV Spanish prompts per speaker) |
| 3 | `SpeechGenerator` | `generated/OMNIVOICE_<speaker>_<text_id>.wav` (24 kHz native) + `generation_metadata.json` |
| 4 | `QualityValidator` | `validated_samples.json` + per-sample WER/CER/NISQA/SpeakerSim |
| 5 | `OutputFormatter` | `LA/ASVspoof2019_LA_{train,dev,eval}/flac/LA_{T,D,E}_NNNNNNN.flac` + protocols |

## Output Format

ASVspoof2019 LA standard, identical to other attack pipelines in this project. Audio IDs occupy the **15_000_000+** range to avoid collisions with FishGram (9M), Qwen (8M), OpenVoice (7M), Chatterbox (6M), OuteTTS (10M), and CosyVoice (11M).

```
data/omnivoice_output/
├── references/
│   └── <speaker>_ref.wav
├── reference_metadata.json
├── text_prompts.json
├── generated/
│   └── OMNIVOICE_<speaker>_<text_id>.wav
├── generation_metadata.json
├── validated_samples.json
└── LA/
    ├── ASVspoof2019_LA_train/
    │   ├── flac/LA_T_15000001.flac
    │   └── ASVspoof2019.LA.cm.train.trl.txt
    ├── ASVspoof2019_LA_dev/
    └── ASVspoof2019_LA_eval/
```

## Quality Thresholds

| Metric | Threshold | Action |
|---|---|---|
| WER | <= 0.15 | Hard rejection |
| CER | <= 0.10 | Hard rejection |
| Audio duration | 0.5 - 30.0 s | Hard rejection |
| Silence | < 1.0 s consecutive | Hard rejection |
| NISQA MOS | >= 2.5 | Informational only |
| Speaker similarity | >= 0.7 | Informational only |

## Notes and Caveats

- **Reference duration is 10s**, not 15s like FishGram/Qwen. OmniVoice docs explicitly warn that references longer than 10s degrade cloning quality.
- **OmniVoice generates at 24 kHz**. Step 5 resamples to 16 kHz on FLAC write via `librosa.load(sr=16000)`.
- **Reference text is pre-computed with Parakeet TDT** (not OmniVoice's internal Whisper). This keeps the project's STT stack consistent and avoids loading two ASR models per run.
- **PyTorch version**: OmniVoice upstream recommends torch 2.8.0+cu128, but this project pins to **cu126** to match the ml-server03 driver (560.35.03). cu126 wheels are forward-compatible.
- **Number normalization**: OmniVoice docs recommend normalizing Arabic numerals to words ("123" -> "one hundred twenty-three") for best results. Mozilla CV Spanish text usually contains words; this is not currently a bottleneck but may be revisited if WER spikes on numeric prompts.
- **Voice design ignored**: OmniVoice supports a `voice design` mode via `instruct=...`. This pipeline uses pure voice cloning only.

## See Also

- [`ARCHITECTURE.md`](./ARCHITECTURE.md) - Technical design and patterns
- [OmniVoice GitHub](https://github.com/k2-fsa/OmniVoice)
- [OmniVoice HuggingFace](https://huggingface.co/k2-fsa/OmniVoice)
