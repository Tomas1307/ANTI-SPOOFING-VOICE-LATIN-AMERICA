# Partial Spoof Pipeline

Creates partially spoofed Latin American Spanish audio by replacing individual words in bonafide HABLA utterances with voice-cloned versions from configurable attack systems.

## Overview

Unlike the existing attack pipelines that generate **fully spoofed** utterances, this pipeline produces **partially spoofed** audio where only 1, 2, or 3 words are replaced with synthetic versions. This is a novel contribution: no partial spoof dataset exists for any variety of Spanish.

## How It Works

1. **Transcribe** bonafide HABLA audio using Parakeet TDT 0.6b-v3
2. **Clone** each utterance using the selected attack system (Fish Speech, Qwen3-TTS, etc.)
3. **Align** both versions to get word-level timestamps
4. **Select** N random words to replace (W1=1, W2=2, W3=3)
5. **Splice** cloned words into bonafide audio with crossfade at boundaries
6. **Validate** splice quality metrics (placeholder)
7. **Format** output to ASVspoof2019 LA structure with `partial_spoof` label

## Word Replacement Tiers

| Tier | Words Replaced | Min Sentence Length | Max Spoof Ratio |
|------|---------------|--------------------|-----------------|
| W1   | 1             | 4 words            | 25%             |
| W2   | 2             | 8 words            | 25%             |
| W3   | 3             | 12 words           | 25%             |

Each bonafide utterance produces up to 3 samples (one per eligible tier).

## Usage

```python
from app.pipeline.partial_spoof import PartialSpoofPipeline
from app.pipeline.partial_spoof.schemas.pipeline_config import PartialSpoofPipelineConfig

config = PartialSpoofPipelineConfig(
    attack_system="fishgram",  # or: qwen, cosyvoice, outetts, chatterbox, openvoice
    tiers=["W1", "W2", "W3"],
)

pipeline = PartialSpoofPipeline(config)
la_path = pipeline.run()
```

## Output Structure

```
data/{attack_name}_partial_spoof/
    LA/
        ASVspoof2019_LA_train/
            flac/
            ASVspoof2019.LA.cm.train.trl.txt
        ASVspoof2019_LA_dev/
        ASVspoof2019_LA_eval/
        partial_spoof_metadata.json
    bonafide_transcripts.json
    cloned_generation_metadata.json
    alignment_metadata.json
    word_selection_metadata.json
    splice_metadata.json
    splice_quality_metadata.json
    cloned/
    spliced/
    references/
```

## Protocol File Format

```
arf_00295 LA_T_12000000 FISHGRAM_PSW1 partial_spoof
arf_00295 LA_T_13000000 FISHGRAM_PSW2 partial_spoof
arf_00295 LA_T_14000000 FISHGRAM_PSW3 partial_spoof
```

## Audio ID Ranges

- W1: 12,000,000 - 12,999,999
- W2: 13,000,000 - 13,999,999
- W3: 14,000,000 - 14,999,999

## Running on ml-server03

```bash
# Check GPU availability
nvidia-smi

# Activate the appropriate venv
export CUDA_VISIBLE_DEVICES=1
source ~/ANTI-SPOOFING-VOICE-LATIN-AMERICA/envs/<env_name>/bin/activate

# For Fish Speech, ensure the server is running first
cd ~/fish-speech && python -m tools.api ...

# Run the pipeline
cd ~/ANTI-SPOOFING-VOICE-LATIN-AMERICA
python -c "
from app.pipeline.partial_spoof import PartialSpoofPipeline
from app.pipeline.partial_spoof.schemas.pipeline_config import PartialSpoofPipelineConfig

config = PartialSpoofPipelineConfig(attack_system='fishgram')
pipeline = PartialSpoofPipeline(config)
pipeline.run()
"
```

## Testing

```bash
pytest app/tests/test_partial_spoof_pipeline.py -v
```
