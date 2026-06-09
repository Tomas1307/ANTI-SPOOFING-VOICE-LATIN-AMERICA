# Partial Spoof Pipeline - Technical Architecture

## High-Level Flow

```
bonafide_dataset_by_speaker/
        |
        v
[Step 1: Transcribe (Parakeet TDT)] --> bonafide_transcripts.json
        |
        v
[Step 2: Clone Speech (Strategy)]   --> cloned/*.wav + cloned_generation_metadata.json
        |
        v
[Step 3: Alignment (Parakeet TDT)]  --> alignment_metadata.json
        |
        v
[Step 4: Select Words (Random)]     --> word_selection_metadata.json
        |
        v
[Step 5: Splice Audio]              --> spliced/*.wav + splice_metadata.json
        |
        v
[Step 6: Validate Quality]          --> splice_quality_metadata.json
        |
        v
[Step 7: Format to LA]              --> data/{attack}_partial_spoof/LA/
```

## Design Patterns

| Pattern | Where | Purpose |
|---------|-------|---------|
| **Facade** | `pipeline_facade.py` | Single entry point orchestrating 7 steps |
| **Strategy** | `strategies/` | Interchangeable voice cloning backends |
| **Factory** | `utils/strategy_factory.py` | Creates strategy from system name |
| **Singleton** | `settings.py` | Shared pipeline configuration |
| **Dependency Injection** | Step constructors | Strategy injected into Step 2 |

## Key Design Decisions

### Why generate the full sentence, not isolated words?

Words generated in isolation have citation-form prosody (flat pitch, wrong duration). Words extracted from a full sentence have natural in-context prosody that matches the surrounding bonafide audio, making the splice harder to detect.

### Why Parakeet TDT for both transcription and alignment?

Parakeet TDT natively produces word-level timestamps via its Token-and-Duration Transducer architecture. This eliminates the need for a separate forced alignment tool (MFA, etc.) and provides consistent behavior across both bonafide and cloned audio.

### Why non-adjacent word selection?

Adjacent replaced words create a single contiguous spoofed segment, which is closer to a segment-level splice attack than true partial spoofing. Non-adjacent selection distributes the spoofed content across the utterance, creating a more challenging detection problem.

### Why absolute word counts (W1/W2/W3) instead of percentages?

Percentages require rounding logic that varies with sentence length. Absolute counts are deterministic, reproducible, and easier to report in the thesis. The percentage becomes a derived metric in metadata.

## Step Details

### Step 1: BonafideTranscriber
- Uses shared `app/utils/parakeet_transcriber.py` (Singleton)
- Filters utterances with < 4 words (below W1 minimum)
- Saves word timestamps alongside transcripts for Step 3 reuse

### Step 2: ClonedSpeechGenerator
- Receives `AttackStrategy` from facade
- Prepares reference audio via `concatenate_with_padding()` from chatterbox utils
- Caches reference clips per speaker
- Supports `skip_existing` for resume capability

### Step 3: ForcedAligner
- Reuses bonafide timestamps from Step 1 (no redundant computation)
- Runs Parakeet on cloned audio for cloned-side timestamps
- Both sets stored in alignment_metadata.json

### Step 4: WordSelector
- Seeded RNG per (utterance + tier) for deterministic results
- Non-adjacency constraint: indices differ by >= 2
- Rejection sampling with 100 attempts max

### Step 5: AudioSplicer
- Processes word replacements right-to-left to preserve sample positions
- Duration mismatch handling: silence stealing -> compression -> truncation
- Crossfade at boundaries (default 5ms)

### Step 6: SpliceQualityValidator (Placeholder)
- Computes spectral flux, energy delta at splice boundaries
- F0 delta placeholder (returns 0.0)
- Does not reject samples; logs metrics for future threshold tuning

### Step 7: OutputFormatter
- ASVspoof2019 LA standard structure
- Protocol label: `partial_spoof` (new, alongside `bonafide` and `spoof`)
- System ID format: `{ATTACK}_PSW{N}` (e.g., `FISHGRAM_PSW1`)
- Detailed metadata JSON saved alongside protocol files

## File Structure

```
app/pipeline/partial_spoof/
    __init__.py                     # Exports PartialSpoofPipeline
    pipeline_facade.py              # 7-step orchestrator
    settings.py                     # PartialSpoofSettings singleton
    schemas/                        # 12 Pydantic models (one per file)
    steps/                          # 7 step classes (one per file)
    utils/                          # Shared utilities
        alignment_engine.py         # (reserved for future alignment backends)
        crossfade.py               # Linear crossfade at splice boundaries
        splice_engine.py           # Core word-level splicing algorithm
        word_selector.py           # (reserved for future selection heuristics)
        strategy_factory.py        # Factory for attack strategies
    strategies/                     # 6 attack strategy implementations
        base_strategy.py           # Abstract AttackStrategy interface
        fishgram_strategy.py       # Fish Speech HTTP API
        qwen_strategy.py           # Qwen3-TTS local model
        cosyvoice_strategy.py      # CosyVoice 2 local model
        chatterbox_strategy.py     # ChatterboxMultilingualTTS
        outetts_strategy.py        # OuteTTS 0.6B
        openvoice_strategy.py      # OpenVoice V2 (MeloTTS + ToneConverter)
```

## Dependencies

- `nemo_toolkit[asr]` (Parakeet TDT)
- `librosa`, `soundfile`, `torchaudio` (audio I/O)
- `numpy` (signal processing)
- Attack-system-specific dependencies (loaded only for selected strategy)
