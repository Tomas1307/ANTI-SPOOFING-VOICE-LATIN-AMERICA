# Partial Spoof Pipeline - Design & Implementation Notes

**Date:** 2026-03-31
**Author:** Tomas Acosta (with Alfred's assistance)
**Pipeline location:** `app/pipeline/partial_spoof/`

---

## 1. Motivation

Existing voice anti-spoofing research (ASVspoof, HISPASpoof, LRLSpoof, SpeechFake-MD) treats utterances as either **fully bonafide** or **fully spoofed**. The PartialSpoof database (Zhang et al., IEEE/ACM TASLP 2023) introduced partially spoofed audio where only fragments are synthetic, but it is English-only and built on ASVspoof 2019 LA.

**No partial spoof dataset exists for any variety of Spanish, let alone Latin American Spanish.**

This pipeline creates partially spoofed audio by replacing individual words in bonafide HABLA utterances with voice-cloned versions from our existing attack systems. This is a novel contribution to the thesis.

### Key References

- **PartialSpoof (English):** Zhang et al., "The PartialSpoof Database and Countermeasures for the Detection of Short Fake Speech Segments Embedded in an Utterance," IEEE/ACM TASLP, Vol. 31, pp. 813-825, 2023. [arXiv:2204.05177](https://arxiv.org/abs/2204.05177) / [Zenodo](https://zenodo.org/records/4817532)
- **HISPASpoof (Spanish):** Risques et al., "HISPASpoof: A New Dataset For Spanish Speech Forensics," arXiv:2509.09155. [Link](https://arxiv.org/html/2509.09155)

---

## 2. Core Concept

Take a bonafide HABLA utterance and replace **1, 2, or 3 words** with voice-cloned versions of those same words, extracted from a full-sentence clone of the same utterance. The key insight: generate the full sentence with the cloned voice, then extract the target words via forced alignment. This preserves natural in-context prosody at the splice boundaries.

### Why not generate words in isolation?

Words generated in isolation have citation-form prosody (flat pitch, wrong duration, no coarticulation). Splicing them into bonafide audio creates obvious artefacts that any detector would catch. Words extracted from a full sentence have natural prosody that matches the surrounding context, making the partial spoof harder to detect and scientifically more valuable.

---

## 3. Word Replacement Tiers

Instead of percentages (which require rounding and vary with sentence length), we use absolute word counts:

| Tier | Words Replaced | Min Sentence Length | Max Spoof Ratio |
|------|---------------|--------------------|-----------------|
| W1   | 1             | 4 words            | 25%             |
| W2   | 2             | 8 words            | 25%             |
| W3   | 3             | 12 words           | 25%             |

The 25% cap ensures the utterance remains predominantly bonafide. A single bonafide utterance produces up to 3 samples (one per eligible tier). The `spoof_ratio` (by duration) is a derived metric in metadata, naturally varying with word length.

**Example:**
```
Bonafide: "el presidente anuncio nuevas medidas economicas para combatir la inflacion del pais entero" (13 words)
  -> W1: 1 word replaced  (1/13 = 7.7%)
  -> W2: 2 words replaced (2/13 = 15.4%)
  -> W3: 3 words replaced (3/13 = 23.1%)
  -> 3 partial spoof samples produced
```

---

## 4. Pipeline Architecture

The pipeline follows the canonical architecture defined in `app/pipeline/ARCHITECTURE.md`: Facade pattern, Strategy-based steps, Pydantic schemas, pipeline-scoped settings.

### 4.1 Directory Structure

```
app/pipeline/partial_spoof/
    __init__.py                         # Exports PartialSpoofPipeline
    pipeline_facade.py                  # 7-step orchestrator (Facade pattern)
    settings.py                         # PartialSpoofSettings singleton (31 parameters)
    README.md                           # User-facing documentation
    ARCHITECTURE.md                     # Technical design document
    schemas/                            # 12 Pydantic models (one per file)
        __init__.py
        pipeline_config.py              # PartialSpoofPipelineConfig
        transcription_result.py         # TranscriptionResult
        cloned_generation_result.py     # ClonedGenerationResult
        alignment_result.py             # AlignmentResult
        word_selection_result.py        # WordSelectionResult
        splice_result.py               # SpliceResult
        splice_quality_result.py       # SpliceQualityResult
        formatting_result.py           # FormattingResult
        word_alignment.py              # WordAlignment
        spliced_word_info.py           # SplicedWordInfo
        splice_metadata_entry.py       # SpliceMetadataEntry
    steps/                              # 7 step classes (one per file)
        __init__.py
        step_01_transcribe_bonafide.py  # BonafideTranscriber
        step_02_generate_cloned_speech.py # ClonedSpeechGenerator
        step_03_forced_alignment.py     # ForcedAligner
        step_04_select_words.py         # WordSelector
        step_05_splice_audio.py         # AudioSplicer
        step_06_validate_splice.py      # SpliceQualityValidator
        step_07_format_output.py        # OutputFormatter
    utils/                              # Shared utilities
        __init__.py
        alignment_engine.py             # (reserved for future alignment backends)
        crossfade.py                   # Linear crossfade at splice boundaries
        splice_engine.py               # Core word-level splicing algorithm
        word_selector.py               # (reserved for future selection heuristics)
        strategy_factory.py            # Factory for attack strategies
    strategies/                         # 6 attack strategy implementations
        __init__.py
        base_strategy.py               # Abstract AttackStrategy interface
        fishgram_strategy.py           # Fish Speech HTTP API
        qwen_strategy.py              # Qwen3-TTS local model
        cosyvoice_strategy.py         # CosyVoice 2 local model
        chatterbox_strategy.py        # ChatterboxMultilingualTTS
        outetts_strategy.py           # OuteTTS 0.6B
        openvoice_strategy.py         # OpenVoice V2 (MeloTTS + ToneConverter)
```

**Total: 37 files.**

### 4.2 Data Flow

```
bonafide_dataset_by_speaker/
        |
[Step 1: Transcribe (Parakeet TDT 0.6b-v3)] --> bonafide_transcripts.json
        |
[Step 2: Clone Speech (Attack Strategy)]     --> cloned/*.wav + cloned_generation_metadata.json
        |
[Step 3: Alignment (Parakeet TDT timestamps)]--> alignment_metadata.json
        |
[Step 4: Select Words (Random, non-adjacent)] --> word_selection_metadata.json
        |
[Step 5: Splice Audio (Crossfade + duration)] --> spliced/*.wav + splice_metadata.json
        |
[Step 6: Validate Quality (Placeholder)]      --> splice_quality_metadata.json
        |
[Step 7: Format to ASVspoof2019 LA]           --> data/{attack}_partial_spoof/LA/
```

---

## 5. Step-by-Step Design

### Step 1: Transcribe Bonafide Audio

**Class:** `BonafideTranscriber`
**ASR Model:** Parakeet TDT 0.6b-v3 (`nvidia/parakeet-tdt-0.6b-v3`)

Transcribes each bonafide HABLA audio file and records word-level timestamps. Parakeet TDT's Token-and-Duration Transducer architecture natively produces word timestamps without needing a separate forced alignment tool. Utterances with fewer than 4 words are filtered out (below W1 minimum).

**Why Parakeet, not Whisper:** Consistency with all existing attack pipelines (same ASR everywhere), proven Spanish performance (3.45% WER on FLEURS), lighter footprint (0.6B vs 1.55B params, ~3-4 GB VRAM).

**Reuses:** `app/utils/parakeet_transcriber.py` (Singleton pattern, shared across all pipelines).

**Output:** `bonafide_transcripts.json` with fields: speaker_id, split, audio_path, transcript, word_count, word_timestamps.

### Step 2: Generate Cloned Speech

**Class:** `ClonedSpeechGenerator`
**Pattern:** Strategy (receives `AttackStrategy` from facade)

For each transcribed bonafide utterance, generates the **same sentence** using the configured voice cloning system with the speaker's reference audio. The reference audio is prepared by concatenating training samples to a target duration (reusing `concatenate_with_padding()` from `chatterbox_attack/utils/`).

**Strategy interface:**
```python
class AttackStrategy(ABC):
    def load_model(device: str) -> None
    def generate(text, reference_audio_path, output_path, reference_text, seed) -> float
    def cleanup() -> None
    def name() -> str            # e.g., "FISHGRAM"
    def needs_reference_transcript() -> bool
```

**6 concrete strategies**, each wrapping the generation logic from its respective existing pipeline:

| Strategy | Backend | Model Loading | Needs Ref Transcript |
|----------|---------|---------------|---------------------|
| FishGramStrategy | HTTP API to Fish Speech server | No local model | No |
| QwenStrategy | Local Qwen3-TTS 1.7B | Singleton + speaker prompt caching | Yes |
| CosyVoiceStrategy | Local CosyVoice 2 | Singleton | Yes |
| ChatterboxStrategy | Local Chatterbox 500M | Singleton + watermark bypass | No |
| OuteTTSStrategy | Local OuteTTS 0.6B | Singleton + speaker profile caching | No |
| OpenVoiceStrategy | Local MeloTTS + ToneColorConverter | Two models + SE caching | No |

**Factory:** `utils/strategy_factory.py` creates the right strategy from a system name string. Uses conditional module imports to avoid loading all 6 TTS frameworks at once.

### Step 3: Forced Alignment

**Class:** `ForcedAligner`

Runs alignment on BOTH bonafide and cloned audio to get word-level timestamps for each version.

- **Bonafide side:** Reuses word timestamps already computed in Step 1 (no redundant ASR call).
- **Cloned side:** Runs Parakeet TDT on the cloned audio to get its word timestamps.

Both sets are stored in `alignment_metadata.json` so Step 5 can extract words from the correct positions in both waveforms.

### Step 4: Select Words to Replace

**Class:** `WordSelector`

For each aligned utterance, determines eligible tiers based on word count and randomly selects N word indices per tier.

**Constraints:**
- Non-adjacency: selected indices must differ by >= 2 (prevents contiguous spoofed blocks)
- Seeded RNG: `seed = RANDOM_SEED + hash(sample_key + tier)` for deterministic per-utterance randomness
- Rejection sampling with 100 attempts max

**Current heuristic:** Pure random selection. Future options (not yet implemented): content-word preference (nouns/verbs over articles), position-based selection, skip first/last word.

### Step 5: Splice Audio

**Class:** `AudioSplicer`
**Core engine:** `utils/splice_engine.py`

For each selection plan, extracts selected word segments from cloned audio and replaces the corresponding regions in bonafide audio.

**Splicing algorithm:**
1. Process word replacements in **reverse order** (right-to-left) so earlier splices don't shift sample positions of later words.
2. For each selected word:
   - Extract cloned word segment using cloned alignment timestamps
   - Identify replacement region in bonafide using bonafide alignment timestamps
   - **Handle duration mismatch:**
     - If cloned shorter: pad with crossfade into surrounding audio
     - If cloned longer: steal silence from adjacent pause (max 50ms), then time-compress (max 10%), then truncate as last resort
3. Apply **crossfade** (5ms default) at each splice boundary using `utils/crossfade.py`
4. Save reconstructed waveform

**Crossfade:** Linear fade-out on the tail of segment_before, linear fade-in on the head of segment_after, summed in the overlap region.

### Step 6: Validate Splice Quality (Placeholder)

**Class:** `SpliceQualityValidator`

Computes continuity metrics at each splice boundary:
- **Spectral flux:** FFT-based spectral change across the boundary
- **Energy delta:** RMS energy difference across the boundary
- **F0 delta:** Pitch continuity (placeholder, returns 0.0)

Currently logs metrics without rejecting samples. Future: configurable thresholds, retry with different TTS seed on failure.

### Step 7: Format Output to LA

**Class:** `OutputFormatter`

Creates the standard ASVspoof2019 LA directory structure:

```
data/{attack}_partial_spoof/
    LA/
        ASVspoof2019_LA_train/
            flac/                                   # FLAC audio files (PCM_16, 16kHz)
            ASVspoof2019.LA.cm.train.trl.txt        # Protocol file
        ASVspoof2019_LA_dev/
            flac/
            ASVspoof2019.LA.cm.dev.trl.txt
        ASVspoof2019_LA_eval/
            flac/
            ASVspoof2019.LA.cm.eval.trl.txt
        partial_spoof_metadata.json                 # Detailed per-sample metadata
```

**Protocol format:**
```
{speaker_id} {audio_id} {SYSTEM}_PSW{N} partial_spoof
```

Example:
```
arf_00295 LA_T_12000000 FISHGRAM_PSW1 partial_spoof
arf_00295 LA_T_13000000 FISHGRAM_PSW2 partial_spoof
arf_00295 LA_T_14000000 FISHGRAM_PSW3 partial_spoof
```

**Audio ID ranges** (avoiding collisions with existing pipelines at 6M-11M):
- W1: 12,000,000 - 12,999,999
- W2: 13,000,000 - 13,999,999
- W3: 14,000,000 - 14,999,999

---

## 6. Per-Sample Metadata

Each partially spoofed sample has detailed metadata in `splice_metadata.json`:

```json
{
  "spk001_utt001_W1": {
    "sample_id": "spk001_utt001_W1",
    "speaker_id": "arf_00295",
    "split": "train",
    "tier": "W1",
    "attack_system": "FISHGRAM",
    "transcript": "el gato negro duerme sobre la mesa",
    "total_words": 7,
    "spoofed_words": [
      {
        "word": "negro",
        "word_index": 2,
        "bonafide_start_s": 0.45,
        "bonafide_end_s": 0.80,
        "cloned_start_s": 0.50,
        "cloned_end_s": 0.88,
        "duration_ratio": 1.09,
        "crossfade_ms": 5.0
      }
    ],
    "spoof_word_ratio": 0.143,
    "spoof_duration_ratio": 0.18,
    "total_duration_s": 2.10
  }
}
```

The `spoof_word_ratio` is the fraction by word count (1/7 = 0.143). The `spoof_duration_ratio` is the fraction by audio duration (derived, varies with word length). Both are available for analysis.

---

## 7. Design Patterns Used

| Pattern | Where | Purpose |
|---------|-------|---------|
| **Facade** | `pipeline_facade.py` | Single entry point orchestrating 7 steps |
| **Strategy** | `strategies/` | Interchangeable voice cloning backends |
| **Factory** | `utils/strategy_factory.py` | Creates strategy from system name |
| **Singleton** | `settings.py`, `ParakeetTranscriber` | Shared configuration and model instances |
| **Dependency Injection** | Step constructors | Strategy injected into ClonedSpeechGenerator |

---

## 8. Reused Code

The pipeline reuses existing shared utilities rather than duplicating logic:

| Utility | Source | Used In |
|---------|--------|---------|
| `ParakeetTranscriber` | `app/utils/parakeet_transcriber.py` | Step 1, Step 3 |
| `WordTimestamp` | `app/utils/word_timestamp.py` | Step 1, Step 3 |
| `concatenate_with_padding()` | `app/pipeline/chatterbox_attack/utils/audio_concatenation.py` | Step 2 |
| `NoOpWatermarker` | `app/pipeline/chatterbox_attack/utils/watermark_remover.py` | ChatterboxStrategy |
| `trim_trailing_noise()` | `app/pipeline/chatterbox_attack/utils/speech_trimmer.py` | ChatterboxStrategy |
| Pipeline settings | `app/pipeline/{attack}/settings.py` | Each concrete strategy |

---

## 9. Test Plan

Test file: `app/tests/test_partial_spoof_pipeline.py`

### Unit Tests (mocked dependencies)

| Test Group | Tests | What's Verified |
|-----------|-------|-----------------|
| Schema validation | 4 tests | Pydantic models accept valid data, nested schemas work |
| Strategy factory | 3 tests | Factory returns correct types, raises on unknown, VALID_SYSTEMS complete |
| Crossfade | 4 tests | Output length, zero-crossfade, short segment error, smooth transition |
| Word selector | 5 tests | Tier minimums, non-adjacency, determinism, exact counts (W1=1, W3=3) |
| Splice engine | 4 tests | Valid output, detail counts per tier, reasonable output duration |
| Output formatter | 2 tests | LA directory structure, protocol format correctness |

### Integration Tests (ml-server03)

- End-to-end pipeline with validation mode (3 speakers)
- Verify LA output has `partial_spoof` label in all protocol entries
- Verify metadata consistency across all JSON files
- Verify tier counts match utterance word lengths

### Running Tests

```bash
# On ml-server03
source ~/ANTI-SPOOFING-VOICE-LATIN-AMERICA/envs/<env_name>/bin/activate
pytest app/tests/test_partial_spoof_pipeline.py -v
```

---

## 10. Usage

```python
from app.pipeline.partial_spoof import PartialSpoofPipeline
from app.pipeline.partial_spoof.schemas.pipeline_config import PartialSpoofPipelineConfig

# Run with Fish Speech
config = PartialSpoofPipelineConfig(
    attack_system="fishgram",
    tiers=["W1", "W2", "W3"],
)
pipeline = PartialSpoofPipeline(config)
la_path = pipeline.run()

# Run with Qwen3-TTS
config = PartialSpoofPipelineConfig(attack_system="qwen")
pipeline = PartialSpoofPipeline(config)
pipeline.run()
```

Step execution can be controlled individually:

```python
config = PartialSpoofPipelineConfig(
    attack_system="fishgram",
    run_step_1=False,  # Skip transcription (reuse existing)
    run_step_2=False,  # Skip cloning (reuse existing)
    run_step_3=True,   # Run alignment
    run_step_4=True,   # Run word selection
    run_step_5=True,   # Run splicing
    run_step_6=True,   # Run validation
    run_step_7=True,   # Run formatting
    skip_existing=True, # Resume from previous partial run
)
```

---

## 11. Open Design Questions

These are not blocking implementation but need decisions before production runs:

1. **Flow metric thresholds (Step 6):** What spectral flux, F0 delta, and energy delta values indicate a bad splice? Requires empirical data from initial runs to calibrate.

2. **Word selection heuristic (Step 4):** Currently pure random. Options to explore:
   - Content-word preference (nouns/verbs over articles/prepositions) for more impactful spoofing
   - Position-based selection (avoid first/last word due to boundary effects)
   - Spread-out distribution (even spacing across utterance)

3. **Retry logic (Step 6):** When splice quality fails, regenerate the full sentence with a different TTS seed and re-splice. Requires Step 6 thresholds first.

---

## 12. Comparison with Existing Datasets

| Aspect | PartialSpoof (Zhang et al.) | Our Partial Spoof |
|--------|---------------------------|-------------------|
| Language | English (VCTK) | Latin American Spanish (7 accents) |
| Construction | VAD + same-speaker segment pairing | Forced alignment + word-level extraction from full clone |
| TTS systems | ASVspoof 2019 LA (A01-A19) | Fish Speech, Qwen3-TTS, CosyVoice, OuteTTS, Chatterbox, OpenVoice |
| Spoof granularity | Segment-level (variable) | Word-level (1, 2, or 3 words) |
| Labels | Utterance + segment-level at 6 resolutions | Utterance + word-level with exact timestamps |
| Bonafide source | ASVspoof 2019 LA (VCTK) | HABLA dataset (162 speakers, 7 Latin American accents) |
| Speakers | VCTK multi-speaker | 162 HABLA speakers (Colombian, Mexican, Argentinian, Chilean, Peruvian, Venezuelan, Puerto Rican) |
| Spoof ratio control | Quantized into 10 levels | Controlled via absolute word count (W1/W2/W3) |
