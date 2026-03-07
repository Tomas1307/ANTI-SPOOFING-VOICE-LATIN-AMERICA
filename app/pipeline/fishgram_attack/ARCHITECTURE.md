# FishGram Attack Pipeline - Architecture

## Overview

This document describes the architecture of the FishGram attack pipeline, which follows the **Facade** and **Strategy** patterns established in the Mozilla speaker selection pipeline.

**Design Principles**:
- Facade pattern for orchestration
- Strategy pattern for interchangeable steps
- Dependency injection for testability
- Pydantic for all data models (no `@dataclass`)
- One class per file
- Separate virtual environment for isolation

## Directory Structure

```
app/pipeline/fishgram_attack/
├── __init__.py                    # Module exports
├── pipeline_facade.py             # Facade orchestrator
├── settings.py                    # Pydantic settings
├── ARCHITECTURE.md                # This file
├── README.md                      # User documentation
│
├── schemas/                       # Pydantic data models
│   ├── __init__.py
│   ├── pipeline_config.py         # Runtime configuration
│   ├── reference_result.py        # Step 1 output
│   ├── text_prompts_result.py     # Step 2 output
│   ├── generation_result.py       # Step 3 output
│   ├── validation_result.py       # Step 4 output
│   └── formatting_result.py       # Step 5 output
│
├── steps/                         # Strategy implementations
│   ├── __init__.py
│   ├── step_01_prepare_references.py  # ReferenceAudioPreparator
│   ├── step_02_prepare_texts.py       # TextPromptPreparator
│   ├── step_03_generate_speech.py     # SpeechGenerator
│   ├── step_04_validate_quality.py    # QualityValidator
│   └── step_05_format_output.py       # OutputFormatter
│
└── utils/                         # Utility functions
    ├── __init__.py
    ├── audio_concatenation.py     # Reference audio preparation
    └── quality_metrics.py         # DNSMOS, speaker verification
```

## Core Design Patterns

### 1. Facade Pattern

**Class**: `FishGramAttackPipeline` ([pipeline_facade.py](pipeline_facade.py:1))

The Facade provides a single entry point for the entire pipeline:

```python
pipeline = FishGramAttackPipeline()
output_dir = pipeline.run()
```

**Responsibilities**:
- Instantiate and orchestrate all 5 steps
- Handle configuration overrides
- Manage sequential execution
- Aggregate and log results

### 2. Strategy Pattern

Each step implements the Strategy pattern with:
- Constructor accepting optional overrides (dependency injection)
- `execute()` method returning typed results (Pydantic models)

**Example**:
```python
class ReferenceAudioPreparator:
    def __init__(self, bonafide_dir: Path | None = None, ...):
        self.bonafide_dir = bonafide_dir or settings.BONAFIDE_DIR

    def execute(self) -> ReferenceResult:
        # Implementation
        return ReferenceResult(...)
```

### 3. Dependency Injection

Steps accept optional parameters for testing and flexibility:

```python
# Production
step = ReferenceAudioPreparator()  # Uses settings defaults

# Testing
step = ReferenceAudioPreparator(
    bonafide_dir=Path("test/data"),
    target_duration=10.0
)
```

### 4. Pydantic Data Models

All schemas use Pydantic `BaseModel` (per CLAUDE.md):

```python
class ReferenceResult(BaseModel):
    reference_metadata_path: Path
    reference_count: int
    split_breakdown: Dict[str, int]
```

**Benefits**:
- Automatic validation
- Type safety
- JSON serialization
- IDE autocompletion

## Pipeline Flow

```
┌─────────────────────────────────────────────────────────────┐
│              FishGramAttackPipeline (Facade)                │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
                ┌──────────────────────┐
                │ Step 1: References   │
                │ ReferenceAudioPrep   │
                │ Output: ReferenceRes │
                └──────────────────────┘
                          │
                          ▼
                ┌──────────────────────┐
                │ Step 2: Texts        │
                │ TextPromptPreparator │
                │ Output: TextPrompts  │
                └──────────────────────┘
                          │
                          ▼
                ┌──────────────────────┐
                │ Step 3: Generate     │
                │ SpeechGenerator      │
                │ (Fish Speech API)    │
                │ Output: Generation   │
                └──────────────────────┘
                          │
                          ▼
                ┌──────────────────────┐
                │ Step 4: Validate     │
                │ QualityValidator     │
                │ Output: Validation   │
                └──────────────────────┘
                          │
                          ▼
                ┌──────────────────────┐
                │ Step 5: Format       │
                │ OutputFormatter      │
                │ Output: Formatting   │
                └──────────────────────┘
```

## Fish Speech API Integration

The Fish Speech TTS model runs as an **external HTTP API server** on ml-server03, separate from the pipeline process. Step 3 (SpeechGenerator) communicates with it via HTTP:

- **Health check**: `GET /` (verifies server is running)
- **Generation**: `POST /v1/tts` (sends text + base64-encoded reference audio)
- **Server startup**: See `guide/how_to_run_fishgram/README.md`

This design decouples the TTS model lifecycle from the pipeline, allowing the server to persist across multiple pipeline runs and avoiding GPU memory allocation/deallocation overhead.

## Configuration Management

### Settings Hierarchy

1. **settings.py**: Default values (Pydantic model)
2. **Runtime overrides**: Via `FishGramPipelineConfig`
3. **Direct modification**: `settings.VALIDATION_MODE = False`

**Example**:
```python
# Option 1: Direct modification
settings.VALIDATION_MODE = True
pipeline = FishGramAttackPipeline()

# Option 2: Config object
config = FishGramPipelineConfig(
    samples_per_speaker_override=3,
    device_override="cuda:1"
)
pipeline = FishGramAttackPipeline(config)
```

### Validation Mode Toggle

**Validation Mode** (`VALIDATION_MODE=True`):
- 3 speakers (Argentina Female)
- 2 samples per speaker = 6 total
- Runtime: ~4 minutes
- Purpose: Quality validation

**Production Mode** (`VALIDATION_MODE=False`):
- 162 speakers (all HABLA bonafide)
- 5 samples per speaker = 810 total
- Runtime: ~61 minutes
- Purpose: Full dataset generation

## Virtual Environment Isolation

**Why Separate Venv**:
- Fish Speech has specific PyTorch version requirements (>= 2.4)
- Prevents conflicts with other augmenters (RIR, Codec, RawBoost)
- Isolates dependencies from other researchers on ml-server03

**Location**: `envs/fishgram_env/`

**Setup**: See `guide/how_to_run_fishgram/README.md` for full installation instructions.

## Step Details

### Step 1: ReferenceAudioPreparator

**Purpose**: Create 15-second reference clips

**Process**:
1. Iterate speakers (3 for validation, 162 for production)
2. Load first 5 training samples
3. Concatenate with 0.1s silence padding
4. Trim to exactly 15.0 seconds
5. Save as `{speaker_id}_ref.wav`

**Determinism**: Alphabetical file sorting ensures reproducibility.

### Step 2: TextPromptPreparator

**Purpose**: Assign Spanish text prompts

**Process**:
1. Load Mozilla CV transcripts (15,000+ unique)
2. Filter by length (5-100 words)
3. Seeded random sampling (N texts per speaker)
4. Save as `text_prompts.json`

**Reproducibility**: `np.random.seed(settings.RANDOM_SEED)` ensures deterministic assignment.

### Step 3: SpeechGenerator

**Purpose**: Generate synthetic speech via Fish Speech HTTP API

**Process**:
1. Verify server health (`GET /`)
2. For each speaker-text pair:
   a. Read reference audio as bytes
   b. Base64-encode and send `POST /v1/tts`
   c. Save returned audio to WAV
   d. Track RTF (generation_time / audio_duration)

### Step 4: QualityValidator

**Purpose**: Filter low-quality samples

**Metrics**:
- **DNSMOS Overall** >= 3.5 (perceptual quality)
- **Speaker Similarity** >= 0.65 (voice cloning accuracy)
- **Silence Detection**: Reject if >1s consecutive silence

**Expected Pass Rate**: 85-95%

### Step 5: OutputFormatter

**Purpose**: Convert to ASVspoof2019 LA format

**Process**:
1. Convert WAV -> FLAC (16kHz, 16-bit)
2. Generate audio IDs (LA_T_9000001, LA_D_9000001, etc.)
3. Create protocol files (SPEAKER_ID AUDIO_ID SYSTEM_ID KEY)
4. Organize into train/dev/eval splits

## Extension Points

### Adding a New Step

1. Create `step_06_new_feature.py` in `steps/`
2. Implement class with `execute()` method
3. Define output schema in `schemas/new_feature_result.py`
4. Update `steps/__init__.py` exports
5. Add to Facade orchestration in `pipeline_facade.py`

### Swapping Quality Metrics

Replace DNSMOS with alternative metric:

```python
# In utils/quality_metrics.py
def compute_pesq(audio_path: Path) -> float:
    # Alternative implementation
    ...

# In step_04_validate_quality.py
quality_score = compute_pesq(audio_path)  # Instead of compute_dnsmos
```

## Testing Strategy

### Unit Testing

Test each step independently:

```python
def test_reference_preparation(tmp_path):
    preparator = ReferenceAudioPreparator(
        bonafide_dir=tmp_path / "mock_habla",
        output_dir=tmp_path / "output"
    )
    result = preparator.execute()
    assert result.reference_count > 0
```

### Integration Testing

Test full pipeline with validation mode:

```python
def test_full_pipeline():
    settings.VALIDATION_MODE = True
    pipeline = FishGramAttackPipeline()
    output = pipeline.run()
    assert output.exists()
    assert (output / "ASVspoof2019_LA_train").exists()
```

## Performance Characteristics

### Validation Mode (6 samples)

| Step | Time | Bottleneck |
|------|------|------------|
| 1 | 1 min | Audio I/O |
| 2 | <1 min | TSV parsing |
| 3 | 1 min | TTS inference |
| 4 | <1 min | Quality metrics |
| 5 | <1 min | FLAC conversion |
| **Total** | **~4 min** | - |

### Production Mode (810 samples)

| Step | Time | Bottleneck |
|------|------|------------|
| 1 | 5 min | Audio I/O (162 speakers) |
| 2 | 1 min | TSV parsing |
| 3 | 40 min | TTS inference (810 samples) |
| 4 | 10 min | Quality metrics (810 samples) |
| 5 | 5 min | FLAC conversion |
| **Total** | **~61 min** | Step 3 (TTS) |

**Hardware**: NVIDIA A40 (46GB VRAM)

## References

- **General Pipeline Architecture**: `app/pipeline/ARCHITECTURE.md`
- **Mozilla Pipeline** (template): `app/pipeline/select_mozilla_speakers/`
- **CLAUDE.md**: Project coding standards
- **How to Run Guide**: `guide/how_to_run_fishgram/README.md`
