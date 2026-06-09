# Mozilla Common Voice Speaker Selection Pipeline - Architecture

## Overview

This pipeline implements the Facade and Strategy patterns to orchestrate the selection of 15,340 acoustically diverse speakers from Mozilla Common Voice, augmenting the HABLA anti-spoofing dataset with Mexico and Spain accents while ensuring speaker independence.

## Directory Structure

```
select_mozilla_speakers/
├── __init__.py                    # Module exports: Pipeline, Config, settings
├── pipeline_facade.py             # Facade orchestrator (entry point)
├── settings.py                    # Centralized configuration (Pydantic)
├── ARCHITECTURE.md                # This file
├── README.md                      # Pipeline documentation for thesis
│
├── schemas/                       # Pydantic data models
│   ├── __init__.py
│   ├── pipeline_config.py         # Runtime configuration schema
│   └── embedding_result.py        # Embedding extraction result schema
│
└── steps/                         # Strategy implementations (one per step)
    ├── __init__.py                # Step class exports
    ├── step_01_extract_habla_embeddings.py
    ├── step_02_extract_cv_embeddings.py
    ├── step_03_filter_by_similarity.py
    ├── step_04_balanced_sampling.py
    └── step_05_integrate_cv_samples.py
```

## Core Components

### 1. Facade Orchestrator

**File**: `pipeline_facade.py`
**Class**: `MozillaSpeakerSelectionPipeline`

The Facade provides a single entry point for the entire pipeline, orchestrating all 5 steps in sequence. It handles:

- Configuration injection (optional overrides for thresholds, seeds, output dirs)
- Step instantiation with dependency injection
- Sequential execution with conditional skipping
- Unified logging and error handling
- Result aggregation

**Usage**:
```python
from app.pipeline.select_mozilla_speakers import (
    MozillaSpeakerSelectionPipeline,
    MozillaSpeakerPipelineConfig
)

# Run all steps with defaults
pipeline = MozillaSpeakerSelectionPipeline()
output_dir = pipeline.run()

# Run with custom configuration
config = MozillaSpeakerPipelineConfig(
    run_step_1=True,
    run_step_2=True,
    run_step_3=True,
    run_step_4=True,
    run_step_5=True,
    similarity_threshold_override=0.80,
    random_seed_override=123
)
pipeline = MozillaSpeakerSelectionPipeline(config)
output_dir = pipeline.run()
```

### 2. Strategy Pattern - Step Classes

Each step implements the Strategy pattern, providing:

- **Constructor**: Accepts optional overrides for dependency injection (paths, thresholds, devices)
- **execute() method**: Performs the step's core logic, returns typed results
- **Defaults from settings**: Falls back to `settings.VARIABLE_NAME` if no override provided

#### Step 1: HABLA Embedding Extraction

**File**: `steps/step_01_extract_habla_embeddings.py`
**Class**: `HablaEmbeddingExtractor`

Extracts ECAPA-TDNN speaker embeddings from HABLA bonafide dataset.

**Inputs**:
- `habla_dir` (Path): Directory with 162 HABLA speakers
- `output_dir` (Path): Output directory for embeddings
- `device` (str): Compute device (cuda/cpu)

**Outputs**: `EmbeddingResult`
- `embeddings_path`: `habla_embeddings.npy` (162, 192)
- `ids_path`: `habla_speaker_ids.json` (162 speaker IDs)

**Process**:
1. Load ECAPA-TDNN model from SpeechBrain
2. For each HABLA speaker, extract embeddings from up to 20 training samples
3. Average embeddings per speaker, L2-normalize
4. Save embeddings and IDs

#### Step 2: Common Voice Embedding Extraction

**File**: `steps/step_02_extract_cv_embeddings.py`
**Class**: `CVEmbeddingExtractor`

Extracts ECAPA-TDNN embeddings from Common Voice speakers (Mexico, Spain, Colombia, Chile, Venezuela only).

**Inputs**:
- `cv_archive` (Path): CV tar.gz archive
- `cv_extracted_dir` (Path): Extraction target
- `output_dir` (Path): Output directory
- `device` (str): Compute device

**Outputs**: `EmbeddingResult`
- `embeddings_path`: `cv_embeddings.npy` (3,299, 192)
- `ids_path`: `cv_client_ids.json` (3,299 client IDs)
- `metadata_path`: `cv_speaker_metadata.json` (metadata dict)

**Process**:
1. Extract CV archive if not already extracted
2. Parse validated.tsv, filter by accent keywords and valid gender/age
3. For each CV speaker, extract embeddings from up to 10 samples
4. Save embeddings, IDs, and metadata (accent, gender, age)

#### Step 3: Similarity-Based Filtering

**File**: `steps/step_03_filter_by_similarity.py`
**Class**: `SimilarityFilter`

Filters CV speakers by cosine similarity to HABLA, ensuring speaker independence.

**Inputs**:
- `threshold` (float): Cosine similarity threshold (default: 0.75)

**Outputs**: Path
- `filtered_speakers.tsv`: CV speakers with max_similarity < threshold

**Process**:
1. Load HABLA embeddings (162, 192) and CV embeddings (3,299, 192)
2. Compute cosine similarity matrix (3,299 × 162)
3. For each CV speaker, find max similarity to any HABLA speaker
4. Filter to speakers with max_similarity < threshold
5. Save filtered client IDs with similarity scores

#### Step 4: Balanced Stratified Sampling

**File**: `steps/step_04_balanced_sampling.py`
**Class**: `BalancedSampler`

Selects exactly 15,340 samples using stratified sampling with fixed accent targets.

**Inputs**:
- `random_seed` (int): Random seed for reproducibility

**Outputs**: Path
- `selected_15340.tsv`: Selected samples with client_id, path, accent, gender, age

**Targets**:
- Colombia: 846 speakers
- Chile: 1,361 speakers
- Venezuela: 2,233 speakers (only 23 available → take all)
- Mexico: 5,450 speakers
- Spain: 5,450 speakers (89% male → female-priority sampling)

**Process**:
1. Load filtered speakers and CV metadata
2. Group by accent
3. For Spain: Sample with female priority to maximize female representation
4. For others: Stratified sampling by gender/age, proportional allocation
5. Save selected samples

#### Step 5: Dataset Integration

**File**: `steps/step_05_integrate_cv_samples.py`
**Class**: `DatasetIntegrator`

Integrates CV speakers into HABLA v2 dataset structure.

**Inputs**:
- `habla_dir` (Path): Original HABLA directory
- `output_dir` (Path): bonafide_dataset_by_speaker_v2/
- `cv_clips_dir` (Path): CV audio clips directory

**Outputs**: Path
- `bonafide_dataset_by_speaker_v2/`: Final integrated dataset
- `cv_speaker_mapping.json`: Client ID → speaker ID mapping

**Process**:
1. Copy all 162 HABLA speakers to v2 directory
2. For each CV speaker:
   - Generate speaker ID: `{accent_code}{gender_code}_{number:05d}` (e.g., mxm_00163)
   - Continue numbering from existing HABLA speakers
   - Split samples into train/val/test (70/15/15)
   - Copy audio files to respective split directories
3. Save speaker mapping with metadata

### 3. Configuration Management

**File**: `settings.py`
**Class**: `MozillaSpeakerSelectionSettings` (Pydantic BaseModel)

Centralized configuration using Pydantic for validation. All constants are defined here and accessed via the module-level singleton `settings`.

**Key Settings**:
```python
HABLA_DIR: Path                           # bonafide_dataset_by_speaker/
BONAFIDE_V2_DIR: Path                     # bonafide_dataset_by_speaker_v2/
OUTPUT_DIR: Path                          # mozilla_speaker_selection/
CV_ARCHIVE: Path                          # cv-corpus-24.0-2025-12-05-es.tar.gz
CV_EXTRACTED_DIR: Path                    # cv-corpus-24.0-2025-12-05/
CV_CLIPS_DIR: Path                        # cv-corpus-24.0-2025-12-05/es/clips/
CV_METADATA_TSV: Path                     # cv-corpus-24.0-2025-12-05/es/validated.tsv

MODEL_SOURCE: str                         # "speechbrain/spkrec-ecapa-voxceleb"
SAMPLE_RATE: int                          # 16000
MAX_SAMPLES_PER_SPEAKER: int              # 20 (HABLA) / 10 (CV)
SIMILARITY_THRESHOLD: float               # 0.75
RANDOM_SEED: int                          # 42
DEVICE: str                               # "cuda" or "cpu"

ACCENT_CODES: Dict[str, str]              # {"Mexico": "mx", "Spain": "es", ...}
ACCENT_TARGETS: Dict[str, int]            # {"Mexico": 5450, "Spain": 5450, ...}
ACCENT_KEYWORDS: Dict[str, List[str]]     # {"Mexico": ["mexico", "méxico"], ...}

TRAIN_RATIO: float                        # 0.70
VAL_RATIO: float                          # 0.15
TEST_RATIO: float                         # 0.15
```

### 4. Schemas (Pydantic Models)

**File**: `schemas/pipeline_config.py`
**Class**: `MozillaSpeakerPipelineConfig`

Runtime configuration for pipeline execution. Allows selective step execution and parameter overrides.

**File**: `schemas/embedding_result.py`
**Class**: `EmbeddingResult`

Typed result from embedding extraction steps (Step 1 and Step 2).

## Data Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                      MozillaSpeakerSelectionPipeline                │
│                            (Facade)                                 │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│ Step 1: HablaEmbeddingExtractor                                     │
│ Input:  bonafide_dataset_by_speaker/ (162 speakers)                │
│ Output: habla_embeddings.npy (162, 192)                            │
│         habla_speaker_ids.json                                      │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│ Step 2: CVEmbeddingExtractor                                        │
│ Input:  cv-corpus-24.0-2025-12-05-es.tar.gz                        │
│ Output: cv_embeddings.npy (3,299, 192)                             │
│         cv_client_ids.json                                          │
│         cv_speaker_metadata.json                                    │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│ Step 3: SimilarityFilter                                            │
│ Input:  habla_embeddings.npy + cv_embeddings.npy                   │
│ Output: filtered_speakers.tsv (3,299 → 3,299 speakers)             │
│         Filters by cosine similarity < 0.75                         │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│ Step 4: BalancedSampler                                             │
│ Input:  filtered_speakers.tsv (3,299 speakers)                     │
│ Output: selected_15340.tsv (1,567 speakers, 15,340 samples)        │
│         Fixed targets: CO=846, CL=1361, VE=23, MX=5450, ES=5450     │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│ Step 5: DatasetIntegrator                                           │
│ Input:  bonafide_dataset_by_speaker/ (162 HABLA)                   │
│         selected_15340.tsv (1,567 CV speakers)                      │
│ Output: bonafide_dataset_by_speaker_v2/ (1,729 total speakers)     │
│         cv_speaker_mapping.json                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## Key Design Decisions

### 1. Facade Pattern for Orchestration

**Why**: Simplifies the client interface. Users only interact with `MozillaSpeakerSelectionPipeline.run()`, not individual steps. The Facade handles:
- Step instantiation
- Dependency passing
- Error propagation
- Logging coordination

**Benefits**:
- Single entry point for entire pipeline
- Easy to add pipeline-level features (logging, profiling, rollback)
- Steps remain decoupled and testable

### 2. Strategy Pattern for Steps

**Why**: Each step is an independent, interchangeable algorithm. Steps can be:
- Tested in isolation
- Swapped for alternative implementations (e.g., different similarity metrics)
- Reused in other pipelines
- Extended without modifying the Facade

**Benefits**:
- Open/Closed Principle: Open for extension, closed for modification
- Dependency Injection: Steps accept overrides for testing
- Single Responsibility: Each step has one clear purpose

### 3. Pydantic for Configuration and Schemas

**Why**: Per CLAUDE.md, Pydantic is the sole standard for data modeling. No `@dataclass` allowed.

**Benefits**:
- Automatic validation (types, ranges, path existence)
- Immutable configuration (frozen=True where appropriate)
- JSON serialization/deserialization
- IDE autocompletion and type checking

### 4. Centralized Settings with Module-Level Singleton

**Why**: Avoid passing 20+ configuration parameters through every function call.

**Implementation**:
```python
# settings.py
class MozillaSpeakerSelectionSettings(BaseModel):
    HABLA_DIR: Path = Field(default=Path("data/bonafide_dataset_by_speaker"))
    # ... 20+ settings

settings = MozillaSpeakerSelectionSettings()  # Module-level singleton
```

**Usage**:
```python
from app.pipeline.select_mozilla_speakers.settings import settings

def some_function():
    habla_dir = settings.HABLA_DIR  # Access directly
```

**Benefits**:
- DRY: No repeated parameter passing
- Easy to override: `settings = MozillaSpeakerSelectionSettings(HABLA_DIR=custom_path)`
- Testable: Mock `settings` for unit tests

### 5. Typed Results with Pydantic

**Why**: Steps return strongly typed results (`EmbeddingResult`, `Path`) instead of tuples or dicts.

**Benefits**:
- Self-documenting: `result.embeddings_path` vs `result[0]`
- Validation: Pydantic ensures paths exist, counts are positive
- IDE support: Autocompletion and type checking

## Extension Points

### Adding a New Step

1. **Create Step Class** in `steps/step_06_new_step.py`:
```python
from pathlib import Path
from app.pipeline.select_mozilla_speakers.settings import settings

class NewStepProcessor:
    """Brief description of what this step does."""

    def __init__(self, param_override: str | None = None) -> None:
        self.param = param_override or settings.DEFAULT_PARAM

    def execute(self) -> Path:
        """Execute the step."""
        # Implementation
        return output_path
```

2. **Add to steps/__init__.py**:
```python
from app.pipeline.select_mozilla_speakers.steps.step_06_new_step import NewStepProcessor

__all__ = [..., "NewStepProcessor"]
```

3. **Update Facade** in `pipeline_facade.py`:
```python
if self.config.run_step_6:
    logger.info("STEP 6/6: New Step Description")
    step_6 = NewStepProcessor()
    result_6 = step_6.execute()
    logger.info(f"✓ New step complete: {result_6}")
```

4. **Update Config Schema** in `schemas/pipeline_config.py`:
```python
class MozillaSpeakerPipelineConfig(BaseModel):
    run_step_6: bool = Field(default=True)
```

### Swapping a Step Implementation

1. Create alternative implementation (e.g., `SimilarityFilterFaiss` using FAISS instead of NumPy)
2. Ensure it has the same interface (`__init__` parameters, `execute()` return type)
3. Update Facade to use new class:
```python
# Before
step_3 = SimilarityFilter(threshold=...)

# After
step_3 = SimilarityFilterFaiss(threshold=...)  # Drop-in replacement
```

### Adding Configuration Parameters

1. **Update settings.py**:
```python
class MozillaSpeakerSelectionSettings(BaseModel):
    NEW_PARAM: int = Field(default=42, description="New parameter for XYZ")
```

2. **Use in Step**:
```python
def __init__(self, new_param: int | None = None):
    self.new_param = new_param or settings.NEW_PARAM
```

## Testing Strategy

### Unit Testing Steps

Each step can be tested in isolation with mocked inputs:

```python
# test_step_01.py
from pathlib import Path
from app.pipeline.select_mozilla_speakers.steps import HablaEmbeddingExtractor

def test_habla_embedding_extraction(tmp_path):
    # Create mock HABLA directory structure
    mock_habla = tmp_path / "mock_habla"
    mock_habla.mkdir()
    # ... create mock speaker directories and audio files

    extractor = HablaEmbeddingExtractor(
        habla_dir=mock_habla,
        output_dir=tmp_path / "output",
        device="cpu"
    )
    result = extractor.execute()

    assert result.embedding_count == expected_count
    assert result.embeddings_path.exists()
```

### Integration Testing Pipeline

Test the full pipeline with a small subset of data:

```python
# test_pipeline_integration.py
from app.pipeline.select_mozilla_speakers import (
    MozillaSpeakerSelectionPipeline,
    MozillaSpeakerPipelineConfig
)

def test_full_pipeline(mock_data_dirs):
    config = MozillaSpeakerPipelineConfig(
        output_dir_override=mock_data_dirs["output"]
    )
    pipeline = MozillaSpeakerSelectionPipeline(config)
    result = pipeline.run()

    assert result.exists()
    # Verify expected speakers in v2 directory
```

## Performance Considerations

### GPU Memory

- **ECAPA-TDNN Model**: ~100 MB VRAM
- **Batch Processing**: Steps 1 and 2 process audio files sequentially (no batching needed)
- **Similarity Matrix**: (3,299 × 162 × 4 bytes) = ~2 MB (easily fits in RAM)

**Resource Requirements**:
- GPU: NVIDIA A40 (46 GB VRAM) - massively overprovisioned, any GPU works
- RAM: 16 GB sufficient for all steps
- Disk: ~50 GB for CV archive + extracted files

### Execution Time

- **Step 1**: ~5 minutes (162 speakers × 20 samples)
- **Step 2**: ~60 minutes (3,299 speakers × 10 samples, includes archive extraction)
- **Step 3**: <1 minute (similarity computation)
- **Step 4**: <1 minute (sampling)
- **Step 5**: ~10 minutes (file copying)

**Total**: ~75 minutes on A40 GPU

## Reproducibility

**Random Seed**: Set via `MozillaSpeakerPipelineConfig(random_seed_override=42)`

**Deterministic Steps**:
- Steps 1-3: Fully deterministic (no randomness)
- Step 4: Seeded random sampling
- Step 5: Deterministic (ID generation uses counter)

**Version Pinning**:
- ECAPA-TDNN model: `speechbrain/spkrec-ecapa-voxceleb` (pinned to SpeechBrain 1.0.3)
- CV archive: `cv-corpus-24.0-2025-12-05-es.tar.gz` (fixed version)

## Maintenance

### Updating ECAPA-TDNN Model

If SpeechBrain releases a new model or if migrating to a different embedding model:

1. Update `settings.MODEL_SOURCE` and `settings.MODEL_SAVE_DIR`
2. Verify embedding dimension (currently 192 for ECAPA-TDNN)
3. Re-run Steps 1-2 to regenerate embeddings
4. Update README.md with new model details

### Updating CV Archive

When a new CV version is released:

1. Download new archive
2. Update `settings.CV_ARCHIVE` path
3. Update `settings.CV_EXTRACTED_DIR` (new date in folder name)
4. Re-run Steps 2-5

### Adding New Accents

To include additional Spanish accents (e.g., Argentina, Peru):

1. Update `settings.ACCENT_KEYWORDS` with new keyword lists
2. Update `settings.ACCENT_CODES` with accent codes (e.g., "Argentina": "ar")
3. Update `settings.ACCENT_TARGETS` with speaker targets
4. Re-run Steps 2-5 (Step 1 unchanged)

## References

- **General Architecture Guide**: `app/pipeline/ARCHITECTURE.md`
- **Pipeline Documentation**: `README.md`
- **Project Guidelines**: `CLAUDE.md` (project root)
- **Hayward Architecture Reference**: `E:\Trabajo\Konecto\hayward\app\pipelines\architecture_pipeline.md` (inspiration source)
