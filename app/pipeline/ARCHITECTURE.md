# Pipeline Architecture Guide - HABLA Anti-Spoofing Project

## 1. Overview

This document defines the standard architecture, conventions, and best practices for creating pipelines in the HABLA Anti-Spoofing Voice project. All pipelines follow the **Facade** pattern for orchestration, **Strategy**-based step classes, **Pydantic** schemas for data modeling, and pipeline-scoped settings.

**Inspiration**: Based on Hayward production architecture patterns, adapted for ML/audio processing workflows.

---

## 2. Directory Structure

Every pipeline must follow this canonical layout:

```
app/pipeline/<pipeline_name>/
    __init__.py                    # Public exports (Facade class + key types)
    pipeline_facade.py             # Facade orchestrator (single entry point)
    settings.py                    # Pipeline-specific configuration (Pydantic BaseModel)
    README.md                      # User-facing documentation
    ARCHITECTURE.md                # Technical design document
    schemas/
        __init__.py                # Schema exports
        pipeline_config.py         # Input configuration schema
        <domain_schemas>.py        # One file per Pydantic model
    steps/
        __init__.py                # Step exports
        step_01_<name>.py          # Step 1 implementation (one class per file)
        step_02_<name>.py          # Step 2 implementation
        ...
    utils/
        __init__.py                # Utility exports
        <utility_modules>.py       # Helper functions (no classes alongside functions)
```

### Naming Conventions

| Element | Convention | Example |
|---------|-----------|---------|
| Pipeline folder | `snake_case` | `select_mozilla_speakers`, `fishgram_attack` |
| Facade class | `PascalCase` + `Pipeline` suffix | `MozillaSpeakerSelectionPipeline` |
| Step files | `step_XX_<verb_noun>.py` (zero-padded) | `step_01_extract_embeddings.py` |
| Step classes | `PascalCase` describing the action | `HablaEmbeddingExtractor`, `SimilarityFilter` |
| Settings class | `PascalCase` + `Settings` suffix | `MozillaSpeakerSettings` |
| Schema files | `snake_case` describing the model | `speaker_embedding.py`, `similarity_result.py` |

---

## 3. Core Components

### 3.1 Pipeline Facade (`pipeline_facade.py`)

The Facade is the single entry point for the pipeline. It orchestrates all steps in sequence and handles lifecycle concerns (initialization, cleanup, error handling).

**Required structure:**

```python
from loguru import logger
from pathlib import Path

from app.pipeline.<pipeline_name>.schemas.pipeline_config import <PipelineConfig>
from app.pipeline.<pipeline_name>.steps.step_01_<name> import <Step1Class>
from app.pipeline.<pipeline_name>.steps.step_02_<name> import <Step2Class>


class <PipelineName>Pipeline:
    """<One-line description of what the pipeline does>.

    This pipeline implements the Facade pattern, orchestrating:
    1. <Step 1 description>
    2. <Step 2 description>
    ...

    Attributes:
        config: Pipeline run configuration.
    """

    def __init__(self, config: <PipelineConfig>) -> None:
        """Initialize the pipeline with run configuration.

        Args:
            config: Pipeline configuration with required parameters.
        """
        self.config = config
        logger.info(f"{self.__class__.__name__} initialized")

    def run(self) -> <ReturnType>:
        """Execute the full pipeline.

        Returns:
            <Description of the output>.

        Raises:
            Exception: If any pipeline step fails.
        """
        logger.info("=" * 70)
        logger.info(f"{self.__class__.__name__.upper()} - START")
        logger.info("=" * 70)

        try:
            # === STEP 1: <Description> ===
            step_1 = <Step1Class>(...)
            result_1 = step_1.execute()

            # === STEP 2: <Description> ===
            step_2 = <Step2Class>(...)
            result_2 = step_2.execute(input=result_1)

            # ... additional steps ...

            logger.info(f"{self.__class__.__name__.upper()} - COMPLETE")
            return result_2

        except Exception as e:
            logger.exception(f"Pipeline failed: {e}")
            raise
```

**Key rules:**
- Constructor validates configuration and initializes dependencies
- `run()` method executes steps sequentially
- Each step receives output of previous step as input
- Use `try/except` for error handling
- Log pipeline start/end with separators

### 3.2 Settings (`settings.py`)

Each pipeline owns its configuration. Pipeline-specific settings must **never** go in `app/config.py`. Only truly global settings (model paths, CUDA device, project-wide paths) belong in global config.

```python
from pydantic import BaseModel, Field


class <Pipeline>Settings(BaseModel):
    """Configuration for the <Pipeline Name> pipeline.

    Attributes:
        PARAM_ONE: Description of parameter one.
        PARAM_TWO: Description of parameter two.
    """

    PARAM_ONE: str = Field(
        default="default_value",
        description="Description of parameter one.",
    )
    PARAM_TWO: int = Field(
        default=10,
        description="Description of parameter two.",
    )


# Module-level singleton
settings = <Pipeline>Settings()
```

**Key rules:**
- All parameter names use `UPPER_SNAKE_CASE`
- Use `Field(default=..., description=...)` for documentation
- Instantiate module-level singleton: `settings = <Pipeline>Settings()`
- Model paths, thresholds, random seeds, output paths all belong here

### 3.3 Schemas (`schemas/`)

All data structures use **Pydantic BaseModel**. The `@dataclass` decorator is strictly forbidden (per project CLAUDE.md).

Each schema lives in its own file (one class per file). The `schemas/__init__.py` re-exports all public models.

**Required schemas:**
- `pipeline_config.py` -- Input configuration for the pipeline
- Domain-specific models for step inputs/outputs

```python
from pydantic import BaseModel, Field


class <SchemaName>(BaseModel):
    """<Description of what this model represents>.

    Attributes:
        field_one: Description.
        field_two: Description.
    """

    field_one: str = Field(..., description="Description.")
    field_two: int = Field(default=0, description="Description.")
```

### 3.4 Steps (`steps/`)

Each step is a single class in its own file. Steps implement the **Strategy** pattern -- they are interchangeable, independently testable units of work.

```python
from loguru import logger
from typing import Dict, List

from app.pipeline.<pipeline_name>.schemas.<input_schema> import <InputSchema>
from app.pipeline.<pipeline_name>.schemas.<output_schema> import <OutputSchema>


class <StepName>:
    """<One-line description of what this step does>.

    <Extended description of the step's purpose and behavior>.

    Attributes:
        <any constructor dependencies>.
    """

    def __init__(self, <dependencies>) -> None:
        """Initialize the step with required dependencies.

        Args:
            <dependencies>: Description.
        """
        self.<dep> = <dep>

    def execute(self, <inputs>) -> <OutputType>:
        """Execute this pipeline step.

        Args:
            <inputs>: Description.

        Returns:
            <Description of the output>.
        """
        logger.info(f"Step {self.__class__.__name__}: Starting")
        # ... implementation ...
        logger.info(f"Step {self.__class__.__name__}: Complete")
        return result
```

**Key rules:**
- Every step class has an `execute()` method as its public interface
- Dependencies (models, database clients) injected via constructor
- Steps must not import or depend on Facade or other steps directly
- Steps receive typed inputs and return typed outputs
- File naming: `step_XX_<descriptive_name>.py` with zero-padded numbering
- One class per file (per CLAUDE.md)

### 3.5 Utils (`utils/`)

Utility functions that support multiple steps or contain shared logic. Functions live here; classes do not coexist with functions in the same file (per CLAUDE.md).

Common utilities include: audio processing, embedding normalization, file I/O helpers, data fusion.

### 3.6 Documentation

Every pipeline must include:

1. **README.md** (User-facing)
   - Overview and motivation
   - Input/output description
   - Step-by-step walkthrough
   - Usage examples
   - Performance benchmarks
   - Troubleshooting

2. **ARCHITECTURE.md** (Technical design)
   - High-level flow diagram (ASCII art)
   - Design patterns used
   - Step implementations detail
   - Configuration parameters
   - File structure
   - Dependencies

---

## 4. Design Patterns Reference

| Pattern | Where | Purpose |
|---------|-------|---------|
| **Facade** | `pipeline_facade.py` | Single entry point orchestrating all steps |
| **Strategy** | `steps/step_XX_*.py` | Interchangeable, independently testable step implementations |
| **Dependency Injection** | Step constructors | External services (models, APIs) injected, not instantiated inside steps |
| **Singleton** | `settings.py` module-level instance | Single shared configuration instance |
| **Factory** | Model loader classes (when applicable) | Creates configured model instances |
| **Adapter** | Data source adapters | Uniform interface to different data sources (HABLA, Mozilla CV) |

---

## 5. Step-by-Step: Creating a New Pipeline

### Step 1: Define the Problem

Before writing code, answer:
- What are the inputs (audio files, embeddings, metadata)?
- What are the outputs (selected samples, generated audio, metrics)?
- How many discrete processing phases does it require?
- What external dependencies are needed (models, APIs, datasets)?

### Step 2: Create the Directory Structure

```bash
app/pipeline/<pipeline_name>/
    __init__.py
    pipeline_facade.py
    settings.py
    README.md
    ARCHITECTURE.md
    schemas/
        __init__.py
        pipeline_config.py
    steps/
        __init__.py
    utils/
        __init__.py
```

### Step 3: Define Schemas

Start with `pipeline_config.py` (input configuration), then define intermediate and output schemas. Each schema in its own file under `schemas/`.

### Step 4: Define Settings

Add all tunable parameters to `settings.py`:
- Model paths and parameters
- Thresholds and hyperparameters
- Random seeds (for reproducibility)
- Output paths
- Device configuration (CUDA/CPU)

### Step 5: Implement Steps

Create one file per step under `steps/`. Number them in execution order. Each step class:
- Accepts dependencies via constructor
- Has an `execute()` method with typed input and output
- Logs progress with `loguru.logger`
- Uses `tqdm` for progress bars when processing collections

### Step 6: Implement the Facade

Wire all steps together in `pipeline_facade.py`. The Facade:
- Validates configuration in constructor
- Initializes external dependencies (load models once)
- Calls steps sequentially in `run()`
- Handles errors and cleanup

### Step 7: Configure Exports

In `__init__.py`, export the Facade class and any types that external consumers need.

### Step 8: Write Documentation

Create both `README.md` (user-facing) and `ARCHITECTURE.md` (technical design).

### Step 9: Test End-to-End

Run the full pipeline with test data:
```python
from app.pipeline.<pipeline_name> import <PipelineName>Pipeline
from app.pipeline.<pipeline_name>.schemas.pipeline_config import <PipelineConfig>

config = <PipelineConfig>(...)
pipeline = <PipelineName>Pipeline(config)
result = pipeline.run()
```

---

## 6. Rules Summary

| Rule | Description |
|------|-------------|
| Pydantic only | All data models use `BaseModel`. No `@dataclass` (per CLAUDE.md). |
| One class per file | Never place multiple classes in the same file (per CLAUDE.md). |
| No loose functions alongside classes | Utility functions go in `utils/` (per CLAUDE.md). |
| All imports at top | No imports inside `try/except`, conditionals, or functions (per CLAUDE.md). |
| Pipeline-scoped settings | Model configs and tunable params in the pipeline's `settings.py`. |
| Comprehensive docstrings | All classes and functions documented in Google/NumPy style (per CLAUDE.md). |
| No emojis in source code | Code and docstrings must be emoji-free (per CLAUDE.md). |
| Loguru for logging | Use `from loguru import logger` throughout. |
| Typed interfaces | Steps accept and return typed objects (Pydantic models or standard types). |
| DRY principle | Don't Repeat Yourself - extract shared logic to utils/ (per CLAUDE.md). |
| Environment variables via settings.py | Use `settings.VARIABLE_NAME` pattern (per CLAUDE.md). |

---

## 7. HABLA Project Specifics

### Audio Processing Conventions

**File Formats:**
- Input: WAV (16kHz, mono) or MP3
- Output: WAV (16kHz, mono) for generated/processed audio
- Use `torchaudio.load()` for reading, `torchaudio.save()` for writing

**Speaker IDs:**
- Format: `{accent_code}{gender_code}_{number:05d}`
- Example: `mxm_00001` (Mexico male speaker 1)
- Accent codes: ar, cl, co, es, mx, pe, ve
- Gender codes: m, f, o

**Directory Structure for Datasets:**
```
data/
├── bonafide_dataset_by_speaker/
│   └── {speaker_id}/
│       ├── train/
│       ├── val/
│       └── test/
```

### Model Loading Best Practices

```python
import torch
from pathlib import Path

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load once in Facade constructor or step __init__
model = load_model(model_path)
model.to(DEVICE)
model.eval()

# Use in step execute()
with torch.no_grad():
    output = model(input_tensor.to(DEVICE))
```

### Progress Tracking

```python
from tqdm import tqdm

# Always provide desc parameter for context
for item in tqdm(items, desc="Processing speakers"):
    process(item)

# Show periodic updates for very long operations
if idx % 100 == 0:
    logger.info(f"[{idx}/{total}] Processed {item_name}")
```

---

## 8. Existing Pipelines

| Pipeline | Purpose | Steps | Status |
|----------|---------|-------|--------|
| `select_mozilla_speakers` | CV speaker selection & HABLA augmentation | 5 steps | ✅ Complete (needs refactoring) |
| `fishgram_attack` | FishGram vocoder attack generation | TBD | 🚧 Planned |
| `whisper_resynth_attack` | Whisper-based resynthesis attack | TBD | 🚧 Planned |

---

## 9. Migration Guide

### Refactoring Existing Code to This Architecture

**Current `select_mozilla_speakers` structure:**
```
app/pipeline/select_mozilla_speakers/
├── 01_extract_habla_embeddings.py
├── 02_extract_cv_embeddings.py
├── 03_filter_by_similarity.py
├── 04_balanced_sampling.py
├── 05_integrate_cv_samples.py
├── run_pipeline.py
└── README.md
```

**Target structure:**
```
app/pipeline/select_mozilla_speakers/
├── __init__.py
├── pipeline_facade.py
├── settings.py
├── README.md
├── ARCHITECTURE.md
├── schemas/
│   ├── __init__.py
│   ├── pipeline_config.py
│   ├── speaker_embedding.py
│   ├── similarity_result.py
│   └── selection_result.py
├── steps/
│   ├── __init__.py
│   ├── step_01_extract_habla_embeddings.py
│   ├── step_02_extract_cv_embeddings.py
│   ├── step_03_filter_by_similarity.py
│   ├── step_04_balanced_sampling.py
│   └── step_05_integrate_cv_samples.py
└── utils/
    ├── __init__.py
    ├── audio_loader.py
    └── embedding_utils.py
```

**Migration steps:**
1. Create new directory structure
2. Extract configuration constants to `settings.py`
3. Define Pydantic schemas for each step's input/output
4. Convert standalone scripts to Step classes
5. Create Facade to orchestrate steps
6. Update README and create ARCHITECTURE.md
7. Test end-to-end equivalence

---

## 10. References

**Architecture Patterns:**
- Gamma et al., "Design Patterns: Elements of Reusable Object-Oriented Software" (GoF Patterns)
- Martin Fowler, "Patterns of Enterprise Application Architecture"

**Python Best Practices:**
- PEP 8: Style Guide for Python Code
- Pydantic Documentation: https://docs.pydantic.dev/
- Loguru Documentation: https://loguru.readthedocs.io/

**ML Pipeline Design:**
- Sculley et al., "Hidden Technical Debt in Machine Learning Systems" (NeurIPS 2015)
- Breck et al., "The ML Test Score: A Rubric for ML Production Readiness" (2017)

---

**Last Updated**: March 2026
**Version**: 1.0
**Based on**: Hayward production architecture patterns
