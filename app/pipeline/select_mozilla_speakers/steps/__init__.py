"""
Step implementations for Mozilla Common Voice speaker selection pipeline.

Each step follows the Strategy pattern with a common interface:
- Constructor accepts optional overrides for dependency injection
- execute() method returns typed results (EmbeddingResult, Path, etc.)
"""
from app.pipeline.select_mozilla_speakers.steps.step_01_extract_habla_embeddings import (
    HablaEmbeddingExtractor
)
from app.pipeline.select_mozilla_speakers.steps.step_02_extract_cv_embeddings import (
    CVEmbeddingExtractor
)
from app.pipeline.select_mozilla_speakers.steps.step_03_filter_by_similarity import (
    SimilarityFilter
)
from app.pipeline.select_mozilla_speakers.steps.step_04_balanced_sampling import (
    BalancedSampler
)
from app.pipeline.select_mozilla_speakers.steps.step_05_integrate_cv_samples import (
    DatasetIntegrator
)

__all__ = [
    "HablaEmbeddingExtractor",
    "CVEmbeddingExtractor",
    "SimilarityFilter",
    "BalancedSampler",
    "DatasetIntegrator",
]
