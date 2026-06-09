"""
Mozilla Common Voice Speaker Selection Pipeline - Facade Orchestrator.

This pipeline selects 15,340 acoustically diverse speakers from Common Voice
to augment the HABLA anti-spoofing dataset, adding Mexico and Spain accents
while maintaining speaker independence.
"""
from loguru import logger
from pathlib import Path

from app.pipeline.select_mozilla_speakers.schemas.pipeline_config import (
    MozillaSpeakerPipelineConfig
)
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
from app.pipeline.select_mozilla_speakers.settings import settings


class MozillaSpeakerSelectionPipeline:
    """Mozilla Common Voice speaker selection and HABLA dataset augmentation pipeline.

    This pipeline implements the Facade pattern, orchestrating:
    1. HABLA speaker embedding extraction (ECAPA-TDNN)
    2. Common Voice speaker embedding extraction
    3. Similarity-based filtering (cosine similarity threshold)
    4. Balanced stratified sampling (fixed targets per accent)
    5. Integration into bonafide_dataset_by_speaker_v2/

    Attributes:
        config: Pipeline run configuration.
    """

    def __init__(self, config: MozillaSpeakerPipelineConfig | None = None) -> None:
        """Initialize the pipeline with run configuration.

        Args:
            config: Pipeline configuration. If None, runs all steps with defaults.
        """
        self.config = config or MozillaSpeakerPipelineConfig()
        logger.info(f"{self.__class__.__name__} initialized")

    def run(self) -> Path:
        """Execute the full pipeline.

        Returns:
            Path to the bonafide_dataset_by_speaker_v2/ directory

        Raises:
            Exception: If any pipeline step fails.
        """
        logger.info("=" * 70)
        logger.info(f"{self.__class__.__name__.upper()} - START")
        logger.info("=" * 70)

        try:
            # === STEP 1: Extract HABLA Embeddings ===
            if self.config.run_step_1:
                logger.info("STEP 1/5: Extract HABLA Speaker Embeddings")
                step_1 = HablaEmbeddingExtractor(
                    output_dir=self.config.output_dir_override or settings.OUTPUT_DIR
                )
                result_1 = step_1.execute()
                logger.info(f"✓ HABLA embeddings: {result_1.embedding_count} speakers")
            else:
                logger.info("STEP 1/5: Skipped")

            # === STEP 2: Extract CV Embeddings ===
            if self.config.run_step_2:
                logger.info("STEP 2/5: Extract Common Voice Speaker Embeddings")
                step_2 = CVEmbeddingExtractor(
                    output_dir=self.config.output_dir_override or settings.OUTPUT_DIR
                )
                result_2 = step_2.execute()
                logger.info(f"✓ CV embeddings: {result_2.embedding_count} speakers")
            else:
                logger.info("STEP 2/5: Skipped")

            # === STEP 3: Filter by Similarity ===
            if self.config.run_step_3:
                logger.info("STEP 3/5: Filter Speakers by Similarity")
                step_3 = SimilarityFilter(
                    threshold=self.config.similarity_threshold_override or settings.SIMILARITY_THRESHOLD
                )
                result_3 = step_3.execute()
                logger.info(f"✓ Filtered speakers saved to: {result_3}")
            else:
                logger.info("STEP 3/5: Skipped")

            # === STEP 4: Balanced Sampling ===
            if self.config.run_step_4:
                logger.info("STEP 4/5: Balanced Stratified Sampling")
                step_4 = BalancedSampler(
                    random_seed=self.config.random_seed_override or settings.RANDOM_SEED
                )
                result_4 = step_4.execute()
                logger.info(f"✓ Selected samples saved to: {result_4}")
            else:
                logger.info("STEP 4/5: Skipped")

            # === STEP 5: Integrate into HABLA v2 ===
            if self.config.run_step_5:
                logger.info("STEP 5/5: Integrate into HABLA 2.0 Dataset")
                step_5 = DatasetIntegrator()
                result_5 = step_5.execute()
                logger.info(f"✓ HABLA v2 dataset created at: {result_5}")
            else:
                logger.info("STEP 5/5: Skipped")

            logger.info("=" * 70)
            logger.info(f"{self.__class__.__name__.upper()} - COMPLETE")
            logger.info("=" * 70)

            return settings.BONAFIDE_V2_DIR

        except Exception as e:
            logger.exception(f"Pipeline failed: {e}")
            raise


if __name__ == "__main__":
    # Example usage
    pipeline = MozillaSpeakerSelectionPipeline()
    result = pipeline.run()
    print(f"✓ Pipeline complete. Output: {result}")
