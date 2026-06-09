"""
Step 2: Prepare Text Prompts

Assigns Spanish text prompts to each speaker from Mozilla Common Voice transcripts.
Text prompts are prefixed with OUTETTS_TEXT_ for unique identification across pipelines.
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger

from app.pipeline.outetts_attack.settings import settings
from app.pipeline.outetts_attack.schemas.text_prompts_result import TextPromptsResult


class TextPromptPreparator:
    """Prepares Spanish text prompts for OuteTTS generation.

    Loads Mozilla Common Voice transcripts and assigns random texts
    to each speaker with reproducible seeding. Text IDs use the
    OUTETTS_TEXT_ prefix to avoid collisions with other pipelines.

    Attributes:
        cv_metadata_path: Path to Mozilla Common Voice validated.tsv.
        output_dir: Output directory for text prompts metadata.
        samples_per_speaker: Number of texts to assign per speaker.
        random_seed: Seed for reproducible sampling.
    """

    def __init__(
        self,
        cv_metadata_path: Path | None = None,
        output_dir: Path | None = None,
        samples_per_speaker: int | None = None,
        random_seed: int | None = None,
    ):
        """Initialize text prompt preparator.

        Args:
            cv_metadata_path: Path to Mozilla CV validated.tsv (default: from settings).
            output_dir: Output directory (default: from settings).
            samples_per_speaker: Texts per speaker (default: from settings).
            random_seed: Random seed (default: from settings).
        """
        self.cv_metadata_path = cv_metadata_path or settings.CV_METADATA_PATH
        self.output_dir = output_dir or settings.OUTPUT_DIR
        self.samples_per_speaker = samples_per_speaker or settings.SAMPLES_PER_SPEAKER
        self.random_seed = random_seed or settings.RANDOM_SEED

    def execute(self) -> TextPromptsResult:
        """Prepare text prompts for all speakers.

        Returns:
            TextPromptsResult with prompts metadata path and total count.

        Raises:
            FileNotFoundError: If reference_metadata.json does not exist.
        """
        logger.info("Step 2: Preparing text prompts...")

        logger.info(f"Loading transcripts from {self.cv_metadata_path}")
        cv_df = pd.read_csv(self.cv_metadata_path, sep="\t")
        transcripts = cv_df["sentence"].drop_duplicates().tolist()
        logger.info(f"Loaded {len(transcripts)} unique Spanish transcripts")

        min_words, max_words = settings.TEXT_LENGTH_RANGE
        transcripts = [
            t for t in transcripts
            if min_words <= len(str(t).split()) <= max_words
        ]
        logger.info(f"Filtered to {len(transcripts)} transcripts ({min_words}-{max_words} words)")

        ref_metadata_path = self.output_dir / "reference_metadata.json"
        with open(ref_metadata_path, "r", encoding="utf-8") as f:
            references = json.load(f)

        speaker_ids = sorted(references.keys())

        if settings.MATCH_BONAFIDE_COUNT:
            logger.info(
                f"Dynamic sample count mode: matching bonafide_count per speaker "
                f"for {len(speaker_ids)} speakers"
            )
        else:
            logger.info(
                f"Assigning {self.samples_per_speaker} texts per speaker "
                f"to {len(speaker_ids)} speakers"
            )

        np.random.seed(self.random_seed)

        prompts = {}
        text_counter = 1

        for speaker_id in speaker_ids:
            if settings.MATCH_BONAFIDE_COUNT:
                n_samples = references[speaker_id].get(
                    "bonafide_count", self.samples_per_speaker
                )
                n_samples = max(1, n_samples)
            else:
                n_samples = self.samples_per_speaker

            if len(transcripts) < n_samples:
                logger.warning(
                    f"Not enough transcripts ({len(transcripts)}) "
                    f"for {speaker_id} ({n_samples} needed), allowing repeats"
                )
                speaker_texts = np.random.choice(
                    transcripts, size=n_samples, replace=True
                ).tolist()
            else:
                speaker_texts = np.random.choice(
                    transcripts, size=n_samples, replace=False
                ).tolist()

            speaker_prompts = []
            for text in speaker_texts:
                speaker_prompts.append({
                    "text_id": f"OUTETTS_TEXT_{text_counter:05d}",
                    "text": text,
                    "length_words": len(str(text).split()),
                    "source": settings.TEXT_SOURCE,
                })
                text_counter += 1

            prompts[speaker_id] = speaker_prompts

        prompts_path = self.output_dir / "text_prompts.json"
        with open(prompts_path, "w", encoding="utf-8") as f:
            json.dump(prompts, f, indent=2, ensure_ascii=False)

        total_prompts = sum(len(texts) for texts in prompts.values())

        logger.info(f"Assigned {total_prompts} text prompts")
        logger.info(f"  Prompts saved to: {prompts_path}")

        return TextPromptsResult(
            prompts_metadata_path=prompts_path,
            total_prompts=total_prompts,
        )
