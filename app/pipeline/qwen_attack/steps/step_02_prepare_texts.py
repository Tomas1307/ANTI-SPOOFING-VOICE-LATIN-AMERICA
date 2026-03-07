"""
Step 2: Prepare Text Prompts

Assigns Spanish text prompts to each speaker from Mozilla Common Voice transcripts.
Applies conservative text length filtering to prevent Qwen3-TTS truncation artifacts.
"""
import json
import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
from app.pipeline.qwen_attack.settings import settings
from app.pipeline.qwen_attack.schemas.text_prompts_result import TextPromptsResult


class TextPromptPreparator:
    """Prepares Spanish text prompts for Qwen3-TTS generation.

    Loads Mozilla Common Voice transcripts and assigns random texts
    to each speaker with reproducible seeding. Applies a conservative
    word count ceiling (default: 40 words) to prevent Qwen3-TTS
    silent truncation artifacts on long texts.

    Attributes:
        cv_metadata_path: Path to Mozilla Common Voice validated.tsv file.
        output_dir: Directory where text prompts metadata is saved.
        samples_per_speaker: Number of text prompts assigned per speaker.
        random_seed: Seed for reproducible random text selection.
    """

    def __init__(
        self,
        cv_metadata_path: Path | None = None,
        output_dir: Path | None = None,
        samples_per_speaker: int | None = None,
        random_seed: int | None = None
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
            Exception: If text preparation fails.
        """
        logger.info("Preparing text prompts...")

        # Load Mozilla CV transcripts
        logger.info(f"Loading transcripts from {self.cv_metadata_path}")
        cv_df = pd.read_csv(self.cv_metadata_path, sep="\t")
        transcripts = cv_df["sentence"].drop_duplicates().tolist()
        logger.info(f"Loaded {len(transcripts)} unique Spanish transcripts")

        # Filter by word length (conservative ceiling for Qwen truncation prevention)
        min_words, max_words = settings.TEXT_LENGTH_RANGE
        transcripts = [
            t for t in transcripts
            if min_words <= len(str(t).split()) <= max_words
        ]
        logger.info(
            f"Filtered to {len(transcripts)} transcripts "
            f"({min_words}-{max_words} words, Qwen truncation prevention)"
        )

        # Load reference metadata to get speaker IDs
        ref_metadata_path = self.output_dir / "reference_metadata.json"
        with open(ref_metadata_path, "r", encoding="utf-8") as f:
            references = json.load(f)

        speaker_ids = sorted(references.keys())
        logger.info(
            f"Assigning {self.samples_per_speaker} texts per speaker "
            f"to {len(speaker_ids)} speakers"
        )

        # Set random seed for reproducibility
        np.random.seed(self.random_seed)

        # Assign texts to speakers
        prompts = {}
        text_counter = 1

        for speaker_id in speaker_ids:
            # Sample texts without replacement
            if len(transcripts) < self.samples_per_speaker:
                logger.warning(
                    f"Not enough transcripts ({len(transcripts)}), allowing repeats"
                )
                speaker_texts = np.random.choice(
                    transcripts,
                    size=self.samples_per_speaker,
                    replace=True
                ).tolist()
            else:
                speaker_texts = np.random.choice(
                    transcripts,
                    size=self.samples_per_speaker,
                    replace=False
                ).tolist()

            # Store prompts with unique text IDs
            speaker_prompts = []
            for text in speaker_texts:
                speaker_prompts.append({
                    "text_id": f"QWEN_TEXT_{text_counter:05d}",
                    "text": text,
                    "length_words": len(str(text).split()),
                    "source": settings.TEXT_SOURCE
                })
                text_counter += 1

            prompts[speaker_id] = speaker_prompts

        # Save prompts metadata
        prompts_path = self.output_dir / "text_prompts.json"
        with open(prompts_path, "w", encoding="utf-8") as f:
            json.dump(prompts, f, indent=2, ensure_ascii=False)

        total_prompts = sum(len(texts) for texts in prompts.values())

        logger.info(f"Assigned {total_prompts} text prompts")
        logger.info(f"  Prompts saved to: {prompts_path}")

        return TextPromptsResult(
            prompts_metadata_path=prompts_path,
            total_prompts=total_prompts
        )
