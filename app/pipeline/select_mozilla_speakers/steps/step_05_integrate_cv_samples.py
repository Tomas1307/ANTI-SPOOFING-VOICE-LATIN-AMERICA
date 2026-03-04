"""
Step 5: Integrate selected CV samples into HABLA v2 dataset structure.

This step copies HABLA speakers and adds CV speakers with proper directory
structure and speaker IDs.
"""
import csv
import json
import shutil
from collections import Counter, defaultdict
from loguru import logger
from pathlib import Path
from typing import Dict, List

from tqdm import tqdm

from app.pipeline.select_mozilla_speakers.settings import settings


class DatasetIntegrator:
    """Integrates CV samples into bonafide_dataset_by_speaker_v2.

    This step creates the v2 dataset by:
    1. Copying all existing HABLA speakers
    2. Adding CV speakers with generated speaker IDs
    3. Splitting CV samples into train/val/test

    Attributes:
        habla_dir: Source HABLA directory
        output_dir: Output v2 directory
        cv_clips_dir: CV audio clips directory
    """

    def __init__(
        self,
        habla_dir: Path | None = None,
        output_dir: Path | None = None,
        cv_clips_dir: Path | None = None
    ) -> None:
        """Initialize the dataset integrator.

        Args:
            habla_dir: Optional override for HABLA source directory
            output_dir: Optional override for output directory
            cv_clips_dir: Optional override for CV clips directory
        """
        self.habla_dir = habla_dir or settings.HABLA_DIR
        self.output_dir = output_dir or settings.BONAFIDE_V2_DIR
        self.cv_clips_dir = cv_clips_dir or settings.CV_CLIPS_DIR

        self.selected_tsv = settings.OUTPUT_DIR / "selected_15340.tsv"
        self.cv_metadata = settings.OUTPUT_DIR / "cv_speaker_metadata.json"
        self.mapping_output = settings.OUTPUT_DIR / "cv_speaker_mapping.json"

    def get_gender_code(self, gender: str) -> str:
        """Extract gender code from gender string."""
        gender_lower = gender.lower()
        if "female" in gender_lower:
            return "f"
        elif "male" in gender_lower:
            return "m"
        else:
            return "o"

    def count_existing_habla_speakers(self) -> Dict[str, int]:
        """Count existing HABLA speakers per accent to continue numbering."""
        speaker_counts = defaultdict(int)

        for speaker_dir in self.habla_dir.iterdir():
            if speaker_dir.is_dir():
                speaker_id = speaker_dir.name
                accent_code = speaker_id[:2]
                try:
                    num = int(speaker_id.split("_")[1])
                    speaker_counts[accent_code] = max(speaker_counts[accent_code], num)
                except (IndexError, ValueError):
                    continue

        return dict(speaker_counts)

    def generate_speaker_id(
        self,
        accent: str,
        gender: str,
        counter: Dict[str, Dict[str, int]]
    ) -> str:
        """Generate new speaker ID for CV speaker."""
        accent_code = settings.ACCENT_CODES[accent]
        gender_code = self.get_gender_code(gender)
        key = f"{accent_code}{gender_code}"

        counter[accent_code][gender_code] += 1
        num = counter[accent_code][gender_code]

        return f"{key}_{num:05d}"

    def split_samples(self, samples: List[Dict]) -> Dict[str, List[Dict]]:
        """Split samples into train/val/test."""
        n = len(samples)

        if n < 3:
            return {"train": samples, "val": [], "test": []}

        train_end = int(n * settings.TRAIN_RATIO)
        val_end = train_end + int(n * settings.VAL_RATIO)

        if train_end == 0:
            train_end = 1
        if val_end <= train_end:
            val_end = train_end + 1

        return {
            "train": samples[:train_end],
            "val": samples[train_end:val_end],
            "test": samples[val_end:],
        }

    def copy_habla_speakers(self):
        """Copy all existing HABLA speakers to v2 directory."""
        logger.info("=" * 70)
        logger.info("Copying Existing HABLA Speakers")
        logger.info("=" * 70)

        habla_speakers = sorted([d for d in self.habla_dir.iterdir() if d.is_dir()])
        logger.info(f"Found {len(habla_speakers)} HABLA speakers")
        logger.info("Copying to bonafide_dataset_by_speaker_v2/...")

        for speaker_dir in tqdm(habla_speakers, desc="Copying HABLA speakers"):
            dest_dir = self.output_dir / speaker_dir.name
            shutil.copytree(speaker_dir, dest_dir, dirs_exist_ok=True)

        logger.info(f"✓ Copied {len(habla_speakers)} HABLA speakers")

    def integrate_cv_speakers(self):
        """Add CV speakers to v2 directory."""
        logger.info("=" * 70)
        logger.info("Integrating Common Voice Speakers")
        logger.info("=" * 70)

        # Load metadata
        logger.info("Loading CV metadata...")
        with open(self.cv_metadata, "r", encoding="utf-8") as f:
            cv_metadata = json.load(f)

        # Load selected samples
        logger.info(f"Loading selected samples from {self.selected_tsv}...")
        samples_by_speaker = defaultdict(list)

        with open(self.selected_tsv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                client_id = row["client_id"]
                samples_by_speaker[client_id].append(row)

        logger.info(f"Loaded {len(samples_by_speaker):,} CV speakers")
        logger.info(f"Total samples: {sum(len(s) for s in samples_by_speaker.values()):,}")

        # Count existing HABLA speakers
        existing_counts = self.count_existing_habla_speakers()
        logger.info("Existing HABLA speaker counts per accent:")
        for accent_code, count in sorted(existing_counts.items()):
            logger.info(f"  {accent_code}: {count}")

        # Initialize counters for CV speaker ID generation
        cv_counters = defaultdict(lambda: defaultdict(int))
        for accent_code, count in existing_counts.items():
            cv_counters[accent_code]["m"] = count
            cv_counters[accent_code]["f"] = count
            cv_counters[accent_code]["o"] = count

        # Process each CV speaker
        speaker_mapping = {}
        accent_stats = Counter()
        gender_stats = Counter()

        logger.info("Creating speaker directories and copying audio files...")

        for client_id, samples in tqdm(samples_by_speaker.items(), desc="Processing CV speakers"):
            metadata = cv_metadata.get(client_id)
            if not metadata:
                logger.warning(f"No metadata for {client_id}, skipping")
                continue

            accent = metadata["accent_category"]
            gender = metadata["gender"]

            # Generate speaker ID
            speaker_id = self.generate_speaker_id(accent, gender, cv_counters)
            speaker_mapping[client_id] = {
                "speaker_id": speaker_id,
                "accent": accent,
                "gender": gender,
                "sample_count": len(samples),
            }

            accent_stats[accent] += 1
            gender_stats[self.get_gender_code(gender)] += 1

            # Create speaker directory
            speaker_dir = self.output_dir / speaker_id
            speaker_dir.mkdir(parents=True, exist_ok=True)

            # Split samples
            splits = self.split_samples(samples)

            # Copy audio files to respective splits
            for split_name, samples_in_split in splits.items():
                if not samples_in_split:
                    continue

                split_dir = speaker_dir / split_name
                split_dir.mkdir(exist_ok=True)

                for sample in samples_in_split:
                    audio_filename = sample["path"]
                    src_path = self.cv_clips_dir / audio_filename

                    if not src_path.exists():
                        continue

                    dest_path = split_dir / audio_filename
                    shutil.copy2(src_path, dest_path)

        logger.info("=" * 70)
        logger.info("Integration Complete")
        logger.info("=" * 70)
        logger.info(f"Total CV speakers added: {len(speaker_mapping):,}")
        logger.info("")
        logger.info("CV speakers by accent:")
        for accent, count in sorted(accent_stats.items(), key=lambda x: -x[1]):
            logger.info(f"  {accent:12s}: {count:,}")
        logger.info("")
        logger.info("CV speakers by gender:")
        for gender, count in sorted(gender_stats.items(), key=lambda x: -x[1]):
            gender_name = {"m": "Male", "f": "Female", "o": "Other"}[gender]
            logger.info(f"  {gender_name:8s}: {count:,}")

        # Save mapping
        logger.info(f"Saving speaker mapping to {self.mapping_output}...")
        with open(self.mapping_output, "w", encoding="utf-8") as f:
            json.dump(speaker_mapping, f, indent=2, ensure_ascii=False)

        return speaker_mapping

    def execute(self) -> Path:
        """Execute dataset integration.

        Returns:
            Path to bonafide_dataset_by_speaker_v2 directory
        """
        logger.info("Step 05 - Dataset Integration: Starting")

        logger.info(f"Creating {self.output_dir}...")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Copy HABLA speakers
        self.copy_habla_speakers()

        # Add CV speakers
        speaker_mapping = self.integrate_cv_speakers()

        # Final summary
        total_speakers = len(list(self.output_dir.iterdir()))
        habla_speakers = len(list(self.habla_dir.iterdir()))
        cv_speakers = len(speaker_mapping)

        logger.info("=" * 70)
        logger.info("Final Summary")
        logger.info("=" * 70)
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Total speakers: {total_speakers:,}")
        logger.info(f"  HABLA: {habla_speakers:,}")
        logger.info(f"  CV:    {cv_speakers:,}")

        logger.info("Step 05 - Dataset Integration: Complete")

        return self.output_dir
