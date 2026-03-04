"""
Step 4: Balanced stratified sampling of Common Voice speakers.

This step selects samples with fixed targets per accent to achieve
balanced representation across all target accents.
"""
import csv
import json
import random
from collections import Counter, defaultdict
from loguru import logger
from pathlib import Path
from typing import Dict, List

from app.pipeline.select_mozilla_speakers.settings import settings


class BalancedSampler:
    """Performs balanced stratified sampling with fixed accent targets.

    This step selects exactly N samples per accent (where N varies by accent to
    match HABLA distribution + additions) with gender and age balance.

    Attributes:
        accent_targets: Dictionary mapping accent names to sample counts
        random_seed: Random seed for reproducibility
        input_dir: Directory containing filtered speakers and metadata
    """

    def __init__(
        self,
        accent_targets: Dict[str, int] | None = None,
        random_seed: int | None = None,
        input_dir: Path | None = None
    ) -> None:
        """Initialize the balanced sampler.

        Args:
            accent_targets: Optional override for accent targets
            random_seed: Optional override for random seed
            input_dir: Optional override for input directory
        """
        self.accent_targets = accent_targets or settings.ACCENT_TARGETS
        self.random_seed = random_seed or settings.RANDOM_SEED
        self.input_dir = input_dir or settings.OUTPUT_DIR

        self.validated_tsv = settings.CV_EXTRACTED_DIR / "es" / "validated.tsv"

    def get_accent_category(self, sample_dict: Dict) -> str:
        """Get accent category from sample metadata."""
        return sample_dict.get("accent_category", "Unknown")

    def sample_with_female_priority(self, samples: List[Dict], target: int) -> List[Dict]:
        """Sample with priority for female speakers (for Spain gender imbalance).

        Args:
            samples: List of sample dictionaries
            target: Target number of samples

        Returns:
            List of selected samples
        """
        females = [s for s in samples if "female" in s.get("gender", "").lower()]
        males = [s for s in samples if "male" in s.get("gender", "").lower() and "female" not in s.get("gender", "").lower()]

        if len(females) >= target // 2:
            selected_females = random.sample(females, target // 2)
            selected_males = random.sample(males, target - target // 2)
        else:
            selected_females = females
            selected_males = random.sample(males, target - len(selected_females))

        return selected_females + selected_males

    def sample_balanced(self, samples: List[Dict], target: int) -> List[Dict]:
        """Sample with gender and age balance.

        Args:
            samples: List of sample dictionaries
            target: Target number of samples

        Returns:
            List of selected samples
        """
        # Group by gender and age
        strata = defaultdict(list)
        for sample in samples:
            gender = sample.get("gender", "unknown")
            age = sample.get("age", "unknown")
            key = f"{gender}_{age}"
            strata[key].append(sample)

        # Proportional allocation
        selected = []
        strata_sizes = {k: len(v) for k, v in strata.items()}
        total = sum(strata_sizes.values())

        for stratum_key, stratum_samples in strata.items():
            proportion = len(stratum_samples) / total
            stratum_target = int(target * proportion)

            if stratum_target > len(stratum_samples):
                stratum_target = len(stratum_samples)

            if stratum_target > 0:
                selected.extend(random.sample(stratum_samples, stratum_target))

        # If we're short, randomly sample from largest strata
        if len(selected) < target:
            remaining = target - len(selected)
            all_remaining = [s for s in samples if s not in selected]
            if len(all_remaining) >= remaining:
                selected.extend(random.sample(all_remaining, remaining))
            else:
                selected.extend(all_remaining)

        return selected[:target]

    def stratified_sample(
        self,
        samples_by_speaker: Dict[str, List[Dict]],
        cv_metadata: dict
    ) -> List[Dict]:
        """Perform stratified sampling with fixed targets per accent.

        Args:
            samples_by_speaker: Dictionary mapping client_id to samples
            cv_metadata: Speaker metadata dictionary

        Returns:
            List of selected sample dictionaries
        """
        # Group samples by accent using cv_metadata (which has accent_category from step 2)
        samples_by_accent = defaultdict(list)

        for client_id, samples in samples_by_speaker.items():
            accent = cv_metadata.get(client_id, {}).get("accent_category", "Unknown")
            if accent in self.accent_targets:
                for sample in samples:
                    sample["accent_category"] = accent
                    samples_by_accent[accent].append(sample)

        logger.info("Available samples by accent:")
        for accent in ["Colombia", "Chile", "Venezuela", "Mexico", "Spain"]:
            logger.info(f"  {accent:12s}: {len(samples_by_accent[accent]):,} samples available")

        selected_samples = []

        for accent, target in self.accent_targets.items():
            available = samples_by_accent[accent]

            if len(available) < target:
                logger.warning(f"{accent} has only {len(available)} samples, need {target}")
                selected = available
            else:
                if accent == "Spain":
                    logger.info(f"  {accent:12s}: sampling {target:,} with female priority...")
                    selected = self.sample_with_female_priority(available, target)
                else:
                    logger.info(f"  {accent:12s}: sampling {target:,} with gender/age balance...")
                    selected = self.sample_balanced(available, target)

            selected_samples.extend(selected)
            logger.info(f"  {accent:12s}: ✓ selected {len(selected):,} samples")

        logger.info(f"Total selected: {len(selected_samples):,}")
        return selected_samples

    def compute_statistics(self, selected_samples: List[Dict], cv_metadata: dict) -> dict:
        """Compute selection statistics.

        Args:
            selected_samples: List of selected samples
            cv_metadata: Speaker metadata

        Returns:
            Dictionary with statistics
        """
        accents = [self.get_accent_category(s) for s in selected_samples]
        genders = [s.get("gender", "Unknown") for s in selected_samples]
        ages = [s.get("age", "Unknown") for s in selected_samples]
        speakers = set(s.get("client_id") for s in selected_samples)

        return {
            "total_samples": len(selected_samples),
            "unique_speakers": len(speakers),
            "accent_distribution": dict(Counter(accents)),
            "gender_distribution": dict(Counter(genders)),
            "age_distribution": dict(Counter(ages)),
        }

    def execute(self) -> Path:
        """Execute balanced stratified sampling.

        Returns:
            Path to selected_15340.tsv
        """
        logger.info("Step 04 - Balanced Sampling: Starting")

        # Set random seed
        random.seed(self.random_seed)

        # Load filtered speakers
        filtered_speakers_path = self.input_dir / "filtered_speakers.json"
        logger.info(f"Loading filtered speakers from {filtered_speakers_path}...")
        with open(filtered_speakers_path, "r", encoding="utf-8") as f:
            filtered_speakers = set(json.load(f))
        logger.info(f"  Filtered speakers: {len(filtered_speakers):,}")

        # Load CV metadata
        cv_metadata_path = self.input_dir / "cv_speaker_metadata.json"
        logger.info(f"Loading CV metadata from {cv_metadata_path}...")
        with open(cv_metadata_path, "r", encoding="utf-8") as f:
            cv_metadata = json.load(f)
        logger.info(f"  Total CV speakers in metadata: {len(cv_metadata):,}")

        # Parse validated.tsv and filter
        logger.info(f"Parsing {self.validated_tsv}...")
        samples_by_speaker = defaultdict(list)
        total_rows = 0

        with open(self.validated_tsv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                total_rows += 1
                client_id = row["client_id"]
                if client_id in filtered_speakers:
                    samples_by_speaker[client_id].append(row)

        logger.info(f"  Total rows in validated.tsv: {total_rows:,}")
        logger.info(f"  Filtered speakers with samples: {len(samples_by_speaker):,}")
        total_available = sum(len(samples) for samples in samples_by_speaker.values())
        logger.info(f"  Total available samples: {total_available:,}")

        # Stratified sampling
        logger.info(f"Performing stratified sampling (target: {sum(self.accent_targets.values()):,})...")
        logger.info("Per-accent targets:")
        for accent, target in self.accent_targets.items():
            logger.info(f"  {accent:12s}: {target:,}")

        selected_samples = self.stratified_sample(samples_by_speaker, cv_metadata)

        # Compute statistics
        logger.info("Computing statistics...")
        stats = self.compute_statistics(selected_samples, cv_metadata)

        logger.info("=" * 70)
        logger.info("Selection Statistics")
        logger.info("=" * 70)
        logger.info(f"Total samples: {stats['total_samples']:,}")
        logger.info(f"Unique speakers: {stats['unique_speakers']:,}")
        logger.info("")
        logger.info("Accent distribution:")
        for accent, count in sorted(stats['accent_distribution'].items(), key=lambda x: -x[1]):
            logger.info(f"  {accent}: {count:,} ({count / stats['total_samples'] * 100:.1f}%)")

        # Save selected samples
        output_tsv = self.input_dir / "selected_15340.tsv"
        logger.info(f"Saving selected samples to {output_tsv}...")

        if selected_samples:
            with open(output_tsv, "w", encoding="utf-8", newline="") as f:
                fieldnames = selected_samples[0].keys()
                writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
                writer.writeheader()
                writer.writerows(selected_samples)
        else:
            logger.warning("No samples selected! Creating empty file.")
            output_tsv.touch()

        # Save statistics
        stats_path = self.input_dir / "selection_stats.json"
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)

        logger.info("Step 04 - Balanced Sampling: Complete")
        return output_tsv
