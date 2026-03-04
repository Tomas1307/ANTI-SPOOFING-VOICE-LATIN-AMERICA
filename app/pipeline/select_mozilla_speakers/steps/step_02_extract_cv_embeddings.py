"""
Step 2: Extract Common Voice Speaker Embeddings using ECAPA-TDNN.

This step extracts the CV archive, filters by accent criteria, and extracts
one representative embedding per speaker.
"""
import csv
import json
import tarfile
from collections import Counter, defaultdict
from loguru import logger
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torchaudio
from speechbrain.inference.speaker import EncoderClassifier
from tqdm import tqdm

from app.pipeline.select_mozilla_speakers.settings import settings
from app.pipeline.select_mozilla_speakers.schemas.embedding_result import EmbeddingResult


class CVEmbeddingExtractor:
    """Extracts ECAPA-TDNN embeddings from Common Voice speakers.

    This step processes Common Voice Spanish dataset, filtering for target accents
    (Colombia, Chile, Venezuela, Mexico, Spain) and extracting one representative
    192-dimensional embedding per speaker.

    Attributes:
        archive_path: Path to CV tar.gz archive
        extracted_dir: Directory where archive is extracted
        output_dir: Directory for output files
        device: Compute device (cuda/cpu)
    """

    def __init__(
        self,
        archive_path: Path | None = None,
        extracted_dir: Path | None = None,
        output_dir: Path | None = None,
        device: str | None = None
    ) -> None:
        """Initialize the CV embedding extractor.

        Args:
            archive_path: Optional override for CV archive path
            extracted_dir: Optional override for extraction directory
            output_dir: Optional override for output directory
            device: Optional override for compute device
        """
        self.archive_path = archive_path or settings.CV_ARCHIVE
        self.extracted_dir = extracted_dir or settings.CV_EXTRACTED_DIR
        self.output_dir = output_dir or settings.OUTPUT_DIR
        self.device = device or settings.DEVICE

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def extract_archive(self) -> None:
        """Extract Common Voice archive if not already extracted."""
        if self.extracted_dir.exists():
            logger.info(f"Archive already extracted at {self.extracted_dir}")
            return

        logger.info(f"Extracting {self.archive_path}...")
        logger.info("Reading archive contents...")

        with tarfile.open(self.archive_path, "r:gz") as tar:
            members = tar.getmembers()
            logger.info(f"Found {len(members):,} files in archive")
            logger.info("Extracting files (this will take several minutes)...")

            for member in tqdm(members, desc="Extracting", unit="files"):
                tar.extract(member, path=self.extracted_dir.parent)

        logger.info("✓ Extraction complete!")

    def is_valid_sample(self, accent: str, gender: str, age: str) -> Tuple[bool, str | None]:
        """Check if sample meets selection criteria.

        Args:
            accent: Accent string from CV metadata
            gender: Gender string from CV metadata
            age: Age string from CV metadata

        Returns:
            Tuple of (is_valid, accent_category)
            accent_category is one of: "Colombia", "Chile", "Venezuela", "Mexico", "Spain", None
        """
        # Exclude Unknown gender or age
        if not gender or not age or gender.lower() == "unknown" or age.lower() == "unknown":
            return False, None

        if not accent:
            return False, None

        accent_lower = accent.lower()

        # Check each accent category
        for kw in settings.COLOMBIA_KEYWORDS:
            if kw in accent_lower:
                return True, "Colombia"

        for kw in settings.CHILE_KEYWORDS:
            if kw in accent_lower:
                return True, "Chile"

        for kw in settings.VENEZUELA_KEYWORDS:
            if kw in accent_lower:
                return True, "Venezuela"

        for kw in settings.MEXICO_KEYWORDS:
            if kw in accent_lower:
                return True, "Mexico"

        for kw in settings.SPAIN_KEYWORDS:
            if kw in accent_lower:
                return True, "Spain"

        return False, None

    def parse_validated_tsv(self) -> Dict[str, List[Dict]]:
        """Parse validated.tsv and filter by accent criteria.

        Returns:
            Dictionary mapping client_id to list of sample dicts
        """
        metadata_tsv = self.extracted_dir / "es" / "validated.tsv"
        logger.info(f"Parsing {metadata_tsv}...")

        speakers = defaultdict(list)
        total_samples = 0
        filtered_samples = 0
        accent_counts = Counter()

        with open(metadata_tsv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")

            for row in reader:
                total_samples += 1
                accent = row.get("accents", "").strip()
                gender = row.get("gender", "").strip()
                age = row.get("age", "").strip()

                is_valid, accent_category = self.is_valid_sample(accent, gender, age)

                if not is_valid:
                    continue

                filtered_samples += 1
                accent_counts[accent_category] += 1
                client_id = row["client_id"]
                speakers[client_id].append({
                    "path": row["path"],
                    "accent": accent,
                    "accent_category": accent_category,
                    "gender": gender,
                    "age": age,
                })

        logger.info(f"Total samples in validated.tsv: {total_samples:,}")
        logger.info(f"Filtered samples (CO/CL/VE/MX/ES, no Unknown): {filtered_samples:,}")
        logger.info(f"Unique speakers: {len(speakers):,}")
        logger.info("")
        logger.info("Samples by accent:")
        for accent, count in sorted(accent_counts.items(), key=lambda x: -x[1]):
            logger.info(f"  {accent:12s}: {count:,}")

        return dict(speakers)

    def load_audio(self, audio_path: Path) -> torch.Tensor:
        """Load and resample audio to target sample rate.

        Args:
            audio_path: Path to audio file

        Returns:
            Waveform tensor (1, samples)
        """
        waveform, sr = torchaudio.load(audio_path)

        if sr != settings.SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(sr, settings.SAMPLE_RATE)
            waveform = resampler(waveform)

        return waveform

    def extract_cv_embeddings(
        self,
        model: EncoderClassifier,
        speakers: Dict[str, List[Dict]]
    ) -> Tuple[np.ndarray, Dict]:
        """Extract embeddings for all CV speakers.

        Args:
            model: ECAPA-TDNN encoder model
            speakers: Dictionary mapping client_id to samples

        Returns:
            Tuple of (embeddings_array, metadata_dict)
        """
        clips_dir = self.extracted_dir / "es" / "clips"

        if not clips_dir.exists():
            # Try alternative path
            clips_dir = self.extracted_dir / "clips"
            if not clips_dir.exists():
                raise FileNotFoundError(f"Could not find clips directory at {clips_dir}")

        logger.info(f"Audio clips directory: {clips_dir}")

        embeddings = []
        metadata = {}
        failed_speakers = []

        total_speakers = len(speakers)
        for idx, (client_id, samples) in enumerate(speakers.items(), 1):
            # Select representative sample (first)
            selected_sample = samples[0]
            audio_filename = selected_sample["path"]
            audio_path = clips_dir / audio_filename

            accent_cat = selected_sample.get("accent_category", "unknown")
            if idx % 100 == 0 or idx <= 10:
                logger.info(f"[{idx}/{total_speakers}] Processing {client_id}: {accent_cat}, {selected_sample.get('gender', 'unknown')}")

            if not audio_path.exists():
                failed_speakers.append(client_id)
                continue

            try:
                waveform = self.load_audio(audio_path)

                with torch.no_grad():
                    embedding = model.encode_batch(waveform.to(self.device))

                embedding = embedding.squeeze().cpu().numpy()
                embedding = embedding / np.linalg.norm(embedding)

                embeddings.append(embedding)
                metadata[client_id] = {
                    "sample_count": len(samples),
                    "accent": selected_sample["accent"],
                    "gender": selected_sample["gender"],
                    "age": selected_sample["age"],
                    "embedding_from": audio_filename,
                    "accent_category": accent_cat,
                }

            except Exception as e:
                logger.warning(f"Failed to process {client_id}: {e}")
                failed_speakers.append(client_id)
                continue

        embeddings_array = np.array(embeddings, dtype=np.float32)

        logger.info(f"Successfully processed: {len(metadata):,} speakers")
        logger.info(f"Failed: {len(failed_speakers):,} speakers")

        return embeddings_array, metadata

    def execute(self) -> EmbeddingResult:
        """Execute CV embedding extraction.

        Returns:
            EmbeddingResult with paths and metadata
        """
        logger.info("Step 02 - CV Embedding Extraction: Starting")
        logger.info(f"Device: {self.device}")

        # Extract archive
        self.extract_archive()

        # Parse validated.tsv
        speakers = self.parse_validated_tsv()

        # Load model
        logger.info("Loading ECAPA-TDNN model...")
        model = EncoderClassifier.from_hparams(
            source=settings.MODEL_SOURCE,
            savedir=str(settings.MODEL_SAVE_DIR),
            run_opts={"device": self.device}
        )
        logger.info(f"Model loaded successfully on {self.device}")

        # Extract embeddings
        embeddings_array, metadata = self.extract_cv_embeddings(model, speakers)

        # Save outputs
        embeddings_path = self.output_dir / "cv_speaker_embeddings.npy"
        metadata_path = self.output_dir / "cv_speaker_metadata.json"

        np.save(embeddings_path, embeddings_array)

        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        logger.info(f"Embeddings shape: {embeddings_array.shape}")
        logger.info(f"Saved to: {embeddings_path}")
        logger.info("Step 02 - CV Embedding Extraction: Complete")

        return EmbeddingResult(
            embeddings_path=embeddings_path,
            ids_path=metadata_path,  # Using metadata_path as ids_path (contains client_ids as keys)
            metadata_path=metadata_path,
            embedding_count=len(metadata),
            embedding_dim=embeddings_array.shape[1]
        )
