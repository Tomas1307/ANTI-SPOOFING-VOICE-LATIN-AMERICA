"""
Step 1: Extract HABLA Speaker Embeddings using ECAPA-TDNN.

This step processes all HABLA bonafide speakers and extracts averaged
speaker embeddings for similarity filtering.
"""
import json
from loguru import logger
from pathlib import Path
from typing import List

import numpy as np
import torch
import torchaudio
from speechbrain.inference.speaker import EncoderClassifier
from tqdm import tqdm

from app.pipeline.select_mozilla_speakers.settings import settings
from app.pipeline.select_mozilla_speakers.schemas.embedding_result import EmbeddingResult


class HablaEmbeddingExtractor:
    """Extracts ECAPA-TDNN embeddings from HABLA bonafide speakers.

    This step loads the ECAPA-TDNN model and processes all 162 HABLA speakers,
    averaging up to 20 training samples per speaker to produce one representative
    192-dimensional embedding per speaker.

    Attributes:
        habla_dir: Directory containing HABLA speakers
        output_dir: Directory for output files
        device: Compute device (cuda/cpu)
    """

    def __init__(
        self,
        habla_dir: Path | None = None,
        output_dir: Path | None = None,
        device: str | None = None
    ) -> None:
        """Initialize the HABLA embedding extractor.

        Args:
            habla_dir: Optional override for HABLA directory
            output_dir: Optional override for output directory
            device: Optional override for compute device
        """
        self.habla_dir = habla_dir or settings.HABLA_DIR
        self.output_dir = output_dir or settings.OUTPUT_DIR
        self.device = device or settings.DEVICE

        self.output_dir.mkdir(parents=True, exist_ok=True)

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

    def extract_speaker_embedding(
        self,
        model: EncoderClassifier,
        audio_files: List[Path]
    ) -> np.ndarray:
        """Extract and average embeddings from multiple audio files.

        Args:
            model: ECAPA-TDNN encoder model
            audio_files: List of audio file paths for one speaker

        Returns:
            Averaged embedding vector, shape (192,)
        """
        embeddings = []

        # Limit to max_samples for efficiency
        selected_files = audio_files[:settings.MAX_SAMPLES_PER_SPEAKER]

        for audio_path in selected_files:
            try:
                waveform = self.load_audio(audio_path)

                with torch.no_grad():
                    embedding = model.encode_batch(waveform.to(self.device))

                embedding = embedding.squeeze().cpu().numpy()
                embeddings.append(embedding)

            except Exception as e:
                logger.warning(f"Failed to process {audio_path.name}: {e}")
                continue

        if not embeddings:
            raise ValueError(f"No valid embeddings extracted from {len(audio_files)} files")

        # Average and L2-normalize
        avg_embedding = np.mean(embeddings, axis=0)
        avg_embedding = avg_embedding / np.linalg.norm(avg_embedding)

        return avg_embedding

    def execute(self) -> EmbeddingResult:
        """Execute HABLA embedding extraction.

        Returns:
            EmbeddingResult with paths and metadata
        """
        logger.info("Step 01 - HABLA Embedding Extraction: Starting")
        logger.info(f"Device: {self.device}")
        logger.info(f"HABLA directory: {self.habla_dir}")

        # Load model
        logger.info(f"Loading ECAPA-TDNN model ({settings.MODEL_SOURCE})...")
        model = EncoderClassifier.from_hparams(
            source=settings.MODEL_SOURCE,
            savedir=str(settings.MODEL_SAVE_DIR),
            run_opts={"device": self.device}
        )
        logger.info("Model loaded successfully")

        # Get speaker directories
        speaker_dirs = sorted([d for d in self.habla_dir.iterdir() if d.is_dir()])
        logger.info(f"Found {len(speaker_dirs)} HABLA speakers")

        speaker_ids = []
        embeddings = []

        # Process each speaker
        for idx, speaker_dir in enumerate(speaker_dirs, 1):
            speaker_id = speaker_dir.name
            speaker_ids.append(speaker_id)

            # Get training audio files
            train_dir = speaker_dir / "train"
            if not train_dir.exists():
                logger.warning(f"No train directory for {speaker_id}, skipping")
                speaker_ids.pop()
                continue

            audio_files = sorted(train_dir.glob("*.wav"))
            if not audio_files:
                logger.warning(f"No audio files for {speaker_id}, skipping")
                speaker_ids.pop()
                continue

            # Extract embedding
            try:
                if idx % 20 == 0 or idx <= 5:
                    logger.info(f"[{idx}/{len(speaker_dirs)}] Processing {speaker_id}: {len(audio_files)} files")

                avg_embedding = self.extract_speaker_embedding(model, audio_files)
                embeddings.append(avg_embedding)

            except Exception as e:
                logger.error(f"Error processing {speaker_id}: {e}")
                speaker_ids.pop()
                continue

        # Convert to numpy array
        embeddings_array = np.array(embeddings, dtype=np.float32)

        # Save outputs
        embeddings_path = self.output_dir / "habla_embeddings.npy"
        ids_path = self.output_dir / "habla_speaker_ids.json"

        np.save(embeddings_path, embeddings_array)

        with open(ids_path, "w", encoding="utf-8") as f:
            json.dump(speaker_ids, f, indent=2)

        logger.info(f"Successfully processed: {len(speaker_ids)} speakers")
        logger.info(f"Embeddings shape: {embeddings_array.shape}")
        logger.info(f"Saved to: {embeddings_path}")
        logger.info("Step 01 - HABLA Embedding Extraction: Complete")

        return EmbeddingResult(
            embeddings_path=embeddings_path,
            ids_path=ids_path,
            embedding_count=len(speaker_ids),
            embedding_dim=embeddings_array.shape[1]
        )
