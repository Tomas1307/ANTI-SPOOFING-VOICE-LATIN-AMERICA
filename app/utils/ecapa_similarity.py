"""
Singleton wrapper for ECAPA-TDNN speaker similarity via SpeechBrain.

ECAPA-TDNN (Emphasized Channel Attention, Propagation and Aggregation in
TDNN) is the standard speaker verification model used in the ASVspoof
community. This wrapper uses SpeechBrain's pre-trained model on VoxCeleb
to extract speaker embeddings and compute cosine similarity between
reference and generated audio.

The model achieves 0.80% EER on VoxCeleb1-test (Cleaned), making it
suitable for evaluating voice cloning fidelity in TTS attack pipelines.

Reference:
    Desplanques et al., "ECAPA-TDNN: Emphasized Channel Attention,
    Propagation and Aggregation in TDNN Based Speaker Verification",
    Interspeech 2020.

Installation (per pipeline venv on ml-server03):
    pip install speechbrain
"""
import numpy as np
import torch
import torchaudio
from pathlib import Path

from loguru import logger


class EcapaSimilarity:
    """Singleton wrapper for ECAPA-TDNN speaker embedding extraction and similarity.

    The model is loaded once on first use and reused across all subsequent
    calls. SpeechBrain downloads the checkpoint on first use to the HuggingFace
    cache (~25MB).

    Usage:
        ecapa = EcapaSimilarity()
        ecapa.load(device="cuda")
        similarity = ecapa.compute_similarity(Path("ref.wav"), Path("gen.wav"))
        embedding = ecapa.extract_embedding(Path("audio.wav"))

    Attributes:
        _instance: Class-level singleton reference.
        _model: Loaded SpeechBrain SpeakerRecognition model (None until load()).
        _device: Computation device string.
    """

    _instance = None
    _model = None
    _device = "cpu"

    def __new__(cls) -> "EcapaSimilarity":
        """Return the singleton instance, creating it if necessary."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def load(self, device: str = "cuda") -> None:
        """Load the ECAPA-TDNN model from HuggingFace if not already loaded.

        Args:
            device: Compute device. Use 'cuda' on ml-server03 A40 GPUs.

        Raises:
            ImportError: If speechbrain is not installed.
        """
        if self._model is not None:
            logger.debug("ECAPA-TDNN model already loaded, skipping.")
            return

        from speechbrain.inference.speaker import SpeakerRecognition

        logger.info(f"Loading ECAPA-TDNN (spkrec-ecapa-voxceleb) on {device}")
        self._device = device
        self._model = SpeakerRecognition.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="pretrained_models/spkrec-ecapa-voxceleb",
            run_opts={"device": device},
        )
        logger.info("ECAPA-TDNN model loaded successfully.")

    def extract_embedding(self, audio_path: Path) -> np.ndarray:
        """Extract a speaker embedding from a single audio file.

        Args:
            audio_path: Path to a WAV/FLAC audio file (16kHz recommended).

        Returns:
            L2-normalized 192-dimensional embedding as a numpy array.

        Raises:
            RuntimeError: If load() has not been called before extracting.
            FileNotFoundError: If audio_path does not exist.
        """
        if self._model is None:
            raise RuntimeError(
                "ECAPA-TDNN model not loaded. Call load() before extracting."
            )
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        signal, sr = torchaudio.load(str(audio_path))
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
            signal = resampler(signal)

        with torch.no_grad():
            embedding = self._model.encode_batch(signal.to(self._device))

        embedding_np = embedding.squeeze().cpu().numpy()
        norm = np.linalg.norm(embedding_np)
        if norm > 0:
            embedding_np = embedding_np / norm
        return embedding_np

    def compute_similarity(self, ref_path: Path, gen_path: Path) -> float:
        """Compute cosine similarity between two audio files.

        Extracts speaker embeddings from both files and computes their
        cosine similarity. Values closer to 1.0 indicate the same speaker.

        Args:
            ref_path: Path to reference (bonafide) audio file.
            gen_path: Path to generated (synthetic) audio file.

        Returns:
            Cosine similarity as a float in [-1.0, 1.0]. Higher means
            more similar speakers. Typical threshold: > 0.7 for same speaker.

        Raises:
            RuntimeError: If load() has not been called before computing.
            FileNotFoundError: If either audio path does not exist.
        """
        ref_emb = self.extract_embedding(ref_path)
        gen_emb = self.extract_embedding(gen_path)
        return float(np.dot(ref_emb, gen_emb))

    def compute_similarity_from_embedding(
        self, ref_embedding: np.ndarray, gen_path: Path
    ) -> float:
        """Compute cosine similarity using a pre-extracted reference embedding.

        This is more efficient when comparing multiple generated samples against
        the same reference speaker, as the reference embedding is extracted once
        and reused.

        Args:
            ref_embedding: Pre-extracted L2-normalized reference speaker embedding.
            gen_path: Path to generated (synthetic) audio file.

        Returns:
            Cosine similarity as a float in [-1.0, 1.0].

        Raises:
            RuntimeError: If load() has not been called before computing.
            FileNotFoundError: If gen_path does not exist.
        """
        gen_emb = self.extract_embedding(gen_path)
        return float(np.dot(ref_embedding, gen_emb))
