"""
Singleton wrapper for NISQA non-intrusive speech quality MOS estimation.

NISQA (Non-Intrusive Speech Quality Assessment) v2.0 predicts a Mean Opinion
Score (MOS) in the range [1.0, 5.0] from audio alone, without requiring a
clean reference signal. This makes it ideal for evaluating TTS output quality
in attack pipelines where no paired reference exists.

Uses the torchmetrics implementation which supports 16kHz input natively
and returns five quality dimensions: overall MOS, noisiness, discontinuity,
coloration, and loudness.

Reference:
    Mittag et al., "NISQA: A Deep CNN-Self-Attention Model for
    Multidimensional Speech Quality Prediction with Crowdsourced Datasets",
    Interspeech 2021.

Installation (per pipeline venv on ml-server03):
    pip install torchmetrics librosa requests
"""
import librosa
import numpy as np
import torch
from pathlib import Path

from loguru import logger


class NisqaScorer:
    """Singleton wrapper for NISQA non-intrusive speech quality MOS estimation.

    The model is loaded once on first use and reused across all subsequent
    calls. NISQA predicts five quality dimensions from a single audio signal
    without requiring a reference.

    Usage:
        scorer = NisqaScorer()
        scorer.load(device="cuda")
        mos = scorer.predict_mos(Path("generated.wav"))

    Attributes:
        _instance: Class-level singleton reference.
        _metric: Loaded torchmetrics NISQA metric (None until load() is called).
        _device: Computation device string.
    """

    _instance = None
    _metric = None
    _device = "cpu"

    def __new__(cls) -> "NisqaScorer":
        """Return the singleton instance, creating it if necessary."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def load(self, device: str = "cuda") -> None:
        """Load the NISQA metric model if not already loaded.

        Args:
            device: Compute device. Use 'cuda' on ml-server03 A40 GPUs.

        Raises:
            ImportError: If torchmetrics is not installed.
        """
        if self._metric is not None:
            logger.debug("NISQA model already loaded, skipping.")
            return

        from torchmetrics.audio import NonIntrusiveSpeechQualityAssessment

        logger.info(f"Loading NISQA model on {device}")
        self._device = device
        self._metric = NonIntrusiveSpeechQualityAssessment(fs=16000)
        self._metric = self._metric.to(device)
        logger.info("NISQA model loaded successfully.")

    def predict_mos(self, audio_path: Path) -> float:
        """Predict overall MOS quality score for a single audio file.

        The audio is loaded at 16kHz mono and passed through the NISQA model.
        Returns only the overall MOS dimension (first of five outputs).

        Args:
            audio_path: Path to a WAV/FLAC audio file.

        Returns:
            Overall MOS score as a float in [1.0, 5.0]. Higher is better.

        Raises:
            RuntimeError: If load() has not been called before predicting.
            FileNotFoundError: If audio_path does not exist.
        """
        if self._metric is None:
            raise RuntimeError("NISQA model not loaded. Call load() before predicting.")
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        audio, _ = librosa.load(str(audio_path), sr=16000)
        waveform = torch.from_numpy(audio).float().to(self._device)

        with torch.no_grad():
            scores = self._metric(waveform)

        overall_mos = float(scores[0].cpu())
        return overall_mos

    def predict_dimensions(self, audio_path: Path) -> dict:
        """Predict all five NISQA quality dimensions for a single audio file.

        Args:
            audio_path: Path to a WAV/FLAC audio file.

        Returns:
            Dictionary with keys: mos, noisiness, discontinuity, coloration,
            loudness. All float values in [1.0, 5.0].

        Raises:
            RuntimeError: If load() has not been called before predicting.
            FileNotFoundError: If audio_path does not exist.
        """
        if self._metric is None:
            raise RuntimeError("NISQA model not loaded. Call load() before predicting.")
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        audio, _ = librosa.load(str(audio_path), sr=16000)
        waveform = torch.from_numpy(audio).float().to(self._device)

        with torch.no_grad():
            scores = self._metric(waveform)

        scores_cpu = scores.cpu().numpy()
        return {
            "mos": float(scores_cpu[0]),
            "noisiness": float(scores_cpu[1]),
            "discontinuity": float(scores_cpu[2]),
            "coloration": float(scores_cpu[3]),
            "loudness": float(scores_cpu[4]),
        }
