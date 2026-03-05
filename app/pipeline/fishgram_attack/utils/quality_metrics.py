"""
Quality metrics for FishGram synthetic speech validation.

Computes DNSMOS and speaker similarity using SpeechBrain models.
"""
import torch
import librosa
import numpy as np
from pathlib import Path
from loguru import logger

# Lazy imports - only load when needed
_dnsmos_estimator = None
_speaker_verifier = None


def compute_dnsmos(audio_path: Path) -> float:
    """Compute DNSMOS overall perceptual quality score.

    Uses Microsoft DNS Challenge P.835 DNSMOS predictor via SpeechBrain.

    Args:
        audio_path: Path to audio file

    Returns:
        DNSMOS overall score (1.0-5.0, where 3.5=good, 3.8=excellent)

    Raises:
        Exception: If DNSMOS computation fails
    """
    global _dnsmos_estimator

    try:
        # Lazy load DNSMOS model
        if _dnsmos_estimator is None:
            from speechbrain.inference.separation import DNSMOSEstimator
            logger.info("Loading DNSMOS model...")
            _dnsmos_estimator = DNSMOSEstimator.from_hparams(
                source="speechbrain/dnsmos-p835",
                savedir="pretrained_models/dnsmos-p835"
            )

        # Compute DNSMOS
        scores = _dnsmos_estimator.estimate_mos(str(audio_path))
        return float(scores["ovrl"])  # Overall score

    except Exception as e:
        logger.error(f"DNSMOS computation failed for {audio_path}: {e}")
        raise


def compute_speaker_similarity(synth_path: Path, ref_path: Path) -> float:
    """Compute cosine similarity between synthetic and reference speaker embeddings.

    Uses SpeechBrain ECAPA-TDNN speaker verification model (same as Mozilla pipeline).

    Args:
        synth_path: Path to synthetic audio file
        ref_path: Path to reference audio file

    Returns:
        Cosine similarity score (0.0-1.0, where 0.65=successful, 0.75=high confidence)

    Raises:
        Exception: If embedding extraction or similarity computation fails
    """
    global _speaker_verifier

    try:
        # Lazy load speaker verification model
        if _speaker_verifier is None:
            from speechbrain.inference.speaker import EncoderClassifier
            logger.info("Loading ECAPA-TDNN speaker verification model...")
            _speaker_verifier = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir="pretrained_models/spkrec-ecapa-voxceleb"
            )

        # Load audio
        synth_audio, sr = librosa.load(synth_path, sr=16000)
        ref_audio, sr = librosa.load(ref_path, sr=16000)

        # Convert to torch tensors
        synth_tensor = torch.tensor(synth_audio).unsqueeze(0)
        ref_tensor = torch.tensor(ref_audio).unsqueeze(0)

        # Extract embeddings
        with torch.no_grad():
            synth_emb = _speaker_verifier.encode_batch(synth_tensor)
            ref_emb = _speaker_verifier.encode_batch(ref_tensor)

        # Compute cosine similarity
        similarity = torch.nn.functional.cosine_similarity(synth_emb, ref_emb, dim=-1)
        return float(similarity.item())

    except Exception as e:
        logger.error(f"Speaker similarity computation failed: {e}")
        raise


def detect_silence(audio: np.ndarray, threshold: float = 0.01, min_duration: float = 1.0, sample_rate: int = 16000) -> bool:
    """Detect if audio contains excessive silence.

    Args:
        audio: Audio signal
        threshold: RMS threshold below which is considered silence
        min_duration: Minimum consecutive silence duration to flag (seconds)
        sample_rate: Audio sample rate

    Returns:
        True if excessive silence detected, False otherwise
    """
    # Compute frame-wise RMS
    frame_length = int(0.025 * sample_rate)  # 25ms frames
    hop_length = int(0.010 * sample_rate)    # 10ms hop

    rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]

    # Find silent frames
    silent_frames = rms < threshold

    # Count consecutive silent frames
    max_consecutive_silence = 0
    current_silence = 0

    for is_silent in silent_frames:
        if is_silent:
            current_silence += 1
        else:
            max_consecutive_silence = max(max_consecutive_silence, current_silence)
            current_silence = 0

    max_consecutive_silence = max(max_consecutive_silence, current_silence)

    # Convert frames to seconds
    silence_duration = (max_consecutive_silence * hop_length) / sample_rate

    return silence_duration >= min_duration
