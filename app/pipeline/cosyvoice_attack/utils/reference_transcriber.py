"""
Reference audio transcription using NVIDIA Parakeet TDT.

Transcribes concatenated reference audio clips to provide prompt_text for
CosyVoice 3.0 zero-shot voice cloning mode (inference_zero_shot requires
both the reference waveform and its text transcription).

Uses the project-wide ParakeetTranscriber singleton so that the same model
instance is reused by Step 4 quality validation, avoiding the cost and VRAM
footprint of loading two separate ASR models per pipeline run.
"""
from pathlib import Path

from loguru import logger

from app.pipeline.cosyvoice_attack.settings import settings
from app.utils.parakeet_transcriber import ParakeetTranscriber


def transcribe_audio(audio_path: Path, language: str = "es") -> str:
    """Transcribe an audio file to text using the Parakeet TDT singleton.

    The Parakeet model is loaded on first call and reused thereafter.
    The language argument is accepted for API compatibility with the previous
    faster-whisper implementation but is not forwarded: Parakeet TDT 0.6b-v3
    auto-detects language and supports Spanish natively.

    Args:
        audio_path: Path to the audio file to transcribe.
        language: ISO 639-1 language code, accepted for API compatibility only.

    Returns:
        Transcribed text as a plain string.

    Raises:
        FileNotFoundError: If audio_path does not exist.
        RuntimeError: If Parakeet transcription fails.
    """
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    transcriber = ParakeetTranscriber()
    transcriber.load(model_id=settings.PARAKEET_MODEL_ID, device=settings.DEVICE)

    transcript = transcriber.transcribe(audio_path)

    logger.debug(
        f"Transcribed {audio_path.name}: {len(transcript)} chars (Parakeet TDT)"
    )

    return transcript
