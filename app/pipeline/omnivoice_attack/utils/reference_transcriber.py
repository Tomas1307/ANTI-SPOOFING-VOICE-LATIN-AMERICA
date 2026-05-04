"""
Reference audio transcription using NVIDIA Parakeet TDT.

Transcribes concatenated reference audio clips to provide ref_text for
OmniVoice voice cloning. OmniVoice can auto-transcribe via internal
Whisper if ref_text is omitted, but pre-computing with Parakeet keeps
the project STT stack consistent (Parakeet is already used in Step 4
quality validation) and avoids loading two ASR models per pipeline run.

Uses the project-wide ParakeetTranscriber singleton so the same model
instance is reused across reference transcription and Step 4 validation.
"""
from pathlib import Path

from loguru import logger

from app.pipeline.omnivoice_attack.settings import settings
from app.utils.parakeet_transcriber import ParakeetTranscriber


def transcribe_audio(audio_path: Path, language: str = "es") -> str:
    """Transcribe an audio file to text using the Parakeet TDT singleton.

    The Parakeet model is loaded on first call and reused thereafter.
    The language argument is accepted for API compatibility but is not
    forwarded: Parakeet TDT 0.6b-v3 auto-detects language and supports
    Spanish natively.

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
