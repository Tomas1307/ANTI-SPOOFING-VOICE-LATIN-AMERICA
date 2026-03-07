"""
Reference audio transcription using faster-whisper.

Transcribes concatenated reference audio clips to provide text context
for Qwen3-TTS full voice cloning mode (x_vector_only_mode=False).
Uses faster-whisper (CTranslate2 backend) for GPU-accelerated Spanish STT.
"""
from pathlib import Path
from loguru import logger
from faster_whisper import WhisperModel
from app.pipeline.qwen_attack.settings import settings

_whisper_model = None


def get_whisper_model() -> WhisperModel:
    """Lazy-load Whisper model as a module-level singleton.

    Loads the model on first call and caches it for subsequent calls.
    Uses settings for model size and compute type configuration.

    Returns:
        Initialized WhisperModel instance ready for transcription.
    """
    global _whisper_model
    if _whisper_model is None:
        logger.info(
            f"Loading Whisper model: {settings.WHISPER_MODEL_SIZE} "
            f"(compute_type={settings.WHISPER_COMPUTE_TYPE})"
        )
        device = "cuda" if "cuda" in settings.DEVICE else "cpu"
        _whisper_model = WhisperModel(
            settings.WHISPER_MODEL_SIZE,
            device=device,
            compute_type=settings.WHISPER_COMPUTE_TYPE,
        )
        logger.info("Whisper model loaded successfully")
    return _whisper_model


def transcribe_audio(audio_path: Path, language: str = "es") -> str:
    """Transcribe an audio file to text using faster-whisper.

    Args:
        audio_path: Path to the audio file to transcribe.
        language: ISO 639-1 language code (default: 'es' for Spanish).

    Returns:
        Transcribed text as a single string with segments joined by spaces.

    Raises:
        FileNotFoundError: If audio_path does not exist.
        RuntimeError: If Whisper transcription fails.
    """
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    model = get_whisper_model()
    segments, info = model.transcribe(
        str(audio_path),
        language=language,
        beam_size=5,
    )

    transcript = " ".join(segment.text.strip() for segment in segments)

    logger.debug(
        f"Transcribed {audio_path.name}: "
        f"{len(transcript)} chars, language={info.language}, "
        f"probability={info.language_probability:.2f}"
    )

    return transcript
