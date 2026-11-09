"""
Reference audio normalisation for the voice-cloning demo.

OpenVoice's ``se_extractor.get_se`` is most reliable on a plain mono WAV
file; the reference clips used for this demo arrive as a phone-recorded
``.mp4``/``.m4a`` container. This module transcodes any input audio
container to a mono 16 kHz WAV via ffmpeg before it reaches the cloner.
"""
import subprocess
from pathlib import Path


def to_mono_wav(source_path: Path, destination_path: Path, sample_rate: int) -> Path:
    """Transcode an arbitrary audio/video container to mono WAV via ffmpeg.

    Args:
        source_path: Input audio or video file (e.g. ``.mp4``, ``.m4a``, ``.wav``).
        destination_path: Output WAV file path. Parent directory is created
            if missing.
        sample_rate: Target sample rate in Hz.

    Returns:
        The destination path, for convenient chaining.

    Raises:
        FileNotFoundError: If ``source_path`` does not exist.
        RuntimeError: If ffmpeg is not installed or the transcoding fails.
    """
    if not source_path.exists():
        raise FileNotFoundError(f"Reference audio not found: {source_path}")

    destination_path.parent.mkdir(parents=True, exist_ok=True)

    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(source_path),
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        str(destination_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed to transcode '{source_path}' to WAV: {result.stderr}"
        )

    return destination_path
