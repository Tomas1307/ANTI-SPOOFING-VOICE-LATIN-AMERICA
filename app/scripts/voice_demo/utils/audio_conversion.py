"""
Reference audio normalisation for the voice-cloning demo.

OpenVoice's ``se_extractor.get_se`` and OmniVoice's cloning conditioning
are both most reliable on a plain mono WAV file within their recommended
reference duration; the reference clips used for this demo arrive as
full-length, phone-recorded ``.mp4``/``.m4a`` containers. This module
transcodes any input audio container to a mono WAV via ffmpeg, optionally
cropping it to a target window, before it reaches the cloner.
"""
import subprocess
from pathlib import Path
from typing import Optional


def to_mono_wav(
    source_path: Path,
    destination_path: Path,
    sample_rate: int,
    start_seconds: float = 0.0,
    duration_seconds: Optional[float] = None,
) -> Path:
    """Transcode an arbitrary audio/video container to mono WAV via ffmpeg.

    Args:
        source_path: Input audio or video file (e.g. ``.mp4``, ``.m4a``, ``.wav``).
        destination_path: Output WAV file path. Parent directory is created
            if missing.
        sample_rate: Target sample rate in Hz.
        start_seconds: Offset into the source audio to start the crop at.
        duration_seconds: If given, crop the output to this many seconds
            starting at ``start_seconds``. If None, the full remaining
            audio is transcoded.

    Returns:
        The destination path, for convenient chaining.

    Raises:
        FileNotFoundError: If ``source_path`` does not exist.
        RuntimeError: If ffmpeg is not installed or the transcoding fails.
    """
    if not source_path.exists():
        raise FileNotFoundError(f"Reference audio not found: {source_path}")

    destination_path.parent.mkdir(parents=True, exist_ok=True)

    command = ["ffmpeg", "-y"]
    if start_seconds:
        command += ["-ss", str(start_seconds)]
    command += ["-i", str(source_path)]
    if duration_seconds is not None:
        command += ["-t", str(duration_seconds)]
    command += ["-ac", "1", "-ar", str(sample_rate), str(destination_path)]

    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed to transcode '{source_path}' to WAV: {result.stderr}"
        )

    return destination_path
