"""
Standalone CLI to clone a reference speaker's voice onto arbitrary text.

This is a thin wrapper around the existing OmniVoice ``Cloner``
(app/pipeline/omnivoice_attack/utils/cloner.py), the same cloning unit
used by the full attack pipeline. It takes one reference audio file from
``tesis/references`` (matched by speaker name, case-insensitively) and
one text string, and writes the synthesised clone to a WAV file.

Must be run on ml-server03 (GPU) with the omnivoice_attack venv
activated, NOT on the local Windows machine -- see CLAUDE.md's
two-machine workflow.

Usage:
    python -m app.scripts.clone_voice ruben "Hola como estan, esta es la presentacion de la tesis de tomas"
    python -m app.scripts.clone_voice mery "Otro texto cualquiera" --ref-text "Lo que mery dice en el audio de referencia"
"""
import argparse
import sys
import tempfile
from pathlib import Path

from loguru import logger

from app.pipeline.omnivoice_attack.settings import settings as omnivoice_settings
from app.pipeline.omnivoice_attack.utils.cloner import Cloner
from app.pipeline.omnivoice_attack.utils.reference_transcriber import transcribe_audio
from app.scripts.voice_demo.schemas.clone_request import CloneRequest
from app.scripts.voice_demo.settings import settings
from app.scripts.voice_demo.utils.audio_conversion import to_mono_wav


def resolve_reference_audio(speaker_name: str, references_dir: Path) -> Path:
    """Find the reference audio file matching a speaker name.

    Matches case-insensitively against the file stem, so ``ruben`` matches
    ``ruben.mp4``, ``RUBEN``, etc.

    Args:
        speaker_name: Speaker identifier as passed on the command line.
        references_dir: Directory to search for the reference file.

    Returns:
        Path to the matching reference audio file.

    Raises:
        FileNotFoundError: If no file in ``references_dir`` has a matching
            stem, or if the directory itself does not exist.
    """
    if not references_dir.exists():
        raise FileNotFoundError(f"References directory not found: {references_dir}")

    target_stem = speaker_name.strip().lower()
    for candidate in references_dir.iterdir():
        if candidate.is_file() and candidate.stem.lower() == target_stem:
            return candidate

    available = sorted(p.stem for p in references_dir.iterdir() if p.is_file())
    raise FileNotFoundError(
        f"No reference audio found for speaker '{speaker_name}' in "
        f"{references_dir}. Available speakers: {available}"
    )


def build_clone_request(
    speaker_name: str,
    text: str,
    reference_text: str,
    ref_start_seconds: float,
    ref_duration_seconds: float,
    output_path: Path | None,
    references_dir: Path,
    output_dir: Path,
) -> CloneRequest:
    """Resolve CLI arguments into a validated CloneRequest.

    Args:
        speaker_name: Speaker identifier as passed on the command line.
        text: Spanish text to synthesise.
        reference_text: Transcript of the reference audio, or "" to
            transcribe the cropped reference with Parakeet.
        ref_start_seconds: Offset into the reference audio to start the crop at.
        ref_duration_seconds: Cropped reference audio duration in seconds.
        output_path: Explicit output path, or None to auto-generate one
            under ``output_dir``.
        references_dir: Directory holding reference audio files.
        output_dir: Default output directory when ``output_path`` is None.

    Returns:
        A validated CloneRequest.
    """
    reference_audio_path = resolve_reference_audio(speaker_name, references_dir)

    if output_path is None:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{speaker_name.lower()}_clone.wav"

    return CloneRequest(
        speaker_name=speaker_name,
        reference_audio_path=reference_audio_path,
        reference_text=reference_text,
        ref_start_seconds=ref_start_seconds,
        ref_duration_seconds=ref_duration_seconds,
        text=text,
        output_path=output_path,
    )


def clone_voice(request: CloneRequest) -> Path:
    """Run the OmniVoice cloner end-to-end for one request.

    Args:
        request: Validated clone request.

    Returns:
        The output WAV path that was written.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        logger.info(
            f"Cropping reference audio to {request.ref_duration_seconds:.1f}s "
            f"starting at {request.ref_start_seconds:.1f}s..."
        )
        wav_reference_path = to_mono_wav(
            source_path=request.reference_audio_path,
            destination_path=Path(tmp_dir) / "reference.wav",
            sample_rate=settings.REFERENCE_WAV_SAMPLE_RATE,
            start_seconds=request.ref_start_seconds,
            duration_seconds=request.ref_duration_seconds,
        )

        reference_text = request.reference_text
        if not reference_text:
            logger.info("No --ref-text given; transcribing the reference with Parakeet...")
            reference_text = transcribe_audio(wav_reference_path)
            logger.info(f"Parakeet transcript: \"{reference_text}\"")

        cloner = Cloner()
        logger.info("Loading OmniVoice model...")
        cloner.load(device=omnivoice_settings.DEVICE)

        logger.info(f"Synthesising: \"{request.text}\"")
        request.output_path.parent.mkdir(parents=True, exist_ok=True)
        generation_time, audio_duration = cloner.clone_single(
            text=request.text,
            reference_audio_path=wav_reference_path,
            output_path=request.output_path,
            reference_text=reference_text,
        )

        cloner.cleanup()

    logger.info(
        f"Done in {generation_time:.2f}s -> {audio_duration:.2f}s of audio. "
        f"Saved to {request.output_path}"
    )
    return request.output_path


def main() -> None:
    """Parse CLI arguments and run the clone."""
    parser = argparse.ArgumentParser(
        description="Clone a reference speaker's voice onto arbitrary Spanish text using OmniVoice."
    )
    parser.add_argument(
        "speaker",
        help="Speaker name matching a file stem under tesis/references (e.g. ruben, mery)",
    )
    parser.add_argument("text", help="Spanish text to synthesise in the speaker's voice")
    parser.add_argument(
        "--ref-text",
        default="",
        help="Transcript of what the speaker says in the reference audio "
        "(omit to auto-transcribe the cropped reference with Parakeet)",
    )
    parser.add_argument(
        "--ref-start",
        type=float,
        default=0.0,
        help="Offset in seconds into the reference audio to start the crop at (default: 0.0)",
    )
    parser.add_argument(
        "--ref-duration",
        type=float,
        default=10.0,
        help="Cropped reference audio duration in seconds (default: 10.0, OmniVoice's recommended max)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output WAV path (default: data/voice_demo_output/<speaker>_clone.wav)",
    )
    args = parser.parse_args()

    try:
        request = build_clone_request(
            speaker_name=args.speaker,
            text=args.text,
            reference_text=args.ref_text,
            ref_start_seconds=args.ref_start,
            ref_duration_seconds=args.ref_duration,
            output_path=args.output,
            references_dir=settings.REFERENCES_DIR,
            output_dir=settings.OUTPUT_DIR,
        )
        clone_voice(request)
    except FileNotFoundError as exc:
        logger.error(str(exc))
        sys.exit(1)


if __name__ == "__main__":
    main()
