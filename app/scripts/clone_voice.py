"""
Standalone CLI to clone a reference speaker's voice onto arbitrary text.

This is a thin wrapper around the existing OpenVoice ``Cloner``
(app/pipeline/openvoice_attack/utils/cloner.py), the same cloning unit
used by the full attack pipeline. It takes one reference audio file from
``tesis/references`` (matched by speaker name, case-insensitively) and
one text string, and writes the synthesised clone to a WAV file.

Must be run on ml-server03 (GPU) with the openvoice_attack venv activated,
NOT on the local Windows machine -- see CLAUDE.md's two-machine workflow.

Usage:
    python -m app.scripts.clone_voice ruben "Hola como estan, esta es la presentacion de la tesis de tomas"
    python -m app.scripts.clone_voice mery "Otro texto cualquiera" --output data/voice_demo_output/mery_test.wav
"""
import argparse
import sys
import tempfile
from pathlib import Path

from loguru import logger

from app.pipeline.openvoice_attack.utils.cloner import Cloner
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
    output_path: Path | None,
    references_dir: Path,
    output_dir: Path,
) -> CloneRequest:
    """Resolve CLI arguments into a validated CloneRequest.

    Args:
        speaker_name: Speaker identifier as passed on the command line.
        text: Spanish text to synthesise.
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
        text=text,
        output_path=output_path,
    )


def clone_voice(request: CloneRequest) -> Path:
    """Run the OpenVoice cloner end-to-end for one request.

    Args:
        request: Validated clone request.

    Returns:
        The output WAV path that was written.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        wav_reference_path = to_mono_wav(
            source_path=request.reference_audio_path,
            destination_path=Path(tmp_dir) / "reference.wav",
            sample_rate=settings.SAMPLE_RATE,
        )

        cloner = Cloner()
        logger.info(f"Loading OpenVoice on {settings.DEVICE}...")
        cloner.load(device=settings.DEVICE)

        logger.info(f"Extracting tone color embedding for '{request.speaker_name}'...")
        cloner.prepare_speaker(
            speaker_id=request.speaker_name,
            reference_audio_path=wav_reference_path,
        )

        logger.info(f"Synthesising: \"{request.text}\"")
        request.output_path.parent.mkdir(parents=True, exist_ok=True)
        generation_time, audio_duration = cloner.clone_single(
            text=request.text,
            reference_audio_path=wav_reference_path,
            output_path=request.output_path,
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
        description="Clone a reference speaker's voice onto arbitrary Spanish text using OpenVoice."
    )
    parser.add_argument(
        "speaker",
        help="Speaker name matching a file stem under tesis/references (e.g. ruben, mery)",
    )
    parser.add_argument("text", help="Spanish text to synthesise in the speaker's voice")
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
