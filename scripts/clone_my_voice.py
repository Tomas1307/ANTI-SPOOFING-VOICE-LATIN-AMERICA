"""
Standalone OmniVoice cloning test.

Loads OmniVoice once, takes a reference audio (MP3 or WAV) plus its
transcript, and synthesises a target sentence in the cloned voice. The
generated audio is written to a single output WAV (default: prueba.wav)
so the user can A/B compare bonafide vs. cloned speech outside the full
production pipeline.

This script is intended for ad-hoc voice-cloning sanity checks. It is NOT
part of the production OmniVoice attack pipeline and does not write
metadata, validation reports, or protocol files.

Execution environment:
    Must run on ml-server03 inside envs/omnivoice_env (the only environment
    with the omnivoice package installed).

Usage example:
    source ~/ANTI-SPOOFING-VOICE-LATIN-AMERICA/envs/omnivoice_env/bin/activate
    export CUDA_VISIBLE_DEVICES=1
    python scripts/clone_my_voice.py \
        --ref-audio my_voice.mp3 \
        --ref-text "Hola, esta es una prueba de mi voz." \
        --text "Buenas tardes, soy una voz sintetica generada con OmniVoice." \
        --output prueba.wav
"""
import argparse
import time
from pathlib import Path

import librosa
import soundfile as sf
import torch
from loguru import logger
from omnivoice import OmniVoice

from app.pipeline.omnivoice_attack.settings import settings


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments.

    Returns:
        Parsed argparse namespace with reference audio path, reference
        transcript, target text, output path, and generation hyper-parameters.
    """
    parser = argparse.ArgumentParser(
        description="Clone an arbitrary voice with OmniVoice (ad-hoc test).",
    )
    parser.add_argument(
        "--ref-audio",
        type=Path,
        required=True,
        help="Path to reference audio file (MP3 or WAV). 3-10s recommended.",
    )
    parser.add_argument(
        "--ref-text",
        type=str,
        required=True,
        help="Exact transcript of the reference audio (required by OmniVoice).",
    )
    parser.add_argument(
        "--text",
        type=str,
        required=True,
        help="Text to synthesise in the cloned voice.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("prueba.wav"),
        help="Output WAV path (default: prueba.wav).",
    )
    parser.add_argument(
        "--num-step",
        type=int,
        default=settings.OMNIVOICE_NUM_STEP,
        help=f"Diffusion sampling steps (default: {settings.OMNIVOICE_NUM_STEP}).",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=settings.OMNIVOICE_SPEED,
        help=f"Speech speed factor (default: {settings.OMNIVOICE_SPEED}).",
    )
    parser.add_argument(
        "--language",
        type=str,
        default=settings.OMNIVOICE_LANGUAGE,
        help=f"Language code for generation (default: {settings.OMNIVOICE_LANGUAGE}).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=settings.DEVICE,
        help=f"Compute device (default: {settings.DEVICE}).",
    )
    return parser.parse_args()


def prepare_reference_audio(ref_audio_path: Path, target_sr: int) -> Path:
    """Normalise reference audio to 16 kHz mono WAV for OmniVoice.

    OmniVoice accepts WAV reference input. Non-WAV inputs (typically MP3
    from phone recordings) are decoded with librosa, downmixed to mono,
    and resampled. The converted file is written next to the original
    with a .converted.wav suffix and its path is returned.

    Args:
        ref_audio_path: Path to the original reference audio.
        target_sr: Target sample rate in Hz for the converted reference.

    Returns:
        Path to a WAV file ready to feed into OmniVoice.

    Raises:
        FileNotFoundError: If the reference audio does not exist.
    """
    if not ref_audio_path.exists():
        raise FileNotFoundError(f"Reference audio not found: {ref_audio_path}")

    if ref_audio_path.suffix.lower() == ".wav":
        logger.info(f"Reference is already WAV: {ref_audio_path}")
        return ref_audio_path

    logger.info(f"Converting {ref_audio_path.suffix} to {target_sr} Hz mono WAV...")
    audio, _ = librosa.load(str(ref_audio_path), sr=target_sr, mono=True)
    converted_path = ref_audio_path.with_suffix(".converted.wav")
    sf.write(str(converted_path), audio, target_sr)
    duration = len(audio) / target_sr
    logger.info(f"Wrote converted reference: {converted_path} ({duration:.2f}s)")
    return converted_path


def load_model(device: str, dtype_name: str, model_id: str) -> OmniVoice:
    """Load OmniVoice from HuggingFace into GPU memory.

    Args:
        device: Compute device string (e.g. "cuda:0", "cpu").
        dtype_name: Torch dtype name (e.g. "float16").
        model_id: HuggingFace model identifier.

    Returns:
        Initialised OmniVoice model ready for inference.
    """
    logger.info(f"Loading OmniVoice ({model_id}) on {device} ({dtype_name})...")
    start = time.time()
    dtype = getattr(torch, dtype_name)
    model = OmniVoice.from_pretrained(
        model_id,
        device_map=device,
        dtype=dtype,
    )
    logger.info(f"Model loaded in {time.time() - start:.1f}s")
    return model


def generate_clone(
    model: OmniVoice,
    text: str,
    ref_audio: Path,
    ref_text: str,
    output_path: Path,
    num_step: int,
    speed: float,
    language: str,
    output_sr: int,
) -> tuple[float, float]:
    """Run a single OmniVoice generation and write the result to disk.

    Args:
        model: Loaded OmniVoice model instance.
        text: Target text to synthesise.
        ref_audio: Reference WAV path.
        ref_text: Reference audio transcript.
        output_path: Output WAV path.
        num_step: Diffusion sampling steps.
        speed: Speech speed factor.
        language: Language code passed to OmniVoice.
        output_sr: Sample rate to write the output WAV at.

    Returns:
        Tuple of (generation_seconds, audio_duration_seconds).
    """
    logger.info(f"Generating: \"{text}\"")
    start = time.time()
    audios = model.generate(
        text=text,
        ref_audio=str(ref_audio),
        ref_text=ref_text,
        num_step=num_step,
        speed=speed,
        language=language,
    )
    generation_time = time.time() - start

    audio = audios[0]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), audio, output_sr)
    audio_duration = len(audio) / output_sr
    return generation_time, audio_duration


def main() -> None:
    """Entry point: parse args, run a single clone, log RTF and output path."""
    args = parse_args()

    ref_path = prepare_reference_audio(
        ref_audio_path=args.ref_audio,
        target_sr=settings.SAMPLE_RATE,
    )

    model = load_model(
        device=args.device,
        dtype_name=settings.DTYPE,
        model_id=settings.OMNIVOICE_MODEL_ID,
    )

    generation_time, audio_duration = generate_clone(
        model=model,
        text=args.text,
        ref_audio=ref_path,
        ref_text=args.ref_text,
        output_path=args.output,
        num_step=args.num_step,
        speed=args.speed,
        language=args.language,
        output_sr=settings.OMNIVOICE_NATIVE_SAMPLE_RATE,
    )

    rtf = generation_time / audio_duration if audio_duration > 0 else 0.0
    logger.info(
        f"Done. Cloned audio: {args.output} "
        f"({audio_duration:.2f}s in {generation_time:.2f}s, RTF={rtf:.3f})"
    )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
