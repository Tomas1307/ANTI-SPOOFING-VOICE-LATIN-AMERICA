"""
Live partial-spoof splice demo for the thesis defense.

Builds two clips from a SINGLE ~10 s recording of Master Tomas's voice,
which serves as both the OmniVoice reference clip and the bonafide host
utterance:

    crudo.wav     - naive cut-and-paste splice (raw ASR boundaries, no
                    valley snap, no crossfade, no loudness matching)
    mitigado.wav  - the same word swap through the real production splice
                    engine (app.pipeline.partial_spoof.utils.splice_engine),
                    with the exact settings used to build the MARSA corpus

Must be run on ml-server03 (GPU) with the omnivoice_attack venv activated
-- it loads both OmniVoice and Parakeet.

Usage:
    python -m app.scripts.partial_spoof_live_demo \\
        tesis/references/Tomas.m4a \\
        --target trescientos mil \\
        --replacement un millon
"""
import argparse
from pathlib import Path

from loguru import logger

from app.scripts.voice_demo.schemas.live_demo_request import LiveDemoRequest
from app.scripts.voice_demo.settings import settings
from app.scripts.voice_demo.utils.live_demo_builder import LiveDemoBuilder


def main() -> None:
    """Parse CLI arguments and build the crude/mitigated demo clips."""
    parser = argparse.ArgumentParser(
        description="Build the crude and mitigated partial-spoof splice demo clips."
    )
    parser.add_argument("host_audio", type=Path, help="Path to the single host/reference recording")
    parser.add_argument(
        "--target", nargs="+", required=True,
        help="Words to remove from the host utterance, in order (e.g. --target trescientos mil)",
    )
    parser.add_argument(
        "--replacement", nargs="+", required=True,
        help="Words that replace --target in the clone (e.g. --replacement un millon)",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output directory (default: data/voice_demo_output/partial_spoof_live_demo)",
    )
    parser.add_argument(
        "--tries", type=int, default=1,
        help="Number of independent OmniVoice renders to generate, keeping the one "
        "with the most silence around the replacement phrase (default: 1)",
    )
    args = parser.parse_args()

    output_dir = args.output or (settings.OUTPUT_DIR / "partial_spoof_live_demo")
    request = LiveDemoRequest(
        host_audio_path=args.host_audio,
        target_phrase=args.target,
        replacement_phrase=args.replacement,
        output_dir=output_dir,
        tries=args.tries,
    )

    builder = LiveDemoBuilder(request)
    paths = builder.build()

    logger.info("Done. Outputs:")
    for label, path in paths.items():
        logger.info(f"  {label}: {path}")


if __name__ == "__main__":
    main()
