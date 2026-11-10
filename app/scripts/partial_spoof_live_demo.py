"""
Live partial-spoof splice demo for the thesis defense.

Builds three clips from a SINGLE ~10 s recording of Master Tomas's voice,
which serves as both the OmniVoice reference clip and the bonafide host
utterance:

    original.wav          - the untouched host recording
    spoof_sin_jitter.wav  - one or more words replaced by cloned speech via
                             the real production splice engine
                             (app.pipeline.partial_spoof.utils.splice_engine),
                             with the exact settings used to build the MARSA
                             corpus, BEFORE boundary jitter
    spoof_con_jitter.wav  - the same spliced clip after the real Step 5b
                             boundary-jitter post-processing homogenizes
                             every internal word boundary

Must be run on ml-server03 (GPU) with the omnivoice_attack venv activated
-- it loads both OmniVoice and Parakeet.

Usage:
    python -m app.scripts.partial_spoof_live_demo \\
        tesis/references/tomas.m4a \\
        --swap "300 mil" "cinco millones de" \\
        --swap "vivienda" "Bancolombia" \\
        --tries 5
"""
import argparse
from pathlib import Path

from loguru import logger

from app.scripts.voice_demo.schemas.live_demo_request import LiveDemoRequest
from app.scripts.voice_demo.settings import settings
from app.scripts.voice_demo.utils.live_demo_builder import LiveDemoBuilder


def main() -> None:
    """Parse CLI arguments and build the original/spoof/spoof+jitter demo clips."""
    parser = argparse.ArgumentParser(
        description="Build the original, pre-jitter spoof, and post-jitter spoof demo clips."
    )
    parser.add_argument("host_audio", type=Path, help="Path to the single host/reference recording")
    parser.add_argument(
        "--swap", nargs=2, action="append", required=True,
        metavar=("TARGET_PHRASE", "REPLACEMENT_PHRASE"),
        help="A target phrase and its replacement, each a quoted space-separated "
        "word group (e.g. --swap \"300 mil\" \"cinco millones de\"); repeatable "
        "for multiple simultaneous swaps.",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output directory (default: data/voice_demo_output/partial_spoof_live_demo)",
    )
    parser.add_argument(
        "--tries", type=int, default=1,
        help="Number of independent OmniVoice renders to generate, keeping the one "
        "with the most silence around every replacement phrase (default: 1)",
    )
    args = parser.parse_args()

    output_dir = args.output or (settings.OUTPUT_DIR / "partial_spoof_live_demo")
    target_phrases = [target.split() for target, _ in args.swap]
    replacement_phrases = [replacement.split() for _, replacement in args.swap]

    request = LiveDemoRequest(
        host_audio_path=args.host_audio,
        target_phrases=target_phrases,
        replacement_phrases=replacement_phrases,
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
