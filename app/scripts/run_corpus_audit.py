"""
Audit a MARSA corpus tier for the invariants that make a reported EER defensible.

This runs step 1 of the training pipeline on its own. It reads only the
protocol files, the metadata CSVs and a directory listing, so it costs about
two minutes, needs no GPU and decodes no audio. Use it before committing GPU
time to a tier, and after any regeneration of that tier.

The JSON report it writes is the reproducible artefact behind the leakage
claims in the data descriptor's technical validation section.

Usage on ml-server03:

    cd ~/ANTI-SPOOFING-VOICE-LATIN-AMERICA
    source envs/fishgram_env/bin/activate
    python -m app.scripts.run_corpus_audit --corpus-root data/augmented/augmented_2x
    deactivate

Exits 0 when every fatal invariant holds, 1 otherwise.
"""
import argparse
import sys
from pathlib import Path

from loguru import logger

from app.pipeline.training.settings import settings
from app.pipeline.training.steps.step_01_audit_leakage import (
    CorpusLeakageAuditor,
)


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Audit a MARSA corpus tier for training-blocking defects."
    )
    parser.add_argument(
        "--corpus-root", type=str, default=settings.CORPUS_ROOT,
        help=f"Corpus directory containing the LA tree (default: {settings.CORPUS_ROOT}).",
    )
    parser.add_argument(
        "--strict-filter", type=str, default=settings.STRICT_FILTER_CSV,
        help="Strict sentence-disjoint filter table; 'none' skips the join check.",
    )
    parser.add_argument(
        "--splits", type=str, nargs="+", default=["train", "dev", "eval"],
        help="Splits to audit.",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Report path (default: <corpus-root>/corpus_audit.json).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    corpus_root = Path(args.corpus_root)
    strict = None if args.strict_filter.lower() == "none" else Path(args.strict_filter)

    auditor = CorpusLeakageAuditor(
        corpus_root=corpus_root, strict_filter_csv=strict, splits=args.splits
    )
    audit_report = auditor.execute()
    auditor.write_report(
        audit_report, args.output or corpus_root / "corpus_audit.json"
    )

    if not audit_report.passed:
        logger.error("Corpus is NOT fit to train on. Do not launch a run.")
        sys.exit(1)

    logger.info("Corpus audit passed. Cleared for training.")
    sys.exit(0)
