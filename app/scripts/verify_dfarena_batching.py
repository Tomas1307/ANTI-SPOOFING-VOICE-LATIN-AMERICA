"""
Verify the batched DF-Arena forward matches the published per-clip forward.

WHY
---
The published DF-Arena backbone hard-codes a batch size of one:

    def forward(self, x):
        out_ssl = self.ssl_model(x.unsqueeze(0))

Handed a real batch it produces a four-dimensional tensor that wav2vec2's
convolutional front end rejects. Every later operation is batch-general, so
``DFArenaDetector.forward`` calls the same submodules in the same order with
genuinely batched input, skipping only that one line.

That is a claim about numerical identity, and claims about numerical identity
should be measured rather than asserted in a docstring. This script runs the
same waveforms through both paths and reports the largest disagreement, in the
logits and in the countermeasure score derived from them.

Random waveforms are sufficient and are used by default: the question is
whether two code paths compute the same function, not whether the model
behaves sensibly on noise. A fixed seed makes the check reproducible.

USAGE
-----
    cd ~/ANTI-SPOOFING-VOICE-LATIN-AMERICA
    source envs/dfarena_env/bin/activate
    export CUDA_VISIBLE_DEVICES=1
    python -m app.scripts.verify_dfarena_batching
    deactivate

Exits 0 when the paths agree within tolerance, 1 otherwise.
"""
import argparse
import sys

import torch
from loguru import logger

from app.pipeline.training.dfarena.dfarena_detector import DFArenaDetector
from app.pipeline.training.dfarena.settings import settings as dfarena_settings


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Check the batched DF-Arena forward against the published one."
    )
    parser.add_argument(
        "--model-id", type=str, default=dfarena_settings.MODEL_ID,
        help="Model repository identifier.",
    )
    parser.add_argument(
        "--clips", type=int, default=8,
        help="Number of waveforms to compare (default: 8).",
    )
    parser.add_argument(
        "--tolerance", type=float, default=1e-3,
        help="Maximum acceptable absolute difference in logits (default: 1e-3).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Seed for the random waveforms (default: 42).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if not torch.cuda.is_available():
        logger.error("No CUDA device visible. Set CUDA_VISIBLE_DEVICES and rerun.")
        sys.exit(1)

    device = torch.device("cuda:0")
    torch.manual_seed(args.seed)

    detector = DFArenaDetector(model_id=args.model_id).to(device)
    detector.eval()

    waveform = torch.randn(
        args.clips, detector.REQUIRED_SAMPLES, device=device, dtype=torch.float32
    )

    with torch.no_grad():
        batched = detector(waveform, torch.full((args.clips,), detector.REQUIRED_SAMPLES))
        published = detector.published_forward(waveform)

    logit_delta = (batched - published).abs().max().item()
    score_delta = (
        (
            DFArenaDetector.score_from_logits(batched)
            - DFArenaDetector.score_from_logits(published)
        )
        .abs()
        .max()
        .item()
    )

    logger.info(f"Compared {args.clips} clips of {detector.REQUIRED_SAMPLES:,} samples")
    logger.info(f"  max |logit difference| : {logit_delta:.3e}")
    logger.info(f"  max |score difference| : {score_delta:.3e}")
    logger.info(f"  tolerance              : {args.tolerance:.3e}")

    if logit_delta > args.tolerance:
        logger.error(
            "The batched forward does NOT match the published one. Do not use "
            "it for scoring; the adapter must be reviewed against the current "
            "published modeling source."
        )
        sys.exit(1)

    logger.info("Batched forward matches the published per-clip forward.")
    sys.exit(0)
