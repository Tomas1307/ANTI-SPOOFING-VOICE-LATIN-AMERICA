"""
Launch a detector training run on the MARSA corpus.

The script builds a run configuration from the pipeline settings, overlays any
command-line arguments and hands it to the pipeline facade. Every artefact of
a run lands in one directory under the runs root: the resolved configuration,
the corpus audit report, the training log, a per-epoch metrics table, rolling
and best checkpoints, and the per-clip score files.

Usage on ml-server03, pinned to one free GPU and detached from the terminal:

    cd ~/ANTI-SPOOFING-VOICE-LATIN-AMERICA
    source envs/dfarena_env/bin/activate
    export CUDA_VISIBLE_DEVICES=1
    nohup python -m app.scripts.run_detector_training \\
        --run-name dfarena_2x_run01 \\
        --corpus-root data/augmented/augmented_2x \\
        > logs/dfarena_2x_run01.out 2>&1 &
    deactivate

An interrupted run resumes from its latest checkpoint by repeating the same
command: the default resume mode is automatic, and the run directory is keyed
by run name.
"""
import argparse
from pathlib import Path

from loguru import logger

from app.pipeline.training.pipeline_facade import DetectorTrainingPipeline
from app.pipeline.training.schemas.pipeline_config import DetectorTrainingConfig
from app.pipeline.training.settings import settings


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Train an anti-spoofing detector on the MARSA corpus."
    )
    parser.add_argument(
        "--run-name", type=str, required=True,
        help="Run identifier; names the output directory under the runs root.",
    )
    parser.add_argument(
        "--corpus-root", type=str, default=settings.CORPUS_ROOT,
        help=f"Corpus directory containing the LA tree (default: {settings.CORPUS_ROOT}).",
    )
    parser.add_argument(
        "--backend", type=str, default=settings.DETECTOR_BACKEND,
        help=f"Detector backend key (default: {settings.DETECTOR_BACKEND}).",
    )
    parser.add_argument(
        "--model-id", type=str, default=None,
        help="Backbone identifier; defaults to whatever the backend settings name.",
    )
    parser.add_argument(
        "--freeze-backbone", action="store_true",
        help="Train only the classifier head; leave the backbone frozen.",
    )
    parser.add_argument(
        "--epochs", type=int, default=settings.EPOCHS,
        help=f"Epochs to train (default: {settings.EPOCHS}).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=settings.BATCH_SIZE,
        help=f"Clips per micro-batch (default: {settings.BATCH_SIZE}).",
    )
    parser.add_argument(
        "--grad-accum", type=int, default=settings.GRADIENT_ACCUMULATION,
        help=f"Micro-batches per optimiser step (default: {settings.GRADIENT_ACCUMULATION}).",
    )
    parser.add_argument(
        "--lr", type=float, default=settings.LEARNING_RATE,
        help=f"Classifier head learning rate (default: {settings.LEARNING_RATE}).",
    )
    parser.add_argument(
        "--backbone-lr", type=float, default=settings.BACKBONE_LEARNING_RATE,
        help=f"Backbone learning rate (default: {settings.BACKBONE_LEARNING_RATE}).",
    )
    parser.add_argument(
        "--crop-seconds", type=float, default=settings.CROP_SECONDS,
        help=f"Training crop length in seconds (default: {settings.CROP_SECONDS}).",
    )
    parser.add_argument(
        "--eval-crop-seconds", type=float, default=settings.EVAL_CROP_SECONDS,
        help="Scoring crop length in seconds; 0 scores whole clips.",
    )
    parser.add_argument(
        "--num-workers", type=int, default=settings.NUM_WORKERS,
        help=f"Data-loader workers (default: {settings.NUM_WORKERS}).",
    )
    parser.add_argument(
        "--amp", type=str, default=settings.AMP_DTYPE, choices=["bf16", "fp16", "none"],
        help=f"Mixed-precision dtype (default: {settings.AMP_DTYPE}).",
    )
    parser.add_argument(
        "--seed", type=int, default=settings.SEED,
        help=f"Master random seed (default: {settings.SEED}).",
    )
    parser.add_argument(
        "--checkpoint-every", type=int, default=settings.CHECKPOINT_EVERY_STEPS,
        help="Optimiser steps between mid-epoch checkpoints; 0 disables them.",
    )
    parser.add_argument(
        "--keep-checkpoints", type=int, default=settings.KEEP_LAST_N_CHECKPOINTS,
        help=f"Rolling checkpoints retained (default: {settings.KEEP_LAST_N_CHECKPOINTS}).",
    )
    parser.add_argument(
        "--resume", type=str, default="auto",
        help="Checkpoint path, 'auto' for the latest, or 'none' to start fresh.",
    )
    parser.add_argument(
        "--eval-splits", type=str, nargs="+", default=["dev", "eval"],
        help="Splits scored once training completes.",
    )
    parser.add_argument(
        "--strict-filter", type=str, default=settings.STRICT_FILTER_CSV,
        help="Strict sentence-disjoint filter table; 'none' disables strict EER.",
    )
    parser.add_argument(
        "--no-class-weighting", action="store_true",
        help="Disable inverse-frequency class weighting in the loss.",
    )
    parser.add_argument(
        "--skip-audit", action="store_true",
        help="Skip the corpus audit. Smoke tests only; results are unreportable.",
    )
    parser.add_argument(
        "--max-train-items", type=int, default=0,
        help="Cap training clips for a smoke test; 0 means no cap.",
    )
    return parser.parse_args()


def _build_config(args: argparse.Namespace) -> DetectorTrainingConfig:
    """Assemble the run configuration from settings and arguments.

    Args:
        args: Parsed command-line arguments.

    Returns:
        The run configuration.
    """
    strict_filter = None if args.strict_filter.lower() == "none" else args.strict_filter
    resume = None if str(args.resume).lower() == "none" else args.resume

    return DetectorTrainingConfig(
        run_name=args.run_name,
        corpus_root=args.corpus_root,
        detector_backend=args.backend,
        model_id=args.model_id,
        freeze_backbone=args.freeze_backbone,
        epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation=args.grad_accum,
        learning_rate=args.lr,
        backbone_learning_rate=args.backbone_lr,
        weight_decay=settings.WEIGHT_DECAY,
        warmup_ratio=settings.WARMUP_RATIO,
        crop_seconds=args.crop_seconds,
        eval_crop_seconds=args.eval_crop_seconds,
        num_workers=args.num_workers,
        seed=args.seed,
        amp_dtype=args.amp,
        class_weighting=not args.no_class_weighting,
        checkpoint_every_steps=args.checkpoint_every,
        keep_last_n_checkpoints=args.keep_checkpoints,
        resume=resume,
        eval_splits=args.eval_splits,
        strict_filter_csv=strict_filter,
        skip_audit=args.skip_audit,
        max_train_items=args.max_train_items,
    )


if __name__ == "__main__":
    arguments = _parse_args()
    configuration = _build_config(arguments)
    outcome = DetectorTrainingPipeline(configuration).run()

    logger.info("=" * 70)
    logger.info(f"Run '{outcome.run_name}' finished")
    logger.info(
        f"Best dev EER {outcome.best_dev_eer:.3f}% at epoch {outcome.best_epoch}"
    )
    for evaluation in outcome.evaluations:
        strict = (
            f", strict {evaluation.strict_eer:.3f}% over "
            f"{evaluation.strict_clip_count:,} clips"
            if evaluation.strict_clip_count
            else ""
        )
        logger.info(
            f"  {evaluation.split}: pooled EER {evaluation.eer:.3f}% over "
            f"{evaluation.clip_count:,} clips{strict}"
        )
    logger.info(f"Artefacts: {Path(settings.RUNS_ROOT) / outcome.run_name}")
