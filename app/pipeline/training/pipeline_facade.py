"""
Facade orchestrating the DF-Arena detector training pipeline.
"""
from pathlib import Path
from typing import Dict, List

import torch
from loguru import logger

from app.pipeline.training.schemas.dataset_split import DatasetSplit
from app.pipeline.training.schemas.pipeline_config import DetectorTrainingConfig
from app.pipeline.training.schemas.training_result import TrainingResult
from app.pipeline.training.settings import settings
from app.pipeline.training.steps.step_01_audit_leakage import (
    CorpusLeakageAuditor,
)
from app.pipeline.training.steps.step_02_build_datasets import (
    ProtocolDatasetBuilder,
)
from app.pipeline.training.steps.step_03_build_model import DetectorFactory
from app.pipeline.training.steps.step_04_train import DetectorTrainer
from app.pipeline.training.steps.step_05_evaluate import DetectorEvaluator
from app.pipeline.training.utils import run_environment


class DetectorTrainingPipeline:
    """Train and evaluate an anti-spoofing detector on the MARSA corpus.

    This pipeline implements the Facade pattern, orchestrating:
    1. Corpus leakage audit, which refuses to spend GPU time on a corpus whose
       invariants no longer hold.
    2. Protocol resolution into typed dataset splits.
    3. Detector construction through a backend factory.
    4. Training with mid-epoch checkpointing and exact resume.
    5. Evaluation reporting pooled, sentence-disjoint strict and per-attack
       equal error rates.

    Attributes:
        config: Run configuration.
        run_dir: Directory holding every artefact of the run.
    """

    def __init__(self, config: DetectorTrainingConfig) -> None:
        """Initialize the pipeline with run configuration.

        Args:
            config: Run configuration with required parameters.
        """
        self.config = config
        self.run_dir = Path(settings.RUNS_ROOT) / config.run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"{self.__class__.__name__} initialized")

    def run(self) -> TrainingResult:
        """Execute the full pipeline.

        Returns:
            The training result, including evaluation of the best checkpoint.

        Raises:
            RuntimeError: If the corpus audit fails, or the environment does
                not satisfy the shared-server and disk requirements.
        """
        run_environment.configure_logging(self.run_dir)
        logger.info("=" * 70)
        logger.info(f"{self.__class__.__name__.upper()} - START")
        logger.info("=" * 70)

        try:
            self._persist_config()
            run_environment.seed_everything(self.config.seed)
            device = run_environment.resolve_device(settings.ENFORCE_SINGLE_GPU)
            run_environment.assert_free_disk(self.run_dir, settings.MIN_FREE_DISK_GB)

            # === STEP 1: Corpus leakage audit ===
            self._audit()

            # === STEP 2: Resolve protocol into dataset splits ===
            splits = self._build_splits()

            # === STEP 3: Build the detector ===
            model = DetectorFactory(self.config, device).execute()

            if self.config.eval_only:
                result = self._evaluate_only(model, device, splits)
            else:
                result = self._train_and_evaluate(model, device, splits)

            self._persist_result(result)
            logger.info(f"{self.__class__.__name__.upper()} - COMPLETE")
            return result

        except Exception as error:
            logger.exception(f"Pipeline failed: {error}")
            raise

    def _train_and_evaluate(
        self, model, device, splits: Dict[str, DatasetSplit]
    ) -> TrainingResult:
        """Train the detector, then score the selected checkpoint.

        Args:
            model: Detector to train.
            device: Compute device.
            splits: Resolved splits keyed by name.

        Returns:
            The training result with its evaluations attached.
        """
        trainer = DetectorTrainer(
            config=self.config,
            model=model,
            device=device,
            train_split=splits["train"],
            dev_split=splits["dev"],
            run_dir=self.run_dir,
        )
        result = trainer.execute()

        checkpoint = result.best_checkpoint or result.last_checkpoint
        if checkpoint:
            state = trainer.checkpoints.load(Path(checkpoint), map_location=str(device))
            model.load_state_dict(state["model"])
            logger.info(f"Evaluating checkpoint: {Path(checkpoint).name}")

        result.evaluations = DetectorEvaluator(
            config=self.config, model=model, device=device, run_dir=self.run_dir
        ).execute(splits, checkpoint)
        return result

    def _evaluate_only(
        self, model, device, splits: Dict[str, DatasetSplit]
    ) -> TrainingResult:
        """Score a detector without training it.

        This is how a zero-shot baseline is produced. No optimiser is built and
        the training split is never read, so the run costs one forward pass per
        clip of the scored splits.

        Args:
            model: Detector to score.
            device: Compute device.
            splits: Resolved splits keyed by name.

        Returns:
            A result carrying only the evaluations.

        Raises:
            FileNotFoundError: If a checkpoint was named but does not exist.
        """
        checkpoint = self.config.eval_checkpoint
        if checkpoint:
            path = Path(checkpoint)
            if not path.exists():
                raise FileNotFoundError(f"Evaluation checkpoint not found: {path}")
            state = torch.load(path, map_location=str(device), weights_only=False)
            model.load_state_dict(state["model"] if "model" in state else state)
            logger.info(f"Eval-only: loaded weights from {path.name}")
        else:
            checkpoint = f"{self.config.detector_backend}:published-weights"
            logger.info(
                "Eval-only: scoring the backend as published, with no fine-tuning"
            )

        result = TrainingResult(run_name=self.config.run_name)
        result.evaluations = DetectorEvaluator(
            config=self.config, model=model, device=device, run_dir=self.run_dir
        ).execute(splits, checkpoint)
        return result

    def _audit(self) -> None:
        """Run the corpus audit and stop the run when it fails.

        Raises:
            RuntimeError: If a fatal invariant does not hold.
        """
        if self.config.skip_audit:
            logger.warning(
                "Corpus audit SKIPPED by configuration. Any error rate this run "
                "produces is unverified and must not be reported."
            )
            return

        auditor = CorpusLeakageAuditor(
            corpus_root=Path(self.config.corpus_root),
            strict_filter_csv=(
                Path(self.config.strict_filter_csv)
                if self.config.strict_filter_csv
                else None
            ),
        )
        report = auditor.execute()
        auditor.write_report(report, self.run_dir / "corpus_audit.json")

        if not report.passed:
            failed = [
                check.name for check in report.checks if check.fatal and not check.passed
            ]
            raise RuntimeError(
                f"Corpus audit failed on: {failed}. Training would produce an "
                "error rate that cannot be defended. See corpus_audit.json."
            )

    def _build_splits(self) -> Dict[str, DatasetSplit]:
        """Resolve every split the run needs.

        Returns:
            Mapping of split name to its resolved description.
        """
        names: List[str] = [] if self.config.eval_only else ["train", "dev"]
        for split in self.config.eval_splits:
            if split not in names:
                names.append(split)

        return ProtocolDatasetBuilder(
            corpus_root=Path(self.config.corpus_root),
            splits=names,
            max_train_items=self.config.max_train_items,
            seed=self.config.seed,
        ).execute()

    def _persist_config(self) -> None:
        """Write the run configuration into the run directory."""
        path = self.run_dir / "config.json"
        path.write_text(self.config.model_dump_json(indent=2), encoding="utf-8")
        logger.info(f"Run configuration written: {path}")

    def _persist_result(self, result: TrainingResult) -> None:
        """Write the training result into the run directory.

        Args:
            result: Completed training result.
        """
        path = self.run_dir / "result.json"
        path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
        logger.info(f"Run result written: {path}")
