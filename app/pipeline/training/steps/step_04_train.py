"""
Step 4: train the detector, with checkpointing and exact resume.
"""
import csv
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from loguru import logger
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Subset

from app.pipeline.training.base_spoof_detector import BaseSpoofDetector
from app.pipeline.training.schemas.dataset_split import DatasetSplit
from app.pipeline.training.schemas.epoch_result import EpochResult
from app.pipeline.training.schemas.pipeline_config import DetectorTrainingConfig
from app.pipeline.training.schemas.training_result import TrainingResult
from app.pipeline.training.settings import settings
from app.pipeline.training.utils import metrics, run_environment, scoring
from app.pipeline.training.utils.audio_dataset import MarsaAudioDataset
from app.pipeline.training.utils.batching import pad_collate
from app.pipeline.training.utils.training_checkpoint_manager import (
    TrainingCheckpointManager,
)


class DetectorTrainer:
    """Run the training loop and keep it restartable.

    Two properties drive the design. First, a run on a shared server will be
    interrupted, so state is checkpointed mid-epoch and a resume replays the
    exact remaining batch order rather than restarting the epoch. Second,
    epoch order is generated from a seeded permutation instead of a shuffling
    sampler, which is what makes that exact replay possible.

    Attributes:
        config: Run configuration.
        model: Detector being trained.
        device: Compute device.
        run_dir: Directory holding checkpoints, logs and metrics.
    """

    def __init__(
        self,
        config: DetectorTrainingConfig,
        model: BaseSpoofDetector,
        device: torch.device,
        train_split: DatasetSplit,
        dev_split: DatasetSplit,
        run_dir: Path,
    ) -> None:
        """Initialize the trainer.

        Args:
            config: Run configuration.
            model: Detector to train.
            device: Compute device.
            train_split: Resolved training split.
            dev_split: Resolved development split, used for model selection.
            run_dir: Directory holding checkpoints, logs and metrics.
        """
        self.config = config
        self.model = model
        self.device = device
        self.run_dir = Path(run_dir)
        self.amp_dtype = run_environment.resolve_amp_dtype(config.amp_dtype)

        crop_samples = int(config.crop_seconds * settings.SAMPLE_RATE)
        eval_crop = int(config.eval_crop_seconds * settings.SAMPLE_RATE)

        if model.required_samples:
            logger.info(
                f"Backend requires exactly {model.required_samples:,} samples "
                f"({model.required_samples / settings.SAMPLE_RATE:.4f} s); "
                f"overriding the configured crop of {crop_samples:,}/{eval_crop:,}"
            )
            crop_samples = model.required_samples
            eval_crop = model.required_samples

        self.train_dataset = MarsaAudioDataset(
            entries=train_split.entries,
            flac_dir=Path(train_split.flac_dir),
            sample_rate=settings.SAMPLE_RATE,
            crop_samples=crop_samples,
            training=True,
            seed=config.seed,
        )
        self.dev_dataset = MarsaAudioDataset(
            entries=dev_split.entries,
            flac_dir=Path(dev_split.flac_dir),
            sample_rate=settings.SAMPLE_RATE,
            crop_samples=eval_crop,
            training=False,
            seed=config.seed,
        )

        self.checkpoints = TrainingCheckpointManager(
            checkpoint_dir=self.run_dir / "checkpoints",
            keep_last_n=config.keep_last_n_checkpoints,
        )
        self.criterion = self._build_criterion()
        self.optimizer = AdamW(
            self.model.parameter_groups(
                config.learning_rate, config.backbone_learning_rate
            ),
            weight_decay=config.weight_decay,
        )
        self.steps_per_epoch = self._steps_per_epoch()
        self.scheduler = self._build_scheduler()
        self.scaler = torch.amp.GradScaler(
            device.type, enabled=config.amp_dtype == "fp16"
        )

        self.metrics_csv = self.run_dir / "metrics.csv"
        self.history_jsonl = self.run_dir / "history.jsonl"

    def execute(self) -> TrainingResult:
        """Train the model, resuming from a checkpoint when one is available.

        Returns:
            The training result, including per-epoch metrics and the paths of
            the best and last checkpoints.
        """
        logger.info(f"Step {self.__class__.__name__}: Starting")
        state = self._restore()
        result = TrainingResult(
            run_name=self.config.run_name,
            best_epoch=state["best_epoch"],
            best_dev_eer=state["best_dev_eer"],
            resumed_from=state["resumed_from"],
        )
        result.epochs = [EpochResult(**record) for record in state["history"]]

        global_step = state["global_step"]
        start_epoch = state["epoch"]
        start_batch = state["batch_in_epoch"]

        logger.info(
            f"Training from epoch {start_epoch}, batch {start_batch}, "
            f"step {global_step}; {self.steps_per_epoch:,} optimiser steps per epoch"
        )

        try:
            for epoch in range(start_epoch, self.config.epochs):
                started = time.time()
                self.train_dataset.set_epoch(epoch)
                train_loss, global_step = self._train_one_epoch(
                    epoch, start_batch, global_step
                )
                start_batch = 0

                dev_scores, dev_labels, _indices, dev_loss = scoring.score_dataset(
                    model=self.model,
                    loader=self._dev_loader(),
                    device=self.device,
                    amp_dtype=self.amp_dtype,
                    criterion=self.criterion,
                )
                dev_eer, _threshold = metrics.compute_eer(
                    dev_scores[dev_labels == 1], dev_scores[dev_labels == 0]
                )

                is_best = result.best_dev_eer < 0 or dev_eer < result.best_dev_eer
                record = EpochResult(
                    epoch=epoch,
                    global_step=global_step,
                    train_loss=train_loss,
                    dev_loss=dev_loss,
                    dev_eer=dev_eer,
                    learning_rate=self.optimizer.param_groups[0]["lr"],
                    seconds=time.time() - started,
                    is_best=is_best,
                )
                result.epochs.append(record)
                self._append_metrics(record)

                logger.info(
                    f"epoch {epoch}: train_loss={train_loss:.4f} "
                    f"dev_loss={dev_loss:.4f} dev_eer={dev_eer:.3f}% "
                    f"({record.seconds / 60.0:.1f} min)"
                    + ("  [best]" if is_best else "")
                )

                if is_best:
                    result.best_dev_eer = dev_eer
                    result.best_epoch = epoch
                    result.best_checkpoint = str(
                        self.checkpoints.save_best(
                            self._state_dict(epoch + 1, 0, global_step, result)
                        )
                    )

                result.last_checkpoint = str(
                    self.checkpoints.save(
                        self._state_dict(epoch + 1, 0, global_step, result), global_step
                    )
                )

        except KeyboardInterrupt:
            logger.warning("Interrupted; writing a checkpoint before exiting")
            self.checkpoints.save(
                self._state_dict(start_epoch, 0, global_step, result), global_step
            )
            raise

        logger.info(
            f"Step {self.__class__.__name__}: Complete "
            f"(best dev EER {result.best_dev_eer:.3f}% at epoch {result.best_epoch})"
        )
        return result

    def _train_one_epoch(
        self, epoch: int, start_batch: int, global_step: int
    ) -> Tuple[float, int]:
        """Run one epoch of optimisation.

        Args:
            epoch: Zero-based epoch index.
            start_batch: Micro-batch to resume from within this epoch.
            global_step: Optimiser steps completed so far.

        Returns:
            A tuple of (mean training loss, updated global step).
        """
        self.model.train()
        if getattr(self.model, "frozen", False):
            self.model.backbone.eval()

        loader = self._train_loader(epoch, start_batch)
        accumulation = self.config.gradient_accumulation
        loss_total = 0.0
        loss_batches = 0

        self.optimizer.zero_grad(set_to_none=True)

        for offset, batch in enumerate(loader):
            batch_in_epoch = start_batch + offset
            waveform = batch["waveform"].to(self.device, non_blocking=True)
            lengths = batch["length"].to(self.device, non_blocking=True)
            target = batch["label"].to(self.device, non_blocking=True)

            with torch.autocast(
                device_type=self.device.type,
                dtype=self.amp_dtype,
                enabled=self.amp_dtype is not None,
            ):
                logits = self.model(waveform, lengths)
                loss = self.criterion(logits.float(), target)

            self.scaler.scale(loss / accumulation).backward()
            loss_total += float(loss.item())
            loss_batches += 1

            if (batch_in_epoch + 1) % accumulation == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), settings.MAX_GRAD_NORM
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % 50 == 0:
                    logger.debug(
                        f"  epoch {epoch} step {global_step} "
                        f"loss {loss_total / max(loss_batches, 1):.4f} "
                        f"lr {self.optimizer.param_groups[0]['lr']:.2e}"
                    )

                if (
                    self.config.checkpoint_every_steps
                    and global_step % self.config.checkpoint_every_steps == 0
                ):
                    self.checkpoints.save(
                        self._state_dict(epoch, batch_in_epoch + 1, global_step, None),
                        global_step,
                    )

        return loss_total / max(loss_batches, 1), global_step

    def _epoch_order(self, epoch: int) -> List[int]:
        """Generate the deterministic clip order for one epoch.

        Args:
            epoch: Zero-based epoch index.

        Returns:
            A permutation of dataset positions.
        """
        generator = np.random.default_rng([self.config.seed, epoch])
        order = np.arange(len(self.train_dataset))
        generator.shuffle(order)
        return order.tolist()

    def _train_loader(self, epoch: int, start_batch: int) -> DataLoader:
        """Build the training loader for one epoch, skipping consumed batches.

        Args:
            epoch: Zero-based epoch index.
            start_batch: Micro-batches already consumed in this epoch.

        Returns:
            A loader over the remaining clips, in the epoch order.
        """
        order = self._epoch_order(epoch)
        consumed = start_batch * self.config.batch_size
        if consumed:
            logger.info(f"  resuming epoch {epoch}: skipping {consumed:,} clips")
        return DataLoader(
            Subset(self.train_dataset, order[consumed:]),
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            collate_fn=pad_collate,
            pin_memory=True,
            drop_last=False,
            worker_init_fn=run_environment.worker_init,
            persistent_workers=self.config.num_workers > 0,
        )

    def _dev_loader(self) -> DataLoader:
        """Build the development-set loader.

        Returns:
            A deterministic loader over the development split.
        """
        return DataLoader(
            self.dev_dataset,
            batch_size=max(1, self.config.batch_size),
            shuffle=False,
            num_workers=self.config.num_workers,
            collate_fn=pad_collate,
            pin_memory=True,
            worker_init_fn=run_environment.worker_init,
        )

    def _build_criterion(self) -> torch.nn.Module:
        """Build the training loss.

        Returns:
            Cross-entropy, weighted by inverse class frequency when the run
            asks for it. The corpus keeps its natural spoof-heavy ratio by
            design, so rebalancing belongs here rather than in the data.
        """
        weight = None
        if self.config.class_weighting:
            weight = self.train_dataset.class_weights()
            if weight is None:
                logger.warning("Class weighting requested but one class is absent")
            else:
                logger.info(
                    f"Class weights: spoof={weight[0]:.3f} bonafide={weight[1]:.3f}"
                )
        return torch.nn.CrossEntropyLoss(
            weight=weight.to(self.device) if weight is not None else None
        )

    def _steps_per_epoch(self) -> int:
        """Compute the optimiser steps performed in one full epoch.

        Returns:
            Steps per epoch, at least one.
        """
        micro_batches = int(
            np.ceil(len(self.train_dataset) / self.config.batch_size)
        )
        return max(1, micro_batches // self.config.gradient_accumulation)

    def _build_scheduler(self) -> LambdaLR:
        """Build a linear warm-up and linear decay schedule.

        Returns:
            The learning-rate scheduler.
        """
        total = max(1, self.steps_per_epoch * self.config.epochs)
        warmup = int(total * self.config.warmup_ratio)

        def factor(step: int) -> float:
            """Compute the learning-rate multiplier for a step.

            Args:
                step: Optimiser step index.

            Returns:
                Multiplier applied to the group learning rate.
            """
            if warmup and step < warmup:
                return step / max(1, warmup)
            progress = (step - warmup) / max(1, total - warmup)
            return max(0.0, 1.0 - progress)

        logger.info(f"Schedule: {total:,} total steps, {warmup:,} warm-up steps")
        return LambdaLR(self.optimizer, lr_lambda=factor)

    def _state_dict(
        self,
        epoch: int,
        batch_in_epoch: int,
        global_step: int,
        result: Optional[TrainingResult],
    ) -> Dict[str, Any]:
        """Assemble everything needed to resume the run.

        Args:
            epoch: Epoch to resume at.
            batch_in_epoch: Micro-batch to resume at within that epoch.
            global_step: Optimiser steps completed.
            result: Result so far, when the caller has one.

        Returns:
            The serialisable training state.
        """
        history = [record.model_dump() for record in result.epochs] if result else []
        return {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "epoch": epoch,
            "batch_in_epoch": batch_in_epoch,
            "global_step": global_step,
            "best_dev_eer": result.best_dev_eer if result else -1.0,
            "best_epoch": result.best_epoch if result else -1,
            "history": history,
            "config": self.config.model_dump(),
            "rng": TrainingCheckpointManager.capture_rng_state(),
        }

    def _restore(self) -> Dict[str, Any]:
        """Load a checkpoint when the configuration asks for one.

        Returns:
            The starting position of the run: epoch, batch, step, best score,
            history and the checkpoint that was restored, if any.

        Raises:
            FileNotFoundError: If an explicit resume path does not exist.
        """
        fresh = {
            "epoch": 0,
            "batch_in_epoch": 0,
            "global_step": 0,
            "best_dev_eer": -1.0,
            "best_epoch": -1,
            "history": [],
            "resumed_from": None,
        }

        requested = self.config.resume
        if not requested:
            logger.info("Starting a fresh run")
            return fresh

        if requested == "auto":
            path = self.checkpoints.latest()
            if path is None:
                logger.info("No checkpoint found; starting a fresh run")
                return fresh
        else:
            path = Path(requested)
            if not path.exists():
                raise FileNotFoundError(f"Resume checkpoint not found: {path}")

        state = self.checkpoints.load(path, map_location=str(self.device))
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        self.scaler.load_state_dict(state["scaler"])
        TrainingCheckpointManager.restore_rng_state(state["rng"])

        logger.info(
            f"Resumed from {path.name}: epoch {state['epoch']}, "
            f"batch {state['batch_in_epoch']}, step {state['global_step']}, "
            f"best dev EER {state['best_dev_eer']:.3f}%"
        )
        return {
            "epoch": state["epoch"],
            "batch_in_epoch": state["batch_in_epoch"],
            "global_step": state["global_step"],
            "best_dev_eer": state["best_dev_eer"],
            "best_epoch": state["best_epoch"],
            "history": state["history"],
            "resumed_from": str(path),
        }

    def _append_metrics(self, record: EpochResult) -> None:
        """Append one epoch to the metrics CSV and the history log.

        Args:
            record: Metrics of the finished epoch.
        """
        write_header = not self.metrics_csv.exists()
        with open(self.metrics_csv, "a", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            if write_header:
                writer.writerow(list(record.model_dump().keys()))
            writer.writerow(list(record.model_dump().values()))

        with open(self.history_jsonl, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record.model_dump()) + "\n")
