"""
Step 5: score held-out splits and report pooled, strict and per-attack error.
"""
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np
import torch
from loguru import logger
from torch.utils.data import DataLoader

from app.pipeline.training.base_spoof_detector import BaseSpoofDetector
from app.pipeline.training.schemas.dataset_split import DatasetSplit
from app.pipeline.training.schemas.evaluation_result import EvaluationResult
from app.pipeline.training.schemas.pipeline_config import DetectorTrainingConfig
from app.pipeline.training.settings import settings
from app.pipeline.training.utils import metrics, protocol_io, run_environment
from app.pipeline.training.utils import scoring
from app.pipeline.training.utils.audio_dataset import MarsaAudioDataset
from app.pipeline.training.utils.batching import pad_collate


class DetectorEvaluator:
    """Score a trained detector and decompose its error.

    Three numbers are reported per split. The pooled equal error rate covers
    every clip. The strict rate covers only clips whose source sentence never
    appears in training, which is the leakage-hardened figure: two thirds of
    evaluation utterances reuse a training sentence, so the pooled number
    alone would overstate the result. The per-attack decomposition shows which
    generator the detector actually struggles with.

    Attributes:
        config: Run configuration.
        model: Trained detector.
        device: Compute device.
        run_dir: Directory the score files are written into.
    """

    def __init__(
        self,
        config: DetectorTrainingConfig,
        model: BaseSpoofDetector,
        device: torch.device,
        run_dir: Path,
    ) -> None:
        """Initialize the evaluator.

        Args:
            config: Run configuration.
            model: Trained detector.
            device: Compute device.
            run_dir: Directory the score files are written into.
        """
        self.config = config
        self.model = model
        self.device = device
        self.run_dir = Path(run_dir)
        self.amp_dtype = run_environment.resolve_amp_dtype(config.amp_dtype)
        self._strict: Optional[Dict[str, Set[str]]] = None

        if config.strict_filter_csv:
            self._strict = protocol_io.read_strict_filter(Path(config.strict_filter_csv))

    def execute(
        self, splits: Dict[str, DatasetSplit], checkpoint: str
    ) -> List[EvaluationResult]:
        """Score every requested split.

        Args:
            splits: Resolved splits keyed by name.
            checkpoint: Checkpoint identifier recorded in the results.

        Returns:
            One result per scored split.
        """
        logger.info(f"Step {self.__class__.__name__}: Starting")
        results: List[EvaluationResult] = []

        for name in self.config.eval_splits:
            if name not in splits:
                logger.warning(f"Split '{name}' was not built; skipping")
                continue
            results.append(self._score_split(splits[name], checkpoint))

        logger.info(f"Step {self.__class__.__name__}: Complete")
        return results

    def _score_split(self, split: DatasetSplit, checkpoint: str) -> EvaluationResult:
        """Score one split and assemble its result.

        Args:
            split: Resolved split to score.
            checkpoint: Checkpoint identifier recorded in the result.

        Returns:
            The evaluation result.
        """
        dataset = MarsaAudioDataset(
            entries=split.entries,
            flac_dir=Path(split.flac_dir),
            sample_rate=settings.SAMPLE_RATE,
            crop_samples=int(self.config.eval_crop_seconds * settings.SAMPLE_RATE),
            training=False,
            seed=self.config.seed,
        )
        loader = DataLoader(
            dataset,
            batch_size=max(1, self.config.batch_size),
            shuffle=False,
            num_workers=self.config.num_workers,
            collate_fn=pad_collate,
            pin_memory=True,
            worker_init_fn=run_environment.worker_init,
        )

        scores, labels, indices, _loss = scoring.score_dataset(
            model=self.model,
            loader=loader,
            device=self.device,
            amp_dtype=self.amp_dtype,
        )

        ordered = [split.entries[position] for position in indices]
        attack_ids = [entry.attack_id for entry in ordered]

        eer, _threshold = metrics.compute_eer(scores[labels == 1], scores[labels == 0])
        per_attack = metrics.compute_per_attack_eer(scores, labels, attack_ids)

        score_file = self.run_dir / "scores" / f"scores_{split.name}.txt"
        protocol_io.write_scores(
            score_file,
            [
                (entry.audio_id, entry.attack_id, entry.key, float(score))
                for entry, score in zip(ordered, scores)
            ],
        )

        result = EvaluationResult(
            split=split.name,
            checkpoint=checkpoint,
            clip_count=int(scores.size),
            eer=eer,
            per_attack_eer=per_attack,
            score_file=str(score_file),
        )
        self._add_strict_eer(result, split.name, ordered, scores, labels)

        logger.info(
            f"{split.name}: pooled EER {result.eer:.3f}% over {result.clip_count:,} clips"
            + (
                f" | strict EER {result.strict_eer:.3f}% over "
                f"{result.strict_clip_count:,} clips"
                if result.strict_clip_count
                else ""
            )
        )
        for attack, value in sorted(per_attack.items(), key=lambda item: -item[1]):
            logger.info(f"    {attack:<24} {value:6.3f}%")
        return result

    def _add_strict_eer(
        self,
        result: EvaluationResult,
        split_name: str,
        ordered: List,
        scores: np.ndarray,
        labels: np.ndarray,
    ) -> None:
        """Attach the sentence-disjoint strict error rate to a result.

        Args:
            result: Result being assembled, modified in place.
            split_name: Split the scores came from.
            ordered: Protocol entries aligned with the score array.
            scores: Countermeasure scores.
            labels: Integer labels aligned with the scores.
        """
        if self._strict is None:
            return

        allowed = self._strict.get(split_name, set())
        if not allowed:
            logger.warning(f"Strict filter has no rows for split '{split_name}'")
            return

        mask = np.array([entry.source_file in allowed for entry in ordered])
        strict_scores = scores[mask]
        strict_labels = labels[mask]
        if strict_scores.size == 0:
            logger.warning(f"Strict filter matched no clips in split '{split_name}'")
            return
        if strict_labels.min() == strict_labels.max():
            logger.warning(
                f"Strict subset of '{split_name}' holds a single class; "
                "no strict EER can be computed"
            )
            return

        strict_eer, _threshold = metrics.compute_eer(
            strict_scores[strict_labels == 1], strict_scores[strict_labels == 0]
        )
        result.strict_clip_count = int(strict_scores.size)
        result.strict_eer = strict_eer
