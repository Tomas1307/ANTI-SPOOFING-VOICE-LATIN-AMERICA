"""
Inference helper shared by the training loop and the evaluation step.
"""
from typing import Optional, Tuple

import numpy as np
import torch
from loguru import logger
from torch.utils.data import DataLoader

from app.pipeline.training.base_spoof_detector import BaseSpoofDetector


def score_dataset(
    model: BaseSpoofDetector,
    loader: DataLoader,
    device: torch.device,
    amp_dtype: Optional[torch.dtype],
    criterion: Optional[torch.nn.Module] = None,
    progress_every: int = 200,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Run the model over a loader and collect scores.

    Args:
        model: Detector to run. Left in evaluation mode on return.
        loader: Data loader yielding padded batches.
        device: Device the batches are moved onto.
        amp_dtype: Autocast dtype, or None to run in full precision.
        criterion: Optional loss to accumulate alongside the scores.
        progress_every: Batches between progress log lines.

    Returns:
        A tuple of (scores, labels, dataset_indices, mean_loss). The mean loss
        is zero when no criterion is supplied.
    """
    model.eval()
    scores: list = []
    labels: list = []
    indices: list = []
    loss_total = 0.0
    loss_batches = 0

    with torch.no_grad():
        for batch_number, batch in enumerate(loader, start=1):
            waveform = batch["waveform"].to(device, non_blocking=True)
            lengths = batch["length"].to(device, non_blocking=True)
            target = batch["label"].to(device, non_blocking=True)

            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=amp_dtype is not None,
            ):
                logits = model(waveform, lengths)

            if criterion is not None:
                loss_total += float(criterion(logits.float(), target).item())
                loss_batches += 1

            scores.append(model.score_from_logits(logits).cpu().numpy())
            labels.append(batch["label"].numpy())
            indices.append(batch["index"].numpy())

            if batch_number % progress_every == 0:
                logger.debug(f"  scored {batch_number:,} batches")

    mean_loss = loss_total / loss_batches if loss_batches else 0.0
    return (
        np.concatenate(scores),
        np.concatenate(labels),
        np.concatenate(indices),
        mean_loss,
    )
