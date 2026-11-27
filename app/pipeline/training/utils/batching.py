"""
Collate helpers for variable-length waveform batches.
"""
from typing import Dict, List

import torch


def pad_collate(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Collate waveforms of differing lengths into one padded batch.

    Padding is zero-filled and accompanied by the true lengths, so the model
    can mask the padded frames out of its pooling. Without that mask a long
    tail of zeros would shift the pooled representation of short clips, and
    clip duration correlates with class in any corpus assembled from mixed
    sources.

    Args:
        batch: Items produced by the dataset.

    Returns:
        Mapping with the padded waveforms, their true lengths, the labels and
        the dataset positions.
    """
    lengths = torch.stack([item["length"] for item in batch])
    longest = int(lengths.max().item())

    padded = torch.zeros(len(batch), longest, dtype=torch.float32)
    for position, item in enumerate(batch):
        waveform = item["waveform"]
        padded[position, : waveform.shape[0]] = waveform

    return {
        "waveform": padded,
        "length": lengths,
        "label": torch.stack([item["label"] for item in batch]),
        "index": torch.stack([item["index"] for item in batch]),
    }
