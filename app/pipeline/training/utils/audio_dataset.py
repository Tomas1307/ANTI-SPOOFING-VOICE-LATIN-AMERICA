"""
Torch dataset over MARSA protocol entries.
"""
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

from app.pipeline.training.schemas.protocol_entry import ProtocolEntry


class MarsaAudioDataset(Dataset):
    """Serve waveforms and labels for one split of the MARSA corpus.

    Training draws a random fixed-length crop from each clip, which both
    bounds memory and acts as a mild regulariser. Evaluation is deterministic:
    either the whole utterance, or a centre crop of fixed length when one is
    requested. Clips shorter than the crop are tiled rather than zero-padded,
    because a trailing block of digital silence is itself a cue a detector can
    learn.

    Attributes:
        entries: Protocol entries served by this dataset.
        flac_dir: Directory holding the split clips.
        sample_rate: Expected sample rate in Hz.
        crop_samples: Fixed crop length in samples, or zero for full clips.
        training: Whether to draw random crops instead of centre crops.
    """

    def __init__(
        self,
        entries: List[ProtocolEntry],
        flac_dir: Path,
        sample_rate: int,
        crop_samples: int,
        training: bool,
        seed: int = 42,
    ) -> None:
        """Initialize the dataset.

        Args:
            entries: Protocol entries to serve.
            flac_dir: Directory holding the split clips.
            sample_rate: Expected sample rate in Hz. A clip stored at any
                other rate raises rather than being silently resampled.
            crop_samples: Fixed crop length in samples. Zero serves whole
                clips, which requires a padding collate function.
            training: Whether to draw random crops instead of centre crops.
            seed: Base seed for the per-item crop generator.
        """
        self.entries = entries
        self.flac_dir = Path(flac_dir)
        self.sample_rate = sample_rate
        self.crop_samples = crop_samples
        self.training = training
        self.seed = seed
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Advance the crop generator so each epoch sees different crops.

        Args:
            epoch: Zero-based epoch index.
        """
        self._epoch = epoch

    def __len__(self) -> int:
        """Return the number of clips served.

        Returns:
            Count of protocol entries.
        """
        return len(self.entries)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """Load and crop one clip.

        Args:
            index: Position of the entry within the dataset.

        Returns:
            Mapping with the waveform, its length, the integer label and the
            position of the entry, the last of which lets the caller recover
            the identifiers without shipping strings through the collate.

        Raises:
            FileNotFoundError: If the clip is missing from disk.
            ValueError: If the clip sample rate differs from the expected one.
        """
        entry = self.entries[index]
        path = self.flac_dir / f"{entry.audio_id}.flac"
        if not path.exists():
            raise FileNotFoundError(f"Clip missing from disk: {path}")

        waveform, rate = sf.read(str(path), dtype="float32", always_2d=False)
        if rate != self.sample_rate:
            raise ValueError(
                f"{path.name} is {rate} Hz, expected {self.sample_rate} Hz"
            )
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)

        if self.crop_samples > 0:
            waveform = self._crop(waveform, index)

        return {
            "waveform": torch.from_numpy(np.ascontiguousarray(waveform)),
            "length": torch.tensor(waveform.shape[0], dtype=torch.long),
            "label": torch.tensor(entry.label, dtype=torch.long),
            "index": torch.tensor(index, dtype=torch.long),
        }

    def _crop(self, waveform: np.ndarray, index: int) -> np.ndarray:
        """Reduce a waveform to the configured fixed length.

        Args:
            waveform: Source samples.
            index: Position of the entry, used to seed the training crop.

        Returns:
            Exactly ``crop_samples`` samples.
        """
        length = waveform.shape[0]
        if length < self.crop_samples:
            repeats = int(np.ceil(self.crop_samples / max(length, 1)))
            waveform = np.tile(waveform, repeats)
            length = waveform.shape[0]

        if length == self.crop_samples:
            return waveform

        if self.training:
            generator = np.random.default_rng([self.seed, self._epoch, index])
            start = int(generator.integers(0, length - self.crop_samples + 1))
        else:
            start = (length - self.crop_samples) // 2
        return waveform[start : start + self.crop_samples]

    def identifiers(self, index: int) -> Tuple[str, str, str, str]:
        """Return the identifiers of one entry.

        Args:
            index: Position of the entry within the dataset.

        Returns:
            A tuple of (audio_id, attack_id, key, source_file).
        """
        entry = self.entries[index]
        return entry.audio_id, entry.attack_id, entry.key, entry.source_file

    def class_counts(self) -> Dict[str, int]:
        """Count entries by ground-truth label.

        Returns:
            Mapping with bonafide and spoof counts.
        """
        counts = {"bonafide": 0, "spoof": 0}
        for entry in self.entries:
            counts[entry.key] = counts.get(entry.key, 0) + 1
        return counts

    def class_weights(self) -> Optional[torch.Tensor]:
        """Compute inverse-frequency class weights for the loss.

        The corpus preserves the natural spoof-heavy ratio by design, so a
        weighted loss is the intended way to rebalance at training time.

        Returns:
            A two-element tensor indexed by label, spoof first, or None when
            either class is absent.
        """
        counts = self.class_counts()
        spoof = counts.get("spoof", 0)
        bonafide = counts.get("bonafide", 0)
        if spoof == 0 or bonafide == 0:
            return None
        total = spoof + bonafide
        return torch.tensor(
            [total / (2.0 * spoof), total / (2.0 * bonafide)], dtype=torch.float32
        )
