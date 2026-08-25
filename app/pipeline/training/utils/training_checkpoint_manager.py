"""
Checkpoint persistence for resumable detector training.

This is deliberately separate from app.utils.checkpoint_manager, which tracks
which items a data-processing job has already emitted. Training checkpoints
carry tensor state (weights, optimiser moments, scheduler position, gradient
scaler) plus the random-number generator states needed to resume a run without
changing its trajectory.
"""
import random
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from loguru import logger


class TrainingCheckpointManager:
    """Write, prune, discover and restore training checkpoints.

    Checkpoints are written to a temporary file and then renamed, so an
    interrupted write can never leave a truncated checkpoint behind. Rolling
    checkpoints are pruned to a fixed depth because a one-billion-parameter
    model carrying AdamW state costs roughly sixteen gigabytes per file, and
    the shared server has limited free disk.

    Attributes:
        checkpoint_dir: Directory holding checkpoint files.
        keep_last_n: Number of rolling checkpoints retained on disk.
    """

    ROLLING_PREFIX = "checkpoint_step"
    BEST_NAME = "checkpoint_best.pt"

    def __init__(self, checkpoint_dir: Path, keep_last_n: int = 2) -> None:
        """Initialize the manager.

        Args:
            checkpoint_dir: Directory holding checkpoint files. Created if it
                does not exist.
            keep_last_n: Number of rolling checkpoints retained on disk. Must
                be at least one.

        Raises:
            ValueError: If keep_last_n is below one.
        """
        if keep_last_n < 1:
            raise ValueError(f"keep_last_n must be at least 1, got {keep_last_n}")
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.keep_last_n = keep_last_n

    @staticmethod
    def capture_rng_state() -> Dict[str, Any]:
        """Capture the random-number generator states of every library in use.

        Returns:
            Mapping of library name to its serialisable generator state.
        """
        state: Dict[str, Any] = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            state["cuda"] = torch.cuda.get_rng_state_all()
        return state

    @staticmethod
    def restore_rng_state(state: Dict[str, Any]) -> None:
        """Restore previously captured random-number generator states.

        Args:
            state: Mapping produced by capture_rng_state.
        """
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch"].cpu() if hasattr(state["torch"], "cpu") else state["torch"])
        if "cuda" in state and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["cuda"])

    def save(self, state: Dict[str, Any], global_step: int) -> Path:
        """Write a rolling checkpoint and prune older ones.

        Args:
            state: Serialisable training state.
            global_step: Optimiser step the checkpoint corresponds to.

        Returns:
            Path of the written checkpoint.
        """
        target = self.checkpoint_dir / f"{self.ROLLING_PREFIX}_{global_step:09d}.pt"
        self._atomic_save(state, target)
        logger.info(f"Checkpoint written: {target.name}")
        self._prune()
        return target

    def save_best(self, state: Dict[str, Any]) -> Path:
        """Write the best-so-far checkpoint, replacing any previous one.

        Args:
            state: Serialisable training state.

        Returns:
            Path of the written checkpoint.
        """
        target = self.checkpoint_dir / self.BEST_NAME
        self._atomic_save(state, target)
        logger.info(f"Best checkpoint updated: {target.name}")
        return target

    def latest(self) -> Optional[Path]:
        """Return the most advanced rolling checkpoint on disk.

        Returns:
            Path of the checkpoint with the highest step number, or None when
            no rolling checkpoint exists.
        """
        candidates = self._rolling_checkpoints()
        return candidates[-1] if candidates else None

    def load(self, path: Path, map_location: str = "cpu") -> Dict[str, Any]:
        """Load a checkpoint from disk.

        Args:
            path: Checkpoint path.
            map_location: Device the tensors are mapped onto.

        Returns:
            The stored training state.

        Raises:
            FileNotFoundError: If the checkpoint does not exist.
        """
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        logger.info(f"Loading checkpoint: {path}")
        return torch.load(path, map_location=map_location, weights_only=False)

    def _rolling_checkpoints(self) -> List[Path]:
        """List rolling checkpoints in ascending step order.

        Returns:
            Sorted list of rolling checkpoint paths.
        """
        return sorted(self.checkpoint_dir.glob(f"{self.ROLLING_PREFIX}_*.pt"))

    def _prune(self) -> None:
        """Delete rolling checkpoints beyond the retention depth."""
        candidates = self._rolling_checkpoints()
        for stale in candidates[: -self.keep_last_n]:
            stale.unlink()
            logger.info(f"Pruned old checkpoint: {stale.name}")

    def _atomic_save(self, state: Dict[str, Any], target: Path) -> None:
        """Serialise state to a temporary file and move it into place.

        Args:
            state: Serialisable training state.
            target: Final checkpoint path.
        """
        temporary = target.with_suffix(".pt.tmp")
        torch.save(state, temporary)
        shutil.move(str(temporary), str(target))
