"""
Run-environment helpers: seeding, device selection, disk headroom, logging.

ml-server03 is shared. These helpers enforce the project rule that a run
occupies exactly one GPU, and refuse to start a job that would fill the disk
partway through.
"""
import os
import random
import shutil
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from loguru import logger


def seed_everything(seed: int) -> None:
    """Seed every random-number generator the run depends on.

    Args:
        seed: Master seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    logger.info(f"Seeded all generators with {seed}")


def worker_init(worker_id: int) -> None:
    """Give each data-loader worker a distinct, reproducible seed.

    Args:
        worker_id: Index of the worker process.
    """
    seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(seed + worker_id)
    random.seed(seed + worker_id)


def resolve_device(enforce_single_gpu: bool) -> torch.device:
    """Select the compute device and enforce the shared-server GPU rule.

    Args:
        enforce_single_gpu: Whether to refuse to run with more than one
            visible CUDA device.

    Returns:
        The device the run will use.

    Raises:
        RuntimeError: If no CUDA device is available, or if more than one is
            visible while the single-GPU rule is enforced.
    """
    if not torch.cuda.is_available():
        raise RuntimeError(
            "No CUDA device visible. Set CUDA_VISIBLE_DEVICES to one free GPU "
            "and rerun; training on CPU is not supported."
        )

    visible = torch.cuda.device_count()
    if enforce_single_gpu and visible > 1:
        raise RuntimeError(
            f"{visible} CUDA devices are visible. ml-server03 is shared: pin "
            "the run to a single free GPU, for example "
            "'export CUDA_VISIBLE_DEVICES=1', and rerun."
        )

    device = torch.device("cuda:0")
    name = torch.cuda.get_device_name(device)
    logger.info(f"Using device {device} ({name}); visible devices: {visible}")
    return device


def resolve_amp_dtype(amp_dtype: str) -> Optional[torch.dtype]:
    """Translate the configured mixed-precision name into a torch dtype.

    Args:
        amp_dtype: One of bf16, fp16 or none.

    Returns:
        The dtype for autocast, or None when mixed precision is disabled.

    Raises:
        ValueError: If the name is not recognised.
    """
    mapping = {"bf16": torch.bfloat16, "fp16": torch.float16, "none": None}
    if amp_dtype not in mapping:
        raise ValueError(f"Unknown amp_dtype '{amp_dtype}'; expected one of {list(mapping)}")
    return mapping[amp_dtype]


def assert_free_disk(path: Path, minimum_gb: float) -> float:
    """Verify the filesystem has room for the run to finish.

    Args:
        path: Any path on the target filesystem.
        minimum_gb: Required free space in gigabytes.

    Returns:
        Free space in gigabytes.

    Raises:
        RuntimeError: If free space is below the requirement.
    """
    path.mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(path).free / (1024 ** 3)
    if free_gb < minimum_gb:
        raise RuntimeError(
            f"Only {free_gb:.1f} GB free on the filesystem holding {path}; "
            f"{minimum_gb:.1f} GB required. Checkpoints for a billion-parameter "
            "model are large, and a run that fills the disk takes the shared "
            "server down with it. Free space or lower KEEP_LAST_N_CHECKPOINTS."
        )
    logger.info(f"Free disk at {path}: {free_gb:.1f} GB")
    return free_gb


def configure_logging(run_dir: Path, verbose: bool = True) -> Path:
    """Route loguru output to both the console and a run log file.

    Args:
        run_dir: Directory of the current run.
        verbose: Whether the console sink reports at debug level.

    Returns:
        Path of the log file.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "train.log"

    logger.remove()
    logger.add(
        sys.stderr,
        level="DEBUG" if verbose else "INFO",
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
    )
    logger.add(
        log_path,
        level="DEBUG",
        rotation="100 MB",
        retention=10,
        enqueue=True,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{line} | {message}",
    )
    logger.info(f"Logging to {log_path}")
    return log_path
