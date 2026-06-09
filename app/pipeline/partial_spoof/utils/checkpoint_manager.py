"""
Atomic per-(attack, partition) checkpoint manager.

Persists progress to OUTPUT_DIR/.checkpoint.json after every successful
WAV commit, so a crashed or killed run can resume without re-doing work
or losing audit trail. Writes via tmp-file + os.replace for crash-safe
atomicity.
"""
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, List, Optional

from loguru import logger

from app.pipeline.partial_spoof.schemas.checkpoint_state import CheckpointState
from app.pipeline.partial_spoof.schemas.generation_failure import GenerationFailure


class CheckpointManager:
    """Atomic checkpoint I/O for a single (attack, partition) pipeline run.

    One instance manages one .checkpoint.json file inside the per-attack
    output directory. Sets (cloned, spliced, jittered) track successful
    work, the failed_generation map tracks recoverable Step 2 errors
    with retry counts. Every successful mark_* call writes to disk
    atomically (temp file plus os.replace) so a process kill mid-write
    never produces a partial checkpoint.

    The on-disk JSON form serialises sets as sorted lists so reloads
    are deterministic; the in-memory CheckpointState restores them
    back to sets.

    Attributes:
        attack: Attack identifier this checkpoint belongs to.
        partition: Partition identifier ('not_jittered' or 'jittered').
        output_dir: Directory holding the .checkpoint.json file.
        checkpoint_path: Full path to the .checkpoint.json file.
        state: In-memory CheckpointState mirror.
    """

    CHECKPOINT_FILENAME = ".checkpoint.json"
    TMP_SUFFIX = ".tmp"

    def __init__(
        self,
        attack: str,
        partition: str,
        output_dir: Path,
    ) -> None:
        """Initialise the manager and load any existing checkpoint.

        If a .checkpoint.json already exists in output_dir, it is loaded
        and the state restored. Otherwise an empty state is created (no
        file is written until the first mark_* call).

        Args:
            attack: Attack system identifier.
            partition: Partition identifier ('not_jittered' / 'jittered').
            output_dir: Directory in which to read/write .checkpoint.json.
        """
        self.attack = attack
        self.partition = partition
        self.output_dir = output_dir
        self.checkpoint_path = output_dir / self.CHECKPOINT_FILENAME
        self.state = self._load_or_initialise()

    def _load_or_initialise(self) -> CheckpointState:
        """Load checkpoint from disk if present, else build a fresh state.

        Returns:
            CheckpointState with restored sets/maps if the file exists,
            otherwise a clean instance keyed by (attack, partition).
        """
        if not self.checkpoint_path.exists():
            return CheckpointState(
                attack=self.attack,
                partition=self.partition,
            )

        try:
            with open(self.checkpoint_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(
                f"Checkpoint at {self.checkpoint_path} is unreadable "
                f"({exc}); starting fresh."
            )
            return CheckpointState(
                attack=self.attack,
                partition=self.partition,
            )

        failed = {
            sk: GenerationFailure(**failure)
            for sk, failure in payload.get("failed_generation", {}).items()
        }
        state = CheckpointState(
            attack=payload.get("attack", self.attack),
            partition=payload.get("partition", self.partition),
            cloned=set(payload.get("cloned", [])),
            spliced=set(payload.get("spliced", [])),
            jittered=set(payload.get("jittered", [])),
            failed_generation=failed,
        )

        if state.attack != self.attack or state.partition != self.partition:
            logger.warning(
                f"Checkpoint at {self.checkpoint_path} belongs to "
                f"{state.attack}/{state.partition} but manager is "
                f"{self.attack}/{self.partition}; ignoring stale checkpoint."
            )
            return CheckpointState(
                attack=self.attack,
                partition=self.partition,
            )

        logger.info(
            f"Resumed checkpoint {self.attack}/{self.partition}: "
            f"cloned={len(state.cloned)}, spliced={len(state.spliced)}, "
            f"jittered={len(state.jittered)}, "
            f"failed={len(state.failed_generation)}."
        )
        return state

    def save(self) -> None:
        """Atomically persist the current state to disk.

        Writes to .checkpoint.json.tmp then os.replace into the final
        path so a partial write never leaves a corrupted checkpoint.
        """
        self.state.last_updated = datetime.now(timezone.utc)
        payload = {
            "attack": self.state.attack,
            "partition": self.state.partition,
            "cloned": sorted(self.state.cloned),
            "spliced": sorted(self.state.spliced),
            "jittered": sorted(self.state.jittered),
            "failed_generation": {
                sk: {
                    "error": failure.error,
                    "retries": failure.retries,
                    "last_attempt_at": failure.last_attempt_at.isoformat(),
                }
                for sk, failure in self.state.failed_generation.items()
            },
            "last_updated": self.state.last_updated.isoformat(),
        }
        self.output_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = self.checkpoint_path.with_suffix(
            self.checkpoint_path.suffix + self.TMP_SUFFIX
        )
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        os.replace(tmp_path, self.checkpoint_path)

    def mark_cloned(self, sample_key: str) -> None:
        """Mark a sample as having a committed Step 2 clone WAV.

        Args:
            sample_key: The per-file primary key.
        """
        self.state.cloned.add(sample_key)
        if sample_key in self.state.failed_generation:
            del self.state.failed_generation[sample_key]
        self.save()

    def mark_spliced(self, sample_key: str) -> None:
        """Mark a sample as having a committed Step 5 spliced WAV.

        Args:
            sample_key: The per-file primary key, including tier suffix
                (e.g. 'arf_00295_TEXT_00001_W2').
        """
        self.state.spliced.add(sample_key)
        self.save()

    def mark_jittered(self, sample_key: str) -> None:
        """Mark a sample as having a committed Step 5b jittered WAV.

        Args:
            sample_key: The per-file primary key, including tier suffix.
        """
        self.state.jittered.add(sample_key)
        self.save()

    def is_cloned(self, sample_key: str) -> bool:
        """Check whether Step 2 has produced a clone for this sample."""
        return sample_key in self.state.cloned

    def is_spliced(self, sample_key: str) -> bool:
        """Check whether Step 5 has produced a spliced WAV for this sample."""
        return sample_key in self.state.spliced

    def is_jittered(self, sample_key: str) -> bool:
        """Check whether Step 5b has produced a jittered WAV for this sample."""
        return sample_key in self.state.jittered

    def record_failure(
        self,
        sample_key: str,
        error: str,
    ) -> int:
        """Register a recoverable Step 2 generation failure and bump retries.

        If the sample_key already has a failure entry the retry counter
        is incremented in place; otherwise a new entry is created. The
        return value is the new retry count and is suitable for the
        caller to gate against MAX_GENERATION_RETRIES.

        Args:
            sample_key: Sample that failed Step 2.
            error: Truncated error message (exception class + head).

        Returns:
            The updated retry counter for this sample.
        """
        existing = self.state.failed_generation.get(sample_key)
        if existing is None:
            failure = GenerationFailure(error=error, retries=1)
        else:
            failure = GenerationFailure(
                error=error,
                retries=existing.retries + 1,
            )
        self.state.failed_generation[sample_key] = failure
        self.save()
        return failure.retries

    def is_abandoned(self, sample_key: str, max_retries: int) -> bool:
        """Check whether a sample exceeded the retry budget.

        Args:
            sample_key: Sample to check.
            max_retries: Maximum retry attempts allowed.

        Returns:
            True if the sample failed and exceeded max_retries, else False.
        """
        failure = self.state.failed_generation.get(sample_key)
        if failure is None:
            return False
        return failure.retries > max_retries

    def pending_retries(self, max_retries: int) -> Iterator[str]:
        """Iterate sample_keys eligible for retry.

        Args:
            max_retries: Maximum retry attempts allowed.

        Yields:
            sample_keys whose retry count is below max_retries and that
            do not yet appear in `cloned`.
        """
        for sample_key, failure in self.state.failed_generation.items():
            if sample_key in self.state.cloned:
                continue
            if failure.retries <= max_retries:
                yield sample_key

    def abandoned_keys(self, max_retries: int) -> List[str]:
        """Return sample_keys that have exhausted the retry budget.

        Args:
            max_retries: Maximum retry attempts allowed.

        Returns:
            Sorted list of abandoned sample_keys.
        """
        return sorted(
            sk for sk in self.state.failed_generation
            if self.is_abandoned(sk, max_retries)
        )

    def reset(self) -> None:
        """Drop all in-memory state and delete the on-disk checkpoint.

        Used when starting a fresh run after a wipe.
        """
        self.state = CheckpointState(
            attack=self.attack,
            partition=self.partition,
        )
        if self.checkpoint_path.exists():
            self.checkpoint_path.unlink()

    def progress_summary(self) -> str:
        """Render a one-line progress string for logging.

        Returns:
            Human-readable progress summary.
        """
        return (
            f"{self.attack}/{self.partition}: "
            f"cloned={len(self.state.cloned)}, "
            f"spliced={len(self.state.spliced)}, "
            f"jittered={len(self.state.jittered)}, "
            f"failed={len(self.state.failed_generation)}"
        )

    @staticmethod
    def truncate_error(exc: BaseException, max_length: int = 200) -> str:
        """Format an exception into a short string suitable for the checkpoint.

        Args:
            exc: The exception instance.
            max_length: Maximum message length to keep.

        Returns:
            'ExceptionClassName: head of message' truncated to max_length.
        """
        msg = f"{exc.__class__.__name__}: {exc}"
        if len(msg) > max_length:
            return msg[: max_length - 3] + "..."
        return msg

    def get_failure(self, sample_key: str) -> Optional[GenerationFailure]:
        """Look up the failure record for a sample, if any.

        Args:
            sample_key: Sample to query.

        Returns:
            The GenerationFailure entry, or None if no failure recorded.
        """
        return self.state.failed_generation.get(sample_key)
