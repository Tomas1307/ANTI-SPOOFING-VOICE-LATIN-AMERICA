"""
Per-sample validation checkpoint for the Chatterbox quality step.

Caches each sample's validation outcome keyed by ``sample_id`` together with a
fingerprint (file size + mtime) of the WAV that was validated. On resume:

    - A sample whose WAV fingerprint is unchanged reuses its cached outcome,
      so an interrupted validation pass continues instead of restarting from
      zero, and a re-run after regenerating a subset only re-validates the
      changed files.
    - The checkpoint is written atomically (temp file + os.replace) and
      flushed periodically, so a crash mid-pass never corrupts it and loses at
      most the last unflushed window of work.

Because validation is the expensive STT + NISQA + ECAPA pass, this cache is
keyed by WAV identity rather than mere existence: a regenerated WAV (new size
or mtime) is always re-validated, while an untouched WAV is trusted.
"""
import json
import os
from pathlib import Path
from typing import Dict, Optional

from loguru import logger


class ValidationCheckpoint:
    """File-backed cache of validation outcomes keyed by WAV fingerprint.

    Attributes:
        path: Destination JSON file for the checkpoint state.
    """

    def __init__(self, path: Path) -> None:
        """Initialise the checkpoint.

        Args:
            path: Path to the checkpoint JSON file (created on first flush).
        """
        self.path = path
        self._records: Dict[str, dict] = {}

    def load(self) -> None:
        """Load the checkpoint from disk, tolerating a missing or corrupt file."""
        if not self.path.exists():
            self._records = {}
            return
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                self._records = json.load(f)
            logger.info(
                f"Loaded validation checkpoint: {len(self._records)} cached outcomes"
            )
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning(
                f"validation_checkpoint.json unreadable ({exc}); starting fresh"
            )
            self._records = {}

    def is_empty(self) -> bool:
        """Return True when no cached outcomes are loaded."""
        return not self._records

    def bootstrap_from_validated(self, validated_path: Path) -> None:
        """Seed the cache from a prior completed ``validated_samples.json``.

        Each prior passed sample is imported as a cached "passed" outcome only
        when its WAV has not been modified since that file was written (WAV
        mtime <= validated_samples.json mtime). A regenerated WAV is therefore
        excluded and will be validated fresh. Rejected samples were never
        persisted, so they are not imported and will be re-validated.

        Args:
            validated_path: Path to a prior ``validated_samples.json``.
        """
        if not validated_path.exists():
            return
        try:
            cutoff_mtime = validated_path.stat().st_mtime
            with open(validated_path, "r", encoding="utf-8") as f:
                prior = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning(f"Could not bootstrap from {validated_path} ({exc})")
            return

        imported = 0
        for sample_id, entry in prior.items():
            wav_path = Path(entry.get("audio_path", ""))
            fingerprint = self.fingerprint(wav_path)
            if fingerprint is None or fingerprint["mtime"] > cutoff_mtime:
                continue
            self._records[sample_id] = {
                "status": "passed",
                "fingerprint": fingerprint,
                "payload": entry,
            }
            imported += 1
        logger.info(
            f"Bootstrapped {imported} passed samples from prior validated_samples.json "
            f"(WAVs unchanged since {validated_path.name})"
        )

    @staticmethod
    def fingerprint(wav_path: Path) -> Optional[dict]:
        """Return a size + mtime fingerprint for a WAV, or None if absent.

        Args:
            wav_path: Path to the WAV file.

        Returns:
            A dict with integer ``size`` and rounded ``mtime``, or None when
            the file does not exist.
        """
        try:
            st = wav_path.stat()
        except OSError:
            return None
        return {"size": st.st_size, "mtime": round(st.st_mtime, 3)}

    def fresh(self, sample_id: str, wav_path: Path) -> bool:
        """Return True when a cached outcome matches the current WAV.

        Args:
            sample_id: Sample identifier.
            wav_path: Current WAV path for the sample.

        Returns:
            True when a record exists and its fingerprint equals the current
            WAV fingerprint (so the cached outcome may be reused).
        """
        record = self._records.get(sample_id)
        if record is None:
            return False
        return record.get("fingerprint") == self.fingerprint(wav_path)

    def outcome(self, sample_id: str) -> dict:
        """Return the cached record for a sample.

        Args:
            sample_id: Sample identifier.

        Returns:
            The cached record dict with ``status`` and ``payload`` keys.
        """
        return self._records[sample_id]

    def record(
        self, sample_id: str, status: str, payload: dict, wav_path: Path
    ) -> None:
        """Store a validation outcome for a sample.

        Args:
            sample_id: Sample identifier.
            status: Either "passed" or "rejected".
            payload: The validated entry (passed) or reject entry (rejected).
            wav_path: WAV path whose fingerprint is recorded.
        """
        self._records[sample_id] = {
            "status": status,
            "fingerprint": self.fingerprint(wav_path),
            "payload": payload,
        }

    def flush(self) -> None:
        """Write the checkpoint to disk atomically (temp file + replace)."""
        tmp_path = self.path.with_name(self.path.name + ".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(self._records, f, ensure_ascii=False)
        os.replace(str(tmp_path), str(self.path))
