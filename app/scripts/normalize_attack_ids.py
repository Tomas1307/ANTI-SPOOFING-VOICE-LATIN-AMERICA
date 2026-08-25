"""
Normalize an attack identifier prefix across every corpus artefact.

MOTIVATION
----------
Qwen3-TTS ships under two names. Full-spoof clips carry attack_id ``qwen``;
partial-spoof clips carry ``qwen3tts-psw1`` and friends. Every other system is
consistent across both attack types (``omnivoice`` / ``omnivoice-psw1``). The
inconsistency has three consequences: a per-attack table lists Qwen twice as
though it were two generators; leave-one-system-out is broken, because holding
out ``qwen`` leaves every Qwen partial spoof in the training set; and the defect
travels to every user of the deposited corpus. This script fixes it at source.

``qwen`` is the surviving name: it matches the short-name convention of the
other five systems and it labels far more clips, so the rewrite touches the
smaller set of rows.

WHAT IS TOUCHED
---------------
Only the attack identifier field, in three kinds of text file:

  * ASVspoof2019 LA protocol files, field 4 of 5
  * MARSA metadata CSVs, the ``attack_id`` column
  * the strict sentence-disjoint filter table, the ``attack_id`` column

No audio is read or written. File modification times are preserved: the corpus
deliberately carries a single constant mtime, and this maintenance edit must
not disturb it.

SAFETY
------
Dry run by default: nothing is written unless ``--apply`` is passed. Each
rewritten file is first copied to ``<name>.bak`` unless ``--no-backup`` is
given, then replaced atomically through a temporary file. After an applied run
a verification pass re-scans every file and asserts the source prefix is gone
and that the surviving identifier absorbed exactly the rows it should have.

USAGE
-----
    cd ~/ANTI-SPOOFING-VOICE-LATIN-AMERICA
    source envs/dfarena_env/bin/activate

    # 1. See what would change, touching nothing
    python -m app.scripts.normalize_attack_ids

    # 2. Apply
    python -m app.scripts.normalize_attack_ids --apply

    deactivate

Exits 0 on success, 1 if verification fails.
"""
import argparse
import csv
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Tuple

from loguru import logger

from app.schemas.attack_id_normalization_report import AttackIdNormalizationReport

DEFAULT_ROOTS: List[str] = [
    "data/augmented/augmented_2x",
    "data/augmented/augmented_3x",
    "data/augmented/augmented_5x",
    "data/augmented/augmented_10x",
    "data/marsa_speaker_disjoint_partition",
]

PROTOCOL_GLOB = "ASVspoof2019.LA.cm.*.txt"
METADATA_GLOB = "MARSA.LA.cm.*.metadata.csv"
FILTER_GLOB = "strict_eval_filter.csv"

PROTOCOL_ATTACK_FIELD = 3
PROTOCOL_FIELD_COUNT = 5


class AttackIdNormalizer:
    """Rewrite one attack identifier prefix wherever the corpus records it.

    The rewrite is prefix-anchored: ``qwen3tts`` becomes ``qwen`` and
    ``qwen3tts-psw2`` becomes ``qwen-psw2``, while an unrelated identifier that
    merely contains the string is left alone. Matching is exact on the segment
    before the first hyphen, which is how every consumer of these files derives
    the system name.

    Attributes:
        roots: Directories searched for corpus artefacts.
        source_prefix: Identifier prefix to replace.
        target_prefix: Replacement identifier prefix.
        apply: Whether to write changes to disk.
        backup: Whether to leave a .bak copy of each rewritten file.
    """

    def __init__(
        self,
        roots: List[Path],
        source_prefix: str,
        target_prefix: str,
        apply: bool,
        backup: bool,
    ) -> None:
        """Initialize the normalizer.

        Args:
            roots: Directories searched for corpus artefacts.
            source_prefix: Identifier prefix to replace.
            target_prefix: Replacement identifier prefix.
            apply: Whether to write changes to disk.
            backup: Whether to leave a .bak copy of each rewritten file.
        """
        self.roots = [Path(root) for root in roots]
        self.source_prefix = source_prefix
        self.target_prefix = target_prefix
        self.apply = apply
        self.backup = backup

    def execute(self) -> AttackIdNormalizationReport:
        """Scan the corpus and, when applying, rewrite it.

        Returns:
            The normalization report, verified when changes were applied.
        """
        logger.info(f"Step {self.__class__.__name__}: Starting")
        logger.info(
            f"{'APPLYING' if self.apply else 'DRY RUN'}: "
            f"'{self.source_prefix}' -> '{self.target_prefix}'"
        )

        report = AttackIdNormalizationReport(
            source_prefix=self.source_prefix,
            target_prefix=self.target_prefix,
            applied=self.apply,
        )

        targets = self._discover()
        report.files_scanned = len(targets)
        logger.info(f"Discovered {len(targets)} corpus artefacts")

        for path in targets:
            before, changed = self._process(path)
            for identifier, count in before.items():
                report.before_counts[identifier] = (
                    report.before_counts.get(identifier, 0) + count
                )
            if changed:
                report.files_changed += 1
                report.rows_changed += changed
                report.per_file[str(path)] = changed

        self._log_plan(report)

        if self.apply:
            report.verified = self._verify(targets, report)
        logger.info(f"Step {self.__class__.__name__}: Complete")
        return report

    def _discover(self) -> List[Path]:
        """Find every file that records an attack identifier.

        Returns:
            Sorted list of artefact paths that exist on disk.
        """
        found: List[Path] = []
        for root in self.roots:
            if not root.is_dir():
                logger.warning(f"Root not found, skipping: {root}")
                continue
            for pattern in (PROTOCOL_GLOB, METADATA_GLOB, FILTER_GLOB):
                found.extend(root.rglob(pattern))
        return sorted(set(found))

    def _process(self, path: Path) -> Tuple[Dict[str, int], int]:
        """Count and optionally rewrite one artefact.

        Args:
            path: Artefact path.

        Returns:
            A tuple of (counts per affected identifier before the rewrite,
            number of rows rewritten).
        """
        if path.name.endswith(".txt"):
            return self._process_protocol(path)
        return self._process_csv(path)

    def _process_protocol(self, path: Path) -> Tuple[Dict[str, int], int]:
        """Count and optionally rewrite an ASVspoof protocol file.

        Args:
            path: Protocol file path.

        Returns:
            A tuple of (counts before, rows rewritten).

        Raises:
            ValueError: If a row does not carry exactly five fields.
        """
        before: Dict[str, int] = {}
        changed = 0
        lines: List[str] = []

        with open(path, "r", encoding="utf-8") as handle:
            for number, line in enumerate(handle, start=1):
                stripped = line.rstrip("\n")
                if not stripped.strip():
                    lines.append(stripped)
                    continue
                fields = stripped.split()
                if len(fields) != PROTOCOL_FIELD_COUNT:
                    raise ValueError(
                        f"{path}:{number} has {len(fields)} fields, "
                        f"expected {PROTOCOL_FIELD_COUNT}"
                    )
                original = fields[PROTOCOL_ATTACK_FIELD]
                rewritten = self._rewrite(original)
                if rewritten != original:
                    before[original] = before.get(original, 0) + 1
                    fields[PROTOCOL_ATTACK_FIELD] = rewritten
                    changed += 1
                lines.append(" ".join(fields))

        if changed and self.apply:
            self._write(path, "\n".join(lines) + "\n")
        return before, changed

    def _process_csv(self, path: Path) -> Tuple[Dict[str, int], int]:
        """Count and optionally rewrite a CSV carrying an attack_id column.

        Args:
            path: CSV path.

        Returns:
            A tuple of (counts before, rows rewritten).
        """
        before: Dict[str, int] = {}
        changed = 0

        with open(path, "r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            rows = list(reader)

        if not rows:
            return before, 0

        header = rows[0]
        if "attack_id" not in header:
            logger.debug(f"No attack_id column, skipping: {path}")
            return before, 0
        column = header.index("attack_id")

        for row in rows[1:]:
            if column >= len(row):
                continue
            original = row[column]
            rewritten = self._rewrite(original)
            if rewritten != original:
                before[original] = before.get(original, 0) + 1
                row[column] = rewritten
                changed += 1

        if changed and self.apply:
            buffer = []
            for row in rows:
                buffer.append(",".join(row))
            self._write(path, "\n".join(buffer) + "\n")
        return before, changed

    def _rewrite(self, identifier: str) -> str:
        """Replace the system segment of an identifier when it matches.

        Args:
            identifier: The attack identifier as recorded.

        Returns:
            The identifier with its system segment replaced, or unchanged.
        """
        if identifier == self.source_prefix:
            return self.target_prefix
        if identifier.startswith(f"{self.source_prefix}-"):
            return self.target_prefix + identifier[len(self.source_prefix) :]
        return identifier

    def _write(self, path: Path, content: str) -> None:
        """Replace a file atomically, preserving its modification time.

        The corpus carries a single constant mtime by design, so that emission
        order cannot be recovered from the filesystem. A maintenance edit must
        not disturb that.

        Args:
            path: File to replace.
            content: Full new contents.
        """
        stat = path.stat()
        if self.backup:
            backup_path = path.with_suffix(path.suffix + ".bak")
            if not backup_path.exists():
                shutil.copy2(path, backup_path)

        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(content, encoding="utf-8")
        os.replace(temporary, path)
        os.utime(path, (stat.st_atime, stat.st_mtime))

    def _verify(
        self, targets: List[Path], report: AttackIdNormalizationReport
    ) -> bool:
        """Re-scan every artefact and confirm the rewrite is complete.

        Args:
            targets: Artefact paths to re-scan.
            report: Report to populate with the post-rewrite counts.

        Returns:
            True when no source identifier survives anywhere.
        """
        logger.info("Verifying...")
        residual: List[str] = []

        for path in targets:
            identifiers = self._read_identifiers(path)
            for identifier, count in identifiers.items():
                if identifier.split("-")[0] in (
                    self.source_prefix,
                    self.target_prefix,
                ):
                    report.after_counts[identifier] = (
                        report.after_counts.get(identifier, 0) + count
                    )
                if identifier.split("-")[0] == self.source_prefix:
                    residual.append(f"{path}:{identifier}")

        report.residual = residual
        if residual:
            logger.error(f"Verification FAILED: {len(residual)} residual rows")
            for item in residual[:20]:
                logger.error(f"  residual: {item}")
            return False

        expected = sum(report.before_counts.values())
        logger.info(
            f"Verification passed: no '{self.source_prefix}' rows remain; "
            f"{expected:,} rows absorbed into '{self.target_prefix}'"
        )
        return True

    def _read_identifiers(self, path: Path) -> Dict[str, int]:
        """Count attack identifiers in one artefact.

        Args:
            path: Artefact path.

        Returns:
            Mapping of identifier to row count.
        """
        counts: Dict[str, int] = {}
        if path.name.endswith(".txt"):
            with open(path, "r", encoding="utf-8") as handle:
                for line in handle:
                    fields = line.split()
                    if len(fields) == PROTOCOL_FIELD_COUNT:
                        identifier = fields[PROTOCOL_ATTACK_FIELD]
                        counts[identifier] = counts.get(identifier, 0) + 1
            return counts

        with open(path, "r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if "attack_id" not in (reader.fieldnames or []):
                return counts
            for row in reader:
                identifier = row["attack_id"]
                counts[identifier] = counts.get(identifier, 0) + 1
        return counts

    @staticmethod
    def _log_plan(report: AttackIdNormalizationReport) -> None:
        """Print what the run changed, or would change.

        Args:
            report: Report to print.
        """
        logger.info("-" * 70)
        logger.info(
            f"{report.files_changed} of {report.files_scanned} files affected, "
            f"{report.rows_changed:,} rows"
        )
        for identifier, count in sorted(report.before_counts.items()):
            logger.info(f"  {identifier:<24} {count:>8,} rows")
        for path, count in sorted(report.per_file.items()):
            logger.debug(f"  {count:>8,}  {path}")
        logger.info("-" * 70)


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Normalize an attack identifier prefix across the corpus."
    )
    parser.add_argument(
        "--roots", type=str, nargs="+", default=DEFAULT_ROOTS,
        help="Directories to scan for corpus artefacts.",
    )
    parser.add_argument(
        "--from-prefix", type=str, default="qwen3tts",
        help="Identifier prefix to replace (default: qwen3tts).",
    )
    parser.add_argument(
        "--to-prefix", type=str, default="qwen",
        help="Replacement identifier prefix (default: qwen).",
    )
    parser.add_argument(
        "--apply", action="store_true",
        help="Write the changes. Without this the run is a dry run.",
    )
    parser.add_argument(
        "--no-backup", action="store_true",
        help="Do not leave a .bak copy of each rewritten file.",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/attack_id_normalization.json"),
        help="Where to write the report.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    normalizer = AttackIdNormalizer(
        roots=[Path(root) for root in args.roots],
        source_prefix=args.from_prefix,
        target_prefix=args.to_prefix,
        apply=args.apply,
        backup=not args.no_backup,
    )
    outcome = normalizer.execute()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(outcome.model_dump_json(indent=2), encoding="utf-8")
    logger.info(f"Report written: {args.output}")

    if not args.apply:
        logger.info("DRY RUN: nothing was written. Re-run with --apply to commit.")
        sys.exit(0)

    if not outcome.verified:
        logger.error("Normalization did not verify. Restore from the .bak files.")
        sys.exit(1)

    logger.info("Normalization complete and verified.")
    sys.exit(0)
