"""
Step 1: assert the corpus invariants that make a trained EER meaningful.
"""
from pathlib import Path
from typing import Dict, List, Set

from loguru import logger

from app.pipeline.training.schemas.leakage_audit_report import (
    LeakageAuditReport,
)
from app.pipeline.training.schemas.leakage_check_result import (
    LeakageCheckResult,
)
from app.pipeline.training.schemas.protocol_entry import ProtocolEntry
from app.pipeline.training.settings import settings
from app.pipeline.training.utils import protocol_io


class CorpusLeakageAuditor:
    """Verify a corpus is fit to train on before any GPU time is spent.

    The audits behind these invariants were established ad hoc in August 2026
    and never committed, so nothing reproducible backed the claims the data
    descriptor makes. Every augmentation tier was moreover regenerated after
    those audits ran, and two silent data-loss bugs had already destroyed two
    earlier runs without announcing themselves. This step re-asserts the
    invariants cheaply, from the protocol and metadata files alone, and writes
    a report that can be cited.

    Attributes:
        corpus_root: Corpus directory containing the LA tree.
        splits: Split names to audit.
        strict_filter_csv: Optional strict filter table to test the join.
    """

    SPLITS = ("train", "dev", "eval")
    MAX_OFFENDERS = 20

    def __init__(
        self,
        corpus_root: Path,
        strict_filter_csv: Path = None,
        splits: List[str] = None,
    ) -> None:
        """Initialize the auditor.

        Args:
            corpus_root: Corpus directory containing the LA tree.
            strict_filter_csv: Strict sentence-disjoint filter table. When
                supplied, the join is exercised as a non-fatal check.
            splits: Split names to audit. Defaults to all three.
        """
        self.corpus_root = Path(corpus_root)
        self.strict_filter_csv = Path(strict_filter_csv) if strict_filter_csv else None
        self.splits = list(splits) if splits else list(self.SPLITS)

    def execute(self) -> LeakageAuditReport:
        """Run every invariant and assemble the report.

        Returns:
            The audit report. Callers decide whether a failing report aborts
            the run; the report is written to disk either way.

        Raises:
            FileNotFoundError: If the corpus root or a protocol file is absent.
        """
        logger.info(f"Step {self.__class__.__name__}: Starting")
        self._assert_corpus_exists()

        entries: Dict[str, List[ProtocolEntry]] = {}
        for split in self.splits:
            entries[split] = protocol_io.read_metadata(
                protocol_io.metadata_path(self.corpus_root, split)
            )
            logger.info(f"  {split}: {len(entries[split]):,} metadata rows")

        checks: List[LeakageCheckResult] = [
            self._check_speaker_disjointness(entries),
            self._check_protocol_correspondence(entries),
            self._check_ondisk_correspondence(entries),
            self._check_clean_fraction_parity(entries),
            self._check_ordering_leak(entries),
            self._check_attack_coverage(entries),
            self._check_class_balance(entries),
        ]
        if self.strict_filter_csv:
            checks.append(self._check_strict_filter_join(entries))

        report = LeakageAuditReport(
            corpus_root=str(self.corpus_root),
            split_sizes={split: len(rows) for split, rows in entries.items()},
            bonafide_fraction={
                split: self._bonafide_fraction(rows) for split, rows in entries.items()
            },
            checks=checks,
            passed=all(check.passed for check in checks if check.fatal),
        )
        self._log_report(report)
        logger.info(f"Step {self.__class__.__name__}: Complete")
        return report

    def write_report(self, report: LeakageAuditReport, destination: Path) -> Path:
        """Persist the audit report as JSON.

        Args:
            report: Report to write.
            destination: Destination file path.

        Returns:
            The path written.
        """
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(report.model_dump_json(indent=2), encoding="utf-8")
        logger.info(f"Audit report written: {destination}")
        return destination

    def _assert_corpus_exists(self) -> None:
        """Verify the corpus directory exists and name the alternatives if not.

        Raises:
            FileNotFoundError: If the corpus root or its LA tree is absent.
        """
        if not (self.corpus_root / "LA").is_dir():
            siblings = []
            if self.corpus_root.parent.is_dir():
                siblings = sorted(
                    child.name
                    for child in self.corpus_root.parent.iterdir()
                    if child.is_dir()
                )
            raise FileNotFoundError(
                f"No LA tree under {self.corpus_root}. "
                f"Available siblings: {siblings}"
            )

    @staticmethod
    def _bonafide_fraction(rows: List[ProtocolEntry]) -> float:
        """Compute the bonafide proportion of a split.

        Args:
            rows: Protocol entries of the split.

        Returns:
            Bonafide rows divided by total rows, or zero for an empty split.
        """
        if not rows:
            return 0.0
        return sum(1 for row in rows if row.key == "bonafide") / len(rows)

    def _check_speaker_disjointness(
        self, entries: Dict[str, List[ProtocolEntry]]
    ) -> LeakageCheckResult:
        """Assert no speaker appears in more than one split.

        Args:
            entries: Protocol entries per split.

        Returns:
            The check result.
        """
        speakers: Dict[str, Set[str]] = {
            split: {row.speaker_id for row in rows} for split, rows in entries.items()
        }
        offenders: List[str] = []
        details: List[str] = []
        names = list(speakers)
        for index, left in enumerate(names):
            for right in names[index + 1 :]:
                shared = speakers[left] & speakers[right]
                details.append(f"{left}/{right}: {len(shared)} shared")
                offenders.extend(
                    f"{left}|{right}|{speaker}" for speaker in sorted(shared)
                )
        return LeakageCheckResult(
            name="speaker_disjointness",
            description=(
                "No speaker appears in more than one split, so no voice heard "
                "in training is scored at evaluation."
            ),
            passed=not offenders,
            detail="; ".join(details)
            + " | sizes: "
            + ", ".join(f"{split}={len(value)}" for split, value in speakers.items()),
            offenders=offenders[: self.MAX_OFFENDERS],
            fatal=True,
        )

    def _check_protocol_correspondence(
        self, entries: Dict[str, List[ProtocolEntry]]
    ) -> LeakageCheckResult:
        """Assert the protocol file and metadata CSV declare the same corpus.

        Args:
            entries: Protocol entries per split.

        Returns:
            The check result.
        """
        offenders: List[str] = []
        details: List[str] = []
        for split, rows in entries.items():
            protocol_rows = protocol_io.read_protocol(
                protocol_io.protocol_path(self.corpus_root, split)
            )
            protocol_map = {
                audio_id: (speaker_id, attack_id, key)
                for speaker_id, audio_id, attack_id, key in protocol_rows
            }
            metadata_map = {
                row.audio_id: (row.speaker_id, row.attack_id, row.key) for row in rows
            }
            only_protocol = set(protocol_map) - set(metadata_map)
            only_metadata = set(metadata_map) - set(protocol_map)
            mismatched = [
                audio_id
                for audio_id in set(protocol_map) & set(metadata_map)
                if protocol_map[audio_id] != metadata_map[audio_id]
            ]
            details.append(
                f"{split}: protocol={len(protocol_map):,} metadata={len(metadata_map):,} "
                f"protocol_only={len(only_protocol)} metadata_only={len(only_metadata)} "
                f"field_mismatch={len(mismatched)}"
            )
            offenders.extend(f"{split}|protocol_only|{i}" for i in sorted(only_protocol))
            offenders.extend(f"{split}|metadata_only|{i}" for i in sorted(only_metadata))
            offenders.extend(f"{split}|mismatch|{i}" for i in sorted(mismatched))
        return LeakageCheckResult(
            name="protocol_metadata_correspondence",
            description=(
                "The ASVspoof protocol and the MARSA metadata CSV declare "
                "exactly the same clips with the same labels."
            ),
            passed=not offenders,
            detail="; ".join(details),
            offenders=offenders[: self.MAX_OFFENDERS],
            fatal=True,
        )

    def _check_ondisk_correspondence(
        self, entries: Dict[str, List[ProtocolEntry]]
    ) -> LeakageCheckResult:
        """Assert every declared clip exists and no undeclared clip does.

        An undeclared clip on disk is the signature of a superseded run whose
        output was not cleared, which has happened before on this corpus.

        Args:
            entries: Protocol entries per split.

        Returns:
            The check result.
        """
        offenders: List[str] = []
        details: List[str] = []
        for split, rows in entries.items():
            directory = protocol_io.flac_dir(self.corpus_root, split)
            on_disk = {path.stem for path in directory.glob("*.flac")}
            declared = {row.audio_id for row in rows}
            missing = declared - on_disk
            orphans = on_disk - declared
            details.append(
                f"{split}: declared={len(declared):,} on_disk={len(on_disk):,} "
                f"missing={len(missing)} orphan={len(orphans)}"
            )
            offenders.extend(f"{split}|missing|{i}" for i in sorted(missing))
            offenders.extend(f"{split}|orphan|{i}" for i in sorted(orphans))
        return LeakageCheckResult(
            name="ondisk_correspondence",
            description=(
                "Every declared clip is present on disk and no undeclared "
                "clip survives from a superseded run."
            ),
            passed=not offenders,
            detail="; ".join(details),
            offenders=offenders[: self.MAX_OFFENDERS],
            fatal=True,
        )

    def _check_clean_fraction_parity(
        self, entries: Dict[str, List[ProtocolEntry]]
    ) -> LeakageCheckResult:
        """Assert augmentation carries no information about the label.

        The corpus is built so that both classes receive the same proportion
        of clean copies. If they diverge, a detector can reach a low error
        rate by learning which clips were augmented rather than which are
        spoofed.

        Args:
            entries: Protocol entries per split.

        Returns:
            The check result.
        """
        details: List[str] = []
        offenders: List[str] = []
        for split, rows in entries.items():
            fractions: Dict[str, float] = {}
            for label in ("bonafide", "spoof"):
                subset = [row for row in rows if row.key == label]
                if not subset:
                    continue
                clean = sum(1 for row in subset if row.aug_id == "-")
                fractions[label] = clean / len(subset)
            if len(fractions) < 2:
                continue
            gap = abs(fractions["bonafide"] - fractions["spoof"])
            details.append(
                f"{split}: bonafide_clean={fractions['bonafide']:.4f} "
                f"spoof_clean={fractions['spoof']:.4f} gap={gap:.4f}"
            )
            if gap > settings.BONAFIDE_FRACTION_TOLERANCE:
                offenders.append(f"{split}|clean_fraction_gap|{gap:.4f}")
        return LeakageCheckResult(
            name="clean_fraction_parity",
            description=(
                "Both classes carry the same proportion of clean copies, so "
                "augmentation status is not a label shortcut."
            ),
            passed=not offenders,
            detail="; ".join(details),
            offenders=offenders,
            fatal=True,
        )

    def _check_ordering_leak(
        self, entries: Dict[str, List[ProtocolEntry]]
    ) -> LeakageCheckResult:
        """Assert clip identifiers do not sort the corpus by label.

        Identifiers assigned class by class let a model, or a careless
        experimenter, recover the label from the file name alone. The test
        counts label runs along the identifier ordering and compares them with
        the count expected of a random arrangement of the same class sizes.

        Args:
            entries: Protocol entries per split.

        Returns:
            The check result.
        """
        details: List[str] = []
        offenders: List[str] = []
        for split, rows in entries.items():
            ordered = sorted(rows, key=lambda row: row.audio_id)
            labels = [row.label for row in ordered]
            if len(set(labels)) < 2:
                continue
            observed = 1 + sum(
                1 for index in range(1, len(labels)) if labels[index] != labels[index - 1]
            )
            positives = sum(labels)
            negatives = len(labels) - positives
            expected = 1.0 + (2.0 * positives * negatives) / len(labels)
            ratio = observed / expected
            details.append(
                f"{split}: runs={observed:,} expected={expected:,.0f} ratio={ratio:.3f}"
            )
            if ratio < settings.ORDERING_LEAK_MIN_RUNS_RATIO:
                offenders.append(f"{split}|run_ratio|{ratio:.3f}")
        return LeakageCheckResult(
            name="audio_id_ordering",
            description=(
                "Clip identifiers interleave the two classes, so identifier "
                "order carries no label information."
            ),
            passed=not offenders,
            detail="; ".join(details),
            offenders=offenders,
            fatal=True,
        )

    def _check_attack_coverage(
        self, entries: Dict[str, List[ProtocolEntry]]
    ) -> LeakageCheckResult:
        """Report whether the splits carry exactly the expected attack systems.

        Both directions matter. A missing system breaks leave-one-system-out
        for that system. An unexpected extra one is usually a naming
        inconsistency rather than a new generator, and it is more insidious:
        it silently splits one system across two rows of a per-attack table
        and lets held-out clips of that system leak back into training. This
        corpus shipped exactly such a defect, with Qwen recorded as ``qwen``
        for full spoof and ``qwen3tts`` for partial spoof.

        Args:
            entries: Protocol entries per split.

        Returns:
            The check result, non-fatal.
        """
        expected = set(settings.EXPECTED_ATTACK_SYSTEMS)
        details: List[str] = []
        offenders: List[str] = []
        for split, rows in entries.items():
            present = {row.attack_id for row in rows if row.key == "spoof"}
            families = {attack.split("-")[0] for attack in present}
            absent = sorted(expected - families)
            unexpected = sorted(families - expected)
            details.append(
                f"{split}: {len(present)} attack ids, {len(families)} systems"
            )
            offenders.extend(f"{split}|absent_system|{system}" for system in absent)
            offenders.extend(
                f"{split}|unexpected_system|{system}" for system in unexpected
            )
        return LeakageCheckResult(
            name="attack_coverage",
            description=(
                "Every split carries exactly the expected attack systems, "
                "neither missing one nor naming one twice."
            ),
            passed=not offenders,
            detail="; ".join(details),
            offenders=offenders,
            fatal=False,
        )

    def _check_class_balance(
        self, entries: Dict[str, List[ProtocolEntry]]
    ) -> LeakageCheckResult:
        """Report whether class balance is consistent across splits.

        A split whose bonafide fraction departs from the others is the
        signature of a discovery bug dropping one file family, which has
        happened on this corpus before.

        Args:
            entries: Protocol entries per split.

        Returns:
            The check result, non-fatal.
        """
        fractions = {split: self._bonafide_fraction(rows) for split, rows in entries.items()}
        if not fractions:
            return LeakageCheckResult(
                name="class_balance",
                description="Class balance is consistent across splits.",
                passed=True,
                detail="no splits",
                fatal=False,
            )
        overall = sum(fractions.values()) / len(fractions)
        offenders = [
            f"{split}|bonafide_fraction|{value:.4f}"
            for split, value in fractions.items()
            if abs(value - overall) > settings.BONAFIDE_FRACTION_TOLERANCE
        ]
        return LeakageCheckResult(
            name="class_balance",
            description="Class balance is consistent across splits.",
            passed=not offenders,
            detail=", ".join(f"{split}={value:.4f}" for split, value in fractions.items()),
            offenders=offenders,
            fatal=False,
        )

    def _check_strict_filter_join(
        self, entries: Dict[str, List[ProtocolEntry]]
    ) -> LeakageCheckResult:
        """Report whether the strict filter still joins onto this corpus.

        Args:
            entries: Protocol entries per split.

        Returns:
            The check result, non-fatal.
        """
        strict = protocol_io.read_strict_filter(self.strict_filter_csv)
        details: List[str] = []
        offenders: List[str] = []
        for split in ("dev", "eval"):
            rows = entries.get(split, [])
            if not rows:
                continue
            allowed = strict.get(split, set())
            matched = sum(1 for row in rows if row.source_file in allowed)
            share = matched / len(rows) if rows else 0.0
            details.append(f"{split}: strict={matched:,}/{len(rows):,} ({share:.1%})")
            if matched == 0:
                offenders.append(f"{split}|no_join|0")
        return LeakageCheckResult(
            name="strict_filter_join",
            description=(
                "The sentence-disjoint filter joins onto this corpus through "
                "source_file, so a strict EER can be reported."
            ),
            passed=not offenders,
            detail="; ".join(details),
            offenders=offenders,
            fatal=False,
        )

    @staticmethod
    def _log_report(report: LeakageAuditReport) -> None:
        """Print the audit outcome as an aligned table.

        Args:
            report: Report to print.
        """
        logger.info("-" * 78)
        logger.info(f"{'CHECK':<34} {'FATAL':<6} {'RESULT':<8} DETAIL")
        logger.info("-" * 78)
        for check in report.checks:
            verdict = "PASS" if check.passed else "FAIL"
            logger.info(
                f"{check.name:<34} {'yes' if check.fatal else 'no':<6} "
                f"{verdict:<8} {check.detail}"
            )
            for offender in check.offenders:
                logger.warning(f"    offender: {offender}")
        logger.info("-" * 78)
        logger.info(f"AUDIT {'PASSED' if report.passed else 'FAILED'}")
