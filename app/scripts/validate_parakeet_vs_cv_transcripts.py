"""
Validate Parakeet TDT transcripts against original Common Voice ground truth.

Full-spoof target text is sourced directly from Common Voice sentences
(see each attack pipeline's step_02_prepare_texts.py), so Parakeet plays no
role there. Partial-spoof, however, transcribes the bonafide audio itself
with Parakeet to obtain word-level timestamps for splice alignment, and
Common Voice's own sentence text cannot substitute for that (it carries no
per-word alignment). For the CV-origin slice of the bonafide corpus we do
have an independent ground truth, so this script measures how accurate the
Parakeet transcription stage is by comparing it against that ground truth.

Reads three artifacts produced by earlier pipelines and performs no audio
processing or model inference:
    1. selected_15340.tsv        (Mozilla speaker selection, Step 4)
    2. cv_speaker_mapping.json   (Mozilla speaker selection, Step 5)
    3. bonafide_transcripts_full.json (partial-spoof manifest pre-flight)

Usage on ml-server03 (no GPU required, CPU-only string comparison):
    source ~/ANTI-SPOOFING-VOICE-LATIN-AMERICA/envs/<any_env_with_jiwer>/bin/activate
    python -m app.scripts.validate_parakeet_vs_cv_transcripts
    deactivate
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

from loguru import logger

from app.pipeline.partial_spoof.settings import settings as partial_spoof_settings
from app.pipeline.partial_spoof.utils.sample_key_builder import SampleKeyBuilder
from app.pipeline.select_mozilla_speakers.settings import settings as mozilla_settings
from app.schemas.cv_parakeet_accent_stat import AccentValidationStat
from app.schemas.cv_parakeet_transcript_outlier import TranscriptOutlier
from app.schemas.cv_parakeet_validation_report import CVParakeetValidationReport
from app.utils.wer_cer import compute_cer, compute_wer

# Mirrors BonafideTranscriber.CACHED_TRANSCRIPTS_FILENAME. Duplicated as a
# literal rather than imported because importing step_01_transcribe_bonafide
# pulls in ParakeetTranscriber's NeMo/torch dependency chain, which this
# CPU-only validation script does not otherwise need.
CACHED_TRANSCRIPTS_FILENAME = "bonafide_transcripts_full.json"


class ParakeetVsCVValidator:
    """Compares Parakeet TDT transcripts against Common Voice ground truth.

    Restricted to the CV-origin subset of the bonafide corpus, where an
    independent ground-truth transcript exists. Produces a
    CVParakeetValidationReport with overall and per-accent WER/CER plus the
    highest-error samples for manual review.

    Attributes:
        selected_tsv_path: Path to selected_15340.tsv.
        speaker_mapping_path: Path to cv_speaker_mapping.json.
        cached_transcripts_path: Path to the cached full-corpus Parakeet
            transcripts JSON produced by the partial-spoof manifest pre-flight.
        report_output_path: Where the resulting report JSON is written.
        worst_n: Number of highest-WER samples to keep for manual review.
    """

    def __init__(
        self,
        selected_tsv_path: Optional[Path] = None,
        speaker_mapping_path: Optional[Path] = None,
        cached_transcripts_path: Optional[Path] = None,
        report_output_path: Optional[Path] = None,
        worst_n: int = 25,
    ) -> None:
        """Initialize the validator, defaulting to the pipelines' own settings.

        Args:
            selected_tsv_path: Override for selected_15340.tsv location.
            speaker_mapping_path: Override for cv_speaker_mapping.json location.
            cached_transcripts_path: Override for the cached Parakeet
                transcripts JSON location.
            report_output_path: Override for the output report JSON location.
            worst_n: Number of highest-WER samples to retain for review.
        """
        self.selected_tsv_path = (
            selected_tsv_path or mozilla_settings.OUTPUT_DIR / "selected_15340.tsv"
        )
        self.speaker_mapping_path = (
            speaker_mapping_path
            or mozilla_settings.OUTPUT_DIR / "cv_speaker_mapping.json"
        )
        self.cached_transcripts_path = (
            cached_transcripts_path
            or partial_spoof_settings.MANIFEST_PATH.parent / CACHED_TRANSCRIPTS_FILENAME
        )
        self.report_output_path = (
            report_output_path
            or partial_spoof_settings.MANIFEST_PATH.parent
            / "cv_parakeet_validation_report.json"
        )
        self.worst_n = worst_n

    def run(self) -> CVParakeetValidationReport:
        """Execute the comparison end to end and persist the report.

        Returns:
            The computed CVParakeetValidationReport.

        Raises:
            FileNotFoundError: If any of the three required input artifacts
                is missing.
            KeyError: If selected_15340.tsv does not carry a 'sentence' column.
        """
        speaker_mapping = self._load_speaker_mapping()
        cv_rows = self._load_cv_rows()
        cached_transcripts = self._load_cached_transcripts()

        matched_pairs: List[dict] = []
        missing_samples = 0

        for row in cv_rows:
            mapping_entry = speaker_mapping.get(row["client_id"])
            if mapping_entry is None:
                missing_samples += 1
                continue

            speaker_id = mapping_entry["speaker_id"]
            sample_key = SampleKeyBuilder.build(speaker_id, Path(row["path"]).stem)
            transcript_entry = cached_transcripts.get(sample_key)
            if transcript_entry is None:
                missing_samples += 1
                continue

            cv_sentence = row["sentence"]
            parakeet_transcript = transcript_entry["transcript"]
            matched_pairs.append(
                {
                    "sample_key": sample_key,
                    "speaker_id": speaker_id,
                    "accent": mapping_entry.get("accent", "unknown"),
                    "wer": compute_wer(reference=cv_sentence, hypothesis=parakeet_transcript),
                    "cer": compute_cer(reference=cv_sentence, hypothesis=parakeet_transcript),
                    "cv_sentence": cv_sentence,
                    "parakeet_transcript": parakeet_transcript,
                }
            )

        report = self._build_report(len(cv_rows), matched_pairs, missing_samples)
        self._save_report(report)
        self._log_summary(report)
        return report

    def _load_speaker_mapping(self) -> Dict[str, dict]:
        """Load the CV client_id -> speaker_id/accent mapping.

        Returns:
            Mapping keyed by Common Voice client_id.

        Raises:
            FileNotFoundError: If speaker_mapping_path does not exist.
        """
        if not self.speaker_mapping_path.exists():
            raise FileNotFoundError(
                f"Speaker mapping not found at {self.speaker_mapping_path}."
            )
        with open(self.speaker_mapping_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _load_cv_rows(self) -> List[dict]:
        """Load the selected Common Voice rows with their ground-truth sentence.

        Returns:
            List of row dicts from selected_15340.tsv.

        Raises:
            FileNotFoundError: If selected_tsv_path does not exist.
            KeyError: If the 'sentence' column is absent.
        """
        if not self.selected_tsv_path.exists():
            raise FileNotFoundError(
                f"Selected CV samples not found at {self.selected_tsv_path}."
            )
        with open(self.selected_tsv_path, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f, delimiter="\t"))
        if rows and "sentence" not in rows[0]:
            raise KeyError(
                f"'sentence' column missing from {self.selected_tsv_path}; "
                f"available columns: {sorted(rows[0].keys())}"
            )
        return rows

    def _load_cached_transcripts(self) -> Dict[str, dict]:
        """Load the cached full-corpus Parakeet transcripts.

        Returns:
            Mapping from sample_key to transcript entry.

        Raises:
            FileNotFoundError: If cached_transcripts_path does not exist.
        """
        if not self.cached_transcripts_path.exists():
            raise FileNotFoundError(
                f"Cached Parakeet transcripts not found at "
                f"{self.cached_transcripts_path}. Run "
                "app/scripts/generate_partial_spoof_manifest.py first."
            )
        with open(self.cached_transcripts_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _build_report(
        self,
        total_cv_samples: int,
        matched_pairs: List[dict],
        missing_samples: int,
    ) -> CVParakeetValidationReport:
        """Assemble the final report from matched sample pairs.

        Args:
            total_cv_samples: Total rows read from selected_15340.tsv.
            matched_pairs: Per-utterance WER/CER records.
            missing_samples: Rows that could not be matched to a cached
                Parakeet transcript.

        Returns:
            Populated CVParakeetValidationReport.
        """
        overall_wer = (
            sum(p["wer"] for p in matched_pairs) / len(matched_pairs)
            if matched_pairs
            else 0.0
        )
        overall_cer = (
            sum(p["cer"] for p in matched_pairs) / len(matched_pairs)
            if matched_pairs
            else 0.0
        )

        return CVParakeetValidationReport(
            total_cv_samples=total_cv_samples,
            matched_samples=len(matched_pairs),
            missing_samples=missing_samples,
            overall_wer=overall_wer,
            overall_cer=overall_cer,
            per_accent=self._aggregate_by_accent(matched_pairs),
            worst_outliers=self._select_worst_outliers(matched_pairs),
        )

    def _aggregate_by_accent(self, matched_pairs: List[dict]) -> List[AccentValidationStat]:
        """Group matched pairs by accent and average their WER/CER.

        Args:
            matched_pairs: Per-utterance WER/CER records.

        Returns:
            One AccentValidationStat per accent, sorted alphabetically.
        """
        grouped: Dict[str, List[dict]] = defaultdict(list)
        for pair in matched_pairs:
            grouped[pair["accent"]].append(pair)

        return [
            AccentValidationStat(
                accent=accent,
                sample_count=len(pairs),
                mean_wer=sum(p["wer"] for p in pairs) / len(pairs),
                mean_cer=sum(p["cer"] for p in pairs) / len(pairs),
            )
            for accent, pairs in sorted(grouped.items())
        ]

    def _select_worst_outliers(self, matched_pairs: List[dict]) -> List[TranscriptOutlier]:
        """Select the highest-WER matched pairs for manual review.

        Args:
            matched_pairs: Per-utterance WER/CER records.

        Returns:
            Up to worst_n TranscriptOutlier entries, sorted by descending WER.
        """
        ranked = sorted(matched_pairs, key=lambda p: p["wer"], reverse=True)
        return [TranscriptOutlier(**pair) for pair in ranked[: self.worst_n]]

    def _save_report(self, report: CVParakeetValidationReport) -> None:
        """Write the report to disk as JSON.

        Args:
            report: The computed report.
        """
        self.report_output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.report_output_path, "w", encoding="utf-8") as f:
            f.write(report.model_dump_json(indent=2))

    def _log_summary(self, report: CVParakeetValidationReport) -> None:
        """Log a human-readable summary of the report.

        Args:
            report: The computed report.
        """
        logger.info("=" * 70)
        logger.info("PARAKEET vs COMMON VOICE GROUND TRUTH - VALIDATION SUMMARY")
        logger.info("=" * 70)
        logger.info(f"Total CV-origin samples : {report.total_cv_samples}")
        logger.info(f"Matched samples         : {report.matched_samples}")
        logger.info(f"Missing samples         : {report.missing_samples}")
        logger.info(f"Overall WER             : {report.overall_wer:.4f}")
        logger.info(f"Overall CER             : {report.overall_cer:.4f}")
        logger.info("Per-accent WER/CER:")
        for stat in report.per_accent:
            logger.info(
                f"  {stat.accent:12s} n={stat.sample_count:5d}  "
                f"WER={stat.mean_wer:.4f}  CER={stat.mean_cer:.4f}"
            )
        logger.info(f"Report written to: {self.report_output_path}")
        logger.info("=" * 70)


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Validate Parakeet TDT transcripts against original Common "
            "Voice ground truth for the CV-origin bonafide subset."
        )
    )
    parser.add_argument(
        "--worst-n",
        type=int,
        default=25,
        help="Number of highest-WER samples to keep for manual review (default: 25).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Override the output report JSON path.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    validator = ParakeetVsCVValidator(worst_n=args.worst_n, report_output_path=args.output)
    validator.run()
