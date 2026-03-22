"""
Writes per-sample validation metrics to CSV for thesis analysis.

Each pipeline's Step 4 (QualityValidator) calls MetricsWriter after the
validation loop completes. The CSV includes ALL samples — both passed and
rejected — so that rejected samples and their partial metrics are not lost.

Output path: <output_dir>/metrics/<SYSTEM_ID>_validation.csv
The file is overwritten on each pipeline run (no append).
"""
import csv
from pathlib import Path
from typing import Any, Dict, List

from loguru import logger


_CSV_COLUMNS = [
    "sample_id",
    "speaker_id",
    "pipeline",
    "split",
    "duration_seconds",
    "wer",
    "cer",
    "nisqa_mos",
    "speaker_similarity",
    "status",
    "rejection_reason",
    "transcription",
    "audio_path",
]


class MetricsWriter:
    """Writes per-sample validation metrics to a CSV file.

    This is a stateless utility with a single static method. It merges data
    from three sources available in Step 4's execute() method:

    - ``generated``: all samples from generation_metadata.json (speaker_id,
      text, audio_path, duration, split).
    - ``validated``: passed samples with full metrics (wer, cer, nisqa_mos,
      speaker_similarity, transcription).
    - ``rejected``: rejected samples with sample_id, reason, and optionally
      partial metrics (wer/cer for WER/CER rejections).

    The CSV is written to ``<output_dir>/metrics/<system_id>_validation.csv``
    and overwrites any previous file at that path.
    """

    @staticmethod
    def write_validation_csv(
        output_dir: Path,
        system_id: str,
        generated: Dict[str, Any],
        validated: Dict[str, Any],
        rejected: List[Dict[str, Any]],
    ) -> Path:
        """Write all validation metrics to a CSV file.

        Args:
            output_dir: Pipeline output directory (e.g. data/chatterbox_output).
            system_id: Pipeline identifier (e.g. CHATTERBOX, FISHGRAM).
            generated: Full dict from generation_metadata.json keyed by sample_id.
            validated: Dict of passed samples keyed by sample_id, with metrics.
            rejected: List of rejected sample dicts with sample_id and reason.

        Returns:
            Path to the written CSV file.
        """
        metrics_dir = output_dir / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        csv_path = metrics_dir / f"{system_id}_validation.csv"

        rejected_lookup = {r["sample_id"]: r for r in rejected}

        rows = []
        for sample_id, sample_data in generated.items():
            row = {
                "sample_id": sample_id,
                "speaker_id": sample_data.get("speaker_id", ""),
                "pipeline": system_id,
                "split": sample_data.get("split", ""),
                "duration_seconds": sample_data.get("duration_seconds", ""),
                "audio_path": sample_data.get("audio_path", ""),
            }

            if sample_id in validated:
                v = validated[sample_id]
                row["status"] = "passed"
                row["rejection_reason"] = ""
                row["wer"] = v.get("wer", "")
                row["cer"] = v.get("cer", "")
                row["nisqa_mos"] = v.get("nisqa_mos", "")
                row["speaker_similarity"] = v.get("speaker_similarity", "")
                row["transcription"] = v.get("transcription", "")
            elif sample_id in rejected_lookup:
                r = rejected_lookup[sample_id]
                row["status"] = "rejected"
                row["rejection_reason"] = r.get("reason", "")
                row["wer"] = r.get("wer", "")
                row["cer"] = r.get("cer", "")
                row["nisqa_mos"] = ""
                row["speaker_similarity"] = ""
                row["transcription"] = r.get("transcription", "")
            else:
                row["status"] = "unknown"
                row["rejection_reason"] = "Sample not found in validated or rejected"
                row["wer"] = ""
                row["cer"] = ""
                row["nisqa_mos"] = ""
                row["speaker_similarity"] = ""
                row["transcription"] = ""

            rows.append(row)

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=_CSV_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)

        logger.info(
            f"Validation metrics CSV written: {csv_path} "
            f"({len(rows)} samples)"
        )
        return csv_path
