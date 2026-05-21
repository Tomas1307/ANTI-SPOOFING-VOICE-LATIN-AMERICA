"""
Step 7: Format Output to ASVspoof2019 LA Structure plus flat CSV exports.

Converts validated partial spoof samples into the standard ASVspoof2019
Logical Access directory structure with protocol files. Each sample is
converted to FLAC format and assigned a unique audio ID within the
tier-specific ID range (W1=12M, W2=13M, W3=14M for non-jittered;
W1=16M, W2=17M, W3=18M for jittered).

Additionally emits two flat CSV exports for downstream analysis and
corpus-level aggregation (consumed by the orchestrator):

    samples.csv        - one row per spliced audio with all key
                         metadata (paths, transcript, metrics, flags).
    spoofed_words.csv  - one row per spoofed word (boundary-label table).

Under the keep-bad-stuff principle, EVERY sample in quality_data is
emitted regardless of WER / NISQA / SIM; the quality_flag column lets
downstream training stratify on quality.
"""
import csv
import json
from pathlib import Path
from typing import Dict, List

import librosa
import soundfile as sf
from loguru import logger
from tqdm import tqdm

from app.pipeline.partial_spoof.settings import settings
from app.pipeline.partial_spoof.schemas.formatting_result import FormattingResult


SPLIT_PREFIXES = {
    "train": "LA_T_",
    "dev": "LA_D_",
    "eval": "LA_E_",
}

TIER_ID_SETTINGS = {
    "W1": "AUDIO_ID_START_W1",
    "W2": "AUDIO_ID_START_W2",
    "W3": "AUDIO_ID_START_W3",
}

TIER_ID_SETTINGS_JITTER = {
    "W1": "AUDIO_ID_START_W1_JITTER",
    "W2": "AUDIO_ID_START_W2_JITTER",
    "W3": "AUDIO_ID_START_W3_JITTER",
}


SAMPLES_CSV_FIELDS = [
    "sample_key",
    "speaker_id",
    "split",
    "partition",
    "tier",
    "attack",
    "bonafide_audio_path",
    "cloned_audio_path",
    "spliced_audio_path",
    "transcript",
    "total_words",
    "num_spoofed_words",
    "spoof_word_ratio",
    "spoof_duration_ratio",
    "total_duration_s",
    "wer",
    "cer",
    "nisqa",
    "ecapa_sim_clone",
    "ecapa_sim_final",
    "quality_flag",
    "has_jitter",
    "jitter_ops_count",
]

SPOOFED_WORDS_CSV_FIELDS = [
    "sample_key",
    "attack",
    "partition",
    "tier",
    "word",
    "word_index",
    "bonafide_start_s",
    "bonafide_end_s",
    "cloned_start_s",
    "cloned_end_s",
    "duration_ratio",
    "crossfade_ms",
    "effective_crossfade_ms",
    "splice_method",
    "margin_before_ms",
    "margin_after_ms",
]


class OutputFormatter:
    """Formats partial spoof samples into ASVspoof2019 LA structure.

    Creates the standard LA directory with train/dev/eval splits, converts
    audio to FLAC, assigns sequential audio IDs per tier, and writes
    protocol files with the 'partial_spoof' label. Also emits two flat
    CSV exports (samples.csv, spoofed_words.csv) for downstream analysis
    and corpus aggregation.

    Attributes:
        output_dir: Base output directory containing splice metadata.
        system_id_prefix: Attack system name prefix for protocol entries.
    """

    def __init__(
        self,
        system_id_prefix: str,
        output_dir: Path | None = None,
    ) -> None:
        """Initialize output formatter.

        Args:
            system_id_prefix: Attack system name (e.g., 'FISHGRAM').
            output_dir: Output directory (default: from settings).
        """
        self.system_id_prefix = system_id_prefix
        self.output_dir = output_dir or settings.OUTPUT_DIR

    def execute(self) -> FormattingResult:
        """Format all spliced samples into ASVspoof2019 LA structure.

        Returns:
            FormattingResult with output paths and sample counts.
        """
        logger.info("Step 7: Formatting output to ASVspoof2019 LA structure...")

        quality_path = self.output_dir / "splice_quality_metadata.json"
        splice_path = self.output_dir / "splice_metadata.json"

        metadata_path = quality_path if quality_path.exists() else splice_path
        with open(metadata_path, "r", encoding="utf-8") as f:
            quality_data = json.load(f)

        with open(splice_path, "r", encoding="utf-8") as f:
            splice_metadata = json.load(f)

        la_dir = self.output_dir / "LA"
        for split in ["train", "dev", "eval"]:
            (la_dir / f"ASVspoof2019_LA_{split}" / "flac").mkdir(parents=True, exist_ok=True)

        tier_id_settings = (
            TIER_ID_SETTINGS_JITTER if settings.ENABLE_BOUNDARY_JITTER else TIER_ID_SETTINGS
        )
        counters = {}
        for tier, setting_name in tier_id_settings.items():
            start_id = getattr(settings, setting_name)
            counters[(tier, "train")] = start_id
            counters[(tier, "dev")] = start_id
            counters[(tier, "eval")] = start_id

        protocols = {"train": [], "dev": [], "eval": []}
        counts = {"train": 0, "dev": 0, "eval": 0}
        clone_sim_map = self._load_clone_similarity_map()
        jitter_map = self._load_jitter_map()
        samples_rows: List[Dict] = []
        spoofed_words_rows: List[Dict] = []
        partition_label = settings.BONAFIDE_FILE_PARTITION
        attack_label = self.system_id_prefix.lower()

        for splice_key, entry in tqdm(splice_metadata.items(), desc="Formatting LA output"):
            if splice_key not in quality_data:
                continue

            quality_entry = quality_data[splice_key]
            if not quality_entry.get("passed", True):
                continue

            audio_path = Path(entry["spliced_audio_path"])
            if not audio_path.exists():
                logger.warning(f"Spliced audio not found: {audio_path}")
                continue

            split = entry["split"]
            if split == "val":
                split = "dev"

            tier = entry["tier"]
            speaker_id = entry["speaker_id"]

            counter_key = (tier, split)
            audio_id_num = counters[counter_key]
            counters[counter_key] += 1

            prefix = SPLIT_PREFIXES[split]
            audio_id = f"{prefix}{audio_id_num:07d}"

            jitter_suffix = "J" if settings.ENABLE_BOUNDARY_JITTER else ""
            system_id = f"{self.system_id_prefix}_PSW{tier[1]}{jitter_suffix}"

            try:
                audio, sr = librosa.load(str(audio_path), sr=settings.SAMPLE_RATE)
                flac_path = (
                    la_dir / f"ASVspoof2019_LA_{split}" / "flac" / f"{audio_id}.flac"
                )
                sf.write(str(flac_path), audio, sr, format="FLAC", subtype="PCM_16")
            except Exception as exc:
                logger.error(f"FLAC conversion failed for {splice_key}: {exc}")
                continue

            protocol_entry = f"{speaker_id} {audio_id} {system_id} partial_spoof"
            protocols[split].append(protocol_entry)
            counts[split] += 1

            jitter_info = jitter_map.get(splice_key, {})
            samples_rows.append(self._build_samples_row(
                splice_key=splice_key,
                entry=entry,
                quality_entry=quality_entry,
                split=split,
                partition=partition_label,
                attack=attack_label,
                clone_sim=clone_sim_map.get(splice_key),
                jitter_info=jitter_info,
            ))
            spoofed_words_rows.extend(self._build_spoofed_words_rows(
                splice_key=splice_key,
                entry=entry,
                partition=partition_label,
                attack=attack_label,
            ))

        protocol_files = {}
        for split, entries in protocols.items():
            protocol_filename = f"ASVspoof2019.LA.cm.{split}.trl.txt"
            protocol_path = la_dir / f"ASVspoof2019_LA_{split}" / protocol_filename
            with open(protocol_path, "w", encoding="utf-8") as f:
                f.write("\n".join(entries))
            protocol_files[split] = protocol_path

        detailed_metadata_path = la_dir / "partial_spoof_metadata.json"
        with open(detailed_metadata_path, "w", encoding="utf-8") as f:
            json.dump(splice_metadata, f, ensure_ascii=False, indent=2)

        self._write_csv(
            path=self.output_dir / "samples.csv",
            fieldnames=SAMPLES_CSV_FIELDS,
            rows=samples_rows,
        )
        self._write_csv(
            path=self.output_dir / "spoofed_words.csv",
            fieldnames=SPOOFED_WORDS_CSV_FIELDS,
            rows=spoofed_words_rows,
        )

        logger.info(
            f"Step 7 complete: LA output at {la_dir}. "
            f"Samples: {counts}. "
            f"samples.csv={len(samples_rows)}, "
            f"spoofed_words.csv={len(spoofed_words_rows)}."
        )

        return FormattingResult(
            output_directory=la_dir,
            protocol_files=protocol_files,
            total_samples=counts,
        )

    def _load_clone_similarity_map(self) -> Dict[str, float]:
        """Load clone_similarity_filter.json into a sample_key -> sim dict.

        Returns:
            Map from sample_key (without tier suffix) to clone ECAPA SIM.
            Empty dict if the file is absent.
        """
        path = self.output_dir / "clone_similarity_filter.json"
        if not path.exists():
            return {}
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return {key: float(entry.get("similarity", 0.0)) for key, entry in raw.items()}

    def _load_jitter_map(self) -> Dict[str, Dict]:
        """Load boundary_jitter_metadata.json into a splice_key -> info dict.

        Returns:
            Map from splice_key (sample_key plus tier) to a dict with
            has_jitter, jitter_ops_count, and the operation breakdown.
            Empty dict if the file is absent or jitter was disabled.
        """
        path = self.output_dir / "boundary_jitter_metadata.json"
        if not path.exists():
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(f"Could not parse jitter metadata at {path}: {exc}")
            return {}
        result: Dict[str, Dict] = {}
        per_sample = payload if isinstance(payload, dict) else {}
        for splice_key, entry in per_sample.items():
            ops = entry.get("operation_counts", {}) if isinstance(entry, dict) else {}
            applied = sum(int(v) for k, v in ops.items() if k != "natural")
            result[splice_key] = {
                "has_jitter": applied > 0,
                "jitter_ops_count": applied,
            }
        return result

    def _build_samples_row(
        self,
        splice_key: str,
        entry: Dict,
        quality_entry: Dict,
        split: str,
        partition: str,
        attack: str,
        clone_sim: float | None,
        jitter_info: Dict,
    ) -> Dict:
        """Flatten one splice + quality entry into a samples.csv row.

        Args:
            splice_key: Per-tier primary key (sample_key plus tier suffix).
            entry: Row from splice_metadata.json.
            quality_entry: Row from splice_quality_metadata.json.
            split: Normalised split label ('train', 'dev', 'eval').
            partition: 'not_jittered' or 'jittered'.
            attack: Lowercase attack identifier.
            clone_sim: Optional ECAPA SIM at Step 2 -> 3 gate.
            jitter_info: Output of _load_jitter_map for this key.

        Returns:
            Dict suitable for csv.DictWriter.writerow.
        """
        sample_key_no_tier = splice_key.rsplit("_", 1)[0]
        return {
            "sample_key": splice_key,
            "speaker_id": entry.get("speaker_id", ""),
            "split": split,
            "partition": partition,
            "tier": entry.get("tier", ""),
            "attack": attack,
            "bonafide_audio_path": entry.get("bonafide_audio_path", ""),
            "cloned_audio_path": entry.get("cloned_audio_path", ""),
            "spliced_audio_path": entry.get("spliced_audio_path", ""),
            "transcript": entry.get("transcript", ""),
            "total_words": entry.get("total_words", 0),
            "num_spoofed_words": len(entry.get("spoofed_words", [])),
            "spoof_word_ratio": entry.get("spoof_word_ratio", 0.0),
            "spoof_duration_ratio": entry.get("spoof_duration_ratio", 0.0),
            "total_duration_s": entry.get("total_duration_s", 0.0),
            "wer": quality_entry.get("wer", ""),
            "cer": quality_entry.get("cer", ""),
            "nisqa": quality_entry.get("nisqa_mos", ""),
            "ecapa_sim_clone": (
                f"{clone_sim:.4f}" if clone_sim is not None else ""
            ),
            "ecapa_sim_final": quality_entry.get("speaker_similarity", ""),
            "quality_flag": quality_entry.get("quality_flag", ""),
            "has_jitter": bool(jitter_info.get("has_jitter", False)),
            "jitter_ops_count": int(jitter_info.get("jitter_ops_count", 0)),
        }

    def _build_spoofed_words_rows(
        self,
        splice_key: str,
        entry: Dict,
        partition: str,
        attack: str,
    ) -> List[Dict]:
        """Expand one splice entry's spoofed_words list into CSV rows.

        Args:
            splice_key: Per-tier primary key.
            entry: Row from splice_metadata.json.
            partition: 'not_jittered' or 'jittered'.
            attack: Lowercase attack identifier.

        Returns:
            List of dicts (one per spoofed word) suitable for DictWriter.
        """
        rows: List[Dict] = []
        tier = entry.get("tier", "")
        for word_info in entry.get("spoofed_words", []):
            rows.append({
                "sample_key": splice_key,
                "attack": attack,
                "partition": partition,
                "tier": tier,
                "word": word_info.get("word", ""),
                "word_index": word_info.get("word_index", ""),
                "bonafide_start_s": word_info.get("bonafide_start_s", ""),
                "bonafide_end_s": word_info.get("bonafide_end_s", ""),
                "cloned_start_s": word_info.get("cloned_start_s", ""),
                "cloned_end_s": word_info.get("cloned_end_s", ""),
                "duration_ratio": word_info.get("duration_ratio", ""),
                "crossfade_ms": word_info.get("crossfade_ms", ""),
                "effective_crossfade_ms": word_info.get(
                    "effective_crossfade_ms", ""
                ),
                "splice_method": word_info.get("splice_method", ""),
                "margin_before_ms": word_info.get("margin_before_ms", ""),
                "margin_after_ms": word_info.get("margin_after_ms", ""),
            })
        return rows

    def _write_csv(
        self,
        path: Path,
        fieldnames: List[str],
        rows: List[Dict],
    ) -> None:
        """Write a list of dict rows to a CSV file.

        Args:
            path: Output CSV path; parent directory created if needed.
            fieldnames: Column order.
            rows: Row dicts in column order.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL,
            )
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
