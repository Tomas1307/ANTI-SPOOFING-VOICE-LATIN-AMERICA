"""
Step 7: Format Output to ASVspoof2019 LA Structure

Converts validated partial spoof samples into the standard ASVspoof2019
Logical Access directory structure with protocol files. Each sample is
converted to FLAC format and assigned a unique audio ID within the
tier-specific ID range (W1=12M, W2=13M, W3=14M).
"""
import json
from pathlib import Path

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


class OutputFormatter:
    """Formats partial spoof samples into ASVspoof2019 LA structure.

    Creates the standard LA directory with train/dev/eval splits, converts
    audio to FLAC, assigns sequential audio IDs per tier, and writes
    protocol files with the 'partial_spoof' label.

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

        for splice_key, entry in tqdm(splice_metadata.items(), desc="Formatting LA output"):
            if splice_key not in quality_data:
                continue

            if not quality_data[splice_key].get("passed", True):
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

        logger.info(
            f"Step 7 complete: LA output at {la_dir}. "
            f"Samples: {counts}"
        )

        return FormattingResult(
            output_directory=la_dir,
            protocol_files=protocol_files,
            total_samples=counts,
        )
