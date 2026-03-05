"""
Step 6: Format Output to ASVspoof2019 LA

Converts validated samples to ASVspoof2019 LA format with protocol files.
"""
import json
import librosa
import soundfile as sf
from pathlib import Path
from loguru import logger
from tqdm import tqdm
from app.pipeline.fishgram_attack.settings import settings
from app.pipeline.fishgram_attack.schemas.formatting_result import FormattingResult


class OutputFormatter:
    """Formats output to ASVspoof2019 LA standard.

    Converts validated samples to FLAC format and generates protocol files
    following ASVspoof2019 LA naming conventions.
    """

    def __init__(
        self,
        output_dir: Path | None = None,
        system_id: str | None = None
    ):
        """Initialize output formatter.

        Args:
            output_dir: Output directory (default: from settings)
            system_id: Attack identifier (default: from settings)
        """
        self.output_dir = output_dir or settings.OUTPUT_DIR
        self.system_id = system_id or settings.FISHGRAM_SYSTEM_ID

    def execute(self) -> FormattingResult:
        """Format output to ASVspoof2019 LA structure.

        Returns:
            FormattingResult with output directory, protocol files, and counts

        Raises:
            Exception: If formatting fails
        """
        logger.info("Formatting output to ASVspoof2019 LA...")

        # Load validated samples
        validated_path = self.output_dir / "validated_samples.json"
        with open(validated_path, "r", encoding="utf-8") as f:
            validated = json.load(f)

        # Create LA directory structure
        la_dir = self.output_dir / "LA"
        for split in ["train", "dev", "eval"]:
            flac_dir = la_dir / f"ASVspoof2019_LA_{split}" / "flac"
            flac_dir.mkdir(parents=True, exist_ok=True)

        # Initialize counters and protocol entries
        counters = {
            "train": settings.AUDIO_ID_START_TRAIN,
            "dev": settings.AUDIO_ID_START_DEV,
            "eval": settings.AUDIO_ID_START_EVAL
        }
        protocols = {"train": [], "dev": [], "eval": []}
        counts = {"train": 0, "dev": 0, "eval": 0}

        logger.info(f"Formatting {len(validated)} validated samples...")

        for sample_id, sample_data in tqdm(validated.items(), desc="Formatting"):
            speaker_id = sample_data["speaker_id"]
            audio_path = Path(sample_data["audio_path"])
            split = sample_data["split"]

            # Map val → dev, keep others
            if split == "val":
                split = "dev"

            # Generate audio ID
            prefix_map = {"train": "LA_T", "dev": "LA_D", "eval": "LA_E"}
            prefix = prefix_map[split]
            audio_id = f"{prefix}_{counters[split]:07d}"
            counters[split] += 1
            counts[split] += 1

            try:
                # Load audio and convert to FLAC
                audio, sr = librosa.load(audio_path, sr=settings.SAMPLE_RATE)
                flac_path = la_dir / f"ASVspoof2019_LA_{split}" / "flac" / f"{audio_id}.flac"
                sf.write(flac_path, audio, sr, format="FLAC", subtype="PCM_16")

                # Add protocol entry: SPEAKER_ID AUDIO_ID SYSTEM_ID KEY
                protocol_entry = f"{speaker_id} {audio_id} {self.system_id} spoof"
                protocols[split].append(protocol_entry)

            except Exception as e:
                logger.error(f"Failed to format {sample_id}: {e}")

        # Write protocol files
        protocol_files = {}
        for split, entries in protocols.items():
            protocol_filename = f"ASVspoof2019.LA.cm.{split}.trl.txt"
            protocol_path = la_dir / f"ASVspoof2019_LA_{split}" / protocol_filename

            with open(protocol_path, "w", encoding="utf-8") as f:
                f.write("\n".join(entries))

            protocol_files[split] = protocol_path

        logger.info(f"✓ Output formatted to ASVspoof2019 LA")
        logger.info(f"  Output directory: {la_dir}")
        logger.info(f"  Train samples: {counts['train']}")
        logger.info(f"  Dev samples: {counts['dev']}")
        logger.info(f"  Eval samples: {counts['eval']}")

        return FormattingResult(
            output_directory=la_dir,
            protocol_files=protocol_files,
            total_samples=counts
        )
