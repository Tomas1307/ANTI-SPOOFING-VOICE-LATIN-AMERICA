"""
Step 5: Splice Audio

For each word selection plan, extracts the selected word segments from
the cloned audio and splices them into the bonafide audio at the aligned
positions. Produces up to 3 partially spoofed samples per bonafide
utterance (one per eligible tier).
"""
import json
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from loguru import logger
from tqdm import tqdm

from app.pipeline.partial_spoof.settings import settings
from app.pipeline.partial_spoof.schemas.splice_result import SpliceResult
from app.pipeline.partial_spoof.utils.splice_engine import splice_words


class AudioSplicer:
    """Splices cloned word segments into bonafide audio.

    For each selection plan (utterance + tier), extracts the selected
    words from the cloned audio and replaces the corresponding regions
    in the bonafide audio, applying crossfade at boundaries and handling
    duration mismatches.

    Attributes:
        output_dir: Directory for pipeline artifacts.
        attack_system_name: Uppercase name of the attack system for file naming.
    """

    def __init__(
        self,
        attack_system_name: str,
        output_dir: Path | None = None,
    ) -> None:
        """Initialize audio splicer.

        Args:
            attack_system_name: Uppercase attack system name (e.g., 'FISHGRAM').
            output_dir: Output directory (default: from settings).
        """
        self.attack_system_name = attack_system_name
        self.output_dir = output_dir or settings.OUTPUT_DIR

    def execute(self) -> SpliceResult:
        """Splice cloned word segments into bonafide audio for all selections.

        Returns:
            SpliceResult with splicing statistics.
        """
        logger.info("Step 5: Splicing cloned word segments into bonafide audio...")

        spliced_dir = self.output_dir / "spliced"
        spliced_dir.mkdir(parents=True, exist_ok=True)

        selection_path = self.output_dir / "word_selection_metadata.json"
        with open(selection_path, "r", encoding="utf-8") as f:
            selections = json.load(f)

        alignment_path = self.output_dir / "alignment_metadata.json"
        with open(alignment_path, "r", encoding="utf-8") as f:
            alignment_data = json.load(f)

        splice_metadata = {}
        failed_splices = []
        tier_counts = {}
        total_spoof_ratio = 0.0
        total_spliced = 0

        for sample_key, selection_entry in tqdm(selections.items(), desc="Splicing audio"):
            if sample_key not in alignment_data:
                failed_splices.append(sample_key)
                continue

            alignment = alignment_data[sample_key]
            bonafide_path = Path(alignment["bonafide_audio_path"])
            cloned_path = Path(alignment["cloned_audio_path"])

            if not bonafide_path.exists() or not cloned_path.exists():
                logger.warning(f"Audio files missing for {sample_key}")
                failed_splices.append(sample_key)
                continue

            try:
                bonafide_audio, _ = librosa.load(
                    str(bonafide_path), sr=settings.SAMPLE_RATE, mono=True
                )
                cloned_audio, _ = librosa.load(
                    str(cloned_path), sr=settings.SAMPLE_RATE, mono=True
                )
            except Exception as exc:
                logger.error(f"Failed to load audio for {sample_key}: {exc}")
                failed_splices.append(sample_key)
                continue

            for sel in selection_entry["selections"]:
                tier = sel["tier"]
                selected_indices = sel["selected_indices"]

                splice_key = f"{sample_key}_{tier}"
                output_filename = (
                    f"{self.attack_system_name}_PSW{tier[1]}_{sample_key}.wav"
                )
                output_path = spliced_dir / output_filename

                try:
                    spliced_audio, splice_details = splice_words(
                        bonafide_audio=bonafide_audio,
                        cloned_audio=cloned_audio,
                        bonafide_words=alignment["bonafide_words"],
                        cloned_words=alignment["cloned_words"],
                        selected_indices=selected_indices,
                        sample_rate=settings.SAMPLE_RATE,
                        crossfade_ms=settings.CROSSFADE_MS,
                        max_silence_steal_ms=settings.MAX_SILENCE_STEAL_MS,
                        max_stretch_ratio=settings.MAX_DURATION_STRETCH_RATIO,
                    )

                    sf.write(str(output_path), spliced_audio, settings.SAMPLE_RATE)

                    total_duration = len(spliced_audio) / settings.SAMPLE_RATE
                    spoofed_duration = sum(
                        d["cloned_end_s"] - d["cloned_start_s"]
                        for d in splice_details
                    )
                    spoof_ratio = spoofed_duration / total_duration if total_duration > 0 else 0.0

                    splice_metadata[splice_key] = {
                        "sample_id": splice_key,
                        "speaker_id": selection_entry["speaker_id"],
                        "split": selection_entry["split"],
                        "tier": tier,
                        "attack_system": self.attack_system_name,
                        "bonafide_audio_path": str(bonafide_path),
                        "cloned_audio_path": str(cloned_path),
                        "spliced_audio_path": str(output_path),
                        "transcript": selection_entry["transcript"],
                        "total_words": selection_entry["word_count"],
                        "spoofed_words": splice_details,
                        "spoof_word_ratio": len(splice_details) / selection_entry["word_count"],
                        "spoof_duration_ratio": spoof_ratio,
                        "total_duration_s": total_duration,
                    }

                    tier_counts[tier] = tier_counts.get(tier, 0) + 1
                    total_spoof_ratio += spoof_ratio
                    total_spliced += 1

                except Exception as exc:
                    logger.error(f"Splice failed for {splice_key}: {exc}")
                    failed_splices.append(splice_key)

        metadata_path = self.output_dir / "splice_metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(splice_metadata, f, ensure_ascii=False, indent=2)

        avg_ratio = total_spoof_ratio / total_spliced if total_spliced > 0 else 0.0

        logger.info(
            f"Step 5 complete: {total_spliced} spliced, "
            f"{len(failed_splices)} failed. Tiers: {tier_counts}. "
            f"Avg spoof duration ratio: {avg_ratio:.3f}"
        )

        return SpliceResult(
            metadata_path=metadata_path,
            total_spliced=total_spliced,
            failed_splices=failed_splices,
            avg_spoof_duration_ratio=avg_ratio,
            tier_counts=tier_counts,
        )
