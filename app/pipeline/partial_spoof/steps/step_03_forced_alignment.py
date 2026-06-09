"""
Step 3: Forced Alignment

Runs forced alignment on both bonafide and cloned audio to obtain
word-level timestamps for each version. The bonafide timestamps come
from Parakeet TDT (already computed in Step 1), while cloned audio
timestamps are computed using the configured alignment engine.

The alignment output enables precise word-level extraction in Step 5.
"""
import json
from pathlib import Path

from loguru import logger
from tqdm import tqdm

from app.pipeline.partial_spoof.settings import settings
from app.pipeline.partial_spoof.schemas.alignment_result import AlignmentResult
from app.utils.parakeet_transcriber import ParakeetTranscriber


class ForcedAligner:
    """Performs forced alignment on bonafide and cloned audio pairs.

    For bonafide audio, reuses word timestamps from Step 1 (Parakeet TDT).
    For cloned audio, runs Parakeet TDT transcription with timestamps,
    then aligns the resulting words to the known transcript.

    Attributes:
        output_dir: Directory for pipeline artifacts.
    """

    def __init__(self, output_dir: Path | None = None) -> None:
        """Initialize forced aligner.

        Args:
            output_dir: Output directory (default: from settings).
        """
        self.output_dir = output_dir or settings.OUTPUT_DIR

    def execute(self) -> AlignmentResult:
        """Align bonafide and cloned audio pairs.

        Loads bonafide transcripts (with pre-computed timestamps from Step 1),
        runs Parakeet alignment on cloned audio, and saves combined alignment
        metadata.

        Returns:
            AlignmentResult with alignment statistics.
        """
        logger.info("Step 3: Running forced alignment on bonafide and cloned audio...")

        transcripts_path = self.output_dir / "bonafide_transcripts.json"
        with open(transcripts_path, "r", encoding="utf-8") as f:
            transcripts = json.load(f)

        cloned_metadata_path = self.output_dir / "cloned_generation_metadata.json"
        with open(cloned_metadata_path, "r", encoding="utf-8") as f:
            cloned_metadata = json.load(f)

        transcriber = ParakeetTranscriber()
        transcriber.load(model_id=settings.PARAKEET_MODEL_ID, device=settings.DEVICE)

        alignment_path = self.output_dir / "alignment_metadata.json"
        if alignment_path.exists():
            try:
                with open(alignment_path, "r", encoding="utf-8") as f:
                    alignment_data = json.load(f)
                logger.info(
                    f"Loaded {len(alignment_data)} existing alignments; new "
                    "samples will be appended and any re-aligned samples overwritten."
                )
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning(
                    f"alignment_metadata.json unreadable ({exc}); starting fresh"
                )
                alignment_data = {}
        else:
            alignment_data = {}

        total_aligned = 0
        failed_alignments = []
        total_words = 0

        for sample_key in tqdm(transcripts, desc="Aligning audio pairs"):
            if sample_key not in cloned_metadata:
                failed_alignments.append(sample_key)
                continue

            bonafide_entry = transcripts[sample_key]
            cloned_entry = cloned_metadata[sample_key]

            bonafide_timestamps = bonafide_entry.get("word_timestamps", [])
            if not bonafide_timestamps:
                logger.warning(f"No bonafide timestamps for {sample_key}, skipping.")
                failed_alignments.append(sample_key)
                continue

            cloned_audio_path = Path(cloned_entry["audio_path"])
            if not cloned_audio_path.exists():
                logger.warning(f"Cloned audio not found: {cloned_audio_path}")
                failed_alignments.append(sample_key)
                continue

            try:
                cloned_text, cloned_word_ts = transcriber.transcribe_with_timestamps(
                    cloned_audio_path
                )

                cloned_timestamps = [
                    {"word": wt.word, "start": wt.start, "end": wt.end}
                    for wt in cloned_word_ts
                ]

                alignment_data[sample_key] = {
                    "speaker_id": bonafide_entry["speaker_id"],
                    "split": bonafide_entry["split"],
                    "transcript": bonafide_entry["transcript"],
                    "bonafide_audio_path": bonafide_entry["audio_path"],
                    "cloned_audio_path": str(cloned_audio_path),
                    "bonafide_words": bonafide_timestamps,
                    "cloned_words": cloned_timestamps,
                    "cloned_transcript": cloned_text,
                    "word_count": bonafide_entry["word_count"],
                }

                total_aligned += 1
                total_words += len(bonafide_timestamps)

            except Exception as exc:
                logger.error(f"Alignment failed for {sample_key}: {exc}")
                failed_alignments.append(sample_key)

        with open(alignment_path, "w", encoding="utf-8") as f:
            json.dump(alignment_data, f, ensure_ascii=False, indent=2)

        avg_words = total_words / total_aligned if total_aligned > 0 else 0.0

        logger.info(
            f"Step 3 complete: {total_aligned} pairs aligned, "
            f"{len(failed_alignments)} failed, avg {avg_words:.1f} words/utterance."
        )

        return AlignmentResult(
            alignment_path=alignment_path,
            total_aligned=total_aligned,
            failed_alignments=failed_alignments,
            avg_words_per_utterance=avg_words,
        )
