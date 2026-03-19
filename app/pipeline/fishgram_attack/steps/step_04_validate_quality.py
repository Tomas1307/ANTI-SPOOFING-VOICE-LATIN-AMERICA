"""
Step 4: Validate Quality with Silence Detection and Transcription Accuracy.

Validates synthetic speech using:
  1. Duration range check: rejects clips outside MIN_AUDIO_DURATION / MAX_AUDIO_DURATION.
  2. Silence detection: rejects near-silent output from Fish Speech.
  3. Parakeet TDT transcription with word-level timestamps.
  4. Spurious prefix trimming: detects and removes extra words hallucinated
     at the beginning of the audio, then re-transcribes.
  5. WER/CER computation against the original text. Target is 0.0 for both.
     Samples exceeding WER_MAX_ACCEPTABLE or CER_MAX_ACCEPTABLE are rejected.
"""
import json
import librosa
import numpy as np
from pathlib import Path
from loguru import logger
from tqdm import tqdm

from app.pipeline.fishgram_attack.settings import settings
from app.pipeline.fishgram_attack.schemas.validation_result import ValidationResult
from app.pipeline.fishgram_attack.utils.quality_metrics import detect_silence
from app.utils.parakeet_transcriber import ParakeetTranscriber
from app.utils.prefix_trimmer import detect_prefix_trim_point, trim_audio_prefix
from app.utils.wer_cer import compute_cer, compute_wer


class QualityValidator:
    """Validates quality of generated Fish Speech synthetic speech.

    Applies silence/duration checks, spurious prefix trimming via
    Parakeet TDT word timestamps, and WER/CER transcription accuracy checks.

    Attributes:
        output_dir: Directory containing generation metadata and audio files.
        wer_max: Hard WER rejection ceiling (samples above this are rejected).
        cer_max: Hard CER rejection ceiling (samples above this are rejected).
    """

    def __init__(
        self,
        output_dir: Path | None = None,
        wer_max: float | None = None,
        cer_max: float | None = None,
    ):
        """Initialize quality validator.

        Args:
            output_dir: Output directory (default: from settings).
            wer_max: Hard WER rejection ceiling (default: from settings).
            cer_max: Hard CER rejection ceiling (default: from settings).
        """
        self.output_dir = output_dir or settings.OUTPUT_DIR
        self.wer_max = wer_max if wer_max is not None else settings.WER_MAX_ACCEPTABLE
        self.cer_max = cer_max if cer_max is not None else settings.CER_MAX_ACCEPTABLE

    def execute(self) -> ValidationResult:
        """Validate quality of all generated FishGram samples.

        Validation sequence per sample:
            1. Check audio file exists.
            2. Load audio; run duration range check and silence detection.
            3. Transcribe with Parakeet TDT (word-level timestamps enabled).
            4. Detect and trim spurious prefix if present; re-transcribe trimmed audio.
            5. Compute WER and CER against original text.
            6. Accept if WER <= wer_max and CER <= cer_max.

        Returns:
            ValidationResult with validated sample path, stats, WER/CER averages,
            and prefix trim count.

        Raises:
            FileNotFoundError: If generation_metadata.json is missing.
            RuntimeError: If Parakeet model fails to load.
        """
        logger.info("Step 4: Validate quality (Parakeet STT + WER/CER)...")
        logger.info(f"  WER hard ceiling : {self.wer_max:.2f}")
        logger.info(f"  CER hard ceiling : {self.cer_max:.2f}")
        logger.info("  Target (ideal)   : WER=0.0, CER=0.0")

        gen_metadata_path = self.output_dir / "generation_metadata.json"
        with open(gen_metadata_path, "r", encoding="utf-8") as f:
            generated = json.load(f)

        transcriber = ParakeetTranscriber()
        transcriber.load(model_id=settings.PARAKEET_MODEL_ID, device=settings.DEVICE)

        validated = {}
        rejected = []
        wer_scores = []
        cer_scores = []
        prefix_trim_count = 0

        logger.info(f"Validating {len(generated)} samples...")

        for sample_id, sample_data in tqdm(generated.items(), desc="Validating"):
            audio_path = Path(sample_data["audio_path"])
            text = sample_data["text"]

            if not audio_path.exists():
                rejected.append({"sample_id": sample_id, "reason": "Audio file not found"})
                continue

            audio, sr = librosa.load(audio_path, sr=settings.SAMPLE_RATE)
            audio_duration = len(audio) / sr

            if audio_duration < settings.MIN_AUDIO_DURATION or audio_duration > settings.MAX_AUDIO_DURATION:
                rejected.append({
                    "sample_id": sample_id,
                    "audio_duration": float(audio_duration),
                    "reason": f"Duration anomaly: {audio_duration:.1f}s",
                })
                continue

            if detect_silence(audio, sample_rate=settings.SAMPLE_RATE):
                rejected.append({
                    "sample_id": sample_id,
                    "audio_duration": float(audio_duration),
                    "reason": "Near-silent output",
                })
                continue

            transcription, word_timestamps = transcriber.transcribe_with_timestamps(audio_path)

            trim_seconds = detect_prefix_trim_point(word_timestamps, text)
            if trim_seconds > 0.0:
                trim_audio_prefix(audio_path, trim_seconds, audio_path)
                transcription, word_timestamps = transcriber.transcribe_with_timestamps(audio_path)
                prefix_trim_count += 1

            sample_wer = compute_wer(text, transcription)
            sample_cer = compute_cer(text, transcription)

            logger.debug(
                f'{sample_id}: WER={sample_wer:.3f} CER={sample_cer:.3f} '
                f'transcript="{transcription[:60]}"'
            )

            if sample_wer <= self.wer_max and sample_cer <= self.cer_max:
                validated[sample_id] = sample_data.copy()
                validated[sample_id]["wer"] = float(sample_wer)
                validated[sample_id]["cer"] = float(sample_cer)
                validated[sample_id]["transcription"] = transcription
                wer_scores.append(sample_wer)
                cer_scores.append(sample_cer)
            else:
                reasons = []
                if sample_wer > self.wer_max:
                    reasons.append(f"WER {sample_wer:.3f} > {self.wer_max:.3f}")
                if sample_cer > self.cer_max:
                    reasons.append(f"CER {sample_cer:.3f} > {self.cer_max:.3f}")
                rejected.append({
                    "sample_id": sample_id,
                    "wer": float(sample_wer),
                    "cer": float(sample_cer),
                    "transcription": transcription,
                    "reason": "; ".join(reasons),
                })

        validated_path = self.output_dir / "validated_samples.json"
        with open(validated_path, "w", encoding="utf-8") as f:
            json.dump(validated, f, indent=2, ensure_ascii=False)

        avg_wer = float(np.mean(wer_scores)) if wer_scores else 0.0
        avg_cer = float(np.mean(cer_scores)) if cer_scores else 0.0
        pass_rate = (100.0 * len(validated) / len(generated)) if generated else 0.0

        logger.info("Validation complete")
        logger.info(f"  Passed          : {len(validated)}/{len(generated)} ({pass_rate:.1f}%)")
        logger.info(f"  Rejected        : {len(rejected)}")
        logger.info(f"  Prefix trims    : {prefix_trim_count}")
        logger.info(f"  Average WER     : {avg_wer:.4f}")
        logger.info(f"  Average CER     : {avg_cer:.4f}")
        logger.info(f"  Output          : {validated_path}")

        return ValidationResult(
            validated_samples_path=validated_path,
            validation_stats={
                "passed": len(validated),
                "rejected": len(rejected),
                "total": len(generated),
            },
            rejected_samples=rejected,
            avg_wer=avg_wer,
            avg_cer=avg_cer,
            prefix_trim_count=prefix_trim_count,
        )
