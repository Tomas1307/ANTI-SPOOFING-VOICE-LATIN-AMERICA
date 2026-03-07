"""
Step 4: Validate Quality with Artifact Detection

Validates synthetic speech quality using DNSMOS, speaker similarity,
and Qwen-specific artifact detection (truncation, low energy, duration anomalies).
"""
import json
import librosa
import numpy as np
from pathlib import Path
from loguru import logger
from tqdm import tqdm
from app.pipeline.qwen_attack.settings import settings
from app.pipeline.qwen_attack.schemas.validation_result import ValidationResult
from app.pipeline.qwen_attack.utils.quality_metrics import (
    compute_dnsmos,
    compute_speaker_similarity,
    detect_silence,
)
from app.pipeline.qwen_attack.utils.artifact_detector import (
    detect_truncation,
    detect_low_energy,
    detect_duration_anomaly,
)


class QualityValidator:
    """Validates quality of generated synthetic speech with artifact detection.

    Extends standard DNSMOS and speaker similarity validation with
    Qwen-specific artifact checks for truncation, near-silent outputs,
    and duration anomalies. These additional checks are necessary because
    Qwen3-TTS silently fails on certain inputs without raising errors.

    Attributes:
        output_dir: Directory containing generation metadata and audio files.
        dnsmos_threshold: Minimum DNSMOS overall score for validation.
        similarity_threshold: Minimum speaker similarity for validation.
    """

    def __init__(
        self,
        output_dir: Path | None = None,
        dnsmos_threshold: float | None = None,
        similarity_threshold: float | None = None
    ):
        """Initialize quality validator.

        Args:
            output_dir: Output directory (default: from settings).
            dnsmos_threshold: Minimum DNSMOS overall score (default: from settings).
            similarity_threshold: Minimum speaker similarity (default: from settings).
        """
        self.output_dir = output_dir or settings.OUTPUT_DIR
        self.dnsmos_threshold = dnsmos_threshold or settings.DNSMOS_THRESHOLD_OVRL
        self.similarity_threshold = similarity_threshold or settings.SPEAKER_SIM_THRESHOLD

    def execute(self) -> ValidationResult:
        """Validate quality of all generated samples.

        Applies five validation checks per sample:
        1. Duration anomaly detection (too short or too long).
        2. Low energy detection (near-silent/garbled output).
        3. Truncation detection (audio too short for text length).
        4. DNSMOS perceptual quality score.
        5. Speaker similarity score.

        Returns:
            ValidationResult with validated samples and statistics.

        Raises:
            Exception: If validation fails.
        """
        logger.info("Validating quality of generated samples...")
        logger.info(f"  DNSMOS threshold: {self.dnsmos_threshold}")
        logger.info(f"  Similarity threshold: {self.similarity_threshold}")
        logger.info(f"  Min duration: {settings.MIN_AUDIO_DURATION}s")
        logger.info(f"  Max duration: {settings.MAX_AUDIO_DURATION}s")

        # Load metadata
        gen_metadata_path = self.output_dir / "generation_metadata.json"
        ref_metadata_path = self.output_dir / "reference_metadata.json"

        with open(gen_metadata_path, "r", encoding="utf-8") as f:
            generated = json.load(f)

        with open(ref_metadata_path, "r", encoding="utf-8") as f:
            references = json.load(f)

        validated = {}
        rejected = []
        dnsmos_scores = []
        similarity_scores = []

        logger.info(f"Validating {len(generated)} samples...")

        for sample_id, sample_data in tqdm(generated.items(), desc="Validating"):
            audio_path = Path(sample_data["audio_path"])
            speaker_id = sample_data["speaker_id"]
            text = sample_data["text"]
            ref_path = Path(references[speaker_id]["reference_path"])

            try:
                # Check if file exists
                if not audio_path.exists():
                    rejected.append({
                        "sample_id": sample_id,
                        "reason": "Audio file not found"
                    })
                    continue

                # Load audio for artifact checks
                audio, sr = librosa.load(audio_path, sr=settings.SAMPLE_RATE)
                audio_duration = len(audio) / sr

                # --- Qwen-specific artifact checks ---

                # Check 1: Duration anomaly
                has_duration_anomaly = detect_duration_anomaly(
                    audio_duration,
                    min_duration=settings.MIN_AUDIO_DURATION,
                    max_duration=settings.MAX_AUDIO_DURATION,
                )

                # Check 2: Low energy (near-silent output)
                has_low_energy = detect_low_energy(
                    audio,
                    threshold=settings.LOW_ENERGY_THRESHOLD,
                )

                # Check 3: Truncation (audio too short for text)
                is_truncated = detect_truncation(
                    audio,
                    text,
                    sample_rate=settings.SAMPLE_RATE,
                    min_words_per_second=settings.MIN_WORDS_PER_SECOND,
                )

                # --- Standard quality checks ---

                # Check 4: DNSMOS
                try:
                    dnsmos_ovrl = compute_dnsmos(audio_path)
                except Exception as e:
                    logger.warning(f"DNSMOS failed for {sample_id}: {e}")
                    dnsmos_ovrl = 0.0

                # Check 5: Speaker similarity
                try:
                    similarity = compute_speaker_similarity(audio_path, ref_path)
                except Exception as e:
                    logger.warning(f"Speaker similarity failed for {sample_id}: {e}")
                    similarity = 0.0

                # Aggregate validation results
                passes_artifacts = (
                    not has_duration_anomaly
                    and not has_low_energy
                    and not is_truncated
                )
                passes_dnsmos = dnsmos_ovrl >= self.dnsmos_threshold
                passes_similarity = similarity >= self.similarity_threshold

                if passes_artifacts and passes_dnsmos and passes_similarity:
                    # PASS
                    validated[sample_id] = sample_data.copy()
                    validated[sample_id]["dnsmos_ovrl"] = float(dnsmos_ovrl)
                    validated[sample_id]["speaker_similarity"] = float(similarity)
                    dnsmos_scores.append(dnsmos_ovrl)
                    similarity_scores.append(similarity)
                else:
                    # FAIL
                    reasons = []
                    if has_duration_anomaly:
                        reasons.append(f"Duration anomaly: {audio_duration:.1f}s")
                    if has_low_energy:
                        reasons.append("Near-silent output")
                    if is_truncated:
                        reasons.append("Truncated audio")
                    if not passes_dnsmos:
                        reasons.append(
                            f"DNSMOS {dnsmos_ovrl:.2f} < {self.dnsmos_threshold}"
                        )
                    if not passes_similarity:
                        reasons.append(
                            f"Similarity {similarity:.2f} < {self.similarity_threshold}"
                        )

                    rejected.append({
                        "sample_id": sample_id,
                        "dnsmos_ovrl": float(dnsmos_ovrl),
                        "speaker_similarity": float(similarity),
                        "audio_duration": float(audio_duration),
                        "reason": "; ".join(reasons)
                    })

            except Exception as e:
                logger.error(f"Validation error for {sample_id}: {e}")
                rejected.append({
                    "sample_id": sample_id,
                    "reason": f"Validation exception: {str(e)}"
                })

        # Save validated samples
        validated_path = self.output_dir / "validated_samples.json"
        with open(validated_path, "w", encoding="utf-8") as f:
            json.dump(validated, f, indent=2, ensure_ascii=False)

        # Compute statistics
        avg_dnsmos = float(np.mean(dnsmos_scores)) if dnsmos_scores else 0.0
        avg_similarity = float(np.mean(similarity_scores)) if similarity_scores else 0.0

        logger.info("Validation complete")
        pass_rate = (100 * len(validated) / len(generated)) if len(generated) > 0 else 0.0
        logger.info(
            f"  Passed: {len(validated)}/{len(generated)} ({pass_rate:.1f}%)"
        )
        logger.info(f"  Rejected: {len(rejected)}")
        logger.info(f"  Average DNSMOS: {avg_dnsmos:.2f}")
        logger.info(f"  Average similarity: {avg_similarity:.2f}")
        logger.info(f"  Validated samples saved to: {validated_path}")

        return ValidationResult(
            validated_samples_path=validated_path,
            validation_stats={
                "passed": len(validated),
                "rejected": len(rejected),
                "total": len(generated)
            },
            rejected_samples=rejected,
            avg_dnsmos=avg_dnsmos,
            avg_similarity=avg_similarity
        )
