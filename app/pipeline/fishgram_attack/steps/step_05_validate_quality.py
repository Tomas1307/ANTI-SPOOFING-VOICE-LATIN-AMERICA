"""
Step 5: Validate Quality

Validates synthetic speech quality using DNSMOS and speaker similarity metrics.
"""
import json
import numpy as np
from pathlib import Path
from loguru import logger
from tqdm import tqdm
from app.pipeline.fishgram_attack.settings import settings
from app.pipeline.fishgram_attack.schemas.validation_result import ValidationResult
from app.pipeline.fishgram_attack.utils.quality_metrics import (
    compute_dnsmos,
    compute_speaker_similarity,
    detect_silence
)


class QualityValidator:
    """Validates quality of generated synthetic speech.

    Uses DNSMOS for perceptual quality and ECAPA-TDNN for speaker similarity.
    Filters out low-quality samples that don't meet thresholds.
    """

    def __init__(
        self,
        output_dir: Path | None = None,
        dnsmos_threshold: float | None = None,
        similarity_threshold: float | None = None
    ):
        """Initialize quality validator.

        Args:
            output_dir: Output directory (default: from settings)
            dnsmos_threshold: Minimum DNSMOS overall score (default: from settings)
            similarity_threshold: Minimum speaker similarity (default: from settings)
        """
        self.output_dir = output_dir or settings.OUTPUT_DIR
        self.dnsmos_threshold = dnsmos_threshold or settings.DNSMOS_THRESHOLD_OVRL
        self.similarity_threshold = similarity_threshold or settings.SPEAKER_SIM_THRESHOLD

    def execute(self) -> ValidationResult:
        """Validate quality of all generated samples.

        Returns:
            ValidationResult with validated samples and statistics

        Raises:
            Exception: If validation fails
        """
        logger.info("Validating quality of generated samples...")
        logger.info(f"  DNSMOS threshold: {self.dnsmos_threshold}")
        logger.info(f"  Similarity threshold: {self.similarity_threshold}")

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
            ref_path = Path(references[speaker_id]["reference_path"])

            try:
                # Check if files exist
                if not audio_path.exists():
                    rejected.append({
                        "sample_id": sample_id,
                        "reason": "Audio file not found"
                    })
                    continue

                # Compute DNSMOS
                try:
                    dnsmos_ovrl = compute_dnsmos(audio_path)
                except Exception as e:
                    logger.warning(f"DNSMOS failed for {sample_id}: {e}")
                    dnsmos_ovrl = 0.0

                # Compute speaker similarity
                try:
                    similarity = compute_speaker_similarity(audio_path, ref_path)
                except Exception as e:
                    logger.warning(f"Speaker similarity failed for {sample_id}: {e}")
                    similarity = 0.0

                # Check for excessive silence
                import librosa
                audio, sr = librosa.load(audio_path, sr=settings.SAMPLE_RATE)
                has_silence = detect_silence(audio, min_duration=1.0)

                # Validate thresholds
                passes_dnsmos = dnsmos_ovrl >= self.dnsmos_threshold
                passes_similarity = similarity >= self.similarity_threshold
                passes_silence = not has_silence

                if passes_dnsmos and passes_similarity and passes_silence:
                    # PASS - add to validated
                    validated[sample_id] = sample_data.copy()
                    validated[sample_id]["dnsmos_ovrl"] = float(dnsmos_ovrl)
                    validated[sample_id]["speaker_similarity"] = float(similarity)
                    dnsmos_scores.append(dnsmos_ovrl)
                    similarity_scores.append(similarity)
                else:
                    # FAIL - add to rejected
                    reasons = []
                    if not passes_dnsmos:
                        reasons.append(f"DNSMOS {dnsmos_ovrl:.2f} < {self.dnsmos_threshold}")
                    if not passes_similarity:
                        reasons.append(f"Similarity {similarity:.2f} < {self.similarity_threshold}")
                    if not passes_silence:
                        reasons.append("Excessive silence detected")

                    rejected.append({
                        "sample_id": sample_id,
                        "dnsmos_ovrl": float(dnsmos_ovrl),
                        "speaker_similarity": float(similarity),
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
        avg_dnsmos = np.mean(dnsmos_scores) if dnsmos_scores else 0.0
        avg_similarity = np.mean(similarity_scores) if similarity_scores else 0.0

        logger.info(f"✓ Validation complete")
        logger.info(f"  Passed: {len(validated)}/{len(generated)} ({100*len(validated)/len(generated):.1f}%)")
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
            avg_dnsmos=float(avg_dnsmos),
            avg_similarity=float(avg_similarity)
        )
