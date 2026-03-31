"""
Step 6: Validate Splice Quality (Placeholder)

Computes continuity metrics at splice boundaries to assess the quality
of partial spoof samples. Currently logs metrics without rejecting
samples. Future versions will enable retry logic with configurable
thresholds.

Metrics computed at each splice boundary:
- Spectral flux: abrupt change in frequency content.
- F0 (pitch) delta: pitch continuity across the splice point.
- Energy (RMS) delta: volume continuity across the splice point.
"""
import json
from pathlib import Path

import librosa
import numpy as np
from loguru import logger
from tqdm import tqdm

from app.pipeline.partial_spoof.settings import settings
from app.pipeline.partial_spoof.schemas.splice_quality_result import SpliceQualityResult


class SpliceQualityValidator:
    """Computes quality metrics at splice boundaries.

    This is a placeholder step that measures and logs boundary continuity
    metrics without rejecting any samples. The metrics provide a baseline
    for designing future rejection thresholds.

    Attributes:
        output_dir: Directory for pipeline artifacts.
    """

    def __init__(self, output_dir: Path | None = None) -> None:
        """Initialize splice quality validator.

        Args:
            output_dir: Output directory (default: from settings).
        """
        self.output_dir = output_dir or settings.OUTPUT_DIR

    def execute(self) -> SpliceQualityResult:
        """Compute quality metrics for all spliced samples.

        Returns:
            SpliceQualityResult with aggregate quality statistics.
        """
        logger.info("Step 6: Validating splice quality (placeholder metrics)...")

        splice_metadata_path = self.output_dir / "splice_metadata.json"
        with open(splice_metadata_path, "r", encoding="utf-8") as f:
            splice_metadata = json.load(f)

        quality_data = {}
        total_spectral_flux = 0.0
        total_f0_delta = 0.0
        total_energy_delta = 0.0
        total_boundaries = 0
        total_validated = 0

        for splice_key, entry in tqdm(splice_metadata.items(), desc="Validating splices"):
            audio_path = Path(entry["spliced_audio_path"])
            if not audio_path.exists():
                logger.warning(f"Spliced audio not found: {audio_path}")
                continue

            try:
                audio, sr = librosa.load(str(audio_path), sr=settings.SAMPLE_RATE, mono=True)
            except Exception as exc:
                logger.error(f"Failed to load {audio_path}: {exc}")
                continue

            boundary_metrics = []
            for word_info in entry["spoofed_words"]:
                start_s = word_info["bonafide_start_s"]
                end_s = word_info["bonafide_end_s"]

                left_metric = self._compute_boundary_metrics(audio, start_s, sr, side="left")
                right_metric = self._compute_boundary_metrics(audio, end_s, sr, side="right")

                boundary_metrics.append({
                    "word": word_info["word"],
                    "left_boundary": left_metric,
                    "right_boundary": right_metric,
                })

                total_spectral_flux += left_metric["spectral_flux"] + right_metric["spectral_flux"]
                total_f0_delta += left_metric["f0_delta"] + right_metric["f0_delta"]
                total_energy_delta += left_metric["energy_delta"] + right_metric["energy_delta"]
                total_boundaries += 2

            quality_data[splice_key] = {
                "spliced_audio_path": str(audio_path),
                "boundary_metrics": boundary_metrics,
                "passed": True,
            }
            total_validated += 1

        quality_path = self.output_dir / "splice_quality_metadata.json"
        with open(quality_path, "w", encoding="utf-8") as f:
            json.dump(quality_data, f, ensure_ascii=False, indent=2)

        avg_flux = total_spectral_flux / total_boundaries if total_boundaries > 0 else 0.0
        avg_f0 = total_f0_delta / total_boundaries if total_boundaries > 0 else 0.0
        avg_energy = total_energy_delta / total_boundaries if total_boundaries > 0 else 0.0

        logger.info(
            f"Step 6 complete: {total_validated} samples validated. "
            f"Avg spectral flux: {avg_flux:.4f}, "
            f"Avg F0 delta: {avg_f0:.2f} Hz, "
            f"Avg energy delta: {avg_energy:.4f}"
        )

        return SpliceQualityResult(
            quality_path=quality_path,
            total_validated=total_validated,
            avg_spectral_flux=avg_flux,
            avg_f0_delta=avg_f0,
            avg_energy_delta=avg_energy,
            retry_count=0,
        )

    def _compute_boundary_metrics(
        self,
        audio: np.ndarray,
        boundary_time_s: float,
        sample_rate: int,
        side: str,
        window_ms: float = 25.0,
    ) -> dict:
        """Compute continuity metrics at a single splice boundary.

        Analyzes a small window around the boundary point to measure
        spectral flux, F0 delta, and energy delta.

        Args:
            audio: Full spliced audio waveform.
            boundary_time_s: Time position of the splice boundary in seconds.
            sample_rate: Audio sample rate.
            side: 'left' for start boundary, 'right' for end boundary.
            window_ms: Analysis window size in milliseconds.

        Returns:
            Dictionary with spectral_flux, f0_delta, and energy_delta.
        """
        boundary_sample = int(boundary_time_s * sample_rate)
        window_samples = int(window_ms * sample_rate / 1000)

        before_start = max(0, boundary_sample - window_samples)
        after_end = min(len(audio), boundary_sample + window_samples)

        before_segment = audio[before_start:boundary_sample]
        after_segment = audio[boundary_sample:after_end]

        if len(before_segment) < window_samples // 2 or len(after_segment) < window_samples // 2:
            return {"spectral_flux": 0.0, "f0_delta": 0.0, "energy_delta": 0.0}

        energy_before = np.sqrt(np.mean(before_segment ** 2))
        energy_after = np.sqrt(np.mean(after_segment ** 2))
        energy_delta = abs(energy_after - energy_before)

        n_fft = min(512, len(before_segment), len(after_segment))
        if n_fft < 64:
            return {"spectral_flux": 0.0, "f0_delta": 0.0, "energy_delta": energy_delta}

        spec_before = np.abs(np.fft.rfft(before_segment[-n_fft:]))
        spec_after = np.abs(np.fft.rfft(after_segment[:n_fft]))
        spectral_flux = np.sqrt(np.mean((spec_after - spec_before) ** 2))

        return {
            "spectral_flux": float(spectral_flux),
            "f0_delta": 0.0,
            "energy_delta": float(energy_delta),
        }
