"""
Step 6: Validate Splice Quality

Validates spliced partial spoof audio using:
1. Rejection of samples with zero spoofed words (no actual splice performed).
2. Parakeet TDT transcription of the spliced audio.
3. WER/CER computation against the original bonafide transcript.
4. Boundary continuity metrics (spectral flux, energy delta).
5. NISQA MOS quality estimation (informational, no rejection).
6. ECAPA-TDNN speaker similarity vs bonafide reference (informational).

Samples are rejected if they have no spoofed words or if WER exceeds
the configured threshold.
"""
import json
from pathlib import Path

import librosa
import numpy as np
from loguru import logger
from tqdm import tqdm

from app.pipeline.partial_spoof.settings import settings
from app.pipeline.partial_spoof.schemas.splice_quality_result import SpliceQualityResult
from app.utils.ecapa_similarity import EcapaSimilarity
from app.utils.nisqa_scorer import NisqaScorer
from app.utils.parakeet_transcriber import ParakeetTranscriber
from app.utils.wer_cer import compute_wer, compute_cer


class SpliceQualityValidator:
    """Validates quality of spliced partial spoof audio.

    Transcribes spliced audio with Parakeet, computes WER/CER against
    the bonafide transcript, measures boundary continuity metrics,
    and computes NISQA MOS and speaker similarity scores.

    Samples with zero spoofed words are rejected outright. Samples
    with WER exceeding the threshold are also rejected.

    Attributes:
        output_dir: Directory for pipeline artifacts.
        wer_max: Hard WER rejection ceiling.
        cer_max: Hard CER rejection ceiling.
    """

    def __init__(
        self,
        output_dir: Path | None = None,
        wer_max: float = 0.30,
        cer_max: float = 0.20,
    ) -> None:
        """Initialize splice quality validator.

        Args:
            output_dir: Output directory (default: from settings).
            wer_max: Maximum acceptable WER (default: 0.30, more lenient
                than full-synthesis pipelines since splicing introduces
                minor boundary artifacts).
            cer_max: Maximum acceptable CER (default: 0.20).
        """
        self.output_dir = output_dir or settings.OUTPUT_DIR
        self.wer_max = wer_max
        self.cer_max = cer_max

    def execute(self) -> SpliceQualityResult:
        """Validate quality of all spliced samples.

        Returns:
            SpliceQualityResult with validation statistics.
        """
        logger.info("Step 6: Validating splice quality (Parakeet STT + WER/CER + NISQA + Speaker Sim)...")
        logger.info(f"  WER hard ceiling: {self.wer_max:.2f}")
        logger.info(f"  CER hard ceiling: {self.cer_max:.2f}")

        splice_metadata_path = self.output_dir / "splice_metadata.json"
        with open(splice_metadata_path, "r", encoding="utf-8") as f:
            splice_metadata = json.load(f)

        transcriber = ParakeetTranscriber()
        transcriber.load(model_id=settings.PARAKEET_MODEL_ID, device=settings.DEVICE)

        nisqa_scorer = NisqaScorer()
        nisqa_scorer.load(device=settings.DEVICE)

        ecapa = EcapaSimilarity()
        ecapa.load(device=settings.DEVICE)

        quality_data = {}
        rejected = []
        wer_scores = []
        cer_scores = []
        nisqa_scores = []
        similarity_scores = []
        total_spectral_flux = 0.0
        total_energy_delta = 0.0
        total_boundaries = 0
        ref_embeddings = {}

        for splice_key, entry in tqdm(splice_metadata.items(), desc="Validating splices"):
            audio_path = Path(entry["spliced_audio_path"])
            if not audio_path.exists():
                logger.warning(f"Spliced audio not found: {audio_path}")
                rejected.append({"sample_id": splice_key, "reason": "Audio file not found"})
                continue

            if not entry["spoofed_words"]:
                rejected.append({
                    "sample_id": splice_key,
                    "reason": "Zero spoofed words — no actual splice performed",
                })
                continue

            try:
                audio, sr = librosa.load(str(audio_path), sr=settings.SAMPLE_RATE, mono=True)
            except Exception as exc:
                logger.error(f"Failed to load {audio_path}: {exc}")
                rejected.append({"sample_id": splice_key, "reason": f"Load error: {exc}"})
                continue

            transcription, _ = transcriber.transcribe_with_timestamps(audio_path)
            original_text = entry["transcript"]
            sample_wer = compute_wer(original_text, transcription)
            sample_cer = compute_cer(original_text, transcription)

            boundary_metrics = self._compute_all_boundary_metrics(
                audio, entry["spoofed_words"], sr
            )
            for bm in boundary_metrics:
                total_spectral_flux += bm["spectral_flux"]
                total_energy_delta += bm["energy_delta"]
                total_boundaries += 1

            if sample_wer > self.wer_max or sample_cer > self.cer_max:
                reasons = []
                if sample_wer > self.wer_max:
                    reasons.append(f"WER {sample_wer:.3f} > {self.wer_max:.3f}")
                if sample_cer > self.cer_max:
                    reasons.append(f"CER {sample_cer:.3f} > {self.cer_max:.3f}")
                rejected.append({
                    "sample_id": splice_key,
                    "wer": float(sample_wer),
                    "cer": float(sample_cer),
                    "transcription": transcription,
                    "reason": "; ".join(reasons),
                })
                continue

            sample_nisqa = nisqa_scorer.predict_mos(audio_path)

            speaker_id = entry["speaker_id"]
            if speaker_id not in ref_embeddings:
                bonafide_path = Path(entry["bonafide_audio_path"])
                ref_embeddings[speaker_id] = ecapa.extract_embedding(bonafide_path)
            sample_sim = ecapa.compute_similarity_from_embedding(
                ref_embeddings[speaker_id], audio_path
            )

            quality_data[splice_key] = {
                "spliced_audio_path": str(audio_path),
                "transcript": original_text,
                "transcription": transcription,
                "wer": float(sample_wer),
                "cer": float(sample_cer),
                "nisqa_mos": float(sample_nisqa),
                "speaker_similarity": float(sample_sim),
                "tier": entry["tier"],
                "spoofed_words_count": len(entry["spoofed_words"]),
                "spoof_duration_ratio": entry["spoof_duration_ratio"],
                "boundary_metrics": boundary_metrics,
                "passed": True,
            }

            wer_scores.append(sample_wer)
            cer_scores.append(sample_cer)
            nisqa_scores.append(sample_nisqa)
            similarity_scores.append(sample_sim)

            logger.debug(
                f"{splice_key}: WER={sample_wer:.3f} CER={sample_cer:.3f} "
                f"NISQA={sample_nisqa:.2f} Sim={sample_sim:.3f}"
            )

        quality_path = self.output_dir / "splice_quality_metadata.json"
        with open(quality_path, "w", encoding="utf-8") as f:
            json.dump(quality_data, f, ensure_ascii=False, indent=2)

        avg_wer = float(np.mean(wer_scores)) if wer_scores else 0.0
        avg_cer = float(np.mean(cer_scores)) if cer_scores else 0.0
        avg_nisqa = float(np.mean(nisqa_scores)) if nisqa_scores else 0.0
        avg_sim = float(np.mean(similarity_scores)) if similarity_scores else 0.0
        avg_flux = total_spectral_flux / total_boundaries if total_boundaries > 0 else 0.0
        avg_energy = total_energy_delta / total_boundaries if total_boundaries > 0 else 0.0

        logger.info("Validation complete")
        logger.info(f"  Passed           : {len(quality_data)}/{len(splice_metadata)}")
        logger.info(f"  Rejected         : {len(rejected)}")
        logger.info(f"  Average WER      : {avg_wer:.4f}")
        logger.info(f"  Average CER      : {avg_cer:.4f}")
        logger.info(f"  Average NISQA    : {avg_nisqa:.2f}")
        logger.info(f"  Average Sim      : {avg_sim:.3f}")
        logger.info(f"  Avg spectral flux: {avg_flux:.4f}")
        logger.info(f"  Avg energy delta : {avg_energy:.4f}")

        return SpliceQualityResult(
            quality_path=quality_path,
            total_validated=len(quality_data),
            total_rejected=len(rejected),
            rejected_samples=rejected,
            avg_wer=avg_wer,
            avg_cer=avg_cer,
            avg_nisqa=avg_nisqa,
            avg_speaker_similarity=avg_sim,
            avg_spectral_flux=avg_flux,
            avg_energy_delta=avg_energy,
        )

    def _compute_all_boundary_metrics(
        self,
        audio: np.ndarray,
        spoofed_words: list,
        sample_rate: int,
    ) -> list:
        """Compute boundary continuity metrics for all splice points.

        Args:
            audio: Full spliced audio waveform.
            spoofed_words: List of spoofed word entries with timestamp info.
            sample_rate: Audio sample rate.

        Returns:
            List of boundary metric dicts, one per spoofed word.
        """
        metrics = []
        window_ms = 25.0

        for word_info in spoofed_words:
            start_s = word_info["bonafide_start_s"]
            end_s = word_info["bonafide_end_s"]

            flux_start = self._spectral_flux_at(audio, start_s, sample_rate, window_ms)
            flux_end = self._spectral_flux_at(audio, end_s, sample_rate, window_ms)
            energy_start = self._energy_delta_at(audio, start_s, sample_rate, window_ms)
            energy_end = self._energy_delta_at(audio, end_s, sample_rate, window_ms)

            metrics.append({
                "word": word_info["word"],
                "spectral_flux": (flux_start + flux_end) / 2,
                "energy_delta": (energy_start + energy_end) / 2,
            })

        return metrics

    def _spectral_flux_at(
        self,
        audio: np.ndarray,
        time_s: float,
        sample_rate: int,
        window_ms: float,
    ) -> float:
        """Compute spectral flux at a boundary point.

        Args:
            audio: Audio waveform.
            time_s: Boundary time in seconds.
            sample_rate: Sample rate.
            window_ms: Window size in milliseconds.

        Returns:
            Spectral flux value.
        """
        boundary = int(time_s * sample_rate)
        window = int(window_ms * sample_rate / 1000)
        before = audio[max(0, boundary - window):boundary]
        after = audio[boundary:min(len(audio), boundary + window)]

        n_fft = min(512, len(before), len(after))
        if n_fft < 64:
            return 0.0

        spec_before = np.abs(np.fft.rfft(before[-n_fft:]))
        spec_after = np.abs(np.fft.rfft(after[:n_fft]))
        return float(np.sqrt(np.mean((spec_after - spec_before) ** 2)))

    def _energy_delta_at(
        self,
        audio: np.ndarray,
        time_s: float,
        sample_rate: int,
        window_ms: float,
    ) -> float:
        """Compute energy delta at a boundary point.

        Args:
            audio: Audio waveform.
            time_s: Boundary time in seconds.
            sample_rate: Sample rate.
            window_ms: Window size in milliseconds.

        Returns:
            Absolute energy delta.
        """
        boundary = int(time_s * sample_rate)
        window = int(window_ms * sample_rate / 1000)
        before = audio[max(0, boundary - window):boundary]
        after = audio[boundary:min(len(audio), boundary + window)]

        if len(before) < window // 2 or len(after) < window // 2:
            return 0.0

        energy_before = np.sqrt(np.mean(before ** 2))
        energy_after = np.sqrt(np.mean(after ** 2))
        return float(abs(energy_after - energy_before))
