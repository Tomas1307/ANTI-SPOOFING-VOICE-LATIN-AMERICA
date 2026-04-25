"""
Energy valley scorer for word boundary quality assessment.

Analyzes the RMS energy envelope of cloned audio to score how cleanly
each word can be extracted. Words with deep energy valleys at both
boundaries (low score) are the best candidates for splicing because
the cut will be inaudible.
"""
import numpy as np
from loguru import logger
from typing import Dict, List

from app.pipeline.partial_spoof.schemas.valley_score import ValleyScore


class ValleyScorer:
    """Scores word boundaries by energy valley depth in cloned audio.

    For each word, computes a valley score at both boundaries (start and end).
    The score is the ratio of the minimum RMS energy to the average RMS energy
    in a window around the boundary. A score near 0 means there is near-silence
    at the boundary (easy to cut). A score near 1 means energy is continuous
    (hard to cut without audible artifacts).

    Attributes:
        sample_rate: Audio sample rate in Hz.
        window_ms: Window size in ms (+/- around each boundary).
        frame_ms: Frame size in ms for RMS energy computation.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        window_ms: float = 100.0,
        frame_ms: float = 5.0,
    ) -> None:
        """Initialize the valley scorer.

        Args:
            sample_rate: Audio sample rate in Hz.
            window_ms: Window size in ms for boundary analysis.
            frame_ms: Frame size in ms for RMS energy computation.
        """
        self.sample_rate = sample_rate
        self.window_ms = window_ms
        self.frame_ms = frame_ms
        self._frame_samples = int(frame_ms * sample_rate / 1000)
        self._window_samples = int(window_ms * sample_rate / 1000)

    def score_words(
        self,
        cloned_audio: np.ndarray,
        cloned_words: List[Dict],
        bonafide_words: List[Dict],
        min_duration_ms: float = 200.0,
        max_stretch_ratio: float = 1.25,
    ) -> List[ValleyScore]:
        """Compute valley scores for all words in the cloned audio.

        Args:
            cloned_audio: Full cloned waveform (1-D float32).
            cloned_words: Word timestamps from cloned audio. Each dict has
                keys 'word' (str), 'start' (float s), 'end' (float s).
            bonafide_words: Word timestamps from bonafide audio (same format).
                Used to compute stretch ratio.
            min_duration_ms: Minimum cloned word duration to be eligible.
            max_stretch_ratio: Maximum acceptable ratio of cloned/bonafide
                duration (or inverse). Words outside [1/ratio, ratio] are
                ineligible.

        Returns:
            List of ValleyScore objects, one per word, sorted by combined_score
            ascending (best candidates first).
        """
        energy = self._compute_energy_envelope(cloned_audio)
        n_words = min(len(cloned_words), len(bonafide_words))
        scores = []

        for i in range(n_words):
            cw = cloned_words[i]
            bw = bonafide_words[i]

            c_dur_ms = (cw["end"] - cw["start"]) * 1000
            b_dur_ms = (bw["end"] - bw["start"]) * 1000

            stretch = c_dur_ms / b_dur_ms if b_dur_ms > 0 else 999.0

            left_score = self._boundary_score(
                energy, cw["start"], i, n_words
            )
            right_score = self._boundary_score(
                energy, cw["end"], i, n_words
            )

            if i == 0:
                combined = right_score
            elif i == n_words - 1:
                combined = left_score
            else:
                combined = max(left_score, right_score)

            eligible = (
                combined <= 1.0
                and c_dur_ms >= min_duration_ms
                and (1.0 / max_stretch_ratio) <= stretch <= max_stretch_ratio
            )

            scores.append(ValleyScore(
                word_index=i,
                word=cw["word"],
                left_score=round(left_score, 4),
                right_score=round(right_score, 4),
                combined_score=round(combined, 4),
                duration_ms=round(c_dur_ms, 1),
                stretch_ratio=round(stretch, 3),
                eligible=eligible,
            ))

        scores.sort(key=lambda s: s.combined_score)

        logger.debug(
            f"Valley scores: {len([s for s in scores if s.eligible])}/{n_words} eligible | "
            f"best={scores[0].combined_score:.3f} ({scores[0].word}) | "
            f"worst={scores[-1].combined_score:.3f} ({scores[-1].word})"
        )

        return scores

    def _compute_energy_envelope(self, audio: np.ndarray) -> np.ndarray:
        """Compute RMS energy envelope in fixed-size frames.

        Args:
            audio: Full audio waveform (1-D float).

        Returns:
            Array of RMS energy values, one per frame.
        """
        n_frames = len(audio) // self._frame_samples
        if n_frames == 0:
            return np.zeros(1, dtype=np.float32)

        frames = audio[:n_frames * self._frame_samples].reshape(n_frames, self._frame_samples)
        return np.sqrt(np.mean(frames ** 2, axis=1)).astype(np.float32)

    def _boundary_score(
        self,
        energy: np.ndarray,
        boundary_s: float,
        word_index: int,
        total_words: int,
    ) -> float:
        """Compute the valley score at a single boundary.

        The score is min_energy / avg_energy within a +/- window around the
        boundary. A score of 0.0 means there is a silent frame at the boundary.
        A score of 1.0 means energy is perfectly uniform (no valley).

        Args:
            energy: RMS energy envelope (from _compute_energy_envelope).
            boundary_s: Boundary position in seconds.
            word_index: Index of the word (for edge detection).
            total_words: Total number of words.

        Returns:
            Valley score in [0, 1].
        """
        center_frame = int(boundary_s * 1000 / self.frame_ms)
        window_frames = int(self.window_ms / self.frame_ms)

        start = max(0, center_frame - window_frames)
        end = min(len(energy), center_frame + window_frames)

        if start >= end:
            return 1.0

        local = energy[start:end]
        avg_val = float(np.mean(local))

        if avg_val < 1e-8:
            return 0.0

        min_val = float(np.min(local))
        return min_val / avg_val
