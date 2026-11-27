"""
RawBoost Augmentation.

IMPORTANT: This module is TRAINING-TIME only. It is NOT called by the offline
augmentation pipeline (``app/scripts/augmentation_pipeline.py``). RawBoost is
applied on-the-fly during model training so that it does not inflate the corpus
size on disk and remains composable with the trainer's data-loading loop.

Wraps the official RawBoost algorithm (Tak et al., 2022) implemented in
``app.augmenter.rawboost_reference``: linear-and-non-linear convolutive noise
(LnL), impulsive signal-dependent additive noise (ISD), and stationary
signal-independent additive noise (SSI). Loudness normalization is NOT done here;
the orchestrator applies one uniform loudness policy on write.

Reference:
    H. Tak, M. Kamble, J. Patino, M. Todisco, N. Evans, "RawBoost: A Raw Data
    Boosting and Augmentation Method applied to Automatic Speaker Verification
    Anti-Spoofing," ICASSP 2022.
"""
import random

import numpy as np

from app.augmenter import rawboost_reference
from app.augmenter.base_augmenter import BaseAugmenter
from app.augmenter.schemas.codec_rawboost_config import RawBoostConfigV2

# RawBoost algorithm id -> component tokens / human-readable name.
_ALGO_OPERATIONS = {
    1: ["LnL"],
    2: ["ISD"],
    3: ["SSI"],
    4: ["LnL", "ISD", "SSI"],
    5: ["LnL", "ISD"],
    6: ["LnL", "SSI"],
    7: ["LnL", "ISD"],
}
_ALGO_NAMES = {
    1: "LnL",
    2: "ISD",
    3: "SSI",
    4: "series_LnL_ISD_SSI",
    5: "series_LnL_ISD",
    6: "series_LnL_SSI",
    7: "parallel_LnL_ISD",
}


class RawBoostAugmenter(BaseAugmenter):
    """
    RawBoost augmentation using the real LnL/ISD/SSI algorithm.

    Attributes:
        config: RawBoostConfigV2 with the algorithm selection and parameter ranges.
    """

    def __init__(self, config: RawBoostConfigV2, sample_rate: int = 16000):
        """
        Initialize the RawBoost augmenter.

        Args:
            config: Configuration object (algorithm choice + parameter ranges).
            sample_rate: Target sample rate for processing.
        """
        super().__init__(sample_rate)
        self.config = config

        print("RawBoostAugmenter initialized:")
        print(f"  - Algo: {config.algo} (0 = random per clip)")
        print(f"  - Algo choices: {config.algo_choices}")

    def _select_algo(self) -> int:
        """Return the fixed algo, or a weighted random one from algo_choices when algo == 0."""
        if self.config.algo != 0:
            return self.config.algo
        weights = [self.config.algo_weights.get(a, 1.0) for a in self.config.algo_choices]
        return random.choices(self.config.algo_choices, weights=weights, k=1)[0]

    def augment(
        self,
        audio: np.ndarray,
        sr: int,
        return_metadata: bool = False
    ) -> np.ndarray:
        """
        Apply RawBoost augmentation to audio.

        Args:
            audio: Input audio signal.
            sr: Sample rate of input audio.
            return_metadata: If True, returns tuple (audio, metadata).

        Returns:
            Augmented audio signal, or tuple (audio, metadata) if return_metadata.
        """
        audio, sr = self._ensure_sample_rate(audio, sr)

        algo = self._select_algo()
        feature = np.asarray(audio, dtype=np.float64)
        augmented = rawboost_reference.process_Rawboost_feature(
            feature, sr, self.config.params, algo
        )

        augmented = np.asarray(augmented, dtype=np.float32)
        augmented = self._clip_audio(augmented, max_val=0.99)

        operations = _ALGO_OPERATIONS.get(algo, [])

        if return_metadata:
            metadata = {
                "operations": operations,
                "num_operations": len(operations),
                "algo": algo,
                "algo_name": _ALGO_NAMES.get(algo, "none"),
            }
            return augmented, metadata

        return augmented

    def get_augmentation_label(self, operations: list) -> str:
        """
        Generate a descriptive label for the RawBoost components applied.

        Args:
            operations: List of component tokens (e.g. ["LnL", "ISD"]).

        Returns:
            Formatted augmentation label, e.g. "RAWBOOST_LnL_ISD".
        """
        if not operations:
            return "RAWBOOST_NONE"
        return f"RAWBOOST_{'_'.join(operations)}"
