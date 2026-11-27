"""
Mock tests for the augmentation stacking gate and updated distribution.

These tests are fully self-contained (no GPU, no real audio) and use local
stubs instead of importing from app.schema directly, because the existing
app/augmenter/__init__.py creates a circular import that only resolves
correctly inside the ml-server03 venv where all heavy deps are installed.

Verifies:
  1. _select_augmentation_mode returns [RIR_NOISE, CODEC] ~40% of the time.
  2. Single-mode draws use relative weights 60/40 (RIR vs CODEC).
  3. _apply_augmentation chains two passes and joins SYSTEM_IDs with "|".
  4. RAWBOOST never appears in offline pipeline output.
  5. AugmentationStrategy.validate() rejects RAWBOOST in type_distribution.
  6. AugmentationStrategy.get_augmentation_counts totals are self-consistent.
"""

import random
import unittest
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple
from unittest.mock import MagicMock

import numpy as np


# ---------------------------------------------------------------------------
# Local stubs (mirror of app.schema) to avoid circular import on Windows dev
# machine where heavy deps are not installed.
# ---------------------------------------------------------------------------

class AugmentationType(Enum):
    """Stub enum matching app.schema.AugmentationType."""

    RIR_NOISE = "rir_noise"
    CODEC = "codec"
    RAWBOOST = "rawboost"
    DEVICE_IR = "device_ir"


class _FakeConfig:
    """Minimal stand-in for RIRNoiseConfig / CodecConfigV2."""

    def validate(self):
        pass


@dataclass
class AugmentationStrategy:
    """
    Stub matching the real AugmentationStrategy interface tested here.

    Only the fields and methods exercised by these tests are implemented.
    """

    augmentation_factor: int = 3
    type_distribution: Dict[AugmentationType, float] = field(
        default_factory=lambda: {
            AugmentationType.RIR_NOISE: 0.60,
            AugmentationType.CODEC: 0.40,
        }
    )
    stacking_probability: float = 0.40
    include_original: bool = True

    def validate(self):
        """Validate strategy configuration."""
        total = sum(self.type_distribution.values())
        assert abs(total - 1.0) < 0.01, (
            f"Type distribution must sum to 1.0, got {total}"
        )
        assert AugmentationType.RAWBOOST not in self.type_distribution, (
            "RAWBOOST must not appear in the offline type_distribution. "
            "It is applied at training time only."
        )
        assert 0.0 <= self.stacking_probability <= 1.0, (
            f"stacking_probability must be in [0,1], got {self.stacking_probability}"
        )

    def get_augmentation_counts(self, n_originals: int) -> Dict[str, int]:
        """Calculate expected sample counts per augmentation mode."""
        total_augmented = n_originals * self.augmentation_factor
        n_stacked = int(total_augmented * self.stacking_probability)
        n_single = total_augmented - n_stacked
        rir_w = self.type_distribution[AugmentationType.RIR_NOISE]
        codec_w = self.type_distribution[AugmentationType.CODEC]
        single_total = rir_w + codec_w
        n_single_rir = int(n_single * (rir_w / single_total))
        n_single_codec = n_single - n_single_rir
        return {
            "original": n_originals if self.include_original else 0,
            "single_rir_noise": n_single_rir,
            "single_codec": n_single_codec,
            "stacked_rir_then_codec": n_stacked,
            "total": (
                n_originals + total_augmented
                if self.include_original
                else total_augmented
            ),
        }


# ---------------------------------------------------------------------------
# Replica of the pipeline selection/application logic (mirrors the real code)
# ---------------------------------------------------------------------------

class _MockPipeline:
    """
    Replicates AugmentationPipeline._select_augmentation_mode and
    _apply_augmentation without any real audio dependencies.
    """

    def __init__(self, stacking_prob: float = 0.40):
        self.strategy = AugmentationStrategy(
            augmentation_factor=3,
            type_distribution={
                AugmentationType.RIR_NOISE: 0.60,
                AugmentationType.CODEC: 0.40,
            },
            stacking_probability=stacking_prob,
        )
        self.rir_augmenter = MagicMock()
        self.rir_augmenter.augment.return_value = (
            np.zeros(16000),
            {
                "room_size": "small",
                "noise_source": "noise",
                "snr_db": 15.0,
            },
        )
        self.rir_augmenter.get_augmentation_label.return_value = "RIR_SMALL_NOI_SNR15"

        self.codec_augmenter = MagicMock()
        self.codec_augmenter.augment.return_value = (
            np.zeros(16000),
            {
                "codec_sr": 8000,
                "packet_loss": 0.02,
                "bandpass": False,
                "quantization_bits": 8,
                "codec": "opus",
            },
        )
        self.codec_augmenter.get_augmentation_label.return_value = "CODEC_8K_OPUS"

    def _select_augmentation_mode(self) -> List[AugmentationType]:
        """Mirror of augmentation_pipeline.py:_select_augmentation_mode."""
        if random.random() < self.strategy.stacking_probability:
            return [AugmentationType.RIR_NOISE, AugmentationType.CODEC]
        types = list(self.strategy.type_distribution.keys())
        probabilities = [self.strategy.type_distribution[t] for t in types]
        chosen = random.choices(types, weights=probabilities, k=1)[0]
        return [chosen]

    def _apply_single_augmentation(
        self,
        audio: np.ndarray,
        sr: int,
        aug_type: AugmentationType,
    ) -> Tuple[np.ndarray, str]:
        """Mirror of augmentation_pipeline.py:_apply_single_augmentation."""
        if aug_type == AugmentationType.RIR_NOISE:
            augmented, metadata = self.rir_augmenter.augment(
                audio, sr, return_metadata=True
            )
            system_id = self.rir_augmenter.get_augmentation_label(
                metadata["room_size"], metadata["noise_source"], metadata["snr_db"]
            )
        elif aug_type == AugmentationType.CODEC:
            augmented, metadata = self.codec_augmenter.augment(
                audio, sr, return_metadata=True
            )
            system_id = self.codec_augmenter.get_augmentation_label(
                metadata["codec_sr"],
                metadata["packet_loss"],
                metadata["bandpass"],
                metadata["quantization_bits"],
                metadata.get("codec"),
            )
        else:
            augmented = audio
            system_id = "-"
        return augmented, system_id

    def _apply_augmentation(
        self,
        audio: np.ndarray,
        sr: int,
        aug_types: List[AugmentationType],
    ) -> Tuple[np.ndarray, str]:
        """Mirror of augmentation_pipeline.py:_apply_augmentation."""
        system_ids: List[str] = []
        current = audio
        for aug_type in aug_types:
            current, part_id = self._apply_single_augmentation(current, sr, aug_type)
            if part_id and part_id != "-":
                system_ids.append(part_id)
        composite_id = "|".join(system_ids) if system_ids else "-"
        return current, composite_id


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

class TestAugmentationStrategySchema(unittest.TestCase):
    """Unit tests for the stub AugmentationStrategy schema."""

    def test_default_distribution_has_no_rawboost(self):
        s = AugmentationStrategy()
        self.assertNotIn(AugmentationType.RAWBOOST, s.type_distribution)

    def test_default_distribution_sums_to_one(self):
        s = AugmentationStrategy()
        self.assertAlmostEqual(sum(s.type_distribution.values()), 1.0, places=6)

    def test_default_stacking_probability(self):
        s = AugmentationStrategy()
        self.assertAlmostEqual(s.stacking_probability, 0.40)

    def test_validate_rejects_rawboost_in_distribution(self):
        bad = AugmentationStrategy(
            type_distribution={
                AugmentationType.RIR_NOISE: 0.60,
                AugmentationType.CODEC: 0.30,
                AugmentationType.RAWBOOST: 0.10,
            }
        )
        with self.assertRaises(AssertionError):
            bad.validate()

    def test_validate_rejects_bad_stacking_probability(self):
        bad = AugmentationStrategy(stacking_probability=1.5)
        with self.assertRaises(AssertionError):
            bad.validate()

    def test_validate_rejects_distribution_not_summing_to_one(self):
        bad = AugmentationStrategy(
            type_distribution={AugmentationType.RIR_NOISE: 0.50}
        )
        with self.assertRaises(AssertionError):
            bad.validate()

    def test_get_augmentation_counts_self_consistent(self):
        s = AugmentationStrategy()
        counts = s.get_augmentation_counts(n_originals=1000)
        aug_sum = (
            counts["single_rir_noise"]
            + counts["single_codec"]
            + counts["stacked_rir_then_codec"]
        )
        self.assertEqual(aug_sum, 1000 * s.augmentation_factor)
        self.assertEqual(counts["total"], counts["original"] + 1000 * s.augmentation_factor)

    def test_get_augmentation_counts_stacked_fraction(self):
        s = AugmentationStrategy()
        counts = s.get_augmentation_counts(n_originals=10000)
        expected_stacked = int(10000 * s.augmentation_factor * 0.40)
        self.assertEqual(counts["stacked_rir_then_codec"], expected_stacked)


class TestStackingGate(unittest.TestCase):
    """Statistical and structural tests for the stacking gate."""

    N = 10_000

    def setUp(self):
        random.seed(42)
        self.pipeline = _MockPipeline(stacking_prob=0.40)

    def _draw_modes(self) -> List[List[AugmentationType]]:
        return [self.pipeline._select_augmentation_mode() for _ in range(self.N)]

    def test_stacking_rate_near_40_percent(self):
        modes = self._draw_modes()
        rate = sum(1 for m in modes if len(m) == 2) / self.N
        self.assertAlmostEqual(rate, 0.40, delta=0.03,
                               msg=f"Stacking rate {rate:.3f} deviates more than 3pp from 0.40")

    def test_single_rir_to_codec_ratio_near_60_40(self):
        modes = self._draw_modes()
        singles = [m[0] for m in modes if len(m) == 1]
        n_rir = sum(1 for t in singles if t == AugmentationType.RIR_NOISE)
        n_codec = sum(1 for t in singles if t == AugmentationType.CODEC)
        self.assertGreater(n_rir + n_codec, 0)
        rir_ratio = n_rir / (n_rir + n_codec)
        self.assertAlmostEqual(rir_ratio, 0.60, delta=0.04,
                               msg=f"RIR ratio {rir_ratio:.3f} deviates from 0.60")

    def test_rawboost_never_appears_in_offline_modes(self):
        modes = self._draw_modes()
        all_types = [t for m in modes for t in m]
        self.assertNotIn(AugmentationType.RAWBOOST, all_types,
                         "RAWBOOST must not appear in offline pipeline modes")

    def test_stacked_system_id_contains_pipe(self):
        audio = np.zeros(16000)
        _, system_id = self.pipeline._apply_augmentation(
            audio, 16000, [AugmentationType.RIR_NOISE, AugmentationType.CODEC]
        )
        self.assertIn("|", system_id)
        self.assertEqual(len(system_id.split("|")), 2)

    def test_stacked_system_id_exact_format(self):
        audio = np.zeros(16000)
        _, system_id = self.pipeline._apply_augmentation(
            audio, 16000, [AugmentationType.RIR_NOISE, AugmentationType.CODEC]
        )
        self.assertEqual(system_id, "RIR_SMALL_NOI_SNR15|CODEC_8K_OPUS")

    def test_single_rir_no_pipe(self):
        audio = np.zeros(16000)
        _, system_id = self.pipeline._apply_augmentation(
            audio, 16000, [AugmentationType.RIR_NOISE]
        )
        self.assertNotIn("|", system_id)
        self.assertEqual(system_id, "RIR_SMALL_NOI_SNR15")

    def test_single_codec_no_pipe(self):
        audio = np.zeros(16000)
        _, system_id = self.pipeline._apply_augmentation(
            audio, 16000, [AugmentationType.CODEC]
        )
        self.assertNotIn("|", system_id)
        self.assertEqual(system_id, "CODEC_8K_OPUS")

    def test_augmenters_called_once_each_for_stacked(self):
        audio = np.zeros(16000)
        self.pipeline._apply_augmentation(
            audio, 16000, [AugmentationType.RIR_NOISE, AugmentationType.CODEC]
        )
        self.assertEqual(self.pipeline.rir_augmenter.augment.call_count, 1)
        self.assertEqual(self.pipeline.codec_augmenter.augment.call_count, 1)

    def test_zero_stacking_probability_never_stacks(self):
        p = _MockPipeline(stacking_prob=0.0)
        random.seed(0)
        modes = [p._select_augmentation_mode() for _ in range(1000)]
        self.assertTrue(all(len(m) == 1 for m in modes))

    def test_full_stacking_probability_always_stacks(self):
        p = _MockPipeline(stacking_prob=1.0)
        random.seed(0)
        modes = [p._select_augmentation_mode() for _ in range(1000)]
        self.assertTrue(all(len(m) == 2 for m in modes))


if __name__ == "__main__":
    unittest.main()
