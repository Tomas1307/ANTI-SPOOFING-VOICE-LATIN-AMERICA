"""
Tests for the Partial Spoof pipeline.

Unit tests for individual steps (with mocked dependencies) and integration
tests for the end-to-end pipeline. Designed to run on ml-server03 with
GPU access for integration tests.

Run with: pytest app/tests/test_partial_spoof_pipeline.py -v
"""
import json
import random
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import soundfile as sf

from app.pipeline.partial_spoof.schemas.pipeline_config import PartialSpoofPipelineConfig
from app.pipeline.partial_spoof.schemas.word_alignment import WordAlignment
from app.pipeline.partial_spoof.schemas.spliced_word_info import SplicedWordInfo
from app.pipeline.partial_spoof.schemas.splice_metadata_entry import SpliceMetadataEntry
from app.pipeline.partial_spoof.schemas.transcription_result import TranscriptionResult
from app.pipeline.partial_spoof.schemas.cloned_generation_result import ClonedGenerationResult
from app.pipeline.partial_spoof.schemas.alignment_result import AlignmentResult
from app.pipeline.partial_spoof.schemas.word_selection_result import WordSelectionResult
from app.pipeline.partial_spoof.schemas.splice_result import SpliceResult
from app.pipeline.partial_spoof.schemas.splice_quality_result import SpliceQualityResult
from app.pipeline.partial_spoof.schemas.formatting_result import FormattingResult
from app.pipeline.partial_spoof.strategies.base_strategy import AttackStrategy
from app.pipeline.partial_spoof.utils.crossfade import apply_crossfade
from app.pipeline.partial_spoof.utils.splice_engine import splice_words
from app.pipeline.partial_spoof.utils.strategy_factory import create_attack_strategy


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_output_dir(tmp_path):
    """Create a temporary output directory for pipeline artifacts."""
    output_dir = tmp_path / "partial_spoof_test"
    output_dir.mkdir()
    return output_dir


@pytest.fixture
def sample_audio_16k(tmp_path):
    """Create a synthetic 16kHz WAV file (3 seconds of sine wave)."""
    sr = 16000
    duration = 3.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    path = tmp_path / "test_audio.wav"
    sf.write(str(path), audio, sr)
    return path


@pytest.fixture
def sample_transcripts_json(tmp_output_dir):
    """Create a mock bonafide_transcripts.json with known word counts."""
    transcripts = {
        "spk001_utt001": {
            "speaker_id": "spk001",
            "split": "train",
            "audio_path": "/fake/spk001/train/utt001.wav",
            "transcript": "el gato negro duerme sobre la mesa",
            "word_count": 7,
            "word_timestamps": [
                {"word": "el", "start": 0.0, "end": 0.15},
                {"word": "gato", "start": 0.15, "end": 0.45},
                {"word": "negro", "start": 0.45, "end": 0.80},
                {"word": "duerme", "start": 0.80, "end": 1.20},
                {"word": "sobre", "start": 1.20, "end": 1.55},
                {"word": "la", "start": 1.55, "end": 1.65},
                {"word": "mesa", "start": 1.65, "end": 2.00},
            ],
        },
        "spk001_utt002": {
            "speaker_id": "spk001",
            "split": "train",
            "audio_path": "/fake/spk001/train/utt002.wav",
            "transcript": "la casa grande esta en la colina verde junto al rio",
            "word_count": 11,
            "word_timestamps": [
                {"word": "la", "start": 0.0, "end": 0.10},
                {"word": "casa", "start": 0.10, "end": 0.40},
                {"word": "grande", "start": 0.40, "end": 0.75},
                {"word": "esta", "start": 0.75, "end": 1.00},
                {"word": "en", "start": 1.00, "end": 1.10},
                {"word": "la", "start": 1.10, "end": 1.20},
                {"word": "colina", "start": 1.20, "end": 1.60},
                {"word": "verde", "start": 1.60, "end": 1.90},
                {"word": "junto", "start": 1.90, "end": 2.20},
                {"word": "al", "start": 2.20, "end": 2.30},
                {"word": "rio", "start": 2.30, "end": 2.60},
            ],
        },
        "spk001_utt003": {
            "speaker_id": "spk001",
            "split": "val",
            "audio_path": "/fake/spk001/val/utt003.wav",
            "transcript": "el presidente anuncio nuevas medidas economicas para combatir la inflacion del pais entero",
            "word_count": 13,
            "word_timestamps": [
                {"word": "el", "start": 0.0, "end": 0.10},
                {"word": "presidente", "start": 0.10, "end": 0.55},
                {"word": "anuncio", "start": 0.55, "end": 0.95},
                {"word": "nuevas", "start": 0.95, "end": 1.25},
                {"word": "medidas", "start": 1.25, "end": 1.65},
                {"word": "economicas", "start": 1.65, "end": 2.15},
                {"word": "para", "start": 2.15, "end": 2.35},
                {"word": "combatir", "start": 2.35, "end": 2.75},
                {"word": "la", "start": 2.75, "end": 2.85},
                {"word": "inflacion", "start": 2.85, "end": 3.30},
                {"word": "del", "start": 3.30, "end": 3.45},
                {"word": "pais", "start": 3.45, "end": 3.75},
                {"word": "entero", "start": 3.75, "end": 4.10},
            ],
        },
    }
    path = tmp_output_dir / "bonafide_transcripts.json"
    with open(path, "w") as f:
        json.dump(transcripts, f)
    return path, transcripts


@pytest.fixture
def sample_alignment_json(tmp_output_dir, sample_transcripts_json):
    """Create a mock alignment_metadata.json with known word timestamps."""
    _, transcripts = sample_transcripts_json
    alignment = {}
    for key, entry in transcripts.items():
        alignment[key] = {
            "speaker_id": entry["speaker_id"],
            "split": entry["split"],
            "transcript": entry["transcript"],
            "bonafide_audio_path": entry["audio_path"],
            "cloned_audio_path": f"/fake/cloned/FISHGRAM_{key}.wav",
            "bonafide_words": entry["word_timestamps"],
            "cloned_words": entry["word_timestamps"],
            "cloned_transcript": entry["transcript"],
            "word_count": entry["word_count"],
        }
    path = tmp_output_dir / "alignment_metadata.json"
    with open(path, "w") as f:
        json.dump(alignment, f)
    return path, alignment


class MockAttackStrategy(AttackStrategy):
    """Mock strategy that copies reference audio as output."""

    def __init__(self):
        self._loaded = False

    def load_model(self, device: str) -> None:
        self._loaded = True

    def generate(
        self, text, reference_audio_path, output_path, reference_text="", seed=None
    ) -> float:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sr = 16000
        duration = len(text.split()) * 0.3
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        audio = 0.3 * np.sin(2 * np.pi * 220 * t).astype(np.float32)
        sf.write(str(output_path), audio, sr)
        return 0.5

    def cleanup(self) -> None:
        self._loaded = False

    def name(self) -> str:
        return "MOCK"

    def needs_reference_transcript(self) -> bool:
        return False


# ---------------------------------------------------------------------------
# Schema Tests
# ---------------------------------------------------------------------------


class TestSchemas:
    """Test Pydantic schema validation."""

    def test_pipeline_config_defaults(self):
        """Verify default config values."""
        config = PartialSpoofPipelineConfig()
        assert config.attack_system == "fishgram"
        assert config.run_step_1 is True
        assert config.tiers == ["W1", "W2", "W3"]

    def test_word_alignment_creation(self):
        """Verify WordAlignment fields."""
        wa = WordAlignment(word="casa", start_seconds=0.5, end_seconds=0.9)
        assert wa.word == "casa"
        assert wa.confidence == 0.0

    def test_spliced_word_info_creation(self):
        """Verify SplicedWordInfo fields."""
        swi = SplicedWordInfo(
            word="gato",
            word_index=1,
            bonafide_start_s=0.15,
            bonafide_end_s=0.45,
            cloned_start_s=0.20,
            cloned_end_s=0.50,
            duration_ratio=1.0,
            crossfade_ms=5.0,
        )
        assert swi.word_index == 1
        assert swi.duration_ratio == 1.0

    def test_splice_metadata_entry_creation(self):
        """Verify SpliceMetadataEntry with nested SplicedWordInfo."""
        entry = SpliceMetadataEntry(
            sample_id="spk001_utt001_W1",
            speaker_id="spk001",
            split="train",
            tier="W1",
            attack_system="FISHGRAM",
            bonafide_audio_path=Path("/fake/bonafide.wav"),
            cloned_audio_path=Path("/fake/cloned.wav"),
            spliced_audio_path=Path("/fake/spliced.wav"),
            transcript="el gato negro",
            total_words=3,
            spoofed_words=[
                SplicedWordInfo(
                    word="gato", word_index=1,
                    bonafide_start_s=0.15, bonafide_end_s=0.45,
                    cloned_start_s=0.20, cloned_end_s=0.50,
                    duration_ratio=1.0, crossfade_ms=5.0,
                )
            ],
            spoof_word_ratio=1 / 3,
            spoof_duration_ratio=0.15,
            total_duration_s=1.5,
        )
        assert entry.tier == "W1"
        assert len(entry.spoofed_words) == 1


# ---------------------------------------------------------------------------
# Strategy Factory Tests
# ---------------------------------------------------------------------------


class TestStrategyFactory:
    """Test attack strategy factory."""

    def test_factory_raises_on_unknown(self):
        """Unknown system raises ValueError."""
        with pytest.raises(ValueError, match="Unknown attack system"):
            create_attack_strategy("nonexistent_system")

    def test_factory_valid_systems(self):
        """All valid system names are recognized (import may fail on local machine)."""
        from app.pipeline.partial_spoof.strategies.base_strategy import VALID_ATTACK_SYSTEMS
        assert "fishgram" in VALID_ATTACK_SYSTEMS
        assert "qwen" in VALID_ATTACK_SYSTEMS
        assert "cosyvoice" in VALID_ATTACK_SYSTEMS
        assert "outetts" in VALID_ATTACK_SYSTEMS
        assert "chatterbox" in VALID_ATTACK_SYSTEMS
        assert "openvoice" in VALID_ATTACK_SYSTEMS

    def test_mock_strategy_implements_interface(self):
        """Mock strategy satisfies AttackStrategy interface."""
        strategy = MockAttackStrategy()
        assert hasattr(strategy, "load_model")
        assert hasattr(strategy, "generate")
        assert hasattr(strategy, "cleanup")
        assert hasattr(strategy, "name")
        assert hasattr(strategy, "needs_reference_transcript")
        assert strategy.name() == "MOCK"
        assert strategy.needs_reference_transcript() is False


# ---------------------------------------------------------------------------
# Crossfade Tests
# ---------------------------------------------------------------------------


class TestCrossfade:
    """Test audio crossfade utility."""

    def test_crossfade_basic(self):
        """Crossfade produces correct output length."""
        a = np.ones(100, dtype=np.float32)
        b = np.ones(100, dtype=np.float32) * 0.5
        result = apply_crossfade(a, b, crossfade_samples=20)
        assert len(result) == 180

    def test_crossfade_zero_samples(self):
        """Zero crossfade is plain concatenation."""
        a = np.ones(50, dtype=np.float32)
        b = np.zeros(50, dtype=np.float32)
        result = apply_crossfade(a, b, crossfade_samples=0)
        assert len(result) == 100

    def test_crossfade_raises_on_short_segment(self):
        """Raises ValueError if segment shorter than crossfade."""
        a = np.ones(5, dtype=np.float32)
        b = np.ones(100, dtype=np.float32)
        with pytest.raises(ValueError, match="segment_before"):
            apply_crossfade(a, b, crossfade_samples=10)

    def test_crossfade_smooth_transition(self):
        """Crossfade region has intermediate values."""
        a = np.ones(100, dtype=np.float32)
        b = np.zeros(100, dtype=np.float32)
        result = apply_crossfade(a, b, crossfade_samples=20)
        overlap = result[80:100]
        assert np.all(overlap >= 0.0)
        assert np.all(overlap <= 1.0)
        assert overlap[0] > overlap[-1]


# ---------------------------------------------------------------------------
# Word Selector Tests
# ---------------------------------------------------------------------------


class TestWordSelector:
    """Test word selection logic."""

    def test_selector_respects_tier_minimums(self, tmp_output_dir, sample_alignment_json):
        """5-word utterance should not produce W2 or W3."""
        from app.pipeline.partial_spoof.steps.step_04_select_words import WordSelector

        _, alignment = sample_alignment_json

        short_alignment = {
            "spk_short": {
                "speaker_id": "spk",
                "split": "train",
                "transcript": "hola que tal amigo",
                "bonafide_words": [
                    {"word": "hola", "start": 0, "end": 0.3},
                    {"word": "que", "start": 0.3, "end": 0.5},
                    {"word": "tal", "start": 0.5, "end": 0.7},
                    {"word": "amigo", "start": 0.7, "end": 1.0},
                ],
                "word_count": 4,
            }
        }
        alignment_path = tmp_output_dir / "alignment_metadata.json"
        with open(alignment_path, "w") as f:
            json.dump(short_alignment, f)

        with patch("app.pipeline.partial_spoof.steps.step_04_select_words.settings") as mock_settings:
            mock_settings.OUTPUT_DIR = tmp_output_dir
            mock_settings.RANDOM_SEED = 42
            mock_settings.REQUIRE_NON_ADJACENT = True
            mock_settings.ENABLED_TIERS = ["W1", "W2", "W3"]
            mock_settings.MIN_WORDS_W1 = 4
            mock_settings.MIN_WORDS_W2 = 8
            mock_settings.MIN_WORDS_W3 = 12

            selector = WordSelector(output_dir=tmp_output_dir)
            result = selector.execute()

        with open(result.selection_path) as f:
            selections = json.load(f)

        assert result.tier_counts.get("W1", 0) == 1
        assert result.tier_counts.get("W2", 0) == 0
        assert result.tier_counts.get("W3", 0) == 0

    def test_selector_non_adjacent_constraint(self):
        """Selected indices should differ by at least 2."""
        from app.pipeline.partial_spoof.steps.step_04_select_words import WordSelector

        selector = WordSelector()
        rng = random.Random(42)

        for _ in range(50):
            indices = selector._select_non_adjacent(n_select=3, total_words=15, rng=rng)
            assert indices is not None
            for i in range(len(indices) - 1):
                assert indices[i + 1] - indices[i] >= 2

    def test_selector_deterministic_with_seed(self):
        """Same seed produces same selection."""
        from app.pipeline.partial_spoof.steps.step_04_select_words import WordSelector

        selector = WordSelector()
        rng1 = random.Random(123)
        rng2 = random.Random(123)

        result1 = selector._select_non_adjacent(3, 20, rng1)
        result2 = selector._select_non_adjacent(3, 20, rng2)
        assert result1 == result2

    def test_selector_w1_selects_exactly_1(self):
        """W1 tier selects exactly 1 word."""
        from app.pipeline.partial_spoof.steps.step_04_select_words import WordSelector

        selector = WordSelector()
        rng = random.Random(42)
        result = selector._select_non_adjacent(1, 10, rng)
        assert result is not None
        assert len(result) == 1

    def test_selector_w3_selects_exactly_3(self):
        """W3 tier selects exactly 3 words."""
        from app.pipeline.partial_spoof.steps.step_04_select_words import WordSelector

        selector = WordSelector()
        rng = random.Random(42)
        result = selector._select_non_adjacent(3, 15, rng)
        assert result is not None
        assert len(result) == 3


# ---------------------------------------------------------------------------
# Splice Engine Tests
# ---------------------------------------------------------------------------


class TestSpliceEngine:
    """Test the core splice_words function."""

    def _make_audio_and_words(self, n_words=5, word_dur_s=0.3, sr=16000):
        """Generate synthetic audio with evenly spaced words."""
        total_samples = int(n_words * word_dur_s * sr)
        audio = np.random.randn(total_samples).astype(np.float32) * 0.1
        words = []
        for i in range(n_words):
            start = i * word_dur_s
            end = (i + 1) * word_dur_s
            words.append({"word": f"word{i}", "start": start, "end": end})
        return audio, words

    def test_splice_output_is_valid(self):
        """Spliced output is a valid numpy array."""
        bonafide, b_words = self._make_audio_and_words(10)
        cloned, c_words = self._make_audio_and_words(10)

        result, details = splice_words(
            bonafide_audio=bonafide,
            cloned_audio=cloned,
            bonafide_words=b_words,
            cloned_words=c_words,
            selected_indices=[2],
            sample_rate=16000,
            crossfade_ms=5.0,
            max_silence_steal_ms=50.0,
            max_stretch_ratio=1.1,
        )
        assert isinstance(result, np.ndarray)
        assert len(result) > 0
        assert len(details) == 1

    def test_splice_w1_has_one_detail(self):
        """W1 splice produces exactly 1 splice detail."""
        bonafide, b_words = self._make_audio_and_words(8)
        cloned, c_words = self._make_audio_and_words(8)

        _, details = splice_words(
            bonafide_audio=bonafide,
            cloned_audio=cloned,
            bonafide_words=b_words,
            cloned_words=c_words,
            selected_indices=[3],
            sample_rate=16000,
            crossfade_ms=2.0,
            max_silence_steal_ms=50.0,
            max_stretch_ratio=1.1,
        )
        assert len(details) == 1
        assert details[0]["word_index"] == 3

    def test_splice_w3_has_three_details(self):
        """W3 splice produces exactly 3 splice details."""
        bonafide, b_words = self._make_audio_and_words(15)
        cloned, c_words = self._make_audio_and_words(15)

        _, details = splice_words(
            bonafide_audio=bonafide,
            cloned_audio=cloned,
            bonafide_words=b_words,
            cloned_words=c_words,
            selected_indices=[2, 7, 12],
            sample_rate=16000,
            crossfade_ms=2.0,
            max_silence_steal_ms=50.0,
            max_stretch_ratio=1.1,
        )
        assert len(details) == 3

    def test_splice_duration_reasonable(self):
        """Output duration within 15% of bonafide duration."""
        bonafide, b_words = self._make_audio_and_words(10, word_dur_s=0.3)
        cloned, c_words = self._make_audio_and_words(10, word_dur_s=0.35)

        result, _ = splice_words(
            bonafide_audio=bonafide,
            cloned_audio=cloned,
            bonafide_words=b_words,
            cloned_words=c_words,
            selected_indices=[3, 7],
            sample_rate=16000,
            crossfade_ms=2.0,
            max_silence_steal_ms=50.0,
            max_stretch_ratio=1.15,
        )
        ratio = len(result) / len(bonafide)
        assert 0.85 < ratio < 1.15


# ---------------------------------------------------------------------------
# Output Formatter Tests
# ---------------------------------------------------------------------------


class TestOutputFormatter:
    """Test LA output formatting."""

    def test_formatter_creates_la_structure(self, tmp_output_dir):
        """Verify LA directory structure is created."""
        from app.pipeline.partial_spoof.steps.step_07_format_output import OutputFormatter

        splice_meta = {
            "spk001_utt001_W1": {
                "sample_id": "spk001_utt001_W1",
                "speaker_id": "spk001",
                "split": "train",
                "tier": "W1",
                "attack_system": "MOCK",
                "spliced_audio_path": "",
                "bonafide_audio_path": "",
                "cloned_audio_path": "",
                "transcript": "test",
                "total_words": 4,
                "spoofed_words": [],
                "spoof_word_ratio": 0.25,
                "spoof_duration_ratio": 0.1,
                "total_duration_s": 1.0,
            }
        }

        sr = 16000
        audio = np.zeros(sr, dtype=np.float32)
        spliced_dir = tmp_output_dir / "spliced"
        spliced_dir.mkdir()
        audio_path = spliced_dir / "MOCK_PSW1_spk001_utt001.wav"
        sf.write(str(audio_path), audio, sr)
        splice_meta["spk001_utt001_W1"]["spliced_audio_path"] = str(audio_path)

        with open(tmp_output_dir / "splice_metadata.json", "w") as f:
            json.dump(splice_meta, f)

        quality_data = {
            "spk001_utt001_W1": {"spliced_audio_path": str(audio_path), "passed": True}
        }
        with open(tmp_output_dir / "splice_quality_metadata.json", "w") as f:
            json.dump(quality_data, f)

        with patch("app.pipeline.partial_spoof.steps.step_07_format_output.settings") as mock_s:
            mock_s.OUTPUT_DIR = tmp_output_dir
            mock_s.SAMPLE_RATE = 16000
            mock_s.AUDIO_ID_START_W1 = 12000000
            mock_s.AUDIO_ID_START_W2 = 13000000
            mock_s.AUDIO_ID_START_W3 = 14000000

            formatter = OutputFormatter(system_id_prefix="MOCK", output_dir=tmp_output_dir)
            result = formatter.execute()

        la_dir = tmp_output_dir / "LA"
        assert la_dir.exists()
        assert (la_dir / "ASVspoof2019_LA_train" / "flac").exists()
        assert (la_dir / "ASVspoof2019_LA_dev" / "flac").exists()
        assert (la_dir / "ASVspoof2019_LA_eval" / "flac").exists()
        assert result.total_samples["train"] == 1

    def test_formatter_protocol_format(self, tmp_output_dir):
        """Protocol entries match expected format."""
        from app.pipeline.partial_spoof.steps.step_07_format_output import OutputFormatter

        sr = 16000
        audio = np.zeros(sr, dtype=np.float32)
        spliced_dir = tmp_output_dir / "spliced"
        spliced_dir.mkdir()
        audio_path = spliced_dir / "test.wav"
        sf.write(str(audio_path), audio, sr)

        splice_meta = {
            "test_W1": {
                "sample_id": "test_W1",
                "speaker_id": "arf_00295",
                "split": "train",
                "tier": "W1",
                "attack_system": "FISHGRAM",
                "spliced_audio_path": str(audio_path),
                "bonafide_audio_path": "",
                "cloned_audio_path": "",
                "transcript": "test",
                "total_words": 4,
                "spoofed_words": [],
                "spoof_word_ratio": 0.25,
                "spoof_duration_ratio": 0.1,
                "total_duration_s": 1.0,
            }
        }
        with open(tmp_output_dir / "splice_metadata.json", "w") as f:
            json.dump(splice_meta, f)
        with open(tmp_output_dir / "splice_quality_metadata.json", "w") as f:
            json.dump({"test_W1": {"spliced_audio_path": str(audio_path), "passed": True}}, f)

        with patch("app.pipeline.partial_spoof.steps.step_07_format_output.settings") as mock_s:
            mock_s.OUTPUT_DIR = tmp_output_dir
            mock_s.SAMPLE_RATE = 16000
            mock_s.AUDIO_ID_START_W1 = 12000000
            mock_s.AUDIO_ID_START_W2 = 13000000
            mock_s.AUDIO_ID_START_W3 = 14000000

            formatter = OutputFormatter(system_id_prefix="FISHGRAM", output_dir=tmp_output_dir)
            result = formatter.execute()

        protocol_path = result.protocol_files["train"]
        content = protocol_path.read_text()
        assert "arf_00295" in content
        assert "FISHGRAM_PSW1" in content
        assert "partial_spoof" in content
        assert "LA_T_12000000" in content
