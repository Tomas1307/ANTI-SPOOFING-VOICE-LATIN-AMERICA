"""
Abstract base class for voice cloning attack strategies.

Each concrete strategy wraps a specific voice cloning system and exposes
a uniform interface for generating cloned speech from text and reference audio.
"""
from abc import ABC, abstractmethod
from pathlib import Path


VALID_ATTACK_SYSTEMS = [
    "fishgram",
    "qwen",
    "cosyvoice",
    "outetts",
    "chatterbox",
    "openvoice",
    "omnivoice",
]


class AttackStrategy(ABC):
    """Abstract interface for voice cloning attack strategies.

    Concrete implementations wrap specific TTS/voice cloning systems
    (Fish Speech, Qwen3-TTS, CosyVoice, OuteTTS, Chatterbox, OpenVoice)
    and provide a consistent API for generating cloned speech.

    The lifecycle is: load_model() -> generate() (N times) -> cleanup().
    """

    @abstractmethod
    def load_model(self, device: str) -> None:
        """Load the voice cloning model onto the specified device.

        Args:
            device: Compute device string (e.g., 'cuda:0', 'cpu').
        """

    @abstractmethod
    def generate(
        self,
        text: str,
        reference_audio_path: Path,
        output_path: Path,
        reference_text: str = "",
        seed: int | None = None,
    ) -> float:
        """Generate cloned speech for the given text using the speaker's reference audio.

        Args:
            text: Text transcript to synthesize.
            reference_audio_path: Path to the speaker's reference audio file.
            output_path: Path where the generated audio will be saved.
            reference_text: Optional transcript of the reference audio (some systems need this).
            seed: Optional random seed for reproducible generation.

        Returns:
            Generation time in seconds.
        """

    @abstractmethod
    def cleanup(self) -> None:
        """Release model resources and GPU memory."""

    @abstractmethod
    def name(self) -> str:
        """Return the attack system identifier for protocol files.

        Returns:
            Uppercase system name (e.g., 'FISHGRAM', 'QWEN3TTS').
        """

    @abstractmethod
    def needs_reference_transcript(self) -> bool:
        """Whether this system requires a text transcript of the reference audio.

        Returns:
            True if the system needs reference_text for optimal voice cloning.
        """
