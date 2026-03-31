"""Abstract base class for voice cloning attack strategies.

Each concrete strategy wraps a specific TTS/voice cloning system and
provides a uniform interface for the Partial Spoof pipeline to generate
cloned speech from text and a speaker reference audio.
"""
from abc import ABC, abstractmethod
from pathlib import Path


class AttackStrategy(ABC):
    """Abstract interface for voice cloning attack systems.

    Concrete implementations wrap specific TTS engines (Fish Speech,
    Qwen3-TTS, CosyVoice, OuteTTS, Chatterbox, OpenVoice) and expose
    a uniform generate() method for the partial spoof pipeline.

    The lifecycle is: load_model() -> generate() (N times) -> cleanup().
    """

    @abstractmethod
    def load_model(self, device: str) -> None:
        """Load the voice cloning model onto the specified device.

        Args:
            device: PyTorch device string (e.g., 'cuda:0', 'cpu').
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
        """Generate cloned speech for the given text and speaker reference.

        Args:
            text: Text to synthesize in the cloned voice.
            reference_audio_path: Path to the speaker reference audio clip.
            output_path: Path where the generated WAV file will be saved.
            reference_text: Optional transcript of the reference audio,
                required by some systems (Qwen, CosyVoice) for better cloning.
            seed: Optional random seed for reproducible generation.

        Returns:
            Generation time in seconds.
        """

    @abstractmethod
    def cleanup(self) -> None:
        """Release model resources and free GPU memory."""

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
            True if reference_text must be non-empty for optimal cloning.
        """
