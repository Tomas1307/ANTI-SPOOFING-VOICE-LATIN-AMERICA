"""
Singleton wrapper for NVIDIA Parakeet TDT 1.1B automatic speech recognition model.

Parakeet TDT (Token-and-Duration Transducer) provides word-level timestamps
natively, making it the preferred model for audio content validation and
spurious prefix detection in TTS attack pipelines.

Installation on ml-server03 (per pipeline venv):
    pip install nemo_toolkit[asr]
    # or lighter variant:
    pip install nvidia-nemo[asr]
"""
from pathlib import Path
from typing import List, Tuple

from loguru import logger

from app.utils.word_timestamp import WordTimestamp


class ParakeetTranscriber:
    """Singleton wrapper for NVIDIA Parakeet TDT 1.1B ASR model.

    The model is loaded once on first use and reused across all subsequent
    calls. Loading NeMo models is expensive (~10s), so repeated instantiation
    must be avoided.

    Usage:
        transcriber = ParakeetTranscriber()
        transcriber.load(model_id="nvidia/parakeet-tdt-1.1b", device="cuda")
        text = transcriber.transcribe(Path("audio.wav"))
        text, timestamps = transcriber.transcribe_with_timestamps(Path("audio.wav"))

    Attributes:
        _instance: Class-level singleton reference.
        _model: Loaded NeMo ASR model (None until load() is called).
    """

    _instance = None
    _model = None

    def __new__(cls) -> "ParakeetTranscriber":
        """Return the singleton instance, creating it if necessary."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def load(self, model_id: str = "nvidia/parakeet-tdt-1.1b", device: str = "cuda") -> None:
        """Load the Parakeet TDT model from HuggingFace if not already loaded.

        Args:
            model_id: HuggingFace model identifier for Parakeet TDT.
                Defaults to the 1.1B parameter variant.
            device: Compute device. Use 'cuda' on ml-server03 A40 GPUs.

        Raises:
            ImportError: If nemo_toolkit[asr] is not installed.
            RuntimeError: If the model fails to load from HuggingFace.
        """
        if self._model is not None:
            logger.debug("Parakeet model already loaded, skipping.")
            return

        import nemo.collections.asr as nemo_asr

        logger.info(f"Loading Parakeet TDT model: {model_id} on {device}")
        self._model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_id)
        self._model.eval()

        if device.startswith("cuda"):
            self._model = self._model.cuda()

        logger.info("Parakeet TDT model loaded successfully.")

    def transcribe(self, audio_path: Path) -> str:
        """Transcribe a single audio file to text.

        Args:
            audio_path: Path to a WAV audio file. Must exist and be readable.

        Returns:
            Transcribed text as a plain string.

        Raises:
            RuntimeError: If load() has not been called before transcribing.
            FileNotFoundError: If audio_path does not exist.
        """
        if self._model is None:
            raise RuntimeError("Parakeet model not loaded. Call load() before transcribing.")
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        results = self._model.transcribe([str(audio_path)])
        result = results[0]
        return result.text if hasattr(result, "text") else str(result)

    def transcribe_with_timestamps(
        self, audio_path: Path
    ) -> Tuple[str, List[WordTimestamp]]:
        """Transcribe a single audio file and return word-level timestamps.

        Parakeet TDT natively supports word-level timestamp prediction via its
        Token-and-Duration Transducer architecture. Timestamps are returned in
        seconds from the beginning of the audio.

        Args:
            audio_path: Path to a WAV audio file. Must exist and be readable.

        Returns:
            Tuple of:
                - Transcribed text as a plain string.
                - List of WordTimestamp objects, one per word, in order.
                  Empty list if the model returns no timestamp data.

        Raises:
            RuntimeError: If load() has not been called before transcribing.
            FileNotFoundError: If audio_path does not exist.
        """
        if self._model is None:
            raise RuntimeError("Parakeet model not loaded. Call load() before transcribing.")
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        results = self._model.transcribe([str(audio_path)], timestamps=True)
        result = results[0]
        text = result.text if hasattr(result, "text") else str(result)

        timestamps: List[WordTimestamp] = []
        if hasattr(result, "timestamp") and result.timestamp:
            for wt in result.timestamp.get("word", []):
                timestamps.append(
                    WordTimestamp(
                        word=wt["word"],
                        start=float(wt["start"]),
                        end=float(wt["end"]),
                    )
                )

        return text, timestamps
