"""
Singleton wrapper for NVIDIA Parakeet TDT automatic speech recognition model.

Parakeet TDT 0.6b-v3 (Token-and-Duration Transducer) provides word-level
timestamps natively and supports 25 languages including Spanish (3.45% WER
on FLEURS). This makes it the preferred model for audio content validation
and spurious prefix detection in TTS attack pipelines.

Installation on ml-server03 (per pipeline venv):
    pip install -U "nemo_toolkit[asr]"
"""
from pathlib import Path
from typing import List, Tuple

from loguru import logger

from app.utils.word_timestamp import WordTimestamp


class ParakeetTranscriber:
    """Singleton wrapper for NVIDIA Parakeet TDT ASR model.

    The model is loaded once on first use and reused across all subsequent
    calls. Loading NeMo models is expensive (~10s), so repeated instantiation
    must be avoided.

    Usage:
        transcriber = ParakeetTranscriber()
        transcriber.load(model_id="nvidia/parakeet-tdt-0.6b-v3", device="cuda")
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

    def load(self, model_id: str = "nvidia/parakeet-tdt-0.6b-v3", device: str = "cuda") -> None:
        """Load the Parakeet TDT model from HuggingFace if not already loaded.

        Args:
            model_id: HuggingFace model identifier for Parakeet TDT.
                Defaults to the 0.6b-v3 multilingual variant (25 languages).
            device: Compute device. Use 'cuda' on ml-server03 A40 GPUs.

        Raises:
            ImportError: If nemo_toolkit[asr] is not installed.
            RuntimeError: If the model fails to load from HuggingFace.
        """
        if self._model is not None:
            logger.debug("Parakeet model already loaded, skipping.")
            return

        import nemo.collections.asr as nemo_asr
        from omegaconf import open_dict

        logger.info(f"Loading Parakeet TDT model: {model_id} on {device}")
        self._model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_id)
        self._model.eval()

        if device.startswith("cuda"):
            self._model = self._model.cuda()

        self._disable_cuda_graphs()
        logger.info("Parakeet TDT model loaded successfully.")

    def _disable_cuda_graphs(self) -> None:
        """Disable CUDA graph decoder and pre-enable timestamps mode.

        Two problems are solved here:

        1. Some pipeline venvs ship PyTorch with CUDA runtime headers newer than
           the server driver (e.g. CUDA 12.8 runtime vs driver 560.35.03 which
           caps at ~12.6). NeMo's CUDA graph greedy decoder calls
           cudaStreamBeginCapture which fails with cudaError 222 in that scenario.

        2. Passing timestamps=True to model.transcribe() causes NeMo to internally
           call change_decoding_strategy, which re-creates the decoder and resets
           use_cuda_graph_decoder to True. By enabling timestamps here alongside
           the CUDA graph disable, we avoid NeMo ever needing to reconfigure.

        Both transcribe() and transcribe_with_timestamps() call model.transcribe()
        without timestamps=True; timestamps are always available in the output.
        """
        from omegaconf import open_dict

        with open_dict(self._model.cfg.decoding):
            self._model.cfg.decoding.greedy.loop_labels = False
            self._model.cfg.decoding.greedy.use_cuda_graph_decoder = False
            self._model.cfg.decoding.compute_timestamps = True
        self._model.change_decoding_strategy(self._model.cfg.decoding)
        logger.debug("CUDA graph decoder disabled, timestamps pre-enabled.")

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

        results = self._model.transcribe([str(audio_path)])
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
