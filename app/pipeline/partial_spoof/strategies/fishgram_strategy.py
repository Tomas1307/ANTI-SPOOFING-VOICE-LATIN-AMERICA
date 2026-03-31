"""Fish Speech (FishGram) attack strategy for the Partial Spoof pipeline.

Wraps the Fish Speech HTTP API server to generate voice-cloned speech.
The server must be running on ml-server03 before using this strategy.
"""
import base64
import time

import requests
import soundfile as sf
from pathlib import Path
from loguru import logger

from app.pipeline.partial_spoof.strategies.base_strategy import AttackStrategy
from app.pipeline.fishgram_attack.settings import settings as fishgram_settings


class FishGramStrategy(AttackStrategy):
    """Voice cloning via Fish Speech HTTP API.

    Delegates generation to an external Fish Speech server instance.
    No local model is loaded; all inference happens server-side.

    Attributes:
        api_url: URL of the Fish Speech HTTP API server.
    """

    def __init__(self) -> None:
        """Initialize FishGram strategy with API URL from fishgram settings."""
        self.api_url = fishgram_settings.FISH_SPEECH_API_URL

    def load_model(self, device: str) -> None:
        """Verify server connectivity (no local model to load).

        Args:
            device: Ignored for HTTP-based strategy.

        Raises:
            ConnectionError: If the Fish Speech server is unreachable.
        """
        try:
            response = requests.get(f"{self.api_url}/", timeout=5)
            if response.status_code != 200:
                raise ConnectionError(
                    f"Fish Speech server at {self.api_url} returned status {response.status_code}."
                )
            logger.info(f"FishGramStrategy: Server healthy at {self.api_url}")
        except requests.ConnectionError as exc:
            raise ConnectionError(
                f"Fish Speech server at {self.api_url} is unreachable. "
                "Start it on ml-server03 before running the pipeline."
            ) from exc

    def generate(
        self,
        text: str,
        reference_audio_path: Path,
        output_path: Path,
        reference_text: str = "",
        seed: int | None = None,
    ) -> float:
        """Generate cloned speech via Fish Speech HTTP API.

        Args:
            text: Text to synthesize in the cloned voice.
            reference_audio_path: Path to the speaker reference audio.
            output_path: Path where the generated WAV will be saved.
            reference_text: Transcript of reference audio (optional for Fish Speech).
            seed: Ignored by Fish Speech API.

        Returns:
            Generation time in seconds.

        Raises:
            RuntimeError: If the API returns an error response.
        """
        start_time = time.time()

        with open(reference_audio_path, "rb") as f:
            ref_audio_bytes = f.read()

        payload = {
            "text": text,
            "references": [
                {
                    "audio": base64.b64encode(ref_audio_bytes).decode("utf-8"),
                    "text": reference_text,
                }
            ],
            "format": fishgram_settings.FISH_SPEECH_FORMAT,
            "top_p": fishgram_settings.FISH_SPEECH_TOP_P,
            "temperature": fishgram_settings.FISH_SPEECH_TEMPERATURE,
            "repetition_penalty": fishgram_settings.FISH_SPEECH_REPETITION_PENALTY,
            "streaming": False,
            "normalize": True,
            "max_new_tokens": 1024,
        }

        response = requests.post(
            f"{self.api_url}/v1/tts",
            json=payload,
            timeout=120,
        )

        if response.status_code != 200:
            raise RuntimeError(
                f"Fish Speech API error (HTTP {response.status_code}): "
                f"{response.text[:200]}"
            )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(response.content)

        generation_time = time.time() - start_time
        return generation_time

    def cleanup(self) -> None:
        """No resources to release for HTTP-based strategy."""
        logger.info("FishGramStrategy: Cleanup complete (no local resources).")

    def name(self) -> str:
        """Return the system identifier.

        Returns:
            'FISHGRAM' for protocol file entries.
        """
        return "FISHGRAM"

    def needs_reference_transcript(self) -> bool:
        """Fish Speech does not require reference transcripts.

        Returns:
            False.
        """
        return False
