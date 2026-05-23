"""
FishGram voice cloning unit -- single source of truth.

This module exposes a Cloner class that encapsulates EXACTLY the per-sample
cloning logic for FishGram (Fish Speech HTTP API). It is consumed by both:

    1. fishgram_attack/steps/step_03_generate_speech.py (the standalone
       full-utterance attack pipeline).
    2. partial_spoof/steps/step_02_generate_cloned_speech.py (via the
       Cloner dispatcher).

Unlike the other attacks, FishGram has no local model: it sends text +
base64-encoded reference audio to a Fish Speech HTTP server which must
be reachable at settings.FISH_SPEECH_API_URL before clone_single is
invoked. ``load(device)`` performs a server health check; ``cleanup()``
is a no-op because there is no GPU resource to release.
"""
import base64
import time
from pathlib import Path
from typing import Optional

import librosa
import requests
from loguru import logger

from app.pipeline.fishgram_attack.settings import settings


class Cloner:
    """FishGram cloning unit: HTTP-based cloning via Fish Speech API.

    The instance holds the API URL and (after load) confirms server
    reachability. There is no model to load locally and no per-speaker
    state to cache: every clone_single call sends the reference audio
    bytes inline in the JSON payload.

    Attributes:
        SYSTEM_ID: Uppercase attack identifier used in output filenames.
        NEEDS_REFERENCE_TRANSCRIPT: FishGram accepts an optional reference
            transcript; the standalone pipeline currently passes empty
            string. Set to False so partial_spoof Step 2 skips the
            reference-text fetch (matching the prior strategy behaviour).
        api_url: Resolved Fish Speech server URL.
    """

    SYSTEM_ID: str = "FISHGRAM"
    NEEDS_REFERENCE_TRANSCRIPT: bool = False

    def __init__(self) -> None:
        """Initialise an empty cloner. Call load() before clone_single()."""
        self.api_url: Optional[str] = None

    def load(self, device: str) -> None:
        """Resolve the Fish Speech API URL and verify the server is reachable.

        Args:
            device: Accepted for interface compatibility; FishGram does
                its inference server-side, so the local device does not
                affect anything here.

        Raises:
            ConnectionError: If the Fish Speech server is not reachable
                at ``settings.FISH_SPEECH_API_URL``.
        """
        self.api_url = settings.FISH_SPEECH_API_URL
        logger.info(f"FishGram Cloner: API URL = {self.api_url}")

        try:
            response = requests.get(f"{self.api_url}/", timeout=5)
            ok = response.status_code == 200
        except (requests.ConnectionError, requests.Timeout):
            ok = False

        if not ok:
            raise ConnectionError(
                f"Fish Speech API server is not reachable at {self.api_url}. "
                "Start the server with: cd ~/fish-speech && "
                "CUDA_VISIBLE_DEVICES=1 python -m tools.api_server "
                "--listen 0.0.0.0:8080 "
                "--llama-checkpoint-path checkpoints/s1-mini "
                "--decoder-checkpoint-path checkpoints/s1-mini/codec.pth "
                "--decoder-config-name modded_dac_vq"
            )
        logger.info("FishGram Cloner: server health check OK")

    def prepare_speaker(
        self,
        speaker_id: str,
        reference_audio_path: Path,
    ) -> None:
        """No-op for FishGram.

        The reference audio is sent inline in every HTTP request; there
        is no per-speaker server-side cache to set up.

        Args:
            speaker_id: HABLA speaker identifier (unused).
            reference_audio_path: Speaker reference audio path (unused).
        """
        return None

    def clone_single(
        self,
        text: str,
        reference_audio_path: Path,
        output_path: Path,
        reference_text: str = "",
        seed: Optional[int] = None,
    ) -> tuple:
        """POST one synthesis request to Fish Speech and write the response.

        Args:
            text: Spanish text to synthesise.
            reference_audio_path: Path to the speaker reference audio,
                loaded and base64-encoded into the request payload.
            output_path: Destination WAV path. Server returns audio in
                FISH_SPEECH_FORMAT (typically wav) and we write the bytes
                straight through.
            reference_text: Optional transcript of the reference; left
                empty by default per the legacy strategy behaviour.
            seed: Accepted for interface compatibility; not supported by
                the Fish Speech API.

        Returns:
            Tuple of (generation_time_seconds, audio_duration_seconds).

        Raises:
            RuntimeError: If the API server returns non-200.
            requests.ConnectionError: If the server becomes unreachable
                mid-run.
            RuntimeError: If load() was not called first.
        """
        if self.api_url is None:
            raise RuntimeError("FishGram Cloner: load() must be called before clone_single()")

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
            "format": settings.FISH_SPEECH_FORMAT,
            "top_p": settings.FISH_SPEECH_TOP_P,
            "temperature": settings.FISH_SPEECH_TEMPERATURE,
            "repetition_penalty": settings.FISH_SPEECH_REPETITION_PENALTY,
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
                f"Fish Speech API error {response.status_code}: {response.text}"
            )

        generation_time = time.time() - start_time

        with open(output_path, "wb") as f:
            f.write(response.content)

        synthetic_audio, _ = librosa.load(output_path, sr=settings.SAMPLE_RATE)
        audio_duration = len(synthetic_audio) / settings.SAMPLE_RATE

        return generation_time, audio_duration

    def cleanup(self) -> None:
        """No-op for FishGram (no GPU model loaded locally)."""
        return None
