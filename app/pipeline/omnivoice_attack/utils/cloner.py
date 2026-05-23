"""
OmniVoice voice cloning unit -- single source of truth.

This module exposes a Cloner class that encapsulates EXACTLY the per-sample
cloning logic for OmniVoice. It is consumed by both:

    1. omnivoice_attack/steps/step_03_generate_speech.py (the standalone
       full-utterance attack pipeline that iterates over Common Voice text
       prompts per speaker).
    2. partial_spoof/steps/step_02_generate_cloned_speech.py (the partial
       spoof pipeline that iterates over bonafide transcripts to obtain
       full-sentence clones for word-level splicing).

Centralising the cloning logic here means a fix or a setting change applies
to BOTH pipelines automatically -- the divergence bugs that bit us when
duplicate strategies fell out of sync cannot recur for the per-sample
generation path.

OmniVoice is imported from the omnivoice package, which is installed only
inside envs/omnivoice_env. The import lives at module top per CLAUDE.md;
importing this module from another venv will fail at import time, which
is the intended isolation behaviour.
"""
import time
from pathlib import Path
from typing import Optional

import soundfile as sf
import torch
from loguru import logger
from omnivoice import OmniVoice

from app.pipeline.omnivoice_attack.settings import settings
from app.utils.base_cloner import BaseCloner


class Cloner(BaseCloner):
    """OmniVoice cloning unit: load, clone_single, cleanup.

    The instance holds the loaded OmniVoice model. OmniVoice does not
    require per-speaker setup -- the reference audio and reference text
    are passed directly per generate() call -- so prepare_speaker is a
    no-op for this attack.

    Attributes:
        SYSTEM_ID: Uppercase attack identifier used in output filenames.
        NEEDS_REFERENCE_TRANSCRIPT: OmniVoice's diffusion conditioning
            needs the transcript of the reference audio, so callers must
            pass ref_text into clone_single.
        model: Loaded OmniVoice instance (None before load() is called).
    """

    SYSTEM_ID: str = "OMNIVOICE"
    NEEDS_REFERENCE_TRANSCRIPT: bool = True

    def __init__(self) -> None:
        """Initialise an empty cloner. Call load() before clone_single()."""
        self.model: Optional[OmniVoice] = None

    def load(self, device: str) -> None:
        """Load the OmniVoice model onto the target device.

        Args:
            device: PyTorch device string (e.g. 'cuda', 'cuda:0', 'cpu').
                OmniVoice routes via HuggingFace's device_map argument.

        Raises:
            RuntimeError: If model loading fails (VRAM, CUDA, download).
        """
        logger.info(f"OmniVoice Cloner: loading model {settings.OMNIVOICE_MODEL_ID}")
        logger.info(f"  Device: {device}")
        logger.info(f"  Dtype: {settings.DTYPE}")

        start_time = time.time()

        dtype = getattr(torch, settings.DTYPE)
        self.model = OmniVoice.from_pretrained(
            settings.OMNIVOICE_MODEL_ID,
            device_map=device,
            dtype=dtype,
        )

        load_time = time.time() - start_time
        logger.info(f"OmniVoice Cloner: model loaded in {load_time:.1f}s")

    def prepare_speaker(
        self,
        speaker_id: str,
        reference_audio_path: Path,
        reference_text: str = "",
    ) -> None:
        """No-op for OmniVoice.

        OmniVoice does not cache per-speaker state: the reference audio
        and reference text are passed per clone_single() call, and the
        diffusion conditioning happens per-call. Implemented for interface
        symmetry with attacks that do need per-speaker setup (Qwen,
        OpenVoice, OuteTTS).

        Args:
            speaker_id: HABLA speaker identifier (unused).
            reference_audio_path: Speaker reference audio path (unused).
            reference_text: Reference transcript (unused; OmniVoice uses
                it only in clone_single via the same parameter there).
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
        """Generate one OmniVoice clone and write it to output_path.

        Args:
            text: Spanish text to synthesise.
            reference_audio_path: Path to the speaker reference audio.
            output_path: Destination WAV path. Written at OmniVoice's
                native sample rate (24 kHz); downstream stages resample
                via librosa on load.
            reference_text: Parakeet transcript of the reference audio.
                Required by OmniVoice's diffusion conditioning to align
                content prosody with the target voice.
            seed: Accepted for interface compatibility; OmniVoice does
                not expose a deterministic sampling seed.

        Returns:
            Tuple of (generation_time_seconds, audio_duration_seconds).

        Raises:
            RuntimeError: If model.generate fails or the loaded model is
                None (load() was never called).
        """
        if self.model is None:
            raise RuntimeError("OmniVoice Cloner: load() must be called before clone_single()")

        start_time = time.time()

        audios = self.model.generate(
            text=text,
            ref_audio=str(reference_audio_path),
            ref_text=reference_text,
            num_step=settings.OMNIVOICE_NUM_STEP,
            speed=settings.OMNIVOICE_SPEED,
            language=settings.OMNIVOICE_LANGUAGE,
        )

        generation_time = time.time() - start_time

        audio = audios[0]
        sf.write(
            str(output_path),
            audio,
            settings.OMNIVOICE_NATIVE_SAMPLE_RATE,
        )

        audio_duration = len(audio) / settings.OMNIVOICE_NATIVE_SAMPLE_RATE
        return generation_time, audio_duration

    def cleanup(self) -> None:
        """Release the OmniVoice model and clear CUDA memory."""
        self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("OmniVoice Cloner: GPU memory released")
