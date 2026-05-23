"""
OuteTTS voice cloning unit -- single source of truth.

This module exposes a Cloner class that encapsulates EXACTLY the per-sample
cloning logic for OuteTTS 1.0 0.6B. It is consumed by both:

    1. outetts_attack/steps/step_03_generate_speech.py (the standalone
       full-utterance attack pipeline, currently running in production).
    2. partial_spoof/steps/step_02_generate_cloned_speech.py (via the
       Cloner dispatcher).

OuteTTS treats speech synthesis as a language-modelling task over DAC
codec tokens. Per-speaker setup builds a "speaker profile" (tempo, energy,
pitch, spectral centroid) from the reference audio; this profile is
expensive enough that prepare_speaker caches it on the Cloner instance.
The output is saved at the DAC native sample rate, then resampled to
settings.SAMPLE_RATE (16 kHz) for downstream consistency.

BEHAVIOUR PARITY NOTE: this module is byte-for-byte equivalent to the
old inline Step 3 logic. The standalone OuteTTS production process has
the OLD code loaded in memory; if it crashes and restarts it will pick
up this new Step 3 (which delegates here). generation_metadata.json
schema is unchanged so resume continues to work.
"""
import time
from pathlib import Path
from typing import Optional

import outetts
import torch
import torchaudio
from loguru import logger

from app.pipeline.outetts_attack.settings import settings


class Cloner:
    """OuteTTS cloning unit: load, per-speaker profile cache, clone_single.

    Attributes:
        SYSTEM_ID: Uppercase attack identifier used in output filenames.
        NEEDS_REFERENCE_TRANSCRIPT: OuteTTS does NOT need the reference
            transcript -- speaker cloning is via the speaker profile.
        interface: Loaded outetts.Interface (None before load()).
        _speaker_profiles: Per-reference-path speaker profile cache.
    """

    SYSTEM_ID: str = "OUTETTS"
    NEEDS_REFERENCE_TRANSCRIPT: bool = False

    def __init__(self) -> None:
        """Initialise an empty cloner. Call load() before clone_single()."""
        self.interface: Optional[outetts.Interface] = None
        self._speaker_profiles: dict = {}

    def load(self, device: str) -> None:
        """Load the OuteTTS Interface via the canonical auto_config path.

        OuteTTS's Interface picks up the active CUDA device from the
        process environment (CUDA_VISIBLE_DEVICES); the explicit
        ``device`` argument is accepted for interface compatibility
        with other Cloner implementations.

        Args:
            device: PyTorch device string (informational; OuteTTS routes
                via the HF backend default).

        Raises:
            RuntimeError: If the model fails to load.
        """
        logger.info(f"OuteTTS Cloner: loading Interface...")
        logger.info(f"  Device          : {device}")
        logger.info(f"  Model version   : {settings.OUTETTS_MODEL_VERSION}")
        logger.info(f"  Model ID        : {settings.OUTETTS_MODEL_ID}")
        logger.info(f"  Temperature     : {settings.TEMPERATURE}")
        logger.info(f"  Top-K           : {settings.TOP_K}")
        logger.info(f"  Top-P           : {settings.TOP_P}")
        logger.info(f"  Rep. penalty    : {settings.REPETITION_PENALTY}")
        logger.info(f"  Max length      : {settings.MAX_LENGTH}")

        self.interface = outetts.Interface(
            outetts.ModelConfig.auto_config(
                model=outetts.Models.VERSION_1_0_SIZE_0_6B,
                backend=outetts.Backend.HF,
            )
        )
        logger.info("OuteTTS Cloner: model loaded")

    def prepare_speaker(
        self,
        speaker_id: str,
        reference_audio_path: Path,
    ) -> None:
        """Create and cache the OuteTTS speaker profile.

        Args:
            speaker_id: HABLA speaker identifier (used in log messages).
            reference_audio_path: Speaker reference audio path. Used as
                the cache key and passed to ``interface.create_speaker``.

        Raises:
            RuntimeError: If load() was not called first or profile
                creation fails.
        """
        if self.interface is None:
            raise RuntimeError(
                "OuteTTS Cloner: load() must be called before prepare_speaker()"
            )

        ref_key = str(reference_audio_path)
        if ref_key in self._speaker_profiles:
            return

        self._speaker_profiles[ref_key] = self.interface.create_speaker(ref_key)
        logger.debug(f"OuteTTS Cloner: created speaker profile for {speaker_id}")

    def clone_single(
        self,
        text: str,
        reference_audio_path: Path,
        output_path: Path,
        reference_text: str = "",
        seed: Optional[int] = None,
    ) -> tuple:
        """Generate one OuteTTS clone using the cached speaker profile.

        Args:
            text: Spanish text to synthesise.
            reference_audio_path: Speaker reference audio path. Used as
                the cache key for the previously-built speaker profile.
            output_path: Destination WAV path, written at
                settings.SAMPLE_RATE (16 kHz) after resampling from the
                DAC native sample rate.
            reference_text: Ignored by OuteTTS.
            seed: Accepted for interface compatibility; OuteTTS does not
                expose a deterministic sampling seed.

        Returns:
            Tuple of (generation_time_seconds, audio_duration_seconds).

        Raises:
            RuntimeError: If load() / prepare_speaker() were not called.
        """
        if self.interface is None:
            raise RuntimeError(
                "OuteTTS Cloner: load() must be called before clone_single()"
            )

        ref_key = str(reference_audio_path)
        if ref_key not in self._speaker_profiles:
            raise RuntimeError(
                f"OuteTTS Cloner: prepare_speaker() must be called before "
                f"clone_single() for reference '{ref_key}'"
            )
        speaker = self._speaker_profiles[ref_key]

        start_time = time.time()

        output = self.interface.generate(
            config=outetts.GenerationConfig(
                text=text,
                speaker=speaker,
                sampler_config=outetts.SamplerConfig(
                    top_k=settings.TOP_K,
                    top_p=settings.TOP_P,
                    temperature=settings.TEMPERATURE,
                    repetition_penalty=settings.REPETITION_PENALTY,
                ),
                max_length=settings.MAX_LENGTH,
            )
        )

        temp_path = output_path.with_suffix(".tmp.wav")
        output.save(str(temp_path))

        wav, native_sr = torchaudio.load(str(temp_path))

        if native_sr != settings.SAMPLE_RATE:
            wav = torchaudio.functional.resample(wav, native_sr, settings.SAMPLE_RATE)

        torchaudio.save(str(output_path), wav, settings.SAMPLE_RATE)

        temp_path.unlink(missing_ok=True)

        generation_time = time.time() - start_time
        audio_duration = wav.shape[-1] / settings.SAMPLE_RATE

        return generation_time, audio_duration

    def cleanup(self) -> None:
        """Release the OuteTTS interface and clear CUDA memory."""
        self.interface = None
        self._speaker_profiles.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("OuteTTS Cloner: GPU memory released")
