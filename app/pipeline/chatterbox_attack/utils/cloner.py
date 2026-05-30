"""
Chatterbox voice cloning unit -- single source of truth.

This module exposes a Cloner class that encapsulates EXACTLY the per-sample
cloning logic for ChatterboxMultilingualTTS. It is consumed by both:

    1. chatterbox_attack/steps/step_03_generate_speech.py (the standalone
       full-utterance attack pipeline, currently running in production).
    2. partial_spoof/steps/step_02_generate_cloned_speech.py (via the
       Cloner dispatcher).

Chatterbox-specific quirks all live inside ``load``:
    - perth_patcher MUST be imported before chatterbox.mtl_tts (the
      ChatterboxMultilingualTTS __init__ calls perth.PerthImplicitWatermarker
      unconditionally and the native Perth binary frequently fails to load).
    - After load, the model's ``watermarker`` is replaced with
      NoOpWatermarker so generated samples carry no neural steganographic
      watermark.
    - The SDPA-to-eager patch is mandatory for transformers >= 4.47
      compatibility -- ``output_attentions=True`` is incompatible with
      SDPA. Both ``_attn_implementation`` and the duplicated
      ``_attn_implementation_internal`` attribute must be flipped.

BEHAVIOUR PARITY NOTE: this module is byte-for-byte equivalent to the
old inline Step 3 logic. The standalone Chatterbox production process
has the OLD code loaded in memory; if it crashes and restarts it will
pick up the new Step 3 (which delegates here). generation_metadata.json
schema is unchanged so resume continues to work.
"""
import os
import time
from pathlib import Path
from typing import Optional

import torch
import torchaudio
from loguru import logger

from app.pipeline.chatterbox_attack.settings import settings
from app.pipeline.chatterbox_attack.utils.perth_patcher import ensure_patched  # noqa: F401 — patches perth on import
from app.pipeline.chatterbox_attack.utils.speech_trimmer import trim_trailing_noise
from app.pipeline.chatterbox_attack.utils.watermark_remover import NoOpWatermarker
from app.utils.base_cloner import BaseCloner
from chatterbox.mtl_tts import ChatterboxMultilingualTTS


class Cloner(BaseCloner):
    """Chatterbox cloning unit: load + watermark bypass + SDPA patch + clone_single.

    Attributes:
        SYSTEM_ID: Uppercase attack identifier used in output filenames.
        NEEDS_REFERENCE_TRANSCRIPT: Chatterbox cloning is purely
            audio-based -- reference transcript is not required.
        model: Loaded ChatterboxMultilingualTTS (None before load()).
    """

    SYSTEM_ID: str = "CHATTERBOX"
    NEEDS_REFERENCE_TRANSCRIPT: bool = False

    def __init__(self) -> None:
        """Initialise an empty cloner. Call load() before clone_single()."""
        self.model: Optional[ChatterboxMultilingualTTS] = None

    def load(self, device: str) -> None:
        """Load ChatterboxMultilingualTTS, bypass watermark, patch SDPA.

        Args:
            device: PyTorch device string passed to
                ``ChatterboxMultilingualTTS.from_pretrained``.

        Raises:
            RuntimeError: If model loading fails.
        """
        logger.info("Chatterbox Cloner: loading ChatterboxMultilingualTTS...")
        logger.info(f"  Device: {device}")
        logger.info(f"  Language: {settings.LANGUAGE_ID}")
        logger.info(f"  Exaggeration: {settings.EXAGGERATION}")
        logger.info(f"  CFG weight: {settings.CFG_WEIGHT}")
        logger.info(f"  Temperature: {settings.TEMPERATURE}")

        self.model = ChatterboxMultilingualTTS.from_pretrained(device=device)
        self.model.watermarker = NoOpWatermarker()

        self._patch_sdpa_to_eager(self.model)

        logger.info("Chatterbox Cloner: model loaded; watermark bypassed for research use")

    def prepare_speaker(
        self,
        speaker_id: str,
        reference_audio_path: Path,
        reference_text: str = "",
    ) -> None:
        """No-op for Chatterbox.

        Chatterbox conditioning happens inline inside each generate()
        call via the ``audio_prompt_path`` argument; there is no
        per-speaker state to cache.

        Args:
            speaker_id: HABLA speaker identifier (unused).
            reference_audio_path: Speaker reference audio path (unused).
            reference_text: Reference transcript (unused).
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
        """Generate one Chatterbox clone, resample to 16 kHz, trim, write.

        Args:
            text: Spanish text to synthesise.
            reference_audio_path: Speaker reference audio path. Passed as
                Chatterbox's ``audio_prompt_path``.
            output_path: Destination WAV path, written at
                settings.SAMPLE_RATE (16 kHz) after resampling from
                model.sr (24 kHz) and trimming trailing noise.
            reference_text: Ignored by Chatterbox.
            seed: Accepted for interface compatibility; Chatterbox does
                not expose a deterministic sampling seed.

        Returns:
            Tuple of (generation_time_seconds, audio_duration_seconds).

        Raises:
            RuntimeError: If load() was not called first or generation fails.
        """
        if self.model is None:
            raise RuntimeError(
                "Chatterbox Cloner: load() must be called before clone_single()"
            )

        start_time = time.time()

        wav = self.model.generate(
            text=text,
            language_id=settings.LANGUAGE_ID,
            audio_prompt_path=str(reference_audio_path),
            exaggeration=settings.EXAGGERATION,
            cfg_weight=settings.CFG_WEIGHT,
            temperature=settings.TEMPERATURE,
            repetition_penalty=settings.REPETITION_PENALTY,
        )

        wav_resampled = torchaudio.functional.resample(
            wav, self.model.sr, settings.SAMPLE_RATE
        )

        wav_trimmed = trim_trailing_noise(
            wav_resampled,
            settings.SAMPLE_RATE,
            margin_ms=settings.VAD_MARGIN_MS,
        )

        # Atomic, durable write: save to a temp file, fsync it to physical
        # storage, then rename into place. The fsync is the load-bearing
        # part on ext4: without it, delayed allocation can journal the
        # inode + (in the caller) the JSON metadata while the data pages
        # are still only in the page cache, so a node crash / OOM-kill
        # leaves a 0-byte file with correct metadata -- exactly the April
        # corruption. os.replace is atomic on the same filesystem, so a
        # crash can never expose a half-written WAV at the real path.
        tmp_path = output_path.with_name(output_path.name + ".tmp")
        torchaudio.save(str(tmp_path), wav_trimmed, settings.SAMPLE_RATE)
        fd = os.open(str(tmp_path), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        os.replace(str(tmp_path), str(output_path))

        generation_time = time.time() - start_time
        audio_duration = wav_trimmed.shape[-1] / settings.SAMPLE_RATE

        return generation_time, audio_duration

    def cleanup(self) -> None:
        """Release the Chatterbox model and clear CUDA memory."""
        del self.model
        self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("Chatterbox Cloner: GPU memory released")

    @staticmethod
    def _patch_sdpa_to_eager(model: ChatterboxMultilingualTTS) -> None:
        """Force eager attention on all internal transformers models.

        ChatterboxMultilingualTTS is a plain Python class (not nn.Module)
        that holds several nn.Module sub-models. This walks all
        instance attributes, finds nn.Module instances, recurses via
        ``named_modules()`` and patches any transformers config that
        uses SDPA attention to use eager instead. Required for
        transformers >= 4.47 where ``output_attentions=True`` is
        incompatible with SDPA. Both ``_attn_implementation`` and the
        duplicated ``_attn_implementation_internal`` attribute must be
        flipped; leaving the latter on ``'sdpa'`` causes transformers
        to revert the choice.

        Args:
            model: Loaded ChatterboxMultilingualTTS instance to patch in-place.
        """
        patched = 0
        for attr_name in vars(model):
            attr = getattr(model, attr_name)
            if not isinstance(attr, torch.nn.Module):
                continue
            for _name, submodule in attr.named_modules():
                config = getattr(submodule, "config", None)
                if config is None:
                    continue
                if getattr(config, "_attn_implementation", None) == "sdpa":
                    config._attn_implementation = "eager"
                    config._attn_implementation_internal = "eager"
                    patched += 1
        if patched > 0:
            logger.info(
                f"Chatterbox Cloner: patched {patched} sub-module(s) from SDPA to eager"
            )
