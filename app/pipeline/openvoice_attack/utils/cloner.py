"""
OpenVoice voice cloning unit -- single source of truth.

This module exposes a Cloner class that encapsulates EXACTLY the per-sample
cloning logic for OpenVoice (MeloTTS + ToneColorConverter). It is consumed
by both:

    1. openvoice_attack/steps/step_03_generate_speech.py (the standalone
       full-utterance attack pipeline).
    2. partial_spoof/steps/step_02_generate_cloned_speech.py (via the
       Cloner dispatcher).

OpenVoice is a two-stage pipeline: MeloTTS synthesises base Spanish speech,
then ToneColorConverter applies the target speaker's tone colour. Per-speaker
state (target_se) is extracted once via se_extractor.get_se and cached on
the Cloner instance keyed by reference audio path.

Output audio is resampled to settings.SAMPLE_RATE (16 kHz) for consistency
with downstream pipeline stages.
"""
import os
import tempfile
import time
from pathlib import Path
from typing import Optional

import librosa
import soundfile as sf
import torch
from loguru import logger
from openvoice import se_extractor
from openvoice.api import ToneColorConverter
from melo.api import TTS

from app.pipeline.openvoice_attack.settings import settings


class Cloner:
    """OpenVoice cloning unit: MeloTTS base + ToneColorConverter transfer.

    The instance holds:
        - The loaded MeloTTS and ToneColorConverter models
        - The base ES source_se (loaded once from checkpoint)
        - melo_speaker_id (resolved at load time)
        - A per-reference-path target_se cache populated by prepare_speaker()

    Attributes:
        SYSTEM_ID: Uppercase attack identifier used in output filenames.
        NEEDS_REFERENCE_TRANSCRIPT: OpenVoice does not require the
            reference transcript -- cloning is audio-based via tone
            colour embedding.
        tts_model: Loaded MeloTTS instance (None before load()).
        tone_color_converter: Loaded ToneColorConverter (None before load()).
        source_se: Base ES tone colour embedding from checkpoint.
        melo_speaker_id: Integer speaker ID for MeloTTS ES voice.
        _target_se_cache: Per-reference-path target_se cache.
    """

    SYSTEM_ID: str = "OPENVOICE"
    NEEDS_REFERENCE_TRANSCRIPT: bool = False

    def __init__(self) -> None:
        """Initialise an empty cloner. Call load() before clone_single()."""
        self.tts_model: Optional[TTS] = None
        self.tone_color_converter: Optional[ToneColorConverter] = None
        self.source_se: Optional[torch.Tensor] = None
        self.melo_speaker_id: Optional[int] = None
        self._target_se_cache: dict = {}

    def load(self, device: str) -> None:
        """Load MeloTTS, ToneColorConverter, and the base ES source_se.

        Args:
            device: PyTorch device string for both sub-models.

        Raises:
            FileNotFoundError: If checkpoint files are missing.
            RuntimeError: If model loading fails.
        """
        converter_config = str(
            settings.OPENVOICE_CHECKPOINT_DIR / "converter" / "config.json"
        )
        converter_ckpt = str(
            settings.OPENVOICE_CHECKPOINT_DIR / "converter" / "checkpoint.pth"
        )
        source_se_path = str(
            settings.OPENVOICE_CHECKPOINT_DIR / "base_speakers" / "ses" / "es.pth"
        )

        logger.info("OpenVoice Cloner: loading ToneColorConverter...")
        self.tone_color_converter = ToneColorConverter(converter_config, device=device)
        self.tone_color_converter.load_ckpt(converter_ckpt)

        logger.info("OpenVoice Cloner: loading MeloTTS (ES)...")
        self.tts_model = TTS(language=settings.MELO_LANGUAGE, device=device)
        speaker_ids = self.tts_model.hps.data.spk2id
        self.melo_speaker_id = speaker_ids[settings.MELO_LANGUAGE]

        self.source_se = torch.load(source_se_path, map_location=device)

        logger.info("OpenVoice Cloner: models loaded")

    def prepare_speaker(
        self,
        speaker_id: str,
        reference_audio_path: Path,
    ) -> None:
        """Extract and cache the target_se for one speaker.

        Idempotent per reference audio path: re-calling for an already-
        cached speaker is a no-op. The cache key is the resolved reference
        audio path string so multiple speakers that happen to share a
        reference share the embedding (extremely unlikely in practice).

        Args:
            speaker_id: HABLA speaker identifier (used only in log messages).
            reference_audio_path: Speaker reference audio file. The
                tone-colour embedding is extracted from this file via
                ``se_extractor.get_se(vad=True)``.

        Raises:
            RuntimeError: If load() was not called first or extraction fails.
        """
        if self.tone_color_converter is None:
            raise RuntimeError(
                "OpenVoice Cloner: load() must be called before prepare_speaker()"
            )

        ref_key = str(reference_audio_path)
        if ref_key in self._target_se_cache:
            return

        target_se, _ = se_extractor.get_se(
            ref_key,
            self.tone_color_converter,
            vad=True,
        )
        self._target_se_cache[ref_key] = target_se
        logger.debug(f"OpenVoice Cloner: extracted target_se for {speaker_id}")

    def clone_single(
        self,
        text: str,
        reference_audio_path: Path,
        output_path: Path,
        reference_text: str = "",
        seed: Optional[int] = None,
    ) -> tuple:
        """Generate one OpenVoice clone via MeloTTS + tone-colour transfer.

        Looks up the cached target_se for ``reference_audio_path``; the
        caller must have invoked prepare_speaker() for this speaker first.

        Args:
            text: Spanish text to synthesise.
            reference_audio_path: Speaker reference audio path. Used as
                the cache key for target_se.
            output_path: Destination WAV path, written at SAMPLE_RATE
                (16 kHz) after the two-stage MeloTTS+ToneColor pipeline.
            reference_text: Ignored by OpenVoice.
            seed: Accepted for interface compatibility; OpenVoice does
                not expose a deterministic sampling seed.

        Returns:
            Tuple of (generation_time_seconds, audio_duration_seconds).

        Raises:
            RuntimeError: If load() / prepare_speaker() were not called,
                or if TTS/conversion fails.
        """
        if self.tts_model is None or self.tone_color_converter is None:
            raise RuntimeError(
                "OpenVoice Cloner: load() must be called before clone_single()"
            )

        ref_key = str(reference_audio_path)
        if ref_key not in self._target_se_cache:
            raise RuntimeError(
                f"OpenVoice Cloner: prepare_speaker() must be called before "
                f"clone_single() for reference '{ref_key}'"
            )
        target_se = self._target_se_cache[ref_key]

        start_time = time.time()

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_base:
            tmp_base_path = tmp_base.name
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_converted:
            tmp_converted_path = tmp_converted.name

        try:
            self.tts_model.tts_to_file(
                text,
                self.melo_speaker_id,
                tmp_base_path,
                speed=settings.MELO_SPEED,
            )

            self.tone_color_converter.convert(
                audio_src_path=tmp_base_path,
                src_se=self.source_se,
                tgt_se=target_se,
                output_path=tmp_converted_path,
                tau=settings.CONVERSION_TAU,
                message="@MyShell",
            )

            audio, _ = librosa.load(tmp_converted_path, sr=settings.SAMPLE_RATE)
            sf.write(str(output_path), audio, settings.SAMPLE_RATE)

        finally:
            if os.path.exists(tmp_base_path):
                os.unlink(tmp_base_path)
            if os.path.exists(tmp_converted_path):
                os.unlink(tmp_converted_path)

        generation_time = time.time() - start_time
        audio_duration = len(audio) / settings.SAMPLE_RATE

        return generation_time, audio_duration

    def cleanup(self) -> None:
        """Release MeloTTS, ToneColorConverter, and clear CUDA memory."""
        del self.tts_model
        del self.tone_color_converter
        self.tts_model = None
        self.tone_color_converter = None
        self.source_se = None
        self.melo_speaker_id = None
        self._target_se_cache.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("OpenVoice Cloner: GPU memory released")
