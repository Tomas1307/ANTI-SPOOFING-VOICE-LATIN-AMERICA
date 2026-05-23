"""OpenVoice attack strategy for the Partial Spoof pipeline.

Wraps the OpenVoice V2 (MeloTTS + ToneColorConverter) pipeline for
voice cloning. Does not require a reference transcript.

Output audio is resampled to SAMPLE_RATE (16 kHz) so downstream stages
(forced alignment, splicing, validation) see a consistent rate across
all attacks - matches the behaviour of
``openvoice_attack/steps/step_03_generate_speech.py``.
"""
import time
from pathlib import Path

import librosa
import soundfile as sf
import torch
from loguru import logger

from app.pipeline.partial_spoof.settings import settings as partial_spoof_settings
from app.pipeline.partial_spoof.strategies.base_strategy import AttackStrategy
from app.pipeline.openvoice_attack.settings import settings as openvoice_settings


class OpenVoiceStrategy(AttackStrategy):
    """Voice cloning via OpenVoice V2 (MeloTTS + ToneColorConverter).

    Two-stage pipeline: MeloTTS generates base speech, then
    ToneColorConverter applies the target speaker's voice timbre.

    Attributes:
        tts_model: MeloTTS model for base speech generation.
        tone_converter: ToneColorConverter for voice timbre transfer.
        se_cache: Cache of speaker embeddings per reference path.
    """

    def __init__(self) -> None:
        """Initialize OpenVoice strategy."""
        self.tts_model = None
        self.tone_converter = None
        self.se_cache = {}

    def load_model(self, device: str) -> None:
        """Load MeloTTS and ToneColorConverter models.

        Args:
            device: PyTorch device string.
        """
        from melo.api import TTS as MeloTTS
        from openvoice.api import ToneColorConverter

        self.tts_model = MeloTTS(
            language=openvoice_settings.MELO_LANGUAGE,
            device=device,
        )
        self.tone_converter = ToneColorConverter(
            openvoice_settings.TONE_CONVERTER_CONFIG,
            device=device,
        )
        self.tone_converter.load_ckpt(openvoice_settings.TONE_CONVERTER_CHECKPOINT)
        self._device = device
        logger.info(f"OpenVoiceStrategy: Models loaded on {device}")

    def generate(
        self,
        text: str,
        reference_audio_path: Path,
        output_path: Path,
        reference_text: str = "",
        seed: int | None = None,
    ) -> float:
        """Generate cloned speech using OpenVoice V2.

        Args:
            text: Text to synthesize.
            reference_audio_path: Speaker reference audio path.
            output_path: Output WAV path.
            reference_text: Ignored by OpenVoice.
            seed: Optional random seed.

        Returns:
            Generation time in seconds.
        """
        start_time = time.time()

        output_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_base_path = output_path.parent / f"_tmp_base_{output_path.stem}.wav"
        tmp_converted_path = (
            output_path.parent / f"_tmp_converted_{output_path.stem}.wav"
        )

        try:
            speaker_ids = self.tts_model.hps.data.spk2id
            speaker_key = list(speaker_ids.keys())[0]
            self.tts_model.tts_to_file(
                text=text,
                speaker_id=speaker_ids[speaker_key],
                output_path=str(tmp_base_path),
                speed=openvoice_settings.MELO_SPEED,
            )

            ref_key = str(reference_audio_path)
            if ref_key not in self.se_cache:
                from openvoice.se_extractor import get_se
                self.se_cache[ref_key] = get_se(
                    str(reference_audio_path),
                    self.tone_converter,
                    vad=True,
                )

            target_se = self.se_cache[ref_key]
            source_se = self.tone_converter.extract_se(str(tmp_base_path))

            self.tone_converter.convert(
                audio_src_path=str(tmp_base_path),
                src_se=source_se,
                tgt_se=target_se,
                output_path=str(tmp_converted_path),
            )

            audio, _ = librosa.load(
                str(tmp_converted_path), sr=partial_spoof_settings.SAMPLE_RATE
            )
            sf.write(
                str(output_path), audio, partial_spoof_settings.SAMPLE_RATE
            )

        finally:
            if tmp_base_path.exists():
                tmp_base_path.unlink()
            if tmp_converted_path.exists():
                tmp_converted_path.unlink()

        return time.time() - start_time

    def cleanup(self) -> None:
        """Release models and clear GPU memory."""
        self.tts_model = None
        self.tone_converter = None
        self.se_cache.clear()
        torch.cuda.empty_cache()
        logger.info("OpenVoiceStrategy: Cleanup complete.")

    def name(self) -> str:
        """Return the system identifier.

        Returns:
            'OPENVOICE' for protocol file entries.
        """
        return "OPENVOICE"

    def needs_reference_transcript(self) -> bool:
        """OpenVoice does not need reference transcripts.

        Returns:
            False.
        """
        return False
