"""OuteTTS attack strategy for the Partial Spoof pipeline.

Wraps the OuteTTS 1.0 0.6B model for voice cloning. Does not require a
reference transcript; cloning is audio-based via the DAC tokenizer.

Mirrors the canonical generation logic from
``outetts_attack/steps/step_03_generate_speech.py``:
    - Uses the unified ``outetts.Interface`` API (the legacy
      ``InterfaceHF`` class does not exist on OuteTTS 1.0).
    - Builds speaker profiles via ``interface.create_speaker``.
    - Passes TOP_K, TOP_P, TEMPERATURE, REPETITION_PENALTY, MAX_LENGTH
      through ``GenerationConfig``/``SamplerConfig`` so every sampling
      parameter is controlled (not library defaults).
    - Resamples the codec-native output to SAMPLE_RATE (16 kHz) so
      downstream stages see consistent rates.
"""
import time
from pathlib import Path

import torch
import torchaudio
from loguru import logger

from app.pipeline.partial_spoof.settings import settings as partial_spoof_settings
from app.pipeline.partial_spoof.strategies.base_strategy import AttackStrategy
from app.pipeline.outetts_attack.settings import settings as outetts_settings


class OuteTTSStrategy(AttackStrategy):
    """Voice cloning via OuteTTS 1.0 0.6B.

    Attributes:
        interface: Loaded OuteTTS Interface instance.
        speaker_cache: Cache of speaker profiles per reference path.
    """

    def __init__(self) -> None:
        """Initialize OuteTTS strategy."""
        self.interface = None
        self.speaker_cache = {}

    def load_model(self, device: str) -> None:
        """Load OuteTTS model via the canonical Interface API.

        Args:
            device: PyTorch device string. OuteTTS routes via the HF
                backend and inherits the active CUDA device from the
                process environment; this argument is accepted for
                signature compatibility.
        """
        import outetts

        self.interface = outetts.Interface(
            outetts.ModelConfig.auto_config(
                model=outetts.Models.VERSION_1_0_SIZE_0_6B,
                backend=outetts.Backend.HF,
            )
        )
        logger.info(
            f"OuteTTSStrategy: Model loaded ({outetts_settings.OUTETTS_MODEL_VERSION})"
        )

    def generate(
        self,
        text: str,
        reference_audio_path: Path,
        output_path: Path,
        reference_text: str = "",
        seed: int | None = None,
    ) -> float:
        """Generate cloned speech using OuteTTS.

        Args:
            text: Text to synthesize.
            reference_audio_path: Speaker reference audio path. Cached
                speaker profile is reused across calls for the same path.
            output_path: Output WAV path (final, resampled to 16 kHz).
            reference_text: Ignored by OuteTTS.
            seed: Accepted for signature compatibility; OuteTTS does not
                expose a deterministic seed for the sampler.

        Returns:
            Generation time in seconds (including resampling).
        """
        import outetts

        start_time = time.time()

        ref_key = str(reference_audio_path)
        if ref_key not in self.speaker_cache:
            self.speaker_cache[ref_key] = self.interface.create_speaker(
                str(reference_audio_path)
            )

        output = self.interface.generate(
            config=outetts.GenerationConfig(
                text=text,
                speaker=self.speaker_cache[ref_key],
                sampler_config=outetts.SamplerConfig(
                    top_k=outetts_settings.TOP_K,
                    top_p=outetts_settings.TOP_P,
                    temperature=outetts_settings.TEMPERATURE,
                    repetition_penalty=outetts_settings.REPETITION_PENALTY,
                ),
                max_length=outetts_settings.MAX_LENGTH,
            )
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = output_path.with_suffix(".tmp.wav")
        output.save(str(temp_path))

        wav, native_sr = torchaudio.load(str(temp_path))
        if native_sr != partial_spoof_settings.SAMPLE_RATE:
            wav = torchaudio.functional.resample(
                wav, native_sr, partial_spoof_settings.SAMPLE_RATE
            )
        torchaudio.save(
            str(output_path), wav, partial_spoof_settings.SAMPLE_RATE
        )

        temp_path.unlink(missing_ok=True)

        return time.time() - start_time

    def cleanup(self) -> None:
        """Release model and clear GPU memory."""
        self.interface = None
        self.speaker_cache.clear()
        torch.cuda.empty_cache()
        logger.info("OuteTTSStrategy: Cleanup complete.")

    def name(self) -> str:
        """Return the system identifier.

        Returns:
            'OUTETTS' for protocol file entries.
        """
        return "OUTETTS"

    def needs_reference_transcript(self) -> bool:
        """OuteTTS does not need reference transcripts.

        Returns:
            False.
        """
        return False
