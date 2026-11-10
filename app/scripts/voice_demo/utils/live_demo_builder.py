"""
Live partial-spoof demo builder.

Builds the two companion clips for the thesis-defense splice demo: a naive
("crudo") cut-and-paste splice and a mitigated splice that reuses the real
production engine (valley snap, energy refinement, crossfade, loudness
matching) from ``app.pipeline.partial_spoof.utils.splice_engine``. Both
start from the SAME single ~10 s recording, which serves as both the
OmniVoice reference clip and the bonafide host utterance, so the cloned
replacement words share the speaker's prosody with the sentence they are
spliced into.
"""
from pathlib import Path
from typing import List, Tuple

import librosa
import numpy as np
import soundfile as sf
from loguru import logger

from app.pipeline.omnivoice_attack.settings import settings as omnivoice_settings
from app.pipeline.omnivoice_attack.utils.cloner import Cloner
from app.pipeline.partial_spoof.settings import settings as partial_spoof_settings
from app.pipeline.partial_spoof.utils.splice_engine import _normalize_word, splice_words
from app.scripts.voice_demo.schemas.live_demo_request import LiveDemoRequest
from app.scripts.voice_demo.settings import settings as voice_demo_settings
from app.scripts.voice_demo.utils.audio_conversion import to_mono_wav
from app.utils.parakeet_transcriber import ParakeetTranscriber
from app.utils.word_timestamp import WordTimestamp


def find_phrase_span(words: List[WordTimestamp], phrase_tokens: List[str]) -> Tuple[int, int]:
    """Locate a contiguous phrase inside a Parakeet word-timestamp list.

    Matches ``phrase_tokens`` against ``words`` in order, using the same
    accent/punctuation normalisation as the production splice engine, so a
    manually chosen phrase (e.g. ``["trescientos", "mil"]``) can be mapped
    onto exact timestamps.

    Args:
        words: Word-level timestamps from Parakeet, in utterance order.
        phrase_tokens: The words to locate, in order.

    Returns:
        A (start_index, end_index) tuple, both inclusive, into ``words``.

    Raises:
        ValueError: If no contiguous match for ``phrase_tokens`` is found.
    """
    target = [_normalize_word(t) for t in phrase_tokens]
    n = len(target)
    for i in range(len(words) - n + 1):
        window = [_normalize_word(words[i + j].word) for j in range(n)]
        if window == target:
            return i, i + n - 1
    raise ValueError(
        f"Phrase {phrase_tokens} not found in transcript: "
        f"{[w.word for w in words]}"
    )


class LiveDemoBuilder:
    """Builds the crude and mitigated splice demo clips for one request.

    Attributes:
        request: The resolved live-demo request.
        transcriber: Shared Parakeet singleton used for both transcriptions.
    """

    def __init__(self, request: LiveDemoRequest) -> None:
        """Initialise the builder for a single demo request.

        Args:
            request: Validated live-demo request.
        """
        self.request = request
        self.transcriber = ParakeetTranscriber()

    def build(self) -> dict:
        """Run the full pipeline and write both demo clips to disk.

        Returns:
            Mapping with keys ``crudo``, ``mitigado``, and the intermediate
            ``host_wav``/``cloned_wav`` paths, all as strings.
        """
        out_dir = self.request.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Preparing host audio (OmniVoice reference + 16kHz processing copy)...")
        host_wav_16k = to_mono_wav(
            self.request.host_audio_path, out_dir / "host_16k.wav", sample_rate=partial_spoof_settings.SAMPLE_RATE
        )
        host_wav_24k = to_mono_wav(
            self.request.host_audio_path,
            out_dir / "host_24k.wav",
            sample_rate=voice_demo_settings.REFERENCE_WAV_SAMPLE_RATE,
        )

        self.transcriber.load(device=partial_spoof_settings.DEVICE)

        logger.info("Transcribing host audio with Parakeet...")
        host_text, host_words = self.transcriber.transcribe_with_timestamps(host_wav_16k)
        logger.info(f"Host transcript: \"{host_text}\"")

        host_start_idx, host_end_idx = find_phrase_span(host_words, self.request.target_phrase)
        host_phrase_start_s = host_words[host_start_idx].start
        host_phrase_end_s = host_words[host_end_idx].end

        synth_words = [w.word for w in host_words]
        synth_words[host_start_idx:host_end_idx + 1] = self.request.replacement_phrase
        synth_text = " ".join(synth_words)
        logger.info(f"Replacement sentence to clone: \"{synth_text}\"")

        cloner = Cloner()
        logger.info("Loading OmniVoice model...")
        cloner.load(device=omnivoice_settings.DEVICE)
        cloned_raw_path = out_dir / "cloned_raw.wav"
        cloner.clone_single(
            text=synth_text,
            reference_audio_path=host_wav_24k,
            output_path=cloned_raw_path,
            reference_text=host_text,
        )
        cloner.cleanup()

        cloned_wav_16k = to_mono_wav(
            cloned_raw_path, out_dir / "cloned_16k.wav", sample_rate=partial_spoof_settings.SAMPLE_RATE
        )

        logger.info("Transcribing cloned replacement with Parakeet...")
        cloned_text, cloned_words = self.transcriber.transcribe_with_timestamps(cloned_wav_16k)
        logger.info(f"Cloned transcript: \"{cloned_text}\"")

        cloned_start_idx, cloned_end_idx = find_phrase_span(
            cloned_words, self.request.replacement_phrase
        )
        cloned_phrase_start_s = cloned_words[cloned_start_idx].start
        cloned_phrase_end_s = cloned_words[cloned_end_idx].end

        host_audio, _ = librosa.load(str(host_wav_16k), sr=partial_spoof_settings.SAMPLE_RATE)
        cloned_audio, _ = librosa.load(str(cloned_wav_16k), sr=partial_spoof_settings.SAMPLE_RATE)
        host_audio = host_audio.astype(np.float32)
        cloned_audio = cloned_audio.astype(np.float32)

        crudo_path = out_dir / "crudo.wav"
        self._build_crudo(
            host_audio, cloned_audio, host_phrase_start_s, host_phrase_end_s,
            cloned_phrase_start_s, cloned_phrase_end_s, crudo_path,
        )

        mitigado_path = out_dir / "mitigado.wav"
        self._build_mitigado(
            host_audio, cloned_audio, host_phrase_start_s, host_phrase_end_s,
            cloned_phrase_start_s, cloned_phrase_end_s, mitigado_path,
        )

        return {
            "host_wav": str(host_wav_16k),
            "cloned_wav": str(cloned_wav_16k),
            "crudo": str(crudo_path),
            "mitigado": str(mitigado_path),
        }

    def _build_crudo(
        self,
        host_audio: np.ndarray,
        cloned_audio: np.ndarray,
        host_start_s: float,
        host_end_s: float,
        cloned_start_s: float,
        cloned_end_s: float,
        output_path: Path,
    ) -> None:
        """Write the naive cut-and-paste splice: raw Parakeet boundaries,
        no valley snap, no crossfade, no loudness matching.

        Args:
            host_audio: Host waveform at the processing sample rate.
            cloned_audio: Cloned waveform at the processing sample rate.
            host_start_s: Host phrase start (raw Parakeet boundary, seconds).
            host_end_s: Host phrase end (raw Parakeet boundary, seconds).
            cloned_start_s: Cloned phrase start (raw Parakeet boundary, seconds).
            cloned_end_s: Cloned phrase end (raw Parakeet boundary, seconds).
            output_path: Destination WAV path.
        """
        sr = partial_spoof_settings.SAMPLE_RATE
        h_start = int(host_start_s * sr)
        h_end = int(host_end_s * sr)
        c_start = int(cloned_start_s * sr)
        c_end = int(cloned_end_s * sr)

        result = np.concatenate([
            host_audio[:h_start],
            cloned_audio[c_start:c_end],
            host_audio[h_end:],
        ]).astype(np.float32)
        sf.write(str(output_path), result, sr)
        logger.info(f"Crude splice written: {output_path}")

    def _build_mitigado(
        self,
        host_audio: np.ndarray,
        cloned_audio: np.ndarray,
        host_start_s: float,
        host_end_s: float,
        cloned_start_s: float,
        cloned_end_s: float,
        output_path: Path,
    ) -> None:
        """Write the mitigated splice using the real production engine.

        Treats the whole target phrase as a single splice unit and feeds it
        through ``splice_words`` with the exact parameters
        ``PartialSpoofSettings`` uses in production (valley snap, energy
        refinement, crossfade, utterance-level loudness matching).

        Args:
            host_audio: Host waveform at the processing sample rate.
            cloned_audio: Cloned waveform at the processing sample rate.
            host_start_s: Host phrase start (raw Parakeet boundary, seconds).
            host_end_s: Host phrase end (raw Parakeet boundary, seconds).
            cloned_start_s: Cloned phrase start (raw Parakeet boundary, seconds).
            cloned_end_s: Cloned phrase end (raw Parakeet boundary, seconds).
            output_path: Destination WAV path.
        """
        placeholder = "__TARGET__"
        bonafide_words = [{"word": placeholder, "start": host_start_s, "end": host_end_s}]
        cloned_words = [{"word": placeholder, "start": cloned_start_s, "end": cloned_end_s}]

        result, details = splice_words(
            bonafide_audio=host_audio,
            cloned_audio=cloned_audio,
            bonafide_words=bonafide_words,
            cloned_words=cloned_words,
            selected_indices=[0],
            sample_rate=partial_spoof_settings.SAMPLE_RATE,
            crossfade_min_ms=partial_spoof_settings.CROSSFADE_MIN_MS,
            crossfade_max_ms=partial_spoof_settings.CROSSFADE_MAX_MS,
            splice_seed=partial_spoof_settings.RANDOM_SEED,
            valley_search_ms=partial_spoof_settings.VALLEY_SEARCH_MS,
            energy_refine_radius_s=partial_spoof_settings.ENERGY_REFINE_RADIUS_S,
            energy_refine_silence_rms=partial_spoof_settings.ENERGY_REFINE_SILENCE_RMS,
            loudness_match_enabled=partial_spoof_settings.LOUDNESS_MATCH_ENABLED,
            loudness_reference_mode=partial_spoof_settings.LOUDNESS_REFERENCE_MODE,
            loudness_voiced_frame_ms=partial_spoof_settings.LOUDNESS_VOICED_FRAME_MS,
            loudness_voiced_gate_fraction=partial_spoof_settings.LOUDNESS_VOICED_GATE_FRACTION,
        )
        if not details:
            raise RuntimeError(
                "Production splice engine rejected the target phrase "
                "(zero-length segment or matching failure); see logs above."
            )
        sf.write(str(output_path), result, partial_spoof_settings.SAMPLE_RATE)
        logger.info(f"Mitigated splice written: {output_path} | details={details[0]}")
