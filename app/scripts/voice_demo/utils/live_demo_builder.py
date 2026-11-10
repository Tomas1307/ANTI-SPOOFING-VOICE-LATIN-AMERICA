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
from app.pipeline.partial_spoof.utils.splice_engine import (
    _normalize_word,
    _silent_run_backward,
    _silent_run_forward,
    splice_words,
)
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

        best = self._generate_best_candidate(cloner, synth_text, host_wav_24k, host_text, out_dir)
        cloner.cleanup()

        if best is None:
            raise RuntimeError(
                f"None of {self.request.tries} OmniVoice render(s) contained the "
                f"replacement phrase {self.request.replacement_phrase!r}."
            )

        logger.info(
            f"Best candidate: \"{best['text']}\" with "
            f"{best['score'] * 1000 / partial_spoof_settings.SAMPLE_RATE:.1f}ms "
            f"total flanking silence around the replacement phrase"
        )

        cloned_wav_16k = out_dir / "cloned_16k.wav"
        sf.write(str(cloned_wav_16k), best["audio"], partial_spoof_settings.SAMPLE_RATE)
        cloned_phrase_start_s = best["phrase_start_s"]
        cloned_phrase_end_s = best["phrase_end_s"]

        host_audio, _ = librosa.load(str(host_wav_16k), sr=partial_spoof_settings.SAMPLE_RATE)
        host_audio = host_audio.astype(np.float32)
        cloned_audio = best["audio"]

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

    def _generate_best_candidate(
        self,
        cloner: Cloner,
        synth_text: str,
        host_wav_24k: Path,
        host_text: str,
        out_dir: Path,
    ) -> dict | None:
        """Render the replacement sentence up to ``request.tries`` times and
        keep the one with the most silence flanking the replacement phrase.

        OmniVoice exposes no deterministic sampling seed (``Cloner.clone_single``'s
        ``seed`` argument is accepted only for interface compatibility), so
        repeated calls with the same text produce different prosody. The
        production splice engine only gets a clean crossfade when the cloned
        word already has silence on both sides in the generated audio; since
        this demo cannot pick a different, better-behaved word (the content
        is fixed by the script), the closest equivalent is to pick the best
        of several independent renders instead.

        Args:
            cloner: Loaded OmniVoice Cloner instance.
            synth_text: Full replacement sentence to synthesise.
            host_wav_24k: OmniVoice reference audio (24 kHz mono).
            host_text: Reference transcript for OmniVoice's conditioning.
            out_dir: Directory to write per-candidate intermediate files to.

        Returns:
            A dict with ``score`` (total flanking-silence samples), ``audio``
            (16 kHz mono waveform), ``text`` (transcript), ``phrase_start_s``,
            and ``phrase_end_s``, or None if no candidate contained the
            replacement phrase.
        """
        sr = partial_spoof_settings.SAMPLE_RATE
        best: dict | None = None

        for attempt in range(self.request.tries):
            candidate_raw = out_dir / f"cloned_candidate_{attempt}_raw.wav"
            cloner.clone_single(
                text=synth_text,
                reference_audio_path=host_wav_24k,
                output_path=candidate_raw,
                reference_text=host_text,
            )
            candidate_16k_path = to_mono_wav(
                candidate_raw, out_dir / f"cloned_candidate_{attempt}_16k.wav", sample_rate=sr
            )

            candidate_text, candidate_words = self.transcriber.transcribe_with_timestamps(
                candidate_16k_path
            )

            try:
                c_start_idx, c_end_idx = find_phrase_span(
                    candidate_words, self.request.replacement_phrase
                )
            except ValueError:
                logger.warning(
                    f"Try {attempt}: replacement phrase not found in "
                    f"\"{candidate_text}\", skipping this render."
                )
                continue

            phrase_start_s = candidate_words[c_start_idx].start
            phrase_end_s = candidate_words[c_end_idx].end

            candidate_audio, _ = librosa.load(str(candidate_16k_path), sr=sr)
            candidate_audio = candidate_audio.astype(np.float32)

            silence_left = _silent_run_backward(
                candidate_audio, int(phrase_start_s * sr), sr,
                max_ms=partial_spoof_settings.CROSSFADE_MAX_MS,
                silence_threshold=partial_spoof_settings.ENERGY_REFINE_SILENCE_RMS,
            )
            silence_right = _silent_run_forward(
                candidate_audio, int(phrase_end_s * sr), sr,
                max_ms=partial_spoof_settings.CROSSFADE_MAX_MS,
                silence_threshold=partial_spoof_settings.ENERGY_REFINE_SILENCE_RMS,
            )
            score = silence_left + silence_right
            logger.info(
                f"Try {attempt}: \"{candidate_text}\" | flanking silence = "
                f"{silence_left * 1000 / sr:.1f}ms + {silence_right * 1000 / sr:.1f}ms"
            )

            if best is None or score > best["score"]:
                best = {
                    "score": score,
                    "audio": candidate_audio,
                    "text": candidate_text,
                    "phrase_start_s": phrase_start_s,
                    "phrase_end_s": phrase_end_s,
                }

        return best

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
