"""
Live partial-spoof demo builder.

Builds the three clips for the thesis-defense splice demo: the untouched
host recording, the same recording with one or more words replaced by
cloned speech (the real production splice engine, before boundary jitter),
and that spliced result after the real boundary-jitter post-processing step
homogenizes every internal word boundary. All three start from the SAME
single ~10 s recording, which serves as both the OmniVoice reference clip
and the bonafide host utterance, so the cloned replacement words share the
speaker's prosody with the sentence they are spliced into.
"""
import difflib
from pathlib import Path
from typing import List, Tuple

import librosa
import numpy as np
import soundfile as sf
import torch
from loguru import logger

from app.pipeline.omnivoice_attack.settings import settings as omnivoice_settings
from app.pipeline.omnivoice_attack.utils.cloner import Cloner
from app.pipeline.partial_spoof.settings import settings as partial_spoof_settings
from app.pipeline.partial_spoof.steps.step_05b_apply_boundary_jitter import BoundaryJitterApplier
from app.pipeline.partial_spoof.utils.splice_engine import _normalize_word, splice_words
from app.scripts.voice_demo.schemas.live_demo_request import LiveDemoRequest
from app.scripts.voice_demo.settings import settings as voice_demo_settings
from app.scripts.voice_demo.utils.audio_conversion import to_mono_wav
from app.utils.parakeet_transcriber import ParakeetTranscriber
from app.utils.word_timestamp import WordTimestamp

SPANISH_NUMBER_WORDS = {
    "cero": "0", "un": "1", "uno": "1", "una": "1", "dos": "2", "tres": "3",
    "cuatro": "4", "cinco": "5", "seis": "6", "siete": "7", "ocho": "8",
    "nueve": "9", "diez": "10", "once": "11", "doce": "12", "trece": "13",
    "catorce": "14", "quince": "15", "veinte": "20", "treinta": "30",
    "cuarenta": "40", "cincuenta": "50", "sesenta": "60", "setenta": "70",
    "ochenta": "80", "noventa": "90", "cien": "100", "ciento": "100",
}
DIGIT_TO_WORDS: dict = {}
for _word, _digit in SPANISH_NUMBER_WORDS.items():
    DIGIT_TO_WORDS.setdefault(_digit, set()).add(_word)


def _token_variants(normalized_token: str) -> set:
    """Expand a normalized token to every spelling Parakeet might use for it.

    Parakeet inconsistently renders small numbers as digits or as spelled-out
    words between otherwise-identical runs of the same TTS text (observed:
    "cinco" <-> "5", "trescientos" <-> "300"), with no way to predict which
    form a given render will use. Without this, a candidate whose audio is
    fine gets discarded purely because its transcript spells the number
    differently than the one used when the phrase was first chosen.

    Args:
        normalized_token: A single token already passed through
            ``_normalize_word``.

    Returns:
        A set containing ``normalized_token`` plus its digit/word
        equivalents, if it is a recognised small number.
    """
    variants = {normalized_token}
    if normalized_token in SPANISH_NUMBER_WORDS:
        variants.add(SPANISH_NUMBER_WORDS[normalized_token])
    if normalized_token in DIGIT_TO_WORDS:
        variants |= DIGIT_TO_WORDS[normalized_token]
    return variants


def find_phrase_span(
    words: List[WordTimestamp],
    phrase_tokens: List[str],
    fuzzy: bool = False,
    fuzzy_threshold: float = 0.55,
) -> Tuple[int, int]:
    """Locate a contiguous phrase inside a Parakeet word-timestamp list.

    Matches ``phrase_tokens`` against ``words`` in order, using the same
    accent/punctuation normalisation as the production splice engine, so
    a manually chosen phrase (e.g. ``["300", "mil"]``) can be mapped onto
    exact timestamps. Exact matches always win; ``fuzzy`` only widens the
    search when no exact match exists.

    Args:
        words: Word-level timestamps from Parakeet, in utterance order.
        phrase_tokens: The words to locate, in order.
        fuzzy: When True and no exact match exists, accept the best
            character-similarity match above ``fuzzy_threshold`` instead.
            Parakeet's SentencePiece tokenizer (8192 tokens) frequently
            mis-renders short proper nouns and acronyms it never saw as a
            whole token during training (observed: "BBVA" -> "BBUA",
            "Bancolombia" -> "Bancolomb", "Nequi" -> "Neki"/"Aneki" --
            Spanish "qu" is pronounced /k/, so the tokenizer's phonetic
            guess is reasonable even though the spelling is wrong) even
            when the underlying audio is fine; without this, a run that
            nails the audio can still be discarded purely because the
            transcript of it is slightly off. 0.55 was picked from real
            mis-transcriptions clustering at 0.6-0.9 similarity, well
            above the ~0.25-0.3 scored by renders that were genuinely
            garbled.
            Should stay False for matching against the host's own
            recording, since that transcript has been exact and
            reproducible across repeated runs.
        fuzzy_threshold: Minimum ``difflib.SequenceMatcher`` ratio (over
            the normalised, space-joined phrase) to accept a fuzzy match.

    Returns:
        A (start_index, end_index) tuple, both inclusive, into ``words``.

    Raises:
        ValueError: If no match (exact, or fuzzy above threshold) is found.
    """
    target = [_normalize_word(t) for t in phrase_tokens]
    n = len(target)

    best_idx = None
    best_ratio = 0.0
    target_str = " ".join(target)
    for i in range(len(words) - n + 1):
        window = [_normalize_word(words[i + j].word) for j in range(n)]
        if all(target[j] in _token_variants(window[j]) for j in range(n)):
            return i, i + n - 1
        if fuzzy:
            ratio = difflib.SequenceMatcher(None, " ".join(window), target_str).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_idx = i

    if fuzzy and best_idx is not None and best_ratio >= fuzzy_threshold:
        matched_words = [words[best_idx + j].word for j in range(n)]
        logger.info(
            f"Fuzzy-matched {phrase_tokens!r} to {matched_words!r} "
            f"(similarity={best_ratio:.2f})"
        )
        return best_idx, best_idx + n - 1

    raise ValueError(
        f"Phrase {phrase_tokens} not found in transcript: "
        f"{[w.word for w in words]}"
    )


class LiveDemoBuilder:
    """Builds the original/spoof/spoof+jitter splice demo clips for one request.

    Attributes:
        request: The resolved live-demo request.
        transcriber: Shared Parakeet singleton used for every transcription.
    """

    def __init__(self, request: LiveDemoRequest) -> None:
        """Initialise the builder for a single demo request.

        Args:
            request: Validated live-demo request.
        """
        self.request = request
        self.transcriber = ParakeetTranscriber()

    def build(self) -> dict:
        """Run the full pipeline and write all three demo clips to disk.

        Returns:
            Mapping with keys ``original``, ``spoof_sin_jitter``,
            ``spoof_con_jitter``, ``host_wav``, and ``cloned_wav``, all as
            strings.
        """
        out_dir = self.request.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        sr = partial_spoof_settings.SAMPLE_RATE

        logger.info("Preparing host audio (OmniVoice reference + 16kHz processing copy)...")
        host_wav_16k = to_mono_wav(
            self.request.host_audio_path, out_dir / "host_16k.wav", sample_rate=sr
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

        host_spans = [find_phrase_span(host_words, tp) for tp in self.request.target_phrases]

        synth_words = [w.word for w in host_words]
        for (start_idx, end_idx), repl in sorted(
            zip(host_spans, self.request.replacement_phrases), key=lambda pair: -pair[0][0]
        ):
            synth_words[start_idx:end_idx + 1] = repl
        synth_text = " ".join(synth_words)
        logger.info(f"Replacement sentence to clone: \"{synth_text}\"")

        host_audio, _ = librosa.load(str(host_wav_16k), sr=sr)
        host_audio = host_audio.astype(np.float32)

        cloner = Cloner()
        logger.info("Loading OmniVoice model...")
        cloner.load(device=omnivoice_settings.DEVICE)

        best = self._generate_best_candidate(
            cloner, synth_text, host_wav_24k, host_text, host_audio, host_words, host_spans, out_dir
        )
        cloner.cleanup()

        if best is None:
            raise RuntimeError(
                f"None of {self.request.tries} OmniVoice render(s) produced a usable splice "
                f"for all of {self.request.replacement_phrases!r}."
            )

        eligible, worst_case_cf, _ = best["rank_key"]
        logger.info(
            f"Best candidate: \"{best['text']}\" | eligible={eligible} | "
            f"worst-case effective crossfade = {worst_case_cf:.1f}ms"
        )
        if not eligible:
            logger.warning(
                "No candidate met the production stretch-ratio gate "
                f"(MAX_STRETCH_RATIO={partial_spoof_settings.MAX_STRETCH_RATIO}); "
                "using the least-bad one. Consider raising --tries."
            )
        for d in best["details"]:
            logger.info(f"Splice detail: {d}")

        cloned_wav_16k = out_dir / "cloned_16k.wav"
        sf.write(str(cloned_wav_16k), best["audio"], sr)
        spoof_audio = best["spoof_audio"]

        original_path = out_dir / "original.wav"
        sf.write(str(original_path), host_audio, sr)

        spoof_path = out_dir / "spoof_sin_jitter.wav"
        sf.write(str(spoof_path), spoof_audio, sr)
        logger.info(f"Spoof (pre-jitter) written: {spoof_path}")

        jitter_path = out_dir / "spoof_con_jitter.wav"
        self._build_jittered(spoof_audio, host_words, jitter_path)

        return {
            "host_wav": str(host_wav_16k),
            "cloned_wav": str(cloned_wav_16k),
            "original": str(original_path),
            "spoof_sin_jitter": str(spoof_path),
            "spoof_con_jitter": str(jitter_path),
        }

    def _generate_best_candidate(
        self,
        cloner: Cloner,
        synth_text: str,
        host_wav_24k: Path,
        host_text: str,
        host_audio: np.ndarray,
        host_words: List[WordTimestamp],
        host_spans: List[Tuple[int, int]],
        out_dir: Path,
    ) -> dict | None:
        """Render the replacement sentence up to ``request.tries`` times and
        keep the one the REAL splice engine handles best.

        OmniVoice exposes no deterministic sampling seed (``Cloner.clone_single``'s
        ``seed`` argument is accepted only for interface compatibility), so
        repeated calls with the same text produce different prosody. A naive
        pre-screen based on raw silence around the replacement phrase is
        unreliable: ``splice_words`` refines each boundary by acoustic energy
        (``ENERGY_REFINE_RADIUS_S``) BEFORE computing the crossfade, and that
        refinement can shift the cut into material a raw-boundary check
        thought was silent. Running the actual production splice on every
        candidate and comparing the real ``effective_crossfade_ms`` it
        achieves is the only measure that matches what will actually be
        heard, so each candidate is fully spliced here and scored by its
        worst-case (minimum) effective crossfade across all swaps -- a
        candidate that nails one swap and fails the other is not "better"
        than one that does acceptably on both.

        Args:
            cloner: Loaded OmniVoice Cloner instance.
            synth_text: Full replacement sentence to synthesise.
            host_wav_24k: OmniVoice reference audio (24 kHz mono).
            host_text: Reference transcript for OmniVoice's conditioning.
            host_audio: Host waveform at the processing sample rate.
            host_words: Host word-level timestamps.
            host_spans: (start_index, end_index) into ``host_words`` for
                each entry of ``request.target_phrases``.
            out_dir: Directory to write per-candidate intermediate files to.

        Returns:
            A dict with ``score`` (worst-case effective_crossfade_ms across
            swaps), ``audio`` (16 kHz mono cloned waveform), ``text``
            (cloned transcript), ``spoof_audio`` (the spliced result), and
            ``details`` (per-swap splice metadata), or None if no candidate
            produced a usable splice for every swap.
        """
        sr = partial_spoof_settings.SAMPLE_RATE
        best: dict | None = None

        for attempt in range(self.request.tries):
            # Cloner.clone_single() accepts no seed OmniVoice itself will
            # honour (see its docstring), and OmniVoiceGenerationConfig
            # exposes no seed/generator field either -- identical (text,
            # reference) inputs were observed to produce byte-identical
            # audio across separate process runs. Forcing torch's global
            # RNG forward here is the only lever available to make
            # successive tries actually explore different renders instead
            # of re-cloning the same output `request.tries` times.
            torch.manual_seed(partial_spoof_settings.RANDOM_SEED + attempt)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(partial_spoof_settings.RANDOM_SEED + attempt)

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
            candidate_audio, _ = librosa.load(str(candidate_16k_path), sr=sr)
            candidate_audio = candidate_audio.astype(np.float32)

            bonafide_words_local = []
            cloned_words_local = []
            all_found = True
            for i, ((h_start_idx, h_end_idx), repl) in enumerate(
                zip(host_spans, self.request.replacement_phrases)
            ):
                try:
                    c_start_idx, c_end_idx = find_phrase_span(candidate_words, repl, fuzzy=True)
                except ValueError:
                    logger.warning(
                        f"Try {attempt}: replacement phrase {repl} not found in "
                        f"\"{candidate_text}\", skipping this render."
                    )
                    all_found = False
                    break
                placeholder = f"__TARGET_{i}__"
                bonafide_words_local.append({
                    "word": placeholder,
                    "start": host_words[h_start_idx].start,
                    "end": host_words[h_end_idx].end,
                })
                cloned_words_local.append({
                    "word": placeholder,
                    "start": candidate_words[c_start_idx].start,
                    "end": candidate_words[c_end_idx].end,
                })

            if not all_found:
                continue

            spoof_audio, details = splice_words(
                bonafide_audio=host_audio,
                cloned_audio=candidate_audio,
                bonafide_words=bonafide_words_local,
                cloned_words=cloned_words_local,
                selected_indices=list(range(len(bonafide_words_local))),
                sample_rate=sr,
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

            if len(details) < len(bonafide_words_local):
                logger.warning(
                    f"Try {attempt}: only {len(details)}/{len(bonafide_words_local)} "
                    "swaps spliced successfully, skipping this render."
                )
                continue

            max_ratio = partial_spoof_settings.MAX_STRETCH_RATIO
            eligible = all(
                (1.0 / max_ratio) <= d["duration_ratio"] <= max_ratio for d in details
            )
            worst_case_cf = min(d["effective_crossfade_ms"] for d in details)
            worst_stretch_penalty = max(abs(np.log(d["duration_ratio"])) for d in details)
            rank_key = (eligible, worst_case_cf, -worst_stretch_penalty)

            logger.info(
                f"Try {attempt}: \"{candidate_text}\" | eligible={eligible} "
                f"(duration ratios: {[round(d['duration_ratio'], 3) for d in details]}, "
                f"gate: {1/max_ratio:.2f}-{max_ratio:.2f}) | "
                f"worst-case effective crossfade = {worst_case_cf:.1f}ms"
            )

            if best is None or rank_key > best["rank_key"]:
                best = {
                    "rank_key": rank_key,
                    "audio": candidate_audio,
                    "text": candidate_text,
                    "spoof_audio": spoof_audio,
                    "details": details,
                }

        return best

    def _build_jittered(
        self,
        spoof_audio: np.ndarray,
        host_words: List[WordTimestamp],
        output_path: Path,
    ) -> None:
        """Apply the real Step 5b boundary-jitter logic to a spliced clip.

        Reuses ``BoundaryJitterApplier``'s internal boundary computation and
        jitter-plan application directly, instead of re-implementing the
        truncate/overlap/bleed logic, so this demo clip is produced by the
        exact same code that jitters the released corpus. Internal
        boundaries are taken from the host's original (pre-splice) word
        timings, matching production's own approximation (Step 5b does not
        re-align boundaries to the post-splice timeline either).

        Args:
            spoof_audio: The spliced (pre-jitter) waveform.
            host_words: Host word-level timestamps (pre-splice).
            output_path: Destination WAV path.
        """
        applier = BoundaryJitterApplier()
        bonafide_words = [{"start": w.start, "end": w.end} for w in host_words]
        boundaries = applier._compute_boundary_samples(bonafide_words)

        seed = partial_spoof_settings.JITTER_SEED + applier._stable_hash("live_demo")
        rng = np.random.RandomState(seed)

        jittered_audio, plan, drift = applier._apply_jitter_plan(
            audio=spoof_audio, boundaries=boundaries, rng=rng
        )
        sf.write(str(output_path), jittered_audio, partial_spoof_settings.SAMPLE_RATE)
        logger.info(
            f"Boundary jitter applied: {len(boundaries)} internal boundaries | "
            f"ops={plan['op_labels']} | drift={drift:+d} samples"
        )
        logger.info(f"Spoof (post-jitter) written: {output_path}")
