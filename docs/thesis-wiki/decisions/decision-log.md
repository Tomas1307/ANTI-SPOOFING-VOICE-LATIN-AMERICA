# Decision Log

**Status:** Active
**Last updated:** 2026-05-06

Chronological record of architectural, methodological, and research decisions.

---

### 2026-02-17: TTS System Selection
**Context:** Needed to select TTS systems for generating synthetic Spanish voice attacks.
**Decision:** 6 systems: Fish Speech (primary), Qwen3-TTS, OpenVoice, Chatterbox, OuteTTS, CosyVoice.
**Alternatives considered:** XTTS, Bark, Parler-TTS. Rejected for poor Spanish support or discontinued development.
**Outcome:** CosyVoice later dropped (generates Chinese for Spanish input). 5 systems in production.

### 2026-03-15: HABLA v2 over v1 for bonafide corpus
**Context:** Original HABLA v1 had 162 speakers. v2 expanded to 1,567 speakers across 7 Latin American accents.
**Decision:** Use v2 exclusively. Update all pipeline BONAFIDE_DIR references.
**Alternatives considered:** Keep v1 for faster iteration. Rejected: v1 lacks accent diversity.
**Outcome:** All 5 attack pipelines use v2. ~35,927 bonafide samples as target.

### 2026-03-20: 1:1 Bonafide Matching (MATCH_BONAFIDE_COUNT=True)
**Context:** Each speaker has variable bonafide counts. Generate exactly as many attack samples as bonafide per speaker.
**Decision:** MATCH_BONAFIDE_COUNT=True in all pipelines. ~35,927 total attack samples per TTS system.
**Alternatives considered:** Fixed N per speaker, or all combinations. Rejected: unbalanced dataset harms detector training.
**Outcome:** Consistent ~35k samples per pipeline. Balanced bonafide:spoof ratio.

### 2026-04-07: Full-sentence cloning for partial spoof (not word-by-word)
**Context:** Need to extract individual words from TTS clone for partial spoof. Two options: generate each word separately, or generate full sentence and extract.
**Decision:** Clone the full sentence, then extract words via forced alignment.
**Alternatives considered:** Word-by-word generation. Rejected: isolated words have flat, citation-form prosody that is trivially detectable.
**Outcome:** Natural in-context prosody preserved. Harder for detectors. But creates the alignment accuracy challenge.

### 2026-04-10: Text-matching splice engine (not positional index)
**Context:** Original splice engine used `cloned_words[idx]` positional matching. If clone has fewer/different words, wrong content gets spliced.
**Decision:** Build text lookup map with NFKD normalization for Spanish accents. Match by word text, not position.
**Alternatives considered:** Strict positional matching. Rejected: fails when TTS drops/merges words.
**Outcome:** 33/33 Spanish character pairs pass normalization. Handles Como/Cómo, niños/ninos, etc.

### 2026-04-15: Non-adjacency constraint for word selection
**Context:** Replacing adjacent words creates a contiguous spoofed block that is easy to detect and sounds unnatural.
**Decision:** Selected word indices must differ by >= 2. Enforced via rejection sampling.
**Alternatives considered:** No constraint (random). Rejected: contiguous blocks are a trivial forensic shortcut.
**Outcome:** W1=1 word, W2=2 non-adjacent, W3=3 non-adjacent. Forces detectors to find scattered replacements.

### 2026-04-22: 7 crossfade techniques with weighted random selection
**Context:** Literature uses varied splicing methods. LlamaPartialSpoof uses 5 fade shapes. We wanted diversity.
**Decision:** 7 techniques: cut-paste (10%), OLA Hanning (20%), linear (15%), cosine (20%), half-sine (15%), logarithmic (10%), parabola (10%). Per-splice random draw.
**Alternatives considered:** Single technique (Hann only). Rejected: detector learns one fingerprint.
**Outcome:** Implemented, but listening tests revealed the technique doesn't matter audibly — the real problem was elsewhere. See 2026-04-25 decisions.

### 2026-04-25: Valley-score word selection (replacing random selection)
**Context:** Listening tests showed all 7 crossfade techniques sounded identical. The audible problem was: (a) words selected at boundaries with no energy valley in the clone, (b) duration mismatch shifting speech rhythm by 100-200ms.
**Decision:** Replace random word selection with energy valley scoring. For each word boundary, `score = min_rms / avg_rms` in +-100ms window. Only select words with deep valleys (score <= 0.65).
**Alternatives considered:** (a) Generate TTS with gaps between words (changes prosody), (b) Better forced aligner (doesn't exist for Spanish at word level). Valley score is a post-hoc filter that doesn't modify the generation.
**Outcome:** Testing. Expected to eliminate bad-sounding splices by only selecting words at clean boundaries.

### 2026-04-25: Clone similarity gate (ECAPA >= 0.60)
**Context:** Bad TTS clones go through the entire pipeline (Steps 3-5) wasting compute and producing obviously detectable splices.
**Decision:** Add ECAPA-TDNN cosine similarity gate between Steps 2 and 3. Reject clones with SIM < 0.60.
**Alternatives considered:** (a) Higher threshold 0.70 — rejects too many (FishGram avg is 0.602), (b) No gate — wastes compute on bad clones.
**Outcome:** Testing. Threshold 0.60 is configurable in settings.py.

### 2026-04-25: Duration-preserving splice (overwrite in place)
**Context:** Previous approach inserted cloned word at natural duration, changing total audio length. A 480ms clone replacing a 640ms bonafide slot shifted everything by 160ms, destroying speech rhythm.
**Decision:** Time-stretch cloned word to fit exact bonafide slot duration. Overwrite in place. Total length never changes.
**Alternatives considered:** (a) Keep natural duration (rhythm breaks), (b) Phase vocoder for higher quality stretch (complexity, marginal benefit for small ratios). Linear interpolation sufficient for ratios 0.75-1.25.
**Outcome:** Testing. All output audio should have identical duration to bonafide source.

### 2026-05-01: OmniVoice (k2-fsa) added as 6th attack pipeline
**Context:** Need an additional TTS attack vector with a fundamentally different architecture from the existing 5 (Fish Speech, Qwen3-TTS, OpenVoice, Chatterbox, OuteTTS). OmniVoice is a diffusion-language-model TTS supporting 646 languages with 27,559 hours of Spanish training data — one of the largest Spanish coverage among open zero-shot TTS models.
**Decision:** Implement `app/pipeline/omnivoice_attack/` following the canonical 5-step structure (prepare_references, prepare_texts, generate_speech, validate_quality, format_output). Use Python API (`OmniVoice.from_pretrained`) in-process; pre-compute reference text with Parakeet TDT (consistent with project STT stack); resample 24 kHz native to 16 kHz on FLAC write. Audio ID range = 15M+ (avoids collision with partial_spoof main W1/W2/W3 at 12-14M).
**Alternatives considered:** (a) HTTP API server like Fish Speech — rejected, OmniVoice in-process is simpler and Fish-style pattern only justified by Fish's heavy native dependencies. (b) Internal Whisper auto-transcription for ref_text — rejected, conflicts with "siempre Parakeet" rule and adds a 2nd ASR model. (c) torch 2.8.0+cu128 (upstream recommendation) — rejected, ml-server03 driver 560.35.03 maps to CUDA 12.6, so cu126 is the safe match.
**Outcome:** Pipeline written, isolated venv `envs/omnivoice_env/` with `omnivoice_requirements.txt`. First run pending on ml-server03 to validate Spanish quality. OmniVoice is intentionally NOT used as the boundary-jitter pilot precisely because its quality is unvalidated.

### 2026-05-01: Boundary jitter feature (Step 5b) for partial spoof
**Context:** The current partial_spoof splice produces detectable artifacts only at the 2 boundaries surrounding each cloned word. The other N-3 boundaries (bonafide-bonafide) are clean. Detectors trained on this distribution can learn the shortcut "the noisy boundary is the splice", as documented in Negroni et al. (2024) (6.16% EER on PartialSpoof with no training, purely spectral-dynamic-range analysis of the join) and the generalization-shortcut analysis in Muller (2024). Master Tomas's idea: apply the same kind of structural artifacts that the splice produces, to ALL internal boundaries (including the spoof one), so the splice no longer stands out.
**Decision:** Add Step 5b `BoundaryJitterApplier` running after Step 5 and before Step 6. For every internal word boundary, throw a coin with `JITTER_PROBABILITY = 0.5`. Heads -> uniformly pick one of three structural manipulations: (a) **truncate** [10-40 ms uniform] — cut a fragment from one side, mimics hard cut/paste; (b) **overlap** [30-80 ms uniform, Hanning fade] — sum left tail with right head, mimics OLA-Hanning crossfade; (c) **bleed** [20-60 ms uniform] — insert a fragment of one adjacent word into the other, mimics tail bleed. Tails -> leave natural. Spoof boundaries receive the same coin flip on top of the existing splice, so the distribution of "manipulations per boundary" is no longer 0 vs 1 but 0/1 (bonafide-bonafide) vs 1/2 (bonafide-spoof) — much harder to use as a signal.
**Alternatives considered:** (a) Channel/codec/RIR augmentation per word (my initial proposal) — rejected by Master Tomas, raises WER significantly and addresses a different threat model than what we want. (b) Apply to entire utterance uniformly (channel matching) — rejected, doesn't address the per-boundary detection signal. (c) Skip spoof boundaries (only manipulate bonafide-bonafide) — rejected, leaves the spoof boundary distinguishable as the only one with a coin-flip-zero-extra-manipulation distribution. (d) Stack multiple operations per boundary — rejected, risks audible chopped audio.
**Outcome:** Implemented. Magnitude ranges grounded: overlap matches LlamaPartialSpoof crossfade range exactly (literature); truncate stays below Spanish syllable nucleus duration (50-90 ms) to preserve intelligibility; bleed covers Spanish VOT (4-29 ms) plus consonant-vowel transition. Boundaries processed right-to-left to avoid invalidating earlier sample positions. Total length drift bounded; recorded per utterance.

### 2026-05-01: Per-speaker bonafide partition for jitter dataset (frases distintas)
**Context:** Master Tomas wanted the jitter dataset to use sentences disjoint from the main partial_spoof dataset, so the combined training pool covers more of the data distribution rather than duplicating utterances with two processings.
**Decision:** Add `BONAFIDE_FILE_PARTITION` setting ('main' or 'jitter') that controls which half of each speaker's bonafide files are used. Files are shuffled per-speaker with a deterministic seed (`BONAFIDE_PARTITION_SEED + sha256(speaker_id)`) and split 50/50; 'main' takes the first half, 'jitter' takes the second half. No speakers are discarded; speakers with only 1 file contribute to whichever partition the shuffle assigned.
**Alternatives considered:** (a) Use the same utterances in both — rejected by Master Tomas, no diversity gain. (b) Different speakers for each partition — rejected, would change the speaker distribution between datasets and break comparability. (c) Discard speakers with too few files — rejected by Master Tomas, "no descartar speakers".
**Outcome:** Implemented in `step_01_transcribe_bonafide._apply_partition`. Trade-off explicit: the jitter dataset is NOT a clean ablation of the main dataset (different inputs, different processing) but covers complementary utterances. A clean ablation experiment would require running both partitions with both processings (4 combinations) — out of scope for now.

### 2026-05-01: Qwen as boundary-jitter pilot, not OmniVoice
**Context:** When piloting a new dataset technique, conflating two unknowns ("does the technique work?" + "does the TTS produce clean clones?") makes the experiment uninterpretable.
**Decision:** Pilot the boundary jitter feature exclusively on Qwen3-TTS, which is already validated and has the highest mean ECAPA similarity (0.720). OmniVoice quality is unknown until its standalone TTS pipeline runs successfully on ml-server03. If jitter improves detector EER on Qwen-based partial_spoof, replicate to Chatterbox, OpenVoice, OuteTTS, FishGram. OmniVoice joins the jitter suite only after passing standalone validation.
**Alternatives considered:** Pilot on OmniVoice (would have been "free" since the new code path is identical) — rejected, because the experiment's signal would be lost in TTS-quality noise.
**Outcome:** `BONAFIDE_FILE_PARTITION` and `ENABLE_BOUNDARY_JITTER` are pipeline-system-agnostic; any of the 6 attack systems can be paired with jitter. First production run will be `attack_system="qwen"` with `BONAFIDE_FILE_PARTITION="jitter"` and `ENABLE_BOUNDARY_JITTER=True`.

### 2026-05-06: Reference end-cut is the actual root cause of OmniVoice prefix bleed (not per-speaker failure)
**Context:** The earlier non-verbal-prefix-detection + retry decision (see entry below) was based on a misdiagnosis. Initial analysis of the validation run that bled on `arf_00295` led to the conclusion that OmniVoice has a deterministic per-speaker failure mode tied to specific reference clips or speaker embeddings. After Master Tomas listened to `arf_00295_ref.wav` and reported "the final phrase was cut", the actual root cause was found in `app/pipeline/omnivoice_attack/utils/audio_concatenation.py`: the concatenation routine sliced the last bonafide file at the exact sample boundary needed to hit a 10 s target duration, landing the cut mid-word. The reference therefore ended abruptly, and OmniVoice's diffusion conditioning attempted to "complete" the cut-off pattern at the start of generation -- producing the reference-voice fragment we observed. The same bug exists in all 7 other attack pipelines but only affects OmniVoice because diffusion-LM is uniquely sensitive to abrupt-end conditioning (autoregressive and VQGAN backends tokenize/quantize the reference and don't extrapolate from its trailing edge).
**Decision:** Rewrite `concatenate_with_padding` to (1) stop at the last file that fits without overflowing 10 s instead of slicing mid-file, (2) snap to the nearest silent frame within +/- 1 s of target if a single file alone exceeds 10 s (edge case), (3) always append 200 ms trailing silence regardless of where the body ended, so the reference always ends on a clean silence boundary. Keep the non-verbal-prefix detector and retry loop as a defensive backstop in case future references still produce bleed.
**Alternatives considered:** (a) Per-speaker reference selection retry (try different bonafide files if generation fails) -- rejected, addresses the symptom not the cause and complicates Step 1. (b) Tighten WER ceiling -- rejected, the bleed is sub-syllabic and Parakeet does not transcribe it, so WER is blind. (c) Forced phoneme alignment (Wav2Vec2 CTC, MFA) for the absorbed-bleed case -- deferred, not currently observed. (d) Retrofit the same fix to the other 7 pipelines -- declined for now (see future-work entry in `methodology/attack-systems.md`); FishGram/Qwen/OpenVoice production runs are complete, Chatterbox/OuteTTS are mid-run, and we have no direct evidence the bug affects them.
**Outcome:** Validation re-run after the fix: 6/6 passed first attempt, zero non-verbal-prefix rejections, zero retries needed. arf_00295 -- which previously bled deterministically across 6 generation attempts -- now passes cleanly. All quality metrics improved (WER 0.0394 -> 0.0185, CER 0.0181 -> 0.0083, NISQA 4.53 -> 4.59, Speaker Sim 0.680 -> 0.696). The earlier "deterministic per-speaker failure mode" finding is retracted; arf_00295's behavior was a consequence of the buggy reference, not a property of the speaker. The detector + retry infrastructure remains in place as defense in depth.

### 2026-05-06: OmniVoice non-verbal prefix detection — reject + retry, not trim
**Context:** Validation of OmniVoice (6 samples) showed 2/6 samples (both from `arf_00295`) had a 200-600 ms reference voice fragment leading the audio. Parakeet TDT did not transcribe the bleed (sub-syllabic), so WER stayed at 0.0 and the existing `detect_prefix_trim_point` (Qwen-style hallucinated-word alignment) returned 0.0. The validator reported "Prefix trims: 0" while the audio was clearly contaminated. The artifact is acoustically valid voice from the same speaker — VAD cannot distinguish it from the prompt content.
**Decision:** Add a second detector `detect_nonverbal_prefix_artifact` to `app/utils/prefix_trimmer.py` that computes RMS dBFS of the audio interval `[0, word_timestamps[0].start]`. If pre-RMS exceeds `NONVERBAL_PREFIX_RMS_FLOOR_DB = -55.0`, mark the sample as **rejected** rather than trimming it in place. Wire OmniVoice into `app/runner/production_runner.py` (key "3") with a new `_execute_omnivoice` method that calls `_retry_rejected` (mirroring the Qwen and FishGram pattern), so rejected samples are regenerated up to `MAX_GENERATION_RETRIES = 5` rounds. Threshold is data-driven: empirically, OmniVoice artifacts measure -22 to -25 dB and clean samples sit at -120 dB silence floor — a -55 dB cut sits 30 dB above the artifact band and 65 dB below the silence band.
**Alternatives considered:** (a) Trim in place using `T_first` as the trim point — rejected, risks cutting natural Spanish vowel onset (e.g., `/e/` of "Eurídice"), no way to know the artifact's true end point. (b) Fixed unconditional 100 ms trim — rejected, too crude, false-positives on clean samples. (c) Forced phoneme alignment (Wav2Vec2 CTC, MFA) — rejected for now, adds another model and another env dependency; deferred until production data shows the absorbed-bleed case (where Parakeet absorbs the bleed into the first word's start time) is common enough to need it. (d) Tighten WER ceiling to catch bleed via downstream WER — rejected, TEXT_00002 had WER=0.0 even with audible bleed.
**Outcome:** Implemented in `prefix_trimmer.py`, `omnivoice_attack/settings.py`, `omnivoice_attack/schemas/validation_result.py`, `omnivoice_attack/steps/step_04_validate_quality.py`, `production_runner.py`. Validation re-run with 5 retry rounds: detector achieved 100% recall (12/12 bleed instances flagged across initial + 5 retry rounds on 2 samples). Both `arf_00295` samples bled deterministically across all 6 generation attempts — surfaces a model-level failure mode (likely tied to specific reference clips or the speaker's diffusion conditioning) that retries cannot fix. Reference-selection follow-up needed before production. Other 4 samples passed cleanly without retry. Detection scope is OmniVoice-only; not retrofitted to other 5 pipelines until evidence shows they exhibit the same artifact.
