# Decision Log

**Status:** Active
**Last updated:** 2026-04-25

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
