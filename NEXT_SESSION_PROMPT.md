# Next Session: Validate Partial Spoof Pipeline v2

## Major Session Discoveries (April 25, 2026)

### The Real Problem Was Not Crossfade
We implemented 7 crossfade techniques (cut-paste, OLA Hanning, linear, cosine, half-sine, log, parabola) and discovered through listening tests that **all 7 sounded identical**. The audible problem was NOT the fade curve — it was:

1. **Blind word selection** — Step 4 picked words randomly. Words at fluid speech boundaries (where TTS generates continuous speech with no pause) sounded terrible when spliced.
2. **No clone quality gate** — Bad TTS clones went through the full pipeline.
3. **Duration mismatch** — Inserting a 480ms cloned word into a 640ms slot shifted everything by 160ms, destroying speech rhythm.

### Three Fixes Implemented

**Fix 1: Valley-Score Word Selection** (Step 4 rewrite)
- For each word boundary, compute `score = min_rms / avg_rms` in ±100ms window of 5ms frames
- Lower score = deeper energy valley = cleaner cut
- Combined score = max(left, right) — both boundaries must be clean
- Only select words below VALLEY_SCORE_THRESHOLD (0.65)
- Filter: duration >= 200ms, stretch ratio within [0.75, 1.25]
- File: `app/pipeline/partial_spoof/utils/valley_scorer.py` (NEW)
- File: `app/pipeline/partial_spoof/steps/step_04_select_words.py` (REWRITTEN)

**Fix 2: Clone Similarity Gate** (between Steps 2 and 3)
- ECAPA-TDNN cosine similarity between bonafide and clone
- Reject clones with SIM < 0.60 before alignment/splicing
- Saves compute on bad TTS outputs
- File: `app/pipeline/partial_spoof/pipeline_facade.py` (method `_filter_clones_by_similarity`)

**Fix 3: Duration-Preserving Splice** (splice_engine.py rewrite)
- Time-stretch cloned word to fit exact bonafide slot duration
- Overwrite in place: `result[b_start:b_end] = fitted` (total length never changes)
- Crossfade happens INSIDE the slot (first/last cf samples blend bonafide↔cloned)
- The 7 SpliceMethod techniques still control the fade curve at boundaries
- File: `app/pipeline/partial_spoof/utils/splice_engine.py` (REWRITTEN)

## What To Do Next

### 1. Validate on ml-server03 (5 speakers, 10+ audios each)

```bash
source ~/ANTI-SPOOFING-VOICE-LATIN-AMERICA/envs/fishgram_env/bin/activate
export CUDA_VISIBLE_DEVICES=1
cd ~/ANTI-SPOOFING-VOICE-LATIN-AMERICA
git pull
python -m app.pipeline.partial_spoof.pipeline_facade
```

Validation speakers: `["arf_00295", "arf_00610", "arf_01523", "arm_00412", "arm_00780"]`

Check:
- All spliced WAVs have identical duration to bonafide source
- `word_selection_metadata.json` contains `valley_score` per word
- `clone_similarity_filter.json` exists with per-sample SIM scores
- Listen to 10+ samples — rhythm should sound natural
- Compare metrics against baseline: WER=3.9%, NISQA=4.72, SIM=0.789

### 2. Check Production Runs
- Chatterbox: was running on GPU 2 (~April 22)
- OuteTTS: was running on GPU 2 (~April 22)

### 3. Run Partial Spoof Production (after validation passes)
All 1,567 speakers with each TTS system that completed production.

## Files Changed This Session

| File | Change |
|------|--------|
| `utils/valley_scorer.py` | NEW: ValleyScorer class |
| `utils/splice_method.py` | NEW: SpliceMethod enum + weights |
| `utils/crossfade.py` | 7 fade curves + _compute_fade_curves |
| `utils/splice_engine.py` | REWRITTEN: duration-preserving overwrite |
| `steps/step_04_select_words.py` | REWRITTEN: valley-score selection |
| `steps/step_05_splice_audio.py` | Updated splice_words call + spoof_duration_ratio |
| `pipeline_facade.py` | Added _filter_clones_by_similarity gate |
| `settings.py` | 7 new fields (valley, similarity, stretch) + 5 validation speakers |
| `schemas/valley_score.py` | NEW: Pydantic schema |
| `schemas/similarity_filter_result.py` | NEW: Pydantic schema |
| `schemas/spliced_word_info.py` | Extended with splice_method, effective_crossfade_ms |
| `presentation/slides/13a-13l` | 12 new slides (visual, challenge, 7 techniques, summary, problem, solution) |

## Presentation Status
34 total slides: 20 original + 06b (Chatterbox) + 06c (OuteTTS) + 13a–13l (12 partial spoof slides).
Run `python presentation/build.py` to rebuild.

## Thesis Wiki
Complete 17-page wiki at `docs/thesis-wiki/`. See `docs/thesis-wiki/index.md` for all pages.
Covers: state-of-art (5), methodology (5), experiments (3), decisions (1), schema/index/log (3).
Maintained per CLAUDE.md protocol — update whenever research decisions or results change.
