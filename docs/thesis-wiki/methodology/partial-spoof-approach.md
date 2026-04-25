# Partial Spoof Approach

**Status:** Active
**Last updated:** 2026-04-25
**Source:** Session discoveries from listening tests + implementation

---

## Overview

The partial spoof pipeline replaces 1-3 individual words in bonafide HABLA utterances with voice-cloned versions from TTS systems. Unlike full-synthesis attacks (where the entire utterance is fake), partial spoof makes detection significantly harder because 90%+ of the audio is genuine.

## Key Design Decisions

### Full-sentence cloning (not word-by-word)
We clone the **full sentence** with TTS, then extract individual words via forced alignment. Words generated in isolation have flat, citation-form prosody that is trivially detectable. In-context words carry natural prosody — making the splice harder to detect.

### Valley-score word selection (not random)

**Discovery (April 25):** Listening tests showed all 7 crossfade techniques sounded identical. The audible problem was blind word selection — randomly picking words at fluid speech boundaries (where the TTS generates continuous speech with no energy dip between words).

**Solution:** Score each word boundary by energy valley depth:

```
score = min_rms / avg_rms    (in +-100ms window, 5ms frames)
combined = max(left_score, right_score)
```

- Score 0.0 = perfect silence at boundary (ideal cut point)
- Score 1.0 = no energy dip (impossible to cut cleanly)
- Threshold: 0.65 (configurable). Words above are ineligible.

**Selection algorithm:** Greedy best-first with non-adjacency constraint. Not random.

### Duration-preserving splice (not variable-length insert)

**Discovery (April 25):** Inserting a 480ms cloned word into a 640ms bonafide slot shifted all subsequent audio by 160ms, destroying speech rhythm.

**Solution:** Time-stretch the cloned word to fit the exact bonafide slot duration. Overwrite in place. Total audio length never changes. Stretch ratio limited to [0.75, 1.25].

### Clone similarity gate

ECAPA-TDNN cosine similarity between bonafide and clone must be >= 0.60. Bad clones are rejected before alignment/splicing (between Steps 2 and 3), saving compute.

## Edge Cases

| Case | Handling |
|------|----------|
| All words have bad valley scores | Reject sample for tier |
| Not enough good words for W2/W3 | Skip higher tiers |
| Word < 200ms | Ineligible (too short to be meaningful) |
| First/last word | Only internal boundary scored (external = silence, artificially good) |
| Stretch ratio outside [0.75, 1.25] | Word ineligible |
| All clones for a speaker fail similarity | Speaker produces no partial spoofs |

## Quality Metrics

From validation run (3 speakers, 7 samples, pre-v2 pipeline):
- WER: 3.9%
- NISQA MOS: 4.72
- Speaker similarity: 0.789
- Pass rate: 7/7 (100%)

**Pending:** v2 validation with 5 speakers, 10+ audios each, including valley-score selection.

## 7 Crossfade Techniques

Still applied per splice for forensic diversity. The technique affects the fade curve at slot boundaries (first/last 15-80ms). Drawn per-splice from weighted distribution:

| Technique | Proportion | Energy behavior |
|-----------|-----------|----------------|
| Direct cut-paste | 10% | No blend |
| OLA Hanning | 20% | Equal-gain (S-curve) |
| Linear | 15% | Equal-gain (amplitude dip) |
| Cosine | 20% | Equal-power |
| Square-root | 15% | Equal-power |
| Logarithmic | 10% | Neither (energy dip) |
| Inv. parabola | 10% | Equal-gain |

## Related Pages
- [Splicing Techniques](../state-of-art/splicing-techniques.md) — literature review
- [Quality Metrics](quality-metrics.md) — thresholds and formulas
- [Decision Log](../decisions/decision-log.md) — chronological decisions
