# Dataset Design

**Status:** Active
**Last updated:** 2026-04-25
**Source:** HABLA v2 corpus, pipeline settings

---

## HABLA 2.0 Corpus

The bonafide corpus is HABLA v2 (HABLA Latinoamericana), a Spanish speech dataset of Latin American speakers.

| Attribute | Value |
|-----------|-------|
| Speakers | 1,567 |
| Accents | 7 Latin American varieties |
| Total bonafide samples | ~35,927 |
| Audio format | WAV (16kHz, mono) + FLAC + MP3 (mxm_* speakers) |
| Directory | `data/bonafide_dataset_by_speaker_v2/` |

### Speaker ID Convention
- `{accent}{gender}_{number}` — e.g. `arf_00295` = Argentina Female #295, `arm_00412` = Argentina Male #412
- Accent codes: ar (Argentina), co (Colombia), mx (Mexico), pe (Peru), cl (Chile), ve (Venezuela), cu (Cuba)
- Gender codes: f (female), m (male)

### Critical Bug History
- **v1 vs v2:** Original code pointed to `bonafide_dataset_by_speaker` (v1, 162 speakers). Must use `bonafide_dataset_by_speaker_v2` (1,567 speakers). Fixed in all 5 pipelines.
- **MP3 glob:** `mxm_*` (Mexican male) speakers use MP3 format. File globs must include `*.wav, *.flac, *.mp3`.
- **MATCH_BONAFIDE_COUNT=True:** Generate exactly as many attack samples as bonafide per speaker for balanced dataset.

## Attack Dataset Structure

Each TTS pipeline generates ~35,927 attack samples (1:1 ratio with bonafide). Output follows ASVspoof2019 LA format:

```
data/attacks/<pipeline>_output/
  LA/
    ASVspoof2019_LA_train/flac/     — training split
    ASVspoof2019_LA_dev/flac/       — validation split
    ASVspoof2019_LA_eval/flac/      — evaluation split
    ASVspoof2019_LA_cm_protocols/   — protocol files (bonafide/spoof labels)
```

## Partial Spoof Tiers

| Tier | Words replaced | Min sentence length | Max spoof ratio |
|------|---------------|---------------------|-----------------|
| W1 | 1 word | 4 words | 25% |
| W2 | 2 words (non-adjacent) | 8 words | 25% |
| W3 | 3 words (non-adjacent) | 12 words | 25% |

## Dataset Scale Summary

| Component | Samples | TTS Systems |
|-----------|---------|-------------|
| Bonafide | ~35,927 | - |
| Full-synthesis attacks | ~35,927 per TTS | 5 (FishGram, Qwen, OpenVoice, Chatterbox, OuteTTS) |
| Partial spoof (W1+W2+W3) | TBD per TTS | 5 (same systems via Strategy pattern) |
| **Total estimated** | **~250,000+** | - |

## Related Pages
- [Production Runs](../experiments/production-runs.md) — actual counts and pass rates
- [Quality Metrics](quality-metrics.md) — validation thresholds
- [Anti-Spoofing Datasets](../state-of-art/anti-spoofing-datasets.md) — comparison with existing datasets
