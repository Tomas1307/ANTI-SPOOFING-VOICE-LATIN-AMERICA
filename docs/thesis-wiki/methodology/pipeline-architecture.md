# Pipeline Architecture

**Status:** Active
**Last updated:** 2026-04-25
**Source:** app/pipeline/ARCHITECTURE.md, codebase

---

## Overview

Every attack pipeline follows a canonical 7-step architecture using GoF design patterns. The pipeline is orchestrated by a Facade class that provides a single entry point.

## Design Patterns

| Pattern | Usage |
|---------|-------|
| **Facade** | `pipeline_facade.py` — single entry point, orchestrates all steps |
| **Strategy** | TTS backends (FishGram, Qwen, OpenVoice, etc.) are interchangeable via `create_attack_strategy()` |
| **Factory** | `strategy_factory.py` — instantiates the correct TTS strategy by name |
| **Singleton** | `settings.py` — Pydantic BaseModel singleton for pipeline-scoped config |

## Full-Synthesis Attack Pipeline (6 steps)

```
Step 1: Scan bonafide speakers → list of (speaker_id, audio_files)
Step 2: Build reference audio  → 15s concatenated reference per speaker
Step 3: Generate TTS clones    → one clone per bonafide sample
Step 4: Validate quality       → Parakeet WER/CER + NISQA MOS + ECAPA SIM
Step 5: Format output          → ASVspoof2019 LA directory structure
Step 6: Generate protocol      → train/val/test protocol files
```

## Partial Spoof Pipeline (7 steps + regen loop)

```
Step 1: Transcribe bonafide       → word-level timestamps (Parakeet TDT)
  REGEN LOOP (up to 3 rounds):
    [Gate]: Clone similarity      → ECAPA SIM >= 0.60
    Step 2: Clone full sentence   → TTS clone of complete utterance
    Step 3: Forced alignment      → word timestamps for both bonafide + clone
    Step 4: Select words          → valley-score ranked, non-adjacent
    Step 5: Splice audio          → duration-preserving overwrite
  Step 6: Validate quality        → WER/CER/NISQA/ECAPA
  Step 7: Format output           → ASVspoof2019 LA structure
```

## File Structure (per pipeline)

```
app/pipeline/<name>/
  pipeline_facade.py      — Facade: orchestrates steps
  settings.py             — Pydantic BaseModel singleton
  strategies/             — TTS strategy implementations
  steps/                  — step_01_*.py through step_0N_*.py
  schemas/                — Pydantic models (one per file)
  utils/                  — helper functions
  README.md               — user-facing docs
  ARCHITECTURE.md         — technical design
```

## Implemented Pipelines

| Pipeline | Directory | Strategy | Steps |
|----------|-----------|----------|-------|
| FishGram | `fishgram_attack/` | FishGramStrategy (HTTP API) | 6 |
| Qwen3-TTS | `qwen_attack/` | QwenStrategy (local model) | 6 |
| OpenVoice | `openvoice_attack/` | OpenVoiceStrategy (MeloTTS + TCC) | 6 |
| Chatterbox | `chatterbox_attack/` | ChatterboxStrategy (GPT + Vocos) | 6 |
| OuteTTS | `outetts_attack/` | OuteTTSStrategy (Llama + WavTokenizer) | 6 |
| Partial Spoof | `partial_spoof/` | Any of the above (Factory) | 7 + regen |

## Related Pages
- [Attack Systems](attack-systems.md) — per-TTS configuration details
- [Partial Spoof Approach](partial-spoof-approach.md) — valley score, duration preserving
- [Decision Log](../decisions/decision-log.md) — architectural decisions
