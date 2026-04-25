# Thesis Wiki Schema

## Purpose

This wiki is the single source of truth for the HABLA 2.0 voice anti-spoofing thesis. It documents the methodology, architectural decisions, experiment results, state of the art, and investigation findings in a structured, cross-referenced format suitable for directly informing the paper writing process.

The codebase is the raw implementation. `investigation.md` is the deep research source. This wiki is the compiled, maintained synthesis — optimized for paper writing.

## Architecture

**Raw sources** — the codebase (`app/pipeline/`), investigation documents (`investigation.md`, `partial_spoof_inv_1.md`, `partial_spoof_inv_2.md`), experiment logs (ml-server03 outputs), and engram MCP memory. These are the ground truth.

**The wiki** — this directory. Markdown files that synthesize, cross-reference, and document the research landscape. Organized by topic, interlinked with relative markdown links. Claude Code maintains this layer. Master Tomas reads and uses it for the paper.

**This schema** — conventions, structure, and workflows for maintaining the wiki.

## Directory Structure

```
docs/thesis-wiki/
  SCHEMA.md               # This file - conventions and workflows
  index.md                # Master catalog of all pages with summaries
  log.md                  # Chronological record of changes and discoveries

  # Literature review and state of the art
  state-of-art/
    tts-systems.md              # 6 TTS systems evaluated (from investigation.md)
    anti-spoofing-datasets.md   # ASVspoof, PartialSpoof, LlamaPartialSpoof, HAD, etc
    detection-methods.md        # AASIST, SSL-based, ECAPA-TDNN, spectral analysis
    partial-spoof-literature.md # Section 8 synthesis: 4 pipelines, 7 questions
    splicing-techniques.md      # 7 crossfade methods, OLA, energy analysis

  # Our methodology
  methodology/
    pipeline-architecture.md    # 7-step pipeline design (Facade, Strategy, Steps)
    attack-systems.md           # Per-TTS implementation details and trade-offs
    partial-spoof-approach.md   # Valley score, duration preserving, clone gate
    dataset-design.md           # HABLA v2, speaker distribution, tiers W1/W2/W3
    quality-metrics.md          # WER, CER, NISQA MOS, ECAPA SIM thresholds

  # Experiment results and logs
  experiments/
    production-runs.md          # Per-pipeline: samples, pass rate, metrics
    validation-results.md       # Partial spoof validation (5 speakers)
    ablation-studies.md         # Crossfade vs valley score, threshold tuning

  # Architectural and research decisions
  decisions/
    decision-log.md             # Chronological: what, why, alternatives considered
```

## Conventions

### Page format

Every page starts with a YAML-like header block:

```markdown
# Page Title

**Status:** Active | Draft | Superseded
**Last updated:** YYYY-MM-DD
**Source:** investigation.md Section N / experiment log / decision

---

Content here...
```

### Cross-references

Use relative markdown links: `[TTS Systems](../state-of-art/tts-systems.md)`.

### Numbers and metrics

Always include the date when metrics were captured. Production numbers change over time.

### Decision records

Each decision in `decision-log.md` follows:

```
### YYYY-MM-DD: Decision title
**Context:** What prompted the decision
**Decision:** What was decided
**Alternatives considered:** What else was evaluated
**Outcome:** Result (if known)
```

## Maintenance workflow

1. After each significant work session, update the relevant wiki pages.
2. Add new decisions to `decisions/decision-log.md`.
3. Update `log.md` with a one-line entry.
4. Update `index.md` if new pages were added.
5. When writing the paper, start from the wiki — it has the curated, cross-referenced version.
