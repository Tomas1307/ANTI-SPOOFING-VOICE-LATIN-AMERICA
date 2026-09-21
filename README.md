# MARSA: Multi-accent Audio Recordings for Spanish Anti-spoofing

Code accompanying the Scientific Data Data Descriptor **"MARSA: A
Multi-accent Corpus of Full and Partial Voice Spoofing Attacks for
Spanish Anti-spoofing Research"** (Acosta Bernal, Vásquez Correa,
Calderón Acosta, and Manrique).

This repository contains the full generation pipeline used to build
MARSA: six full-spoof TTS voice-cloning attack systems, an
energy-aware partial-spoof splicing engine with a boundary-jitter
variant, an acoustic-augmentation layer (reverberation, noise, and
telephony/VoIP codecs), a corpus leakage auditor, and the detector
training and zero-shot baseline evaluation harness reported in the
paper's Technical Validation section. It does not contain the
generated audio itself, the corpus is deposited separately (below).

## Dataset

The corpus, protocol files, frame-level labels, and per-sample
metadata are openly available on Zenodo:

- **Zenodo record:** `[ZENODO_DOI_PENDING]`

The Zenodo record's own description links back to this repository as
the code that produced it.

## Citation

If you use this code or the MARSA corpus, please cite the Data
Descriptor:

```
[CITATION_PENDING_UNTIL_PUBLISHED]
```

## Repository structure

```
ANTI-SPOOFING-VOICE-LATIN-AMERICA/
├── app/
│   ├── pipeline/
│   │   ├── fishgram_attack/       # Fish-Speech full-spoof pipeline
│   │   ├── qwen_attack/           # Qwen3-TTS full-spoof pipeline
│   │   ├── openvoice_attack/      # OpenVoice v2 full-spoof pipeline
│   │   ├── chatterbox_attack/     # Chatterbox full-spoof pipeline
│   │   ├── outetts_attack/        # OuteTTS full-spoof pipeline
│   │   ├── omnivoice_attack/      # OmniVoice full-spoof pipeline
│   │   ├── partial_spoof/         # Partial-spoof splicing pipeline
│   │   └── training/              # Detector training / eval harness
│   │       ├── dfarena/           # DF-Arena backend adapter
│   │       ├── lcnn/              # LFCC-LCNN, LSTM-sum backend
│   │       ├── lcnn_fixedcrop/    # LFCC-LCNN, fixed-crop backend
│   │       └── lcnn_selfattn/     # LFCC-LCNN, self-attention backend
│   ├── augmenter/                 # RIR, codec, and RawBoost augmentation
│   ├── runner/                    # Production orchestrators
│   ├── scripts/                   # Entry-point scripts (see Usage)
│   ├── schemas/                   # Pydantic data models
│   └── utils/                     # Shared utilities
├── envs/                          # Per-pipeline isolated venv requirements
├── requirements.txt                # Base requirements (corpus audit, stats)
└── README.md                      # This file
```

Each attack pipeline and the training harness follow the same internal
convention: a `pipeline_facade.py` entry point, `steps/step_XX_*.py`
for each pipeline stage, `schemas/` for its Pydantic models, and
`settings.py` for its configuration. See `app/pipeline/ARCHITECTURE.md`
for the full design rationale.

## Installation

Each attack system depends on a different, sometimes conflicting, set
of model libraries, so each runs in its own isolated virtual
environment rather than one shared one.

```bash
# Example: setting up one attack's environment
python3 -m venv envs/fishgram_env
source envs/fishgram_env/bin/activate
pip install -r envs/fishgram_requirements.txt
deactivate
```

Repeat for `qwen_env`, `openvoice_env`, `chatterbox_env`, `outetts_env`,
and `omnivoice_env`, using the matching `envs/<name>_requirements.txt`.
The corpus auditor, augmentation pipeline, and detector-training
harness (`dfarena_env`) use the base `requirements.txt` and do not
need model-specific dependencies.

GPU note: every generation, augmentation, and training script expects
exactly one visible CUDA device. Set `CUDA_VISIBLE_DEVICES` before
running anything that touches a GPU.

## Reproducing the corpus

The pipeline runs in this order. Each stage reads the previous stage's
output.

### 1. Bonafide source corpus

The bonafide speech underlying MARSA combines the original HABLA
corpus with a Common Voice extension used to fill under-represented
accent groups. See the paper's Methods section for the exact
speaker-selection and de-duplication procedure.

### 2. Full-spoof generation (six attack pipelines)

Each attack pipeline is run independently, inside its own venv:

```bash
source envs/fishgram_env/bin/activate
python -m app.scripts.run_attack --system fishgram
deactivate
```

Repeat per system (`qwen`, `openvoice`, `chatterbox`, `outetts`,
`omnivoice`), each inside its own venv.

### 3. Partial-spoof splicing

The partial-spoof pipeline consumes the full-spoof output and the
bonafide word-level timestamps to produce spliced attacks. A manifest
is generated once, then each (attack, partition) cell is dispatched:

```bash
source envs/dfarena_env/bin/activate
python -m app.scripts.generate_partial_spoof_manifest

python -m app.runner.partial_spoof_orchestrator --mode runbook
# prints the exact per-attack, per-partition commands to run,
# each inside that attack's own venv

python -m app.scripts.aggregate_partial_spoof_corpus
deactivate
```

### 4. Acoustic augmentation

```bash
source envs/dfarena_env/bin/activate
python -m app.scripts.run_augmentation_batch --factors 2x 3x 5x 10x
deactivate
```

### 5. Corpus leakage audit

Run before trusting any tier for training or evaluation. This is a
hard, citable gate, not an optional check:

```bash
source envs/dfarena_env/bin/activate
python -m app.scripts.run_corpus_audit --corpus-root data/augmented/augmented_2x
deactivate
```

## Baseline anti-spoofing evaluation

The zero-shot cross-corpus baseline reported in the paper's Technical
Validation section is reproduced with:

```bash
source envs/dfarena_env/bin/activate
export CUDA_VISIBLE_DEVICES=<free GPU index>

python -m app.scripts.run_detector_training \
    --run-name dfarena_zeroshot --backend dfarena \
    --eval-only --eval-splits dev eval

python -m app.scripts.run_detector_training \
    --run-name lcnn_selfattn_zeroshot --backend lcnn_selfattn \
    --eval-only --eval-splits dev eval --eval-crop-seconds 4.0375

deactivate
```

Substitute `--backend lcnn` or `--backend lcnn_fixedcrop` for the other
two LFCC-LCNN pooling variants. Results, including the corpus audit
report and per-attack breakdown, are written under
`data/training_runs/<run-name>/`.

## Output format

The released corpus follows the ASVspoof2019 Logical Access (LA)
convention: a flat directory of FLAC files per split, accompanied by a
protocol file naming the speaker, audio ID, generating system, and
class label (`bonafide`, `spoof`, or `partial_spoof`). Full details are
in the paper's Data Records section.

## License

The MARSA corpus is released under **Creative Commons Attribution 4.0
International (CC BY 4.0)**, consistent with the license of the
upstream HABLA corpus it derives from. The bonafide recordings
originate, in part, from Mozilla Common Voice (CC0); Mozilla's own
data-stewardship guidance asks that Common Voice audio not be mirrored
independently of this derivative use, which this release respects by
distributing it only as part of the processed MARSA corpus, not as a
standalone re-export.

The code in this repository is released under `[CODE_LICENSE_PENDING]`.

## Contact

Corresponding author: `[EMAIL_PENDING]`
