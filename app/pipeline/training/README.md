# Detector Training Pipeline

Trains and evaluates anti-spoofing detectors on the MARSA corpus, and reports
the baseline equal error rate the Scientific Data descriptor needs for its
Technical Validation section.

One shared harness serves every detector. The corpus audit, protocol handling,
training loop, checkpointing and evaluation live at the package root; each
detector contributes only a subpackage with its model adapter and settings.
`dfarena/` is the first. LFCC-LCNN, Nes2Net and HoliAntiSpoof sit beside it.

---

## 1. What it does

| Step | Class | Purpose |
|------|-------|---------|
| 1 | `CorpusLeakageAuditor` | Refuses to spend GPU time on a corpus whose invariants no longer hold |
| 2 | `ProtocolDatasetBuilder` | Resolves protocol and metadata files into typed splits |
| 3 | `DetectorFactory` | Instantiates the configured detector backend |
| 4 | `DetectorTrainer` | Trains, checkpoints mid-epoch, resumes exactly |
| 5 | `DetectorEvaluator` | Reports pooled, strict and per-attack EER |

---

## 2. Inputs

An augmentation tier laid out as the augmentation pipeline emits it:

```
data/augmented/augmented_2x/
  LA/
    ASVspoof2019_LA_train/
      flac/LA_T_0000001.flac ...
      ASVspoof2019.LA.cm.train.trn.txt
      MARSA.LA.cm.train.metadata.csv
    ASVspoof2019_LA_dev/   ...
    ASVspoof2019_LA_eval/  ...
```

Optionally `data/marsa_speaker_disjoint_partition/strict_eval_filter.csv`, which
joins through the metadata `source_file` column and enables the strict EER.

## 3. Outputs

Everything for one run lands in `data/training_runs/<run-name>/`:

```
config.json         resolved run configuration
corpus_audit.json   the audit report, written pass or fail
train.log           full loguru log
metrics.csv         one row per epoch
history.jsonl       the same, append-only
checkpoints/        checkpoint_step_*.pt (rolling), checkpoint_best.pt
scores/             scores_dev.txt, scores_eval.txt in ASVspoof score format
result.json         epochs, best checkpoint, evaluations
```

---

## 4. The audit gate

Step 1 asserts eight invariants from the protocol and metadata files alone.
No audio is decoded and no GPU is touched, so it costs about two minutes.

| Check | Fatal | What a failure means |
|-------|-------|----------------------|
| `speaker_disjointness` | yes | A voice heard in training is scored at evaluation |
| `protocol_metadata_correspondence` | yes | The two declarations of the corpus disagree |
| `ondisk_correspondence` | yes | Missing clips, or orphans from a superseded run |
| `clean_fraction_parity` | yes | Augmentation status predicts the label |
| `audio_id_ordering` | yes | The file name predicts the label |
| `attack_coverage` | no | A system is missing from a split; LOSO not viable |
| `class_balance` | no | One split lost a file family |
| `strict_filter_join` | no | The filter no longer joins onto this tier |

Run it standalone before committing to a tier:

```bash
python -m app.scripts.run_corpus_audit --corpus-root data/augmented/augmented_2x
```

`--skip-audit` exists for smoke tests only. A run that skips the audit logs a
warning and its error rate must not be reported.

The auditor is verified against synthetic corpora: a clean fixture passes all
eight checks; a fixture with a planted cross-split speaker, orphan FLAC and
class-by-class identifier ordering fails exactly those three and exits 1.

---

## 5. Environment

DF-Arena needs a recent `transformers`, which the attack-pipeline venvs pin
older for their own reasons. Give it its own environment.

```bash
cd ~/ANTI-SPOOFING-VOICE-LATIN-AMERICA
python3 -m venv envs/dfarena_env
source envs/dfarena_env/bin/activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install transformers soundfile numpy pydantic loguru
deactivate
```

The audit script alone has no torch dependency beyond what `fishgram_env`
already carries, so it can run there.

---

## 6. Usage

Pin the run to one free GPU. ml-server03 is shared, and the pipeline refuses
to start when it can see more than one device.

```bash
cd ~/ANTI-SPOOFING-VOICE-LATIN-AMERICA
source envs/dfarena_env/bin/activate
export CUDA_VISIBLE_DEVICES=1
mkdir -p logs

nohup python -m app.scripts.run_detector_training \
    --run-name dfarena_2x_run01 \
    --corpus-root data/augmented/augmented_2x \
    > logs/dfarena_2x_run01.out 2>&1 &

deactivate
```

Smoke test first, on a few thousand clips:

```bash
python -m app.scripts.run_detector_training \
    --run-name smoke01 --max-train-items 4000 --epochs 1 --eval-splits dev
```

Resume after an interruption by repeating the original command. `--resume`
defaults to `auto`, which picks up the latest checkpoint in the run directory
and replays the exact remaining batch order of the interrupted epoch.

---

## 7. Key options

| Flag | Default | Notes |
|------|---------|-------|
| `--batch-size` | 8 | Micro-batch; raise until the A40 is full |
| `--grad-accum` | 4 | Effective batch is the product of the two |
| `--backend` | dfarena | Registered backend key |
| `--freeze-backbone` | off | Head-only training; far cheaper, weaker |
| `--lr` / `--backbone-lr` | 1e-4 / 1e-6 | A pretrained backbone needs the smaller rate |
| `--crop-seconds` | 4.0 | Random crop in training, centre crop in scoring |
| `--eval-crop-seconds` | 0.0 | Zero scores whole utterances |
| `--keep-checkpoints` | 2 | Each checkpoint costs roughly 16 GB at 1B parameters |
| `--amp` | bf16 | Native on A40; `fp16` engages the gradient scaler |

---

## 8. Troubleshooting

**`No LA tree under ...`** — the tier directory name is wrong. The error lists
the sibling directories that do exist; pass the right one to `--corpus-root`.

**`N CUDA devices are visible`** — set `CUDA_VISIBLE_DEVICES` to a single free
GPU. Check `nvidia-smi` first; GPUs 0 and 2 are usually taken by others.

**`Only X GB free`** — checkpoints are large and the server partition runs
close to full. Lower `--keep-checkpoints`, or clear an old run directory.

**`Corpus audit failed on: [...]`** — read `corpus_audit.json`. Offending
speakers or clip identifiers are listed per check, capped at twenty.

**Clip sample rate errors** — the dataset refuses to resample silently. A clip
at anything other than 16 kHz means the tier was built wrong.

**Out of memory** — lower `--batch-size` and raise `--grad-accum` by the same
factor to keep the effective batch constant, or shorten `--crop-seconds`.
