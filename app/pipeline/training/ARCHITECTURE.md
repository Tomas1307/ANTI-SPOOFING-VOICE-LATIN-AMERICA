# Detector Training Pipeline - Technical Design

## 1. Flow

```
                        DetectorTrainingConfig
                                 |
                    DetectorTrainingPipeline (Facade)
                                 |
   +-------------+---------------+---------------+--------------+
   |             |               |               |              |
 STEP 1        STEP 2          STEP 3          STEP 4         STEP 5
 Corpus        Protocol        Detector        Training       Evaluation
 audit         to splits       factory         loop           and scoring
   |             |               |               |              |
LeakageAudit  Dict[str,      BaseSpoof       TrainingResult  List[Evaluation
 Report       DatasetSplit]   Detector                        Result]
   |
 FATAL FAILURE -> RuntimeError, no GPU time spent
```

Run artefacts, all under `data/training_runs/<run-name>/`:

```
config.json  corpus_audit.json  train.log  metrics.csv  history.jsonl
checkpoints/{checkpoint_step_*.pt, checkpoint_best.pt}
scores/{scores_dev.txt, scores_eval.txt}  result.json
```

---

## 2. One harness, many detectors

The corpus is the constant; the detector is the variable. Everything that
touches the corpus — the audit, protocol I/O, the dataset, the training loop,
checkpointing, the metrics — is written once at the package root and shared.
A detector contributes only a subpackage holding its model adapter and its own
settings.

Copying the harness per detector was considered and rejected: it would give us
one copy of the audit and one copy of the training loop per model to keep in
step, and they would diverge.

```
app/pipeline/training/
    pipeline_facade.py                Facade orchestrator (backend-agnostic)
    settings.py                       MarsaTrainingSettings + singleton
    base_spoof_detector.py            BaseSpoofDetector (ABC) - the contract
    schemas/                          One Pydantic model per file
        pipeline_config.py            DetectorTrainingConfig
        protocol_entry.py             ProtocolEntry
        dataset_split.py              DatasetSplit
        leakage_check_result.py       LeakageCheckResult
        leakage_audit_report.py       LeakageAuditReport
        epoch_result.py               EpochResult
        evaluation_result.py          EvaluationResult
        training_result.py            TrainingResult
    steps/                            One step class per file
        step_01_audit_leakage.py      CorpusLeakageAuditor
        step_02_build_datasets.py     ProtocolDatasetBuilder
        step_03_build_model.py        DetectorFactory
        step_04_train.py              DetectorTrainer
        step_05_evaluate.py           DetectorEvaluator
    utils/
        protocol_io.py                Protocol, metadata, filter, score I/O
        metrics.py                    DET curve, EER, per-attack EER
        scoring.py                    Inference loop shared by steps 4 and 5
        batching.py                   Padding collate
        audio_dataset.py              MarsaAudioDataset
        training_checkpoint_manager.py  TrainingCheckpointManager
        run_environment.py            Seeding, device, disk, logging
    dfarena/                          Backend subpackage
        dfarena_detector.py           DFArenaDetector
        settings.py                   DFArenaBackendSettings + singleton
```

Future backends sit beside `dfarena/`: `lcnn/`, `nes2net/`, `holiantispoof/`.

---

## 3. Design patterns

| Pattern | Where | Why |
|---------|-------|-----|
| Facade | `pipeline_facade.py` | One entry point; owns lifecycle, logging, run directory |
| Strategy | `steps/step_*.py` | Each step is an interchangeable unit with an `execute()` method |
| Factory | `DetectorFactory` | Backends registered by key; adding one touches nothing else |
| Adapter | `<backend>/*_detector.py` | Wraps a foreign model behind `BaseSpoofDetector` |
| Singleton | `settings.py` | Module-level Pydantic settings, shared and per-backend |

---

## 4. Deviations from the canonical guide, and why

**A backend subpackage per detector.** The guide names `schemas/`, `steps/` and
`utils/`. Model definitions belong in none of them: they are not workflow and
not helpers. Keeping each backend in its own subpackage is also what lets a
second detector be added without editing any step but the factory.

**The `__init__.py` files re-export nothing.** The guide asks for re-exports.
Here they would be actively harmful: an eager re-export in `utils/__init__.py`
makes importing `protocol_io` pull in `soundfile` and `torch`, and one in
`steps/__init__.py` makes importing the corpus auditor pull in `transformers`.
The audit is designed to run in about two minutes on a machine with none of
that installed. This was caught by an import smoke test after the first draft
shipped with eager re-exports. Import from the defining module.

Everything else follows the guide: one class per file, Pydantic throughout, no
dataclasses, all imports at module top.

---

## 5. Decisions worth knowing

**The audit gates the GPU.** Two silent data-loss bugs already destroyed two
augmentation runs on this corpus, and every tier was regenerated after the
August 2026 leak audits were run by hand and never committed. Step 1 re-asserts
those invariants from text files in about two minutes and writes a report that
can be cited. A fatal failure raises before a model is built.

**Labels: bonafide is class 1.** The countermeasure score is the bonafide
log-likelihood ratio, so higher means more genuine. This matches the ASVspoof
convention and lets the official evaluation package consume the score files.

**Epoch order is a seeded permutation, not a shuffling sampler.** The order for
epoch `n` is `default_rng([seed, n])`, so an interrupted epoch resumes by
slicing the already-consumed prefix off the same permutation. A shuffling
sampler would have made mid-epoch resume either approximate or expensive.

**Padding is masked everywhere.** The collate pads to the longest clip in the
batch and carries true lengths; the detector masks both its input
standardisation and its temporal pooling. Unmasked pooling would let clip
duration leak into the representation, and duration correlates with class in
any corpus assembled from mixed sources.

**Short clips are tiled, not zero-padded, when cropping.** A trailing block of
digital silence is itself a cue a detector can learn.

**Class imbalance is handled in the loss.** The corpus deliberately preserves
its natural spoof-heavy ratio, so rebalancing belongs at training time.
Inverse-frequency weights are on by default.

**Checkpoints carry RNG state.** Python, NumPy, Torch and CUDA generator states
are saved and restored, so a resumed run follows the trajectory it would have
followed uninterrupted. Writes are atomic: a temporary file is renamed into
place, so an interrupted write cannot leave a truncated checkpoint.

**Retention is deliberately shallow.** A one-billion-parameter model with AdamW
state costs roughly 16 GB per checkpoint. The server partition runs near full,
so the default keeps two rolling checkpoints plus the best one, and the run
refuses to start below a free-disk floor.

**One GPU, enforced.** The pipeline raises if more than one CUDA device is
visible, which makes the shared-server rule a property of the code rather than
a habit.

---

## 6. Adding a backend

1. Create `app/pipeline/training/<name>/` with `__init__.py`, a
   `<name>_detector.py` holding a `BaseSpoofDetector` subclass that implements
   `forward(waveform, lengths) -> logits` and `parameter_groups`, and a
   `settings.py` for its own parameters.
2. Register it in `DetectorFactory._registry` under its key, importing it at
   module top.
3. Run with `--backend <name>`.

Nothing else changes. Steps 1, 2, 4 and 5 never learn which model they drive.

---

## 7. Verification status

The auditor was exercised against synthetic corpora before first use: a clean
fixture passes all eight checks, and a fixture with a planted cross-split
speaker, a planted orphan FLAC and a planted class-by-class identifier ordering
fails exactly those three checks, names the offenders, and exits non-zero.

---

## 8. Open items

**Eval-only mode is not implemented.** Scoring a pretrained checkpoint without
training it — needed for the zero-shot baselines — requires a path that skips
step 4. Small addition to the facade.

**DF-Arena head.** `DFArenaDetector` loads the backbone with `AutoModel` and
adds its own pooled classifier. If the published checkpoint ships a native
detection head, reusing it would start from a far better initialisation.
Confirm against the model's `config.json` (`architectures`, `hidden_size`,
`num_labels`) before the first full-length run, and add a second backend entry
rather than mutating this one if both variants are worth comparing.
