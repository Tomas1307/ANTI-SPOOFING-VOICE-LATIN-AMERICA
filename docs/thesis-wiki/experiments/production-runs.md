# Production Runs

**Status:** Active
**Last updated:** 2026-07-09
**Source:** ml-server03 logs, pipeline output metadata, on-server file counts (2026-07-09)

---

## Summary Table — Full-Spoof Production (MARSA corpus)

Counts AUTHORITATIVE per `LA/` FLAC + metrics CSV `status==passed` (verified 2026-07-14).
Supersedes the 2026-07-09 raw `find` counts. Metrics are means over passed samples.

| Pipeline | Status | Passed | Pass Rate | WER | NISQA | SIM | RTF |
|----------|--------|--------|-----------|-----|-------|-----|-----|
| FishGram | DONE | 34,197 | 95.2% | 2.17% | 4.57 | 0.602 | 2-3x |
| Qwen3-TTS | DONE | 31,568 | 87.9% | 1.46% | 4.37 | 0.720 | 3-5x |
| OpenVoice | DONE (watermark-free re-run 2026-07-14) | 29,796 | 83.1% | 1.50% | 4.38 | 0.388 | 0.07-0.10x |
| Chatterbox | DONE | 31,701 | ~88.2% | 1.20% | 4.39 | 0.714 | 31-45x |
| OuteTTS | DONE | 25,642 | ~71.4% | 2.36% | 4.45 | 0.462 | ~5.6x |
| OmniVoice | DONE | 33,743 | ~93.9% | 1.06% | 4.28 | 0.686 | ~0.025x |
| CosyVoice | DROPPED | - | - | - | - | - | - |
| **Total full spoof** | | **186,647** | | | | | |

**CORRECTION 2026-07-14:** OmniVoice metrics were previously logged from the 6-sample validation
run (NISQA 4.59 / SIM 0.696 / WER 1.85%); the full production means are NISQA **4.28** / SIM 0.686 /
WER 1.06% -- OmniVoice is the LOWEST NISQA of the suite, not the highest. FishGram (4.57) is highest.
Chatterbox SIM 0.714 is second only to Qwen (0.720). OpenVoice re-run clean: 29,796 (was 29,626),
metrics unchanged within noise -> confirms watermark removal is quality-neutral.

**Partial-spoof production (AUTHORITATIVE, `corpus_samples.csv`, 2026-07-14):** **18,421 spliced
samples** (28,720 spoofed words). The earlier "86,766" was a raw file miscount (counted intermediates:
`cloned/` + `spliced/` + `references/` + `LA/flac/` ~= 4.7x the sample count). Breakdown -- by attack:
omnivoice 10,990 / chatterbox 3,423 / qwen 1,565 / fishgram 1,317 / outetts 1,073 / **openvoice 53**
(ECAPA 0.60 gate nearly excludes it, SIM 0.388). By tier: W1 10,276 / W2 5,991 / W3 2,154.
By partition: jittered 9,965 / not_jittered 8,456. By quality_flag: high 4,875 / medium 13,072 / low 474.

**Recommended clean partial-spoof subset (created 2026-07-14):** applying the full-spoof intelligibility
gate (WER<=0.15 & CER<=0.10) to the 18,421 keeps **15,641** (84.9%); losses ~9-18% per system so
composition is preserved. Copied to `data/partial_spoof_clean/` (audio + `corpus_samples_clean.csv` +
`corpus_spoofed_words_clean.csv`); the full 18,421 in `data/partial_spoof_output/` is left intact.
Paper reports 18,421 total with 15,641 as the recommended clean partition (deposit all, train on clean).

**Corpus grand total (pre-augmentation):** 35,927 bonafide + 186,647 full spoof
+ 18,421 partial spoof = **240,995 utterances**.
Augmentation `augmented_*_balanced_5050/` folders are the OLD pre-Option-B approach and must be
re-run uniform (Tesis 2 OE1) before deposit.

All six TTS pipelines are COMPLETE. The 2026-05-06 wiki state that listed Chatterbox/OuteTTS
as RUNNING and OmniVoice as validation-only is superseded.

**Hardware:** ml-server03, NVIDIA A40 (46GB VRAM), CUDA 12.6
**Bonafide corpus:** HABLA v2, 1,567 speakers, 7 accents (cross-continental: Spain + LatAm), ~35,927 samples

## Per-Pipeline Notes

### FishGram (Fish Speech / OpenAudio-S1)
- Best pass rate (95.2%). Highest NISQA (4.57).
- Moderate speaker similarity (0.602) — voice quality is excellent but timbre transfer is imperfect.
- Runs as HTTP API server on a separate port. RTF 2-3x.
- Completed on GPU 1.

### Qwen3-TTS
- Highest speaker similarity (0.720). Lowest WER (1.46%).
- x_vector_only_mode=True required (ref_text mismatch with concatenated reference).
- All sampling params needed: do_sample=True, temperature, top_k, top_p, repetition_penalty.
- Completed on GPU 3.

### OpenVoice V2
- Fastest pipeline (RTF 0.07-0.10x, 10-14x real-time).
- Lowest speaker similarity (0.394) — MeloTTS base voice bleeds through ToneColorConverter.
- Lowest CER (0.45%). Very consistent (lowest std on NISQA).
- Completed on GPU 1.

### Chatterbox (Resemble.ai)
- Started April 12 on GPU 2. Currently at ~14,818/35,927 (41%).
- RTF 31-45x — dramatically slower than all other pipelines.
- GPT-style autoregressive with CFG. EnCodec tokens + Vocos vocoder.
- ETA: ~May 13 (18 more days).

### OuteTTS
- Started April 13 on GPU 2. Currently at ~23,561/35,927 (66%).
- RTF ~5.6x (consistent). Llama 3.1-based, WavTokenizer codes.
- Known issue: PyLoudNorm clipping warnings.
- ETA: ~May 1 (6 more days).

### CosyVoice 3.0
- Dropped. Generates Chinese output for Spanish input text. No Spanish support despite multilingual claims.

### OmniVoice (k2-fsa) -- VALIDATED 2026-05-06 (after reference-cut fix)
- Standalone TTS pipeline at `app/pipeline/omnivoice_attack/`. Audio ID range 15M-15.99M (avoids collision with partial_spoof main 12M-14M).
- Initial validation appeared to pass 6/6 but listening surfaced reference-voice bleed in 2 samples that Parakeet did not transcribe.
- Root cause identified 2026-05-06: `concatenate_with_padding` sliced the last bonafide file mid-word to hit a 10 s target, leaving the reference ending abruptly. OmniVoice's diffusion conditioning attempted to "complete" the cut-off pattern at the start of generation, manifesting as reference-voice bleed.
- Fix: stop at last fitting file, snap to silence in edge case, always append 200 ms trailing silence. References are now 3-10 s ending in silence.
- Post-fix validation: **6/6 passed first attempt**, zero non-verbal-prefix rejections, zero retries needed. Metrics improved across the board.
- Final post-fix metrics: avg WER 1.85%, avg CER 0.83%, avg NISQA MOS 4.59, avg ECAPA SIM 0.696.
- ECAPA SIM 0.696 is still below the 0.70 informational floor -- a stable property of OmniVoice's diffusion conditioning, not a bug. OmniVoice remains the weakest cloner of the suite by speaker similarity (Qwen 0.720, FishGram 0.602, OpenVoice 0.394). Useful contrast for the paper.
- Cleared for production run AND boundary jitter pilot.

---

## Operational Runbook (continue from another PC / ml-server03)

This section is the executable handover. Read top to bottom; run blocks in order on ml-server03.

### Pre-requisites

Before running anything, verify these are in place on ml-server03:

| Item | Path | Check |
|---|---|---|
| Repo cloned | `~/ANTI-SPOOFING-VOICE-LATIN-AMERICA/` | `git pull` from `feat/attacks` |
| Bonafide v2 dataset | `data/bonafide_dataset_by_speaker_v2/` | `ls | wc -l` returns ~1567 |
| Mozilla CV transcripts | `data/cv-corpus-24.0-2025-12-05/es/validated.tsv` | exists |
| MUSAN noise (RIR augmentation, not jitter) | `data/noise_dataset/musan/` | exists |
| qwen_env | `envs/qwen_env/` | `source .../bin/activate` works |
| omnivoice_env | `envs/omnivoice_env/` | exists (built 2026-05-06, validated) |
| GPU availability | `nvidia-smi` | confirm one A40 free (avoid 0 and 2 if shared) |

### Job 1: OmniVoice standalone validation -- COMPLETED 2026-05-06

**Outcome:** PASSED. 6/6 samples on GPU 1. Avg WER 3.94%, CER 1.81%, NISQA 4.53, ECAPA SIM 0.680.

**Venv build recipe (proven on ml-server03 2026-05-06):**

The straightforward `pip install -r envs/omnivoice_requirements.txt` does NOT yield a working environment because NeMo and modern transformers fight over `huggingface_hub` versions. The proven order is:

```bash
cd ~/ANTI-SPOOFING-VOICE-LATIN-AMERICA
git pull
python3 -m venv envs/omnivoice_env
source envs/omnivoice_env/bin/activate

pip install --upgrade pip

# 1. Torch first, matched to driver 560.35.03 / CUDA 12.6
pip install torch==2.8.0 torchaudio==2.8.0 --extra-index-url https://download.pytorch.org/whl/cu126

# 2. NeMo with all transitive deps (also pulls setuptools<81 to keep pkg_resources working)
pip install "nemo_toolkit[asr]>=2.7.0" "setuptools<81"

# 3. OmniVoice without deps (avoids version fight with NeMo's transformers pin)
pip install --no-deps omnivoice

# 4. Force transformers to 5.3.0 (NeMo runs fine on 5.3+)
pip install "transformers>=5.3.0" --force-reinstall --no-deps

# 5. CRITICAL: align huggingface_hub with transformers 5.3.0, otherwise
#    `is_offline_mode` ImportError cascades through lightning/torchmetrics
pip install "huggingface_hub==1.5.0" --no-deps

# 6. Pipeline deps not pulled by NeMo
pip install "speechbrain>=1.0.0" "torchmetrics>=1.0.0" "jiwer>=3.0.0" "pydantic>=2.0.0"

# Smoke check (all four must pass)
python -c "from huggingface_hub import is_offline_mode; print('hf_hub OK')"
python -c "import nemo.collections.asr; print('NeMo ASR OK')"
python -c "from omnivoice import OmniVoice; print('OmniVoice OK')"
python -c "from speechbrain.inference.speaker import SpeakerRecognition; print('SpeechBrain OK')"
deactivate
```

After the env is healthy, regenerate the requirements lock:
```bash
source envs/omnivoice_env/bin/activate
pip freeze > envs/omnivoice_requirements.txt
deactivate
```

**Run validation mode (3 speakers, 6 samples):**
```bash
source envs/omnivoice_env/bin/activate
export CUDA_VISIBLE_DEVICES=1   # check nvidia-smi first

python -u -c "
from app.pipeline.omnivoice_attack import OmniVoiceAttackPipeline, settings
settings.VALIDATION_MODE = True
settings.SAMPLES_PER_SPEAKER = 2
settings.MATCH_BONAFIDE_COUNT = False

pipe = OmniVoiceAttackPipeline()
print('Output:', pipe.run())
" 2>&1 | tee logs/omnivoice_validation_$(date +%Y_%m_%d).log
deactivate
```

**Validation results (2026-05-06):**
- 6/6 samples passed validation (100% pass rate, no rejections)
- Avg WER 3.94%, Avg CER 1.81% (well below the 15%/10% rejection ceilings)
- Avg NISQA MOS 4.53 (best of suite -- exceeds FishGram's 4.57 in single run; full production needed for fair comparison)
- Avg ECAPA SIM 0.680 (below 0.70 informational floor; weakest cloner of suite)
- 0 prefix trims, all 6 in train split (artifact of the validation speaker selection)

**Next step:** Promote to production mode (`VALIDATION_MODE=False`, `MATCH_BONAFIDE_COUNT=True`).

### Job 2: Qwen partial_spoof main run with file partition

**Why second:** Before producing the jitter dataset, we need the **main** dataset to use the same `BONAFIDE_FILE_PARTITION="main"` policy so the partition is enforced on both sides. The original Qwen partial_spoof run (35,927 samples) used the full bonafide pool — overlap with the future jitter dataset is possible. Either:

- (A) **Re-run main with `partition="main"`**: produces a clean half-pool main dataset, but invalidates the existing 35,927 sample dataset. Costly.
- (B) **Keep existing main dataset, only run jitter with `partition="jitter"`**: jitter dataset overlaps with ~half of main's sentences. Acceptable for production but breaks the "frases distintas" guarantee.

**Recommended:** Option B for now (cheaper), with the caveat documented. If a fully clean ablation is needed later, re-run main.

### Job 3: Qwen partial_spoof JITTER pilot

```bash
cd ~/ANTI-SPOOFING-VOICE-LATIN-AMERICA
source envs/qwen_env/bin/activate
export CUDA_VISIBLE_DEVICES=3   # check nvidia-smi first

python -c "
from app.pipeline.partial_spoof import PartialSpoofPipeline
from app.pipeline.partial_spoof.schemas.pipeline_config import PartialSpoofPipelineConfig

config = PartialSpoofPipelineConfig(
    attack_system='qwen',
    enable_boundary_jitter_override=True,
    bonafide_file_partition_override='jitter',
)
PartialSpoofPipeline(config=config).run()
"
deactivate
```

**Output:**
- `data/qwen_partial_spoof_jitter/jittered/*.wav` (jittered audio)
- `data/qwen_partial_spoof_jitter/boundary_jitter_metadata.json` (per-utterance jitter plans)
- `data/qwen_partial_spoof_jitter/LA/` (final ASVspoof2019 LA structure with system_id `QWEN3TTS_PSW{1,2,3}J`)

**Success criteria:**
- Mean per-utterance duration drift |dt| < 100 ms (otherwise jitter is too aggressive)
- WER on jittered audio <= main_qwen WER + 0.03 absolute (otherwise jitter degrades intelligibility too much)
- Operation counts roughly balanced across truncate/overlap/bleed (~33% each among the manipulated boundaries)

**If success:** Replicate Job 3 for the 4 remaining validated attacks: chatterbox, openvoice, outetts, fishgram.
**If failure (drift too large or WER spike):** Tune magnitude ranges in `settings.py` (e.g., reduce `JITTER_OVERLAP_RANGE_MS` to (30, 60)). Re-run.

### Job 4: Detector EER comparison (deferred)

Once jitter datasets exist for at least 2 attacks, train AASIST/RawNet3 on:
1. Main partial_spoof only -> measure EER on main eval and jitter eval.
2. Jitter only -> measure EER on main eval and jitter eval.
3. Combined main + jitter -> measure EER on each eval set.

The diagonal (train_main, eval_main) is the baseline. Off-diagonal degradation reveals whether the detector learned the boundary-anomaly shortcut.

This is deferred until the dataset construction settles. Capture the experimental design in `experiments/ablation-studies.md` once underway.

---

## Pending Work Checklist

Use this as the bookmark when picking up later. Items struck through are done.

- [ ] OmniVoice venv setup on ml-server03
- [ ] OmniVoice validation run (3 speakers, 6 samples)
- [ ] OmniVoice listening test sign-off by Master Tomas
- [ ] OmniVoice production run (decide after listening test)
- [ ] Qwen partial_spoof JITTER pilot run on ml-server03
- [ ] Audit jitter run: drift, WER, op-count distribution
- [ ] Replicate jitter to chatterbox, openvoice, outetts, fishgram (after Qwen succeeds)
- [ ] Decide ablation-vs-disjoint-sentences strategy for jitter (Option A re-run vs Option B accept overlap)
- [ ] Magnitude ablation: sweep `JITTER_PROBABILITY` in {0.3, 0.5, 0.7, 1.0}
- [ ] Detector EER comparison on main vs jitter datasets
- [ ] Update this page with results after each milestone

---

## MARSA Production Sweep Runbook (2026-05-20)

Twelve jobs total: six attacks (OmniVoice 40%, Qwen 20%, FishGram 10%, OpenVoice 10%, Chatterbox 10%, OuteTTS 10%) crossed with two partitions (`not_jittered`, `jittered`). Driven by the dispatch manifest at `data/manifests/partial_spoof_plan.csv` so every job processes a disjoint slice of bonafide files.

### Step 0 -- Generate the manifest (one-time, any GPU venv)

```bash
export CUDA_VISIBLE_DEVICES=1
source ~/ANTI-SPOOFING-VOICE-LATIN-AMERICA/envs/fishgram_env/bin/activate
cd ~/ANTI-SPOOFING-VOICE-LATIN-AMERICA
python -m app.scripts.generate_partial_spoof_manifest
deactivate
```

Outputs:
- `data/manifests/partial_spoof_plan.csv` -- the dispatch table (~35,927 rows)
- `data/manifests/partial_spoof_plan_summary.json` -- target vs actual attack marginals, speaker coverage, tier potential
- `data/manifests/bonafide_transcripts_full.json` -- cached Parakeet output reused by every per-attack run

Audit before launching: target weights should match `settings.ATTACK_WEIGHTS` exactly; actual marginal should be within ~1% of target by construction; every speaker must appear in the manifest (or the speaker had too few words to clear `MIN_WORDS_W1=4`).

### Step 1 -- Run the 12 per-attack jobs

Two launch paths: serial (one job per terminal, full manual control) or **parallel on one GPU** (recommended, ~3-4x wall-clock reduction).

**Parallel on one GPU.** Each pipeline consumes ~6-8 GB VRAM. On a 46 GB A40 the safe ceiling is 4 concurrent (~32 GB) with the 5th slot tight; 5 fits if no surprise allocations. The launcher dispatches into a free slot whenever any child finishes (reap-and-relaunch) so Chatterbox / OuteTTS occupy their slots for days while the fast attacks rotate through the others.

```bash
export CUDA_VISIBLE_DEVICES=1
source ~/ANTI-SPOOFING-VOICE-LATIN-AMERICA/envs/fishgram_env/bin/activate
cd ~/ANTI-SPOOFING-VOICE-LATIN-AMERICA
python -m app.runner.partial_spoof_orchestrator \
    --mode parallel --gpu 1 --max-concurrent 4 --order slow_first
deactivate
```

- `--max-concurrent 4` is the default (safe). Bump to 5 only after watching `nvidia-smi` during a stable steady-state segment.
- `--order slow_first` front-loads Chatterbox + OuteTTS so they don't get stuck waiting behind faster jobs.
- Resume after crash / kill: re-run the same command. Cells with a present `samples.csv` are skipped; use `--no-skip-complete` to force redispatch.
- Per-child logs land in `logs/parallel_<attack>_<partition>.log`. Tail a specific job with `tail -f logs/parallel_chatterbox_jittered.log`.
- SIGINT in the launcher propagates SIGTERM to all children. Send SIGINT twice to force-kill.

**Serial (one job at a time).** Useful for debugging a single attack or when GPU contention with another researcher needs strict capping. The orchestrator's `runbook` mode prints the 12 individual commands:

```bash
python -m app.runner.partial_spoof_orchestrator --mode runbook --gpu 1
```

A representative single-job command (substitute attack + venv + partition):

```bash
export CUDA_VISIBLE_DEVICES=1
source ~/ANTI-SPOOFING-VOICE-LATIN-AMERICA/envs/omnivoice_env/bin/activate
cd ~/ANTI-SPOOFING-VOICE-LATIN-AMERICA
python -m app.runner.partial_spoof_orchestrator \
    --mode single --attack omnivoice --partition not_jittered
deactivate
```

Each job (serial or parallel) writes to `data/partial_spoof_output/<attack>/<partition>/` and is restartable via the per-cell checkpoint at `.checkpoint.json`. Progress across all cells:

```bash
python -m app.runner.partial_spoof_orchestrator --mode status
```

### Step 2 -- Aggregate corpus tables

After all 12 cells have produced `samples.csv` and `spoofed_words.csv`:

```bash
source ~/ANTI-SPOOFING-VOICE-LATIN-AMERICA/envs/fishgram_env/bin/activate
python -m app.runner.partial_spoof_orchestrator --mode aggregate
deactivate
```

Outputs in `data/partial_spoof_output/`:
- `corpus_samples.csv` -- master per-sample table (~82k rows expected)
- `corpus_spoofed_words.csv` -- master per-spoofed-word table (~200k rows expected, frame-level label source)
- `corpus_summary.json` -- final marginals: per-attack totals, per-partition totals, per-quality-flag counts, target vs actual weights

### Operational notes

- GPU pinning: prefer GPUs 1 and 3 (per `CLAUDE.md`, GPUs 0 and 2 are shared with other researchers). Run Chatterbox + OuteTTS in parallel on different GPUs since their RTF is highest.
- Wall-clock estimate: dominated by Chatterbox (~1.5-2.5 days per partition) and OuteTTS (~10-15 hours per partition). OmniVoice and the others combined should complete in under 24h.
- Keep-bad-stuff: `ENABLE_STEP_6_REJECTION = False`. Low-quality samples land in the corpus labeled `quality_flag='low'`; the upstream clone gate (ECAPA SIM >= 0.60) still drops obvious non-attacks but nothing past Step 5 gets filtered for WER/NISQA.

## Related Pages

- [Partial Spoof Approach](../methodology/partial-spoof-approach.md) -- algorithmic details and justifications for jitter
- [Attack Systems](../methodology/attack-systems.md) -- per-pipeline configuration and venv paths
- [Decision Log](../decisions/decision-log.md) -- chronological design decisions
- [TTS Systems](../state-of-art/tts-systems.md) -- TTS evaluation including OmniVoice section
