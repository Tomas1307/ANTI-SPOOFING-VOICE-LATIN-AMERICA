# Production Runs

**Status:** Active
**Last updated:** 2026-05-06
**Source:** ml-server03 logs, pipeline output metadata

---

## Summary Table

| Pipeline | Status | Samples | Passed | Pass Rate | WER | NISQA | SIM | RTF |
|----------|--------|---------|--------|-----------|-----|-------|-----|-----|
| FishGram | DONE | 35,927 | 34,197 | 95.2% | 2.17% | 4.57 | 0.602 | 2-3x |
| Qwen3-TTS | DONE | 35,927 | 31,568 | 87.9% | 1.46% | 4.37 | 0.720 | 3-5x |
| OpenVoice | DONE | 35,544 | 29,626 | 83.4% | 1.50% | 4.41 | 0.394 | 0.07-0.10x |
| Chatterbox | RUNNING | ~14,818 | TBD | TBD | TBD | TBD | TBD | 31-45x |
| OuteTTS | RUNNING | ~23,561 | TBD | TBD | TBD | TBD | TBD | ~5.6x |
| OmniVoice | VALIDATED | 6 | 6 | 100.0% | 3.94% | 4.53 | 0.680 | TBD |
| CosyVoice | DROPPED | - | - | - | - | - | - | - |

**Hardware:** ml-server03, NVIDIA A40 (46GB VRAM), CUDA 12.6
**Bonafide corpus:** HABLA v2, 1,567 speakers, 7 Latin American accents, ~35,927 samples

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

### OmniVoice (k2-fsa) -- VALIDATED 2026-05-06
- Standalone TTS pipeline written. Code at `app/pipeline/omnivoice_attack/`.
- Audio ID range 15M-15.99M (avoids collision with partial_spoof main 12M-14M).
- Status: validation PASSED 2026-05-06 on GPU 1, ml-server03. 6/6 samples (3 speakers, 2 each).
- Metrics: avg WER 3.94%, avg CER 1.81%, avg NISQA MOS 4.53, avg ECAPA SIM 0.680.
- All 3 validation speakers (arf_00295, arf_00610, arf_01523) live in train split per the canonical HABLA partition, so train=6 / dev=0 / eval=0 in this run. Production mode will hit all splits.
- ECAPA SIM 0.680 sits below the informational floor of 0.70 -- OmniVoice is the **weakest cloner of the suite** by speaker similarity. Content quality (NISQA 4.53) is the highest of any attack. Useful contrast for the paper.
- Cleared for boundary jitter pilot (no longer blocked).

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

## Related Pages

- [Partial Spoof Approach](../methodology/partial-spoof-approach.md) -- algorithmic details and justifications for jitter
- [Attack Systems](../methodology/attack-systems.md) -- per-pipeline configuration and venv paths
- [Decision Log](../decisions/decision-log.md) -- chronological design decisions
- [TTS Systems](../state-of-art/tts-systems.md) -- TTS evaluation including OmniVoice section
