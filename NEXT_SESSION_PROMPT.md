# Next Session Context — Data Augmentation Pipeline (handoff 2026-06-08)

Paste this (or say "read NEXT_SESSION_PROMPT.md") to resume. Role/conventions live in CLAUDE.md
(Alfred style, two-machine workflow, Pydantic-not-@dataclass, wiki protocol). Branch: `feat/attacks`.

## What this session did

### 1. FishGram health-probe hardening (start of session)
- Root-caused the 12-job parallel run's lone failure (`fishgram/jittered`): NOT a dead Fish Speech
  server — the `GET /` health probe in `app/pipeline/fishgram_attack/utils/cloner.py` had a 5 s
  timeout, single attempt, required exactly 200, and lost a race against in-flight inference under
  shared-server load.
- Fix: retry with backoff, longer per-attempt timeout, accept ANY HTTP response as alive. New settings
  `FISH_SPEECH_HEALTH_RETRIES/TIMEOUT/BACKOFF` in `fishgram_attack/settings.py`.

### 2. Data-augmentation audit + rebuild (the main work)
Audited `app/augmenter/` (triggered by a student investigation `spoofing_DA.pdf` /
`feedback_da_antispoofing.pdf`). Found and fixed:

- **Two label-leak shortcuts (CRITICAL):**
  - Clean fraction coupled to class (balanced mode used different per-class factors → clean%=1/factor
    diverged). Fixed with `AugmentationModeCalculator.calculate_equal_clean_blocks()` (equal clean
    fraction both classes) + a `--mode {balanced,uniform}` switch in `augmentation_pipeline.py`.
  - Loudness coupled to augmentation type. Fixed with one `utils.normalize_loudness()` applied to
    EVERY clip in the orchestrator save path (`_save_audio_and_protocol`).
- **"RawBoost" was not RawBoost** → vendored faithful reference `app/augmenter/rawboost_reference.py`
  (LnL/ISD/SSI, Tak et al. 2022), wrapped in `rawboost_augmenter.py`.
- **"Codec" was not a codec** → real codecs via `app/augmenter/codec_backend.py`
  (`torchaudio.io.AudioEffector` + ffmpeg, availability probe + graceful skip), rewired
  `codec_augmenter.py`.
- **Correctness:** RIR `convolve_with_rir` now `mode='full'` aligned to the direct-path peak (was
  `'same'`, smeared/shifted); codec packet-loss single-draw (applied==logged); deleted the dead
  `_apply_random_gain` (RMS-normalize cancelled it).
- **New Pydantic config** `app/augmenter/schemas/codec_rawboost_config.py`
  (`RawBoostParams`, `RawBoostConfigV2`, `CodecConfigV2`, `CodecSpec`), wired into
  `AugmentationStrategy` in `app/schema.py` (old `@dataclass` RawBoostConfig/CodecConfig now orphaned).
- **Weighted selection** (added after initial uniform): `CodecConfigV2.codec_weights`
  (g711_ulaw 25 / g711_alaw 15 / amr_nb 20 / ilbc 5 / opus 25 / aac 10 → ~65% NB / ~35% BB);
  `RawBoostConfigV2.algo_weights` `{4:0.5, 5:0.3, 7:0.2}` (so SSI fires 50%).

### 3. Presentation
- 5 new slides `presentation/slides/15a..15e_*.html`: augmentation overview, RIR+Noise, Codec,
  RawBoost, leak-safe design. Each category slide shows subcategory **percentage breakdowns**
  (RIR room/noise/SNR; codec per-codec table; RawBoost algo mix + component firing).
- Demo generator `app/scripts/demo_augmentations.py` → isolated per-augmentation clips to
  `data/demo_augmentations/<label>/{original,rir,noise,babble,codec_g711,codec_opus,rawboost}.wav`.
- Rebuilt deck with `presentation/build.py` (canonical: header + sorted slides + footer, auto IDs +
  count). Renamed `14_next_steps.html` → `16_next_steps.html` so augmentation (s47–s51) precedes the
  finale (s52). 52 slides total.

### 4. Wiki
- New `docs/thesis-wiki/methodology/data-augmentation.md`; decision-log entries 2026-06-06 (RIR
  sourcing = openSLR RIRS_NOISES / Ko et al. 2017, kept small/med/large names, rejected ASVspoof a/b/c
  rename) and 2026-06-08 (augmentation audit + Option B). NOTE: written pre-verification — revise after
  the server run confirms behavior.

## PENDING / DO NEXT

1. **Verify on ml-server03** (nothing has been executed — local has no torch/ffmpeg):
   - 4 unit tests: `python -m app.tests.test_{rawboost,rir,codec}_augmenter`, `..._augmentation_batch`.
   - Codec probe: `python -c "from app.augmenter import codec_backend as cb; print(cb.probe_available_codecs())"`.
   - Dry run: `run_augmentation --mode balanced --target_ratio 0.50 --min_factor 3x --output data/augmented_dryrun`.
   - Acceptance: per-class clean% equal (leak #1), uniform ~-23 dBFS RMS (leak #2), real codec names in SYSTEM_IDs.
2. **Run `python -m app.scripts.demo_augmentations`** to populate the slide audio.
3. **Diff `rawboost_reference.py` vs upstream** TakHemlata/RawBoost-antispoofing before any production run.
4. **Tier D (deferred):** migrate `app/schema.py` off `@dataclass` → Pydantic; remove import-time
   `mp.set_start_method('spawn')`; DECOUPLE the augmentation import path from `transformers`/`datasets`
   (today importing `app.augmenter` drags them in via `schema.py` → augmentation can't run on a lean
   env); verify the "Sánchez 2024" citation in `augmentation_config.py`.
5. **git:** PR `feat/attacks` → dev; new branch `feat/augmentation-presentation` for slides + demo.
6. **Confirm which venv** built `data/augmented/` (needs torch+torchaudio+transformers+datasets+ffmpeg).

## OPEN DECISIONS / CONSTRAINTS
- Balancing = **Option B** (uniform augmentation, rebalance in trainer) is the eventual target, but the
  detector/trainer is **months away and unselected** — so for now produce a self-contained leak-free
  corpus. Revisit B vs C when the trainer exists.
- **Do NOT train on the existing `data/augmented/`** — it has both leaks baked in. Regenerate.
- Hard rule for augmentation aggression: must preserve the spoofing artifact (no artifact-destroying
  codec/clip stacks).

## KEY FILES
- `app/augmenter/{base_augmenter,rawboost_augmenter,codec_augmenter,rir_augmenter,rawboost_reference,codec_backend}.py`
- `app/augmenter/schemas/codec_rawboost_config.py`
- `app/utils/{utils,augmentation_calculator}.py`, `app/scripts/{augmentation_pipeline,run_augmentation,run_augmentation_batch,demo_augmentations}.py`
- `app/schema.py` (AugmentationStrategy wiring; Tier D target)
- `presentation/slides/15a..15e_*.html`, `presentation/build.py`, `presentation.html`
- Plan: `C:\Users\ASUS\.claude\plans\nooo-lets-fix-things-gleaming-cray.md`
- Wiki: `docs/thesis-wiki/methodology/data-augmentation.md`, `docs/thesis-wiki/decisions/decision-log.md`
