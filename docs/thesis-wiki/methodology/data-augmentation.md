# Data Augmentation Pipeline

**Status:** Audited 2026-06-08 — remediation pending (Tier A mandatory before any trustworthy training run)
**Code:** `app/augmenter/` (base + 3 augmenters), `app/config/augmentation_config.py`, `app/schema.py`, `app/utils/utils.py`, `app/utils/augmentation_calculator.py`, `app/scripts/augmentation_pipeline.py`
**Output:** `data/augmented/augmented_{factor}_balanced_{ratio}/LA/` in ASVspoof2019 LA format (flac + cm protocol)

This page is the compiled source of truth for the channel/device data-augmentation layer that is applied to the ASVspoof2019-LA-format corpus (both bonafide and spoof) to train robust detectors. It synthesizes three inputs: (1) the as-built code, (2) a student investigation (`spoofing_DA.pdf`, `feedback_da_antispoofing.pdf`), and (3) the engineering audit of 2026-06-08. It describes intent vs. reality vs. feedback per augmentation, the two cross-cutting shortcuts, the correctness bugs, the locked decisions, and the remediation roadmap.

---

## 1. Design philosophy: robustness prior, not realism prior

Two competing objectives sit behind every augmentation choice:

- **Realism prior (student's framing):** an augmentation is good if it faithfully mimics a real channel/device. Most of the student's "Problema" notes descend from this.
- **Robustness prior (this project's framing):** augmentation exists to regularize the detector and expand the support of the training distribution so it generalizes to unseen attacks and channels. Physical plausibility is optional; aggressive, even non-physical perturbations are valuable.

The robustness prior is the better-supported one for anti-spoofing generalization, and the student's own citations confirm it: RawBoost (Tak et al. 2022) and SpecAugment (Park et al. 2019) are deliberately non-physical and work precisely because of it. The student flagged the FIR filter's "arbitrary spectral dips" as a problem while proposing SpecAugment masking as the alternative — the same mechanism (force spectral invariance), so the realism critique is internally inconsistent.

**Key reframe from the audit:** the student's empirical observations (NL barely changes the spectrum, clipping looks like the original, white noise present, AGC does nothing) are all *correct symptoms* — but of a **weak/incorrect implementation**, not of insufficient realism. The cure is correctness + aggression (real RawBoost, real codecs, harder distortion), which serves the robustness prior. The single place the realism prior adds genuine value is **targeted telephony coverage**, which is folded into the "all threats" codec decision below.

**Hard constraint the robustness prior must still respect — label preservation:** augmentation must not be so destructive that it erases the spoofing artifact the detector keys on (which would turn spoof into something indistinguishable from bonafide and inject label noise). Unrealistic is fine; artifact-destroying is not.

---

## 2. Architecture (as built → pending redesign)

- `BaseAugmenter` (ABC) with `augment()`, plus helpers `_normalize_audio` (RMS to -20 dB), `_ensure_sample_rate`, `_clip_audio`.
- Three concrete augmenters: `RIRAugmenter`, `CodecAugmenter`, `RawBoostAugmenter`.
- `AugmentationConfigManager` (singleton) holds strategies `3x/5x/10x`, all with type distribution **RIR_NOISE 60% / CODEC 30% / RAWBOOST 10%** (pre-2026-06-09 design — see Section 2a for the locked new design).
- `AugmentationPipeline` orchestrator: balanced-mode factor computation (`AugmentationModeCalculator`), speaker-disjoint train/dev/eval, **dev/eval are 100% clean**, originals always preserved, ASVspoof LA protocol emitted with per-clip SYSTEM_ID labels.

Augmentation type is selected per clip by a label-blind draw (same distribution for both classes).

### 2a. Locked redesign (2026-06-09) — replaces the above when implemented

**Offline type distribution (new):** `RIR_NOISE 60% / CODEC 40% / RAWBOOST 0%`

RawBoost moves entirely to **training-time on-the-fly augmentation** (applied per batch in the training loop, never pre-baked to disk). It is a fast CPU operation with no GPU or disk I/O requirement; pre-computing it adds corpus size with no benefit.

**Stacking gate — applied per augmented clip:**

```
roll p ~ U(0,1)
if p < 0.40:   # stacked (40%)
    apply RIR_NOISE → then CODEC on top
    SYSTEM_ID = "RIR_<params>|CODEC_<params>"
else:           # single (60%)
    pick type from {RIR_NOISE, CODEC} with relative weights {60, 40}
    apply that type alone
    SYSTEM_ID = "RIR_<params>"  or  "CODEC_<params>"
```

Output structure is **unchanged** — all FLAC files in one flat folder. The `|` character in SYSTEM_ID is the only encoding of stacked vs. single; no separate directory is needed. Trainers read the protocol file, not the directory tree, so single vs. stacked can be filtered at any time by checking for `|` in SYSTEM_ID.

---

## 3. The three augmentation families: intent vs. reality vs. feedback

### 3.1 RIR + Noise (60% of augmented clips)

- **Intent:** room reverberation (convolve with a Room Impulse Response) + additive background noise at a controlled SNR.
- **Reality:**
  - RIRs from `data/noise_dataset/RIR/simulated_rirs/{smallroom,mediumroom,largeroom}` — confirmed **openSLR RIRS_NOISES (SLR28), Ko et al. ICASSP 2017**. Only the room-size class is exposed; T60/Ds are baked in, not per-RIR metadata. `t60_range` in the config is **dead (never read)**.
  - Noise from MUSAN `noise/`, `speech/`, `music/` at sources distribution **NOISE 50% / SPEECH 30% / MUSIC 20%**.
  - SNR distribution: **low (0-5 dB) 10% / mid (5-30 dB) 80% / high (30-35 dB) 10%** — i.e. 80% of mass in a very wide mostly-clean band, 10% nearly clean (30-35 dB, near-inaudible noise).
  - **Correctness bug:** `convolve_with_rir` uses `fftconvolve(..., mode='same')` (`utils.py:138`), which centers the convolution — smearing the impulse response symmetrically and time-shifting by ~half the RIR length. Reverb should be causal: use `mode='full'` and trim/align to the direct-path peak.
- **Student feedback:** asked for the RIR source (answered: Ko et al. 2017); proposed renaming small/medium/large to ASVspoof `a/b/c` (room size, T60, Ds); asked whether MUSAN intelligible speech is wanted; proposed SNR sampling uniform over {20,15,10,5,0} dB.
- **Decisions/verdict:** keep `small/medium/large` (canonical, citeable), cite Ko et al. 2017 / SLR28, state the T60/Ds metadata limitation; **reject** the `a/b/c` rename (can't populate the triplet; `a/b/c` is ASVspoof2019 *Physical Access*, would misattribute provenance). Fix the `mode='same'` reverb bug. Re-weight SNR toward the harder 0-20 dB range (the 30-35 dB bucket is near-useless). MUSAN intelligible speech is a threat-model decision (see all-threats below).

### 3.2 Codec (30% of augmented clips)

- **Intent:** telephone/VoIP codec degradation.
- **Reality — this is NOT a codec.** `codec_types = ["g711","amr","opus"]` is **never referenced anywhere**. The actual operations are: downsample to 8 kHz and back (`_apply_telephone_codec`), a fixed 300-3400 Hz Butterworth bandpass (p=0.7), packet loss by **zeroing** 20 ms packets (p=0.5), and **linear** PCM quantization at 8/12 bits (p=0.3). No G.711 mu-law/A-law companding, no AMR, no Opus, no real PLC. 12-bit linear quantization is essentially inaudible.
  - **Correctness bug:** the packet-loss rate written to metadata is drawn separately from the rate actually applied (`codec_augmenter.py:172-174` vs `utils.py:219`), so the `LOSS{pct}` label is fabricated.
  - Final RMS-normalization to -20 dB (see shortcut #2).
- **Student feedback (largely correct):** implement real codecs and companding (A-law/mu-law 50/50), variable packet loss (up to ~20% for iLBC), PLC notions, complex quantization (2-8 bits, parametric), sample-rate-dependent bandpass; provided a VoIP codec reference table (G.711, G.722, Opus, AMR, iLBC, ...).
- **Decision:** rebuild with **proven libraries** — `torchaudio` / ffmpeg real codecs — covering **all threats**: narrowband telephony (G.711 mu-law/A-law, AMR-NB, iLBC, 8 kHz) AND broadband (Opus full-band, AAC).

### 3.3 RawBoost (10% of augmented clips)

- **Intent:** RawBoost (Tak et al. 2022) — LnL convolutive noise + ISD + SSI.
- **Reality — this is NOT RawBoost.** None of the three real components are present:
  - Real LnL = a bank of multiband notch FIRs + a Hammerstein polynomial nonlinearity. Code is a single random FIR of length **5-25** with sum-normalized Gaussian taps (`rawboost_augmenter.py:51`).
  - Real ISD = impulsive, sampled at P points with the -log `D_R{-1,1}` distribution. **Absent.**
  - Real SSI = white noise colored by a random FIR at a chosen SNR. **Absent.**
  - What's actually there: that random FIR + `tanh(alpha*x)` with **alpha in [0.1, 0.5]** (so weak it is nearly linear -> explains the student's "enriquecimiento poco evidente") + Gaussian noise at level 0.001-0.01 (-40 to -60 dB, inaudible) with both a signal-dependent and a white term + a **dead gain op** (see shortcut/bug below) + clipping at threshold 0.9 with p=0.2.
- **Student feedback (his strongest contribution, and correct):** the "alternatives" he proposes — Hammerstein/multiband, impulsive signal-dependent, white-through-FIR — **are literally the three real RawBoost components.** Also: raise the `tanh` alpha for real saturation; clipping barely engages; AGC/gain does nothing.
- **Decision:** replace with the **official RawBoost reference implementation** (LnL + ISD + SSI). Do not increase RawBoost's 10% weight until it actually works.

---

## 4. Two cross-cutting shortcuts (CRITICAL — invalidate training until fixed)

A "shortcut" is a feature that correlates with the label but is not the thing the detector should learn; a lazy network exploits it instead of learning real spoofing artifacts. Two were manufactured by the pipeline itself.

### Shortcut #1 — clean fraction is coupled to the class label

`calculate_balanced_mode` hits the target bonafide:spoof ratio by giving the two classes **different augmentation factors** (e.g. bonafide 8x, spoof 2x when spoof >> bonafide). Since each original is saved once as clean plus `factor-1` augmented copies, the clean fraction per class is `1/factor`:

- bonafide 8x -> 12.5% clean
- spoof 2x -> 50% clean

So "is this clip augmented?" leaks the label (among clean clips, ~80% are spoof; among augmented clips, ~64% are bonafide). The detector can score well by detecting reverb/noise/codec presence — never learning synthesis artifacts. At eval (100% clean) the learned rule misfires on everything. Note: integer `ceil()` rounding also means "balanced_5050" never actually lands on 50/50.

**Important:** a *minimum augmentation factor for the minority* does NOT fix this — the leak is on the **clean-fraction** lever, not the factor lever. Raising the bonafide factor lowers its clean fraction and *widens* the gap.

**FIX — Option B (chosen 2026-06-08):** augment both classes with the **same factor** (identical clean fraction, type mix, loudness -> zero coupling), keep the natural bonafide:spoof imbalance in the corpus, and correct it **in the trainer** via class-weighted loss or a balanced/weighted sampler. EER is a threshold-sweep metric and fairly robust to the class prior, so this is safe; class weighting is cheap insurance against the minority being under-learned. (Option C — a self-contained balanced corpus with equal clean fraction via minority oversampling — was the fallback for trainers that cannot reweight; not chosen.)

### Shortcut #2 — loudness normalization differs per augmentation type

- Clean originals: saved at **natural loudness** (no normalization).
- RawBoost and Codec: RMS-normalized to **-20 dB**.
- RIR: only peak-clipped, not RMS-normalized.

Loudness therefore correlates with augmentation type and (via #1) with class — a second learnable leak, directly analogous to the loudness shortcut addressed in the partial-spoof work.

**FIX:** apply ONE uniform loudness policy to **all four paths including clean**, so loudness carries zero information about augmentation or class.

---

## 5. Correctness bugs (independent of design philosophy)

- **RIR `mode='same'`** smears/time-shifts the reverb (should be `mode='full'` aligned to the direct-path peak).
- **Packet-loss metadata double-draw** — logged loss rate != applied loss rate.
- **`_apply_random_gain` is mathematically dead** — a constant scalar gain (0.7-1.3x) followed by RMS-normalization cancels exactly. Zero effect on output. (This is why the student measured ~0.043 dB within-audio gain variability.)
- **`t60_range` config field is never read.**
- For the record, NOT a bug: `mix_audio_with_snr` computes the SNR scaling correctly (RMS ratio, /20).

## 6. What is done right (keep)

- Augmentation **type** selection is class-symmetric (same distribution for both classes) — no type->class leak.
- **Speaker-disjoint** train/dev/eval splits.
- **dev/eval are 100% clean** (correct evaluation hygiene).
- Originals always preserved.

## 7. Architecture / project-rule debt (Tier D)

- `app/schema.py` is wall-to-wall `@dataclass`, which **violates the project rule (Pydantic only, no @dataclass)**; it also packs ~9 classes into one file and runs **`mp.set_start_method('spawn', force=True)` at import time** (global side effect for every importer), plus imports `transformers`/`torch`/`datasets` into a schema module.
- The config cites *"Sanchez et al. (2024)"* — verify this reference actually resolves before it reaches the paper.

---

## 8. Locked decisions

| # | Decision | Date | Rationale |
|---|----------|------|-----------|
| 1 | **Option B** — uniform augmentation factor for both classes; class balancing in the trainer (weighted loss / sampler) | 2026-06-08 | Kills shortcut #1 at the source; keeps augmentation label-blind; preserves full spoof diversity |
| 2 | Uniform loudness policy across all paths incl. clean | 2026-06-08 | Kills shortcut #2 |
| 3 | **Proven libraries**, no hand-rolled DSP — official RawBoost + torchaudio/ffmpeg codecs | 2026-06-08 | Hand-rolled imitations are exactly why the pipeline is broken |
| 4 | Codec coverage = **all threats** (narrowband G.711/AMR/iLBC + broadband Opus/AAC) | 2026-06-08 | Maximum channel coverage; aligns with robustness prior |
| 5 | Detector front-end **deferred** — corpus stays **waveform-level**, no polarity inversion or SpecAugment baked in | 2026-06-08 | Those are model/training-time choices |
| 6 | RIR: keep small/medium/large, cite Ko et al. 2017 / SLR28, state T60/Ds metadata limitation; reject a/b/c rename | 2026-06-06 | Canonical provenance; can't populate the triplet; a/b/c is ASVspoof PA |
| 7 | **RawBoost removed from offline corpus** — moved to training-time on-the-fly augmentation | 2026-06-09 | Fast CPU op; no pre-baking benefit; frees 10% offline weight |
| 8 | **CODEC raised 30% → 40%** (absorbs freed RawBoost slot) — offline distribution: RIR_NOISE 60% / CODEC 40% | 2026-06-09 | Stronger channel coverage in offline corpus |
| 9 | **Stacking gate**: 60% single augmentation / 40% RIR_NOISE+CODEC stacked; SYSTEM_ID uses `\|` separator; flat folder preserved | 2026-06-09 | Models real-world reverb+telephony degradation; no directory restructuring needed |

## 9. Remediation roadmap

- **Tier A (MANDATORY, first — before any trustworthy run):** implement Option B (uniform factor + trainer-side class weighting) and the uniform loudness policy. Fold in the quick correctness fixes (RIR `mode='full'`, packet-loss metadata, delete the dead gain op). Outcome: an honest baseline corpus with zero augmentation->class leakage.
- **Tier B (the real rebuild):** real RawBoost (LnL+ISD+SSI, official code) and real codecs (torchaudio/ffmpeg, all-threats set). Resolves the mislabeled-SYSTEM_ID contamination automatically.
- **Tier C:** correctness fixes (folded into B).
- **Tier D (hygiene, non-blocking):** migrate `schema.py` off `@dataclass` to Pydantic + one-class-per-file, remove the import-time `mp.set_start_method`, verify citations.

Library installs for Tier B are server-side (ml-server03); the venv that runs this pipeline must be confirmed (candidate: repo-root `venv`) and ffmpeg must be present for the torchaudio codec backend.

## 10. Open questions

- Which trainer consumes this corpus, and does it support class weights / a weighted sampler? (Confirmed direction is Option B, which assumes yes.)
- MUSAN intelligible-speech (`speech/` at 30%): keep as babble/multi-speaker channel augmentation, or exclude to avoid second-speaker label confusion? (Threat-model dependent; "all threats" leans toward keeping it.)
- Detector front-end (deferred) — will later decide whether polarity inversion / SpecAugment are added at training time.

## See also

- [[decision-log]] entries 2026-06-06 (RIR sourcing) and 2026-06-08 (augmentation audit + Option B)
- [[dataset-design]], [[quality-metrics]], [[partial-spoof-approach]] (loudness-shortcut precedent)
