# Acoustic Augmentation Methodology

**Status:** Active
**Last updated:** 2026-06-01
**Source:** Deep research 2026-06-01 (109 agents, 25 claims verified); codec table investigation; augmentation pipeline code review

---

## Overview

The augmentation pipeline transforms the raw HABLA-Spoof corpus (bonafide + spoof attacks)
into a balanced, channel-degraded training set in ASVspoof2019 LA format. Its purpose is to
close the domain gap between studio-quality corpus audio and the VoIP, telephony, and
reverberant conditions a deployed anti-spoofing detector actually encounters.

Empirical support: Cohen et al. (2022) demonstrated that codec and channel augmentation
reduced baseline EER by 50% and min t-DCF by 16% on ASVspoof 2021 LA compared to
clean-only training. RawBoost (Tak et al., ICASSP 2022) provides convergent evidence with a
44% relative EER reduction using signal-level proxies for the same class of degradation.

Related pages:
- [Dataset Design](dataset-design.md) — HABLA v2 speaker distribution, bonafide counts
- [Detection Methods](../state-of-art/detection-methods.md) — AASIST, SSL-based detectors
- [Decision Log](../decisions/decision-log.md) — augmentation design decisions

---

## Probability Tree

```
For each original file in {train_bonafide, train_spoof}:
    emit 1 copy  system_id="-"             (clean original, always included)
    for k in range(factor - 1):
        type = sample_one(Layer 2)
        apply Layer 3a / 3b / 3c accordingly
        emit with system_id = descriptive label

For each file in {val, eval}:
    emit 1 copy  system_id="-"             (100% clean, no augmentation)
```

---

## LAYER 1 — Copies-per-file

### Method: Balanced mode

```
total_min  = (N_bonafide + N_spoof) × min_factor
B_target   = total_min × target_ratio
S_target   = total_min × (1 − target_ratio)
B_factor   = max(1, ceil(B_target / N_bonafide))
S_factor   = max(1, ceil(S_target / N_spoof))
```

### Split policy

| Split | Factor | Augmentation |
|-------|--------|--------------|
| train | B_factor / S_factor | applied |
| val (dev) | 1 | none — 100% clean |
| eval (test) | 1 | none — 100% clean |

### Rationale for 25%-clean preservation

All originals are always included as clean copies. With min_factor=3x and 50/50 target,
approximately 33% of the train set is clean, 67% augmented. This prevents the detector from
losing sensitivity to the original voice manifold.

---

## LAYER 2 — Augmentation type

For each augmented copy (not the original), sample exactly one type:

```
RIR_NOISE   60%
CODEC       30%
RAWBOOST    10%
```

Types do not compose at this level. One copy = one type.

### Rationale for 60/30/10

- RIR_NOISE majority: room acoustics are ubiquitous; most voice recordings have some degree
  of reverberation and ambient noise. MUSAN+RIR is the canonical ASVspoof augmentation recipe.
- CODEC 30%: a substantial share of real LATAM voice traffic passes through a lossy codec
  (VoLTE, WhatsApp/Opus, PSTN). Justified by GSMA Mobile Economy LATAM 2024 trend data.
- RAWBOOST 10%: signal-level proxies for device and microphone distortion. Tak et al. (2022).

---

## LAYER 3a — RIR_NOISE

### Sub-distribution

Three INDEPENDENT samples per file:

**Room size** (source: standard RIR augmentation practice, ASVspoof recipe)
```
smallroom    30%
mediumroom   50%
largeroom    20%
```

**Noise source** (MUSAN — Ko et al. 2017)
```
noise    50%
speech   30%
music    20%
```

**SNR** (mixture of three uniform bands)
```
low band   ( 0–5  dB)   10%   uniform within band
mid band   ( 5–30 dB)   80%   uniform within band
high band  (30–35 dB)   10%   uniform within band
```

### Processing chain
```
audio → convolve(random RIR from room_size) → mix(random MUSAN noise at SNR) → clip[-1,1]
```

### Output label format
```
RIR_<ROOM>_<SRC3>_SNR<INT>
e.g. RIR_SMALL_NOI_SNR15
     RIR_MEDIUM_SPE_SNR8
```

---

## LAYER 3b — CODEC (VoIP/Telephony)

This layer was completely redesigned based on the deep research pass (2026-06-01).
The previous implementation used independent Bernoulli gates (downsample + bandpass + packet
loss + quantization) with no codec identity. The new design uses a real ffmpeg codec
round-trip with deployment-realistic codec selection.

### Academic precedent

Chen et al. (2021, UR Channel-Robust System, ASVspoof 2021) provide the peer-reviewed
taxonomy this design follows:
- **Landline**: mu-law, A-law, ADPCM (G.726)
- **Cellular**: GSM-FR, AMR-NB, AMR-WB
- **VoIP**: G.722, G.729, SILK, SILK-WB, Opus

ASVspoof 2021 LA used 7 channel conditions (C1 clean, C2 A-law 8 kHz, C3 PSTN+mu-law,
C4 G.722 16 kHz, C5 mu-law 8 kHz, C6 GSM-FR 8 kHz, C7 Opus 16 kHz) — establishing these
families as the canonical anti-spoofing channel benchmark set.

### Step (i) — Sample LOSS TIER

```
clean      (0%     packet loss)   35%
light      (0.5–3% packet loss)   35%
moderate   (3–8%   packet loss)   20%
heavy      (8–15%  packet loss)    8%
extreme    (15–20% packet loss)    2%
```

**Confidence:** ENGINEERING PRIOR — no primary source publishes empirical LATAM voice loss
distributions. Tiers informed by:
- iLBC design tolerance (15–20%) — from the codec reference table
- General VoIP literature on residential loss rates (<3% normal, >5% degraded)

**IMPORTANT — burst vs. Bernoulli:** Packet loss in real networks is bursty, not i.i.d.
(Bolot 1993, SIGCOMM; Sun & Ifeachor, IET Communications 2005; corroborated through
2022 LTE/WiFi traces). The loss simulator MUST use a Gilbert-Elliott two-state Markov
model, not uniform random sampling. At low loss rates, burst patterns cause less
perceptual damage than Bernoulli; at high rates, more damage. Using Bernoulli produces
audible transient artifacts at regular intervals that detectors learn as a spurious feature
rather than a real codec artifact.

Gilbert-Elliott parameters per tier (to be calibrated from Bolot 1993 / Hasslinger 2008):
```
State G (good):  loss_prob ≈ 0.01,  p(G→B) ≈ 0.05
State B (bad):   loss_prob ≈ 0.50,  p(B→G) ≈ 0.10
(per-tier scaling applied to transition probabilities)
```

### Step (ii) — Sample CODEC (tier-filtered)

**Base weights** (sum = 100, engineering prior based on LATAM deployment trends):

| Codec | Weight | Eligible tiers | Basis |
|-------|--------|---------------|-------|
| opus_voip | 38% | all (in-band FEC) | OTT apps dominate LATAM consumer voice (WhatsApp, Telegram, Zoom, Meet, Discord). GSMA LATAM 2024 confirms smartphone OTT usage dominance. |
| amrwb | 25% | all (FEC) | VoLTE standard codec. GSMA IR.92 mandate. Mandatory on all 4G/5G voice-capable UEs. |
| amrnb | 8% | all (FEC) | 2G/3G mobile. GSMA LATAM 2024: 11 operators completed 2G shutdown, 10 more by 2030 — low but nonzero. |
| g711_ulaw | 10% | clean/light/moderate | PSTN/landline. NA-influenced countries. No primary source for exact LATAM share. |
| g711_alaw | 8% | clean/light/moderate | PSTN/landline. EU-influenced countries. |
| g722 | 5% | all (adaptive mute) | HD voice on some VoLTE / enterprise SIP. |
| g729a | 3% | clean/light/moderate | Legacy enterprise VoIP / PBX. |
| aac_ld | 2% | clean/light | Skype, broadcast contribution. Rare. |
| g726 | 1% | clean/light/moderate | DECT cordless handsets. Tail coverage. |
| g723.1 | <1% | clean/light/moderate | H.323 legacy video conferencing. Tail. |
| ilbc | <1% | all (frame-independent PLC) | Legacy Google Voice/Talk. Frame-independent PLC tolerates 15–20% loss. |

**NOTE:** All weights are ENGINEERING PRIORS. No peer-reviewed or regulatory source
publishes per-codec voice-traffic shares for LATAM. Thesis must disclose this explicitly
and accompany with a sensitivity analysis (±25% weight perturbation, report EER delta).

**Geographic note on G.711 variant:** mu-law (NA-influenced: MX, CO, VE) vs A-law
(EU-influenced: AR, CL, BR, ES). No LATAM regulator primary source confirmed the
per-country split. Both companding laws are included without per-country assignment.
Disclosed as methodology limitation.

**Tier-eligibility filter** — weights renormalize over eligible codecs per tier:

```
clean / light     → all codecs eligible (use base weights)
moderate          → drop {aac_ld}; renormalize
heavy             → keep {opus, amrwb, amrnb, g722, ilbc}; renormalize
extreme           → keep {opus, ilbc}; renormalize → ~opus 97%, ilbc 3%
```

Rationale: G.711 and G.729 have no PLC and degrade severely at high loss rates;
pairing them with heavy/extreme loss is an unrealistic combination. Codecs with FEC
(OPUS, AMR) and frame-independent coding (iLBC) are explicitly designed for high loss.

### Step (iii) — Sample MODE/BITRATE within codec

**AMR-NB — 8 modes, two-tier weighting:**

Citable: GSMA IR.92 mandates {12.2, 7.4, 5.9, 4.75 kbps} for IMS/CS-interworking.
Non-IMS modes ({5.15, 6.7, 7.95, 10.2 kbps}) cannot be routed transcoder-free through
legacy CS GSM-FR but exist in standards and may appear in direct SIP / fringe deployments.

```
12.2 kbps    35%   (GSMA IR.92 mandatory, highest quality)
 7.4 kbps    25%   (GSMA IR.92 mandatory)
 5.9 kbps    20%   (GSMA IR.92 mandatory)
 4.75 kbps   10%   (GSMA IR.92 mandatory, lowest quality)
10.2 kbps     4%   (fringe)
 7.95 kbps    3%   (fringe)
 6.7 kbps     2%   (fringe)
 5.15 kbps    1%   (fringe, rarest)
```

Source for mandatory 4-mode set: GSMA IR.92 v12.0 (consistent across v9–v16).

**AMR-WB — 9 modes, two-tier weighting:**

Citable: GSMA IR.36 v4.0 mandates 5 modes for CS HD-voice: {6.6, 8.85, 12.65, 15.85,
23.85 kbps}. Modes {18.25, 19.85, 23.05} and SID/DTX exist in the AMR-WB standard (3GPP
TS 26.190) but are not part of the mandatory CS/VoLTE deployment profile.

```
12.65 kbps   35%   (GSMA IR.36 WB-Set 0 operational baseline)
 8.85 kbps   20%   (GSMA IR.36 WB-Set 0)
15.85 kbps   15%   (GSMA IR.36 CS 5-mode set)
 6.6 kbps    10%   (GSMA IR.36 WB-Set 0 minimum)
23.85 kbps    5%   (GSMA IR.36 CS 5-mode set, highest quality)
18.25 kbps    6%   (fringe)
23.05 kbps    5%   (fringe)
19.85 kbps    3%   (fringe)
SID/DTX       1%   (comfort noise / discontinuous transmission)
```

Source: GSMA IR.92 v12.0, GSMA IR.36 v4.0, 3GPP TS 26.103 (Config-WB-Code 0).

**OPUS — 4 bitrates:**

Citable: IETF RFC 7587 Section 3.1.1 defines speech sweet-spots at 8–12 kbps NB,
16–20 kbps WB, 28–40 kbps FB. The 12/16/24/32 kbps grid covers NB top through FB bottom.
48k/64k are the mono-music regime (RFC 7587) — not appropriate for a speech-only corpus.

```
16 kbps   35%   (WB sweet-spot lower bound — most WhatsApp/Telegram calls)
24 kbps   30%   (WB-to-FB transition — Zoom, Teams typical)
12 kbps   20%   (NB upper bound — low-bandwidth conditions)
32 kbps   15%   (FB lower bound — high-quality VoIP)
```

**G.722** — bitrate ∈ {48, 56, 64} kbps, weights {15, 25, 60}
**G.726** — bitrate ∈ {16, 24, 32, 40} kbps, weights {10, 20, 40, 30}
**G.729a** — fixed 8 kbps (single mode per standard)
**G.711 mu/A-law** — fixed 64 kbps (single mode per standard)
**G.723.1** — bitrate ∈ {5.3, 6.3} kbps, weights {30, 70}
**iLBC** — frame_ms ∈ {20, 30}, weights {60, 40}
**AAC-LD** — fixed 64 kbps

### Step (iv) — Apply codec round-trip

```
audio (any SR)
  → resample to codec.native_sr     (8 / 16 / 48 kHz per codec)
  → ENCODE via ffmpeg subprocess    (real codec artifacts)
  → PACKET-LOSS via Gilbert-Elliott (burst-aware, frame-size aware per codec)
  → PLC reconstruction              (strategy per codec, see below)
  → DECODE back to PCM
  → resample to 16 kHz              (dataset standardization)
```

**PLC strategy per codec:**

| Strategy | Codecs | Description |
|----------|--------|-------------|
| inband_fec | opus | libopus -fec 1; decoder reconstructs from redundant copy inside next packet. Losses up to ~15% often inaudible. |
| fec | amrwb, amrnb | Last good frame's LPC+pitch+excitation reused. Short losses (20 ms) often imperceptible. |
| adaptive_mute | g722 | Low band extrapolated, high band muted; audible as "blurry" HF dropout. |
| pitch_cycle | g729a, g723.1 | Last pitch period repeated and gain-faded. "Vocal hold" artifact. |
| repetition | g711_ulaw, g711_alaw, g726 | Previous frame PCM copied. Stutter artifact at loss boundaries. |
| frame_independent | ilbc | Each frame coded independently; loss is local, no cascade. Zero-fill acceptable. |
| application | aac_ld | No standard PLC defined. Zero-fill. |

### Output label format

```
CODEC_<NAME>_<MODE>_BR<KBPS>K_LOSS<PCT>PCT_PLC_<STRATEGY>
e.g.
  CODEC_OPUSVOIP_BR16K_LOSS5PCT_PLC_FEC
  CODEC_G711U_BR64K_LOSS2PCT_PLC_REP
  CODEC_AMRWB_M2_BR12.65K_LOSS0PCT_PLC_FEC
  CODEC_AMRNB_M7_BR12.2K_LOSS3PCT_PLC_FEC
  CODEC_AACLD_BR64K_LOSS0PCT
```

---

## LAYER 3c — RAWBOOST

Based on Tak et al. (2022), ICASSP. Five INDEPENDENT Bernoulli gates — composable,
multiple can fire on the same file.

```
Linear filtering              P = 0.50
  Random FIR, length ∈ U[5,25] samples, coeffs ~ N(0,1) normalized by sum(|coeff|)

Nonlinear distortion          P = 0.30
  output = tanh(α × audio), α ∈ U[0.1, 0.5]; renormalize to 0.99 peak

Additive noise                P = 0.60
  noise_level ∈ U[0.001, 0.01]
  signal-dependent: audio × N(0,1) × noise_level
  independent:               N(0,1) × noise_level × 0.5

Gain variation                P = 0.40
  output = audio × U[0.7, 1.3]

Clipping                      P = 0.20
  hard clip to [-0.9, 0.9]

Always applied:
  normalize to -20 dBFS RMS
  clip to [-0.99, 0.99]
```

Expected effects per file = 2.0. P(zero effects) ≈ 6.7% → labeled RAWBOOST_NONE.

### Output label format

```
RAWBOOST_<OP1>_<OP2>_...
e.g.
  RAWBOOST_LF_AN_GV
  RAWBOOST_NL_AN
  RAWBOOST_NONE
```

---

## Cross-layer output conventions

| Property | Value |
|----------|-------|
| Bit depth | 16-bit PCM |
| Sample rate | 16 kHz (forced on write) |
| Container | FLAC |
| Filename | LA_T_<7d>.flac / LA_D_* / LA_E_* |
| Protocol line | `<speaker_id> <audio_id> <system_id> <bonafide\|spoof>` |

---

## What is citable vs. what is an engineering prior

### Citable (primary sources)

| Claim | Source |
|-------|--------|
| AMR-NB mandatory 4-mode set {12.2, 7.4, 5.9, 4.75 kbps} | GSMA IR.92 v12.0 |
| AMR-WB mandatory 5-mode set {6.6, 8.85, 12.65, 15.85, 23.85 kbps} | GSMA IR.92, IR.36 v4.0 |
| OPUS speech sweet-spots 8–12 / 16–20 / 28–40 kbps | IETF RFC 7587 §3.1.1 |
| Codec families (landline/cellular/VoIP) for anti-spoofing augmentation | Chen et al. 2021 (arxiv:2107.12018) |
| ASVspoof 2021 LA channel conditions (C1–C7) as academic precedent | ASVspoof 2021 Eval Plan + TASLP paper (arxiv:2109.00535, 2210.02437) |
| Codec augmentation reduces EER 50%, min t-DCF 16% | Cohen et al. 2022 (Speech Communication, DOI:10.1016/j.specom.2022.04.005) |
| Packet loss is bursty; Gilbert-Elliott model required | Bolot 1993 (SIGCOMM); Sun & Ifeachor 2005 (IET Comm.) |
| 2G sunset LATAM (11 done, 10 by 2030) | GSMA Mobile Economy LATAM 2024 |

### Engineering priors (must be disclosed in thesis)

| Claim | Status |
|-------|--------|
| Codec deployment weights (opus 38%, amrwb 25%, …) | No primary source for LATAM per-codec traffic share |
| Loss-tier percentages (35/35/20/8/2) | No primary source; derived from VoIP literature directionally |
| OPUS per-bitrate weights (35/30/20/15) | No per-app RTP capture study available |
| G.711 mu-law/A-law per-country assignment | No regulator-level source found |
| AMR-NB/WB fringe-mode weights (10% / 15% tail) | Coverage-over-precision engineering choice |

### Thesis defense posture

Primary defense: "AMR-NB/WB mode sets follow GSMA IR.92/IR.36 mandatory deployment
profiles. OPUS bitrates follow IETF RFC 7587 speech sweet-spots. Codec families follow
the Chen et al. (2021) landline/cellular/VoIP taxonomy established for ASVspoof 2021."

Secondary defense: "Per-codec sampling weights are engineering priors reflecting GSMA
LATAM 2024 deployment trends (declining 2G/3G; OTT-app dominance). A sensitivity analysis
over ±25% weight perturbations demonstrates detector robustness to these prior assumptions."

---

## Open questions / future work

1. **Gilbert-Elliott parameters** — calibrate `(p_G→B, p_B→G, loss_in_B)` per tier from
   Bolot 1993 and Hasslinger & Hohlfeld 2008. No LATAM-specific mobile measurement found.

2. **Sensitivity analysis** — retrain detector under ±25% codec weight perturbations and
   report EER delta. If EER varies <2% absolute, weights are declared robust.

3. **ffmpeg codec availability** — G.723.1, G.729, AMR-NB, AMR-WB, iLBC require
   non-default ffmpeg build flags. Pending inventory from ml-server03 (see Gate 1 commands
   in the implementation plan).

4. **ASVspoof 5 (2024) / newer benchmarks** — check if any 2024–2026 benchmark publishes
   per-codec sample-level weights, which would supersede ASVspoof 2021 as the academic
   weight-distribution anchor.
