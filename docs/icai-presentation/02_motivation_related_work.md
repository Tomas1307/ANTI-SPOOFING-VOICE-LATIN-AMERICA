# Motivation and Related Work

## The problem (Introduction)

Speech-based biometric systems (automatic speaker verification, ASV) can be
attacked at the logical access point by two families of synthetic speech:

- **Text-to-speech (TTS):** produces spoken output from written text.
- **Voice conversion (VC):** modifies non-linguistic characteristics of a
  real recording to impersonate a target speaker's voice.

Both can lead to unauthorized access, financial fraud, or identity theft.
GANs and VAEs have lowered the barrier to generating perceptually convincing
synthetic speech, which is why anti-spoofing countermeasures (CMs) matter.

## The specific gap this paper targets

The ASVspoof challenge series has driven most anti-spoofing progress, but
every ASVspoof benchmark is English-only. Prior work (Tamayo et al.) already
showed that countermeasures trained on English degrade severely on Spanish
speech, with EER exceeding 30% even for strong baselines. That degradation
is attributed to accent-driven differences in vowel space, formant
structure, and speaking rate that affect cepstral front-end features. What
was missing before this paper: nobody had systematically measured **which**
front-end/back-end combinations degrade how much, or **why** (which attack
family drives the collapse), or whether the effect is really about accent
versus something else (like speaker gender).

## Prior datasets (Table `datasets-taxonomy`)

| Dataset | Language | Attack types |
|---|---|---|
| HABLA Spoof | Spanish | Logical Access, Deepfake |
| ASVspoof 2019 | English | Logical Access, Physical Access |
| ASVspoof 2021 | English | Logical Access, Physical Access, Deepfake |
| SAS | English | Logical Access |
| FoR | English | Logical Access |
| VCTK | English | Logical Access |

HABLA Spoof is the only Spanish entry in this table, and it is the corpus
this paper's Spanish-side experiments are built on.

## Prior architecture building blocks

**Front-ends** (DSP feature extractors that turn raw audio into a compact
feature sequence):
- **MFCC:** Mel-scale filter bank + Discrete Cosine Transform. Perceptually
  motivated, compact in low frequencies.
- **LFCC:** same idea but with a uniform linear-scale filter bank instead of
  Mel, weighting all frequencies equally.
- **CQCC:** Constant-Q Transform before cepstral analysis, giving
  geometrically spaced frequency resolution suited to spectral artifacts in
  synthetic speech.
- **Spectrogram:** the raw Short-Time Fourier Transform magnitude, the
  richest but highest-dimensional representation.

LFCC and CQCC were the official ASVspoof baseline front-ends.

**Back-ends** (classify the front-end's feature sequence as bona fide or
spoofed):
- **LCNN** (Light Convolutional Neural Network): uses Max-Feature-Map (MFM)
  activations to suppress irrelevant channels; the standard strong baseline.
- **LSTM-augmented LCNN:** adds long-range temporal modeling.
- **RawNet2:** end-to-end, operates directly on the raw waveform.
- **GMM:** classical, still competitive for physical-access scenarios.

## The stated research gap this paper closes

Tamayo et al. demonstrated the cross-lingual EER drop but did not
systematically evaluate front-end/back-end combinations or filter-bank
scales. This paper directly targets that gap, plus a second one: whether
regional accent (as opposed to some other speaker attribute) is what drives
residual Spanish-side error once bilingual training is in place.
