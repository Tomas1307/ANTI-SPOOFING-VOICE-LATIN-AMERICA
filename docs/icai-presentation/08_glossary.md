# Glossary — Terms an ICAI Audience May Not Know Cold

Grouped by theme. Written for putting directly on a "definitions" slide or
speaker notes, not as an exhaustive textbook entry.

## Attack types

- **TTS (text-to-speech):** generates spoken audio from written text. No
  real recording of the target speaker is required as input at inference
  time (though the model was trained on real speech).
- **VC (voice conversion):** takes a real recording of one speaker (the
  source) and alters it to sound like a different target speaker, while
  keeping the linguistic content.
- **Logical Access (LA):** spoofing attack injected directly into the
  system's audio input pipeline (as opposed to being played through a
  microphone).
- **Physical Access (PA):** attack replayed through a loudspeaker and
  recaptured by a microphone.
- **Deepfake (DF):** speech generated to convincingly imitate a person,
  usually evaluated outside the strict ASV pipeline context.

## Front-ends (feature extraction)

- **MFCC (Mel-Frequency Cepstral Coefficients):** Mel-scale filter bank +
  DCT. Standard in speech recognition; perceptually weighted toward how
  humans hear frequency.
- **LFCC (Linear-Frequency Cepstral Coefficients):** same pipeline as MFCC
  but with a linearly spaced filter bank — no perceptual weighting.
- **CQCC (Constant-Q Cepstral Coefficients):** uses the Constant-Q
  Transform, which gives geometrically spaced (better low-frequency)
  resolution; historically strong for spotting synthesis artifacts.
- **Spectrogram:** the raw time-frequency magnitude representation, no
  cepstral compression at all.
- **Filter-bank scale (ascendant/constant/descendant):** how much gain the
  filter bank applies at high versus low frequencies.

## Back-ends (classification)

- **LCNN (Light CNN):** convolutional network using Max-Feature-Map
  activations, which suppress less-informative channels; a strong,
  lightweight standard anti-spoofing back-end.
- **MFM (Max-Feature-Map):** an activation that takes the element-wise
  maximum across paired feature channels, acting as a feature selector.
- **Pooling strategy (Attention / trim-pad / LSTM-sum):** how a
  variable-length sequence of frame-level features is collapsed into one
  utterance-level decision.
- **RawNet2:** an end-to-end back-end that skips hand-designed front-ends
  entirely and learns directly from the raw waveform.

## Loss functions

- **AM-Softmax / OC-Softmax:** margin-based classification losses designed
  to push bona fide and spoofed embeddings further apart than plain
  softmax would.
- **P2S-Grad (Pair-to-Sample Gradient, MSE-based):** a loss formulated as a
  regression toward a target score, rather than a classification margin.
- **Sigmoid (cross-entropy):** the plain binary-classification baseline
  loss, used here as a lower bound for comparison.

## Evaluation

- **EER (Equal Error Rate):** the single operating threshold where the
  False Rejection Rate equals the False Acceptance Rate. Lower is better;
  0% is a perfect detector at that threshold.
- **min t-DCF:** minimum normalized tandem detection cost function; unlike
  EER, it weights errors by their real cost to a downstream speaker
  verification system, so it can disagree with EER's ranking.
- **Wilcoxon signed-rank test:** a non-parametric test for whether two sets
  of paired measurements (here, EER from the same seed across two configs)
  differ systematically, without assuming a normal distribution.
- **Cohen's d:** a standard effect-size measure — how large a difference
  is, independent of whether it's statistically significant.
- **Bonferroni correction:** a conservative adjustment to the significance
  threshold when many pairwise comparisons are run at once, to control the
  chance of a false positive appearing purely by multiple testing.
- **Seed / random seed:** the initialization value controlling a model's
  random weight initialization and training randomness; running the same
  configuration under different seeds measures how much of the result is
  architecture versus luck.

## Datasets

- **ASVspoof 2019 / 2021:** the standard English-language anti-spoofing
  benchmark series.
- **HABLA / HABLA Spoof:** the Spanish (Latin American) multi-accent
  anti-spoofing corpus this paper's Spanish-side experiments use.
- **MARSA:** the follow-up multi-accent Spanish corpus under construction
  (Tomás's own ongoing thesis work), named in this paper's Future Work
  section as the response to its attack-coverage limitation.
