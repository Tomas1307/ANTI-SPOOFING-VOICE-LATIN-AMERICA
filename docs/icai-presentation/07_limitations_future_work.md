# Limitations and Future Work — and the Link to MARSA

The paper is unusually candid about its own limitations. This is worth
presenting as a strength (rigor), not hidden. It also gives you a clean
bridge from this ICAI talk into your own ongoing thesis work (MARSA),
which the paper's own Future Work section names explicitly.

## Limitation 1: attack coverage

Section on cross-language evaluation shows conclusions depend strongly on
which attack families a corpus contains, and HABLA covers only four attack
types. The paper's stated response is that the authors are building
**MARSA**, a multi-accent Spanish corpus that adds:
- Full-spoof attacks from **six contemporary TTS systems** (not just the
  four HABLA covers).
- A **partial-spoof tier** where individual words are spliced into
  otherwise bona fide utterances.
- An **acoustic augmentation layer** covering reverberation, noise, and
  codec degradation.

The paper explicitly notes that evaluating the partial-spoof tier requires
**frame-level scoring**, which utterance-level EER (used throughout this
paper) cannot express. This is a direct, citable statement in Jaime's paper
that motivates the metric choices in your own MARSA Data Descriptor.

**Presentation tip:** this is the natural closing slide if you want to tease
your own thesis work as "what comes next after this ICAI paper" — the
paper itself says so, you don't need to invent the connection.

## Limitation 2: training/evaluation protocol

Two protocol issues the paper flags for future work:
- **Language-balanced retraining** would separate the effect of bilingual
  data from the effect of corpus composition (the 57.7%/42.3% imbalance
  flagged throughout the results).
- **A protocol that holds out target identities** used by conversion
  attacks would remove the residual optimism identified in the partition
  audit (see `03_methodology.md` — target identities are shared between
  train and test today).
- The gender asymmetry found in the accent/gender analysis suggests
  **demographic conditioning** is a more promising adaptation target than
  regional/accent conditioning.

## Beyond these two lines

The paper also names, as clearly lower-priority future directions:
- Transformer-based back-ends.
- Self-supervised front-end representations: **wav2vec 2.0** and
  **HuBERT**.
- Extending evaluation to deepfake detection using the **ASVspoof 2021**
  dataset.

## Honest framing for Q&A

If asked "why isn't this a bigger dataset / why only HABLA," the honest
answer already exists in the paper: attack-family coverage was a known,
acknowledged limitation from the start, and it's precisely what MARSA is
being built to close. If asked "why not just retrain on a balanced corpus
now," the honest answer is that it wasn't done in this paper and is named
as explicit future work, not swept under the near-zero EER numbers.
