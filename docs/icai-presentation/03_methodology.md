# Methodology

## The 48-architecture grid

Every model = one front-end + one back-end + one loss function.

| Axis | Options (4 / 3 / 4) |
|---|---|
| Front-end | CQCC, MFCC, LFCC, Spectrogram |
| Back-end (pooling) | LCNN-Attention, LCNN-trim-pad, LCNN-LSTM-sum |
| Loss/activation | P2S-Grad (MSE-based), AM-Softmax, OC-Softmax, Sigmoid |

4 x 3 x 4 = **48 configurations**, each trained under three data conditions
(English-only, Spanish-only, combined bilingual) — so effectively up to 144
distinct training runs before multiplying by seeds.

## Front-end details

- MFCC and LFCC: frame-by-frame DFT -> log-magnitude -> DCT. MFCC uses a
  Mel-scale (log-spaced) filter bank; LFCC uses a linear-spaced one.
- Filter-bank **scale response** is varied in three modes for MFCC/LFCC:
  **ascendant** (gain rises toward high frequencies), **constant** (flat),
  **descendant** (gain falls toward high frequencies, i.e. low-frequency
  emphasis). This is the axis behind the filter-bank scale study
  (`05_accent_gender_filterbank.md`).
- CQCC: Constant-Q Transform before cepstral analysis, finer low-frequency
  resolution where spoofing artifacts often live.
- Spectrogram: full STFT magnitude fed directly to the LCNN as a 2-D image.

Shared framing across all four front-ends (so differences are attributable
to the filter bank/cepstral stage, not to framing): Hann window, 320
samples, 160-sample shift, 512-point FFT. MFCC/LFCC/CQCC keep 20 static
coefficients plus first- and second-order deltas; Spectrogram uses 60 static
coefficients with no deltas. All four present **60 dimensions per frame** to
the back-end. No voice activity detection, no cepstral mean normalization,
no feature-level normalization at any stage.

## Back-end (LCNN) architecture

Shared LCNN stem (5-block conv/MFM/BatchNorm/MaxPool stack, ending at a
54x38x32 feature map) followed by one of three pooling strategies:
- **Attention:** learns to weight frames by relevance.
- **Trim-pad:** forces fixed-length input.
- **LSTM-sum:** bidirectional LSTM over the sequence, then sum.

The pooled embedding feeds a classification head shaped by the loss
function: AM-Softmax, OC-Softmax, MSE-based P2S-Grad, or Sigmoid
cross-entropy.

## Datasets and partitions

| Training set | Train | Dev | Test |
|---|---|---|---|
| English (ASVspoof 2019 LA) | 25,380 | 24,844 | 71,237 |
| Spanish (HABLA Spoof) | 34,684 | 3,658 | 4,809 |
| Combined | 60,064 | 28,502 | — |

- **English:** standard ASVspoof2019 LA split, 19 TTS/VC systems in
  training, 48 in evaluation.
- **Spanish (HABLA Spoof):** 5 Latin-American accents (Argentina, Chile,
  Colombia, Peru, Venezuela).
- **Combined:** English + Spanish training sets concatenated (60,064
  utterances); evaluated separately per language's own test partition. Not
  language-balanced: Spanish is 57.7% of it, English 42.3%.

### A partition caveat the paper is explicit about — important for Q&A

The authors independently audited the HABLA partition file lists (not just
trusted the published description). Result: bonafide speakers and VC
*source* speakers are strictly disjoint across train/dev/test (29/16/14
speakers, zero overlap). But the VC **target** identities being imitated are
*not* disjoint — all 113 test-set target identities also appear in training.
This affects 95.5% of Spanish test utterances. No test speaker's genuine
voice leaks into training, but a conversion aimed at a familiar target
identity does. Consequence stated explicitly in the paper: **the near-zero
Spanish EER results are an upper bound on performance under a strictly
identity-disjoint protocol**, though comparisons *between* architectures
(which all share this exposure equally) remain internally valid. In the
opposite direction, CycleGAN conversions appear in training but not in the
Spanish test split, making part of that evaluation an unseen-attack
condition.

Have this ready if someone in the ICAI audience asks "isn't 0.14% EER on
Spanish suspiciously low?" — the paper already anticipated and flagged it.

## Training protocol

- Adam optimizer (beta1=0.9, beta2=0.999, eps=1e-8).
- Learning rate 3e-4, halved every 10 epochs.
- Batch size 64, utterances grouped by similar duration.
- Up to 100 epochs; early stopping after 50 epochs with no dev-loss
  improvement; best-dev-loss checkpoint is kept.
- Representative run: early stopping triggered at epoch 72.
- **Every one of the 48 configurations is trained 6 times** with different
  random seeds; results reported as mean plus/minus standard deviation EER.

## Evaluation metric and statistical treatment

- Primary metric: **Equal Error Rate (EER)** — the operating point where
  False Rejection Rate equals False Acceptance Rate.
- Secondary metric: **min t-DCF** (minimum normalized tandem detection cost
  function), which weights CM decisions by their effect on a downstream
  speaker-verification system.
- Because Spanish-side EER concentrates near zero (exact zeros occur,
  skewness exceeds 1 in a quarter of cells), the authors deliberately avoid
  any normality assumption. Instead: median and observed range for
  description, **Wilcoxon signed-rank test** on seed-matched pairs for
  significance, **Cohen's d** for effect size, **Bonferroni correction** for
  multiple comparisons.
- With only 6 seeds, the smallest attainable two-sided p-value is 0.031 —
  the study is explicitly underpowered for small effects, and the authors
  say so rather than overclaiming significance.

This statistical discipline is why so many results below are reported as
"a leading family of N configurations" rather than "the best configuration."
That is a deliberate, defensible choice, not a hedge — be ready to say that
plainly if asked why there's no single winner.
