# Results, by Experimental Condition

Five conditions in total. Numbers below are copied directly from the paper's
tables; nothing here is re-derived or rounded further than the source.

## 1. English train / English eval (Table `mean-train-en-eval-en`)

- Performance is consistently low (good) across all 48 configurations.
- Cepstral front-ends (LFCC, MFCC) consistently beat CQCC and Spectrogram.
- Lowest mean EER: **LFCC + LCNN-LSTM-sum + P2S-Grad, 2.67% (std 0.60)**.
- But it is **not statistically separable** from the next seven
  configurations: seed-matched Wilcoxon tests show no significant
  difference against any of them (p > 0.31 throughout), effect size against
  the runner-up is only d = 0.22.
- Median EER 2.51% (range 2.12-3.70%); the next seven span medians
  2.61-2.97% with overlapping ranges; their median min t-DCF (0.063-0.075)
  doesn't even reproduce the EER ranking. **Read these eight as one
  indistinguishable group, not a ranking.**
- What the data does support: P2S-Grad and OC-Softmax outperform AM-Softmax
  and Sigmoid; Spectrogram has higher seed-to-seed variance than cepstral
  front-ends; CQCC performs comparably to Spectrogram here.

## 2. Cross-language: English-trained models evaluated on Spanish (Section
`cross-language`, Table `per-attack`)

- **All 48 configurations exceed 30% EER.** Lowest configuration mean:
  34.4%. 28 of 48 exceed 40%.
- By front-end: Spectrogram degrades least (best mean approx. 36%). CQCC
  and LFCC degrade most and are effectively tied (45.5% and 45.4% mean).
  MFCC/LFCC — strong English baselines — land at 34-55% cross-lingual EER.
- This reproduces the Tamayo et al. degradation finding.
- **The headline result of the whole paper:** decomposing by attack family
  shows the collapse is not uniform. Voice conversion (VC) drives it almost
  entirely; text-to-speech (TTS) attacks stay detectable.

| Back-end | Training | CycleGAN (VC) | DiffVC (VC) | StarGAN (VC) | TTS |
|---|---|---|---|---|---|
| LCNN-Attention | English | 53.09 | 50.33 | 64.77 | 46.11 |
| LCNN-LSTM-sum | English | 51.72 | 53.99 | 65.83 | **1.79** |
| LCNN-trim-pad | English | 51.28 | 43.22 | 63.77 | 16.92 |
| LCNN-Attention | Spanish | 0.03 | 0.00 | 0.01 | 0.00 |
| LCNN-LSTM-sum | Spanish | 0.01 | 0.01 | 0.01 | 0.00 |
| LCNN-trim-pad | Spanish | 0.03 | 0.01 | 0.01 | 0.01 |

- Best English-trained model reaches **1.79% EER on Spanish TTS** — within
  its own English-side range — while sitting at or below chance on
  CycleGAN, DiffVC and StarGAN.
- StarGAN EER is *above* chance (63-66%): the decision is inverted, not
  merely uninformative.
- TTS detection varies by an order of magnitude across back-ends (1.79% to
  46.11%), so cross-language TTS transfer depends on the pooling strategy,
  not on the front-end alone.
- Interpretation offered in the paper: vocoder artifacts are largely
  language-independent and transfer; conversion artifacts are entangled
  with speaker/accent characteristics that differ between English and
  Spanish, so they don't transfer.
- Figure `det`: DET curves on the full HABLA evaluation set (32,327
  utterances, larger than the 4,809-utterance partition used elsewhere).
  Grey = English-trained, black = Spanish-trained, circles mark EER.

## 3. Spanish train / Spanish eval (Table `mean-train-es-eval-es`)

- Low EER for most architectures.
- Best: **LFCC + LCNN-Attention + P2S-Grad, 0.14% EER (std 0.09)**.
- Exception: Spectrogram front-ends consistently land at 12-20% EER —
  synthetic Spanish speech apparently lacks the texture-level cues
  Spectrogram exploits in English.
- **The architecture ranking shifts between languages**: LSTM-sum pooling
  was best for English, Attention pooling is best for Spanish. This
  language-specific sensitivity is the paper's own motivation for testing
  combined bilingual training next.

## 4. Combined bilingual training (Section `combined`)

**Spanish evaluation** (Table `mean-train-un-eval-es`):
- MFCC and LFCC reach near-zero EER (<= 0.5%).
- Lowest mean: **MFCC + LCNN-LSTM-sum + AM-Softmax, 0.087% (std 0.04)**.
- Not separable from its two nearest competitors (both 0.17%): Wilcoxon p =
  0.0625; a paired t-test flags three of seven at p < 0.05, but none
  survives Bonferroni correction (needs p < 0.0071).
- Reported conclusion: a **leading family of four cepstral configurations
  below 0.2%**, median min t-DCF between 0.0022 and 0.0035 — not a single
  best system.
- CQCC also performs well (<= 1.3%); Spectrogram remains weakest, mirroring
  the Spanish-only result.

**English evaluation** (Table `mean-train-un-eval-en`):
- Performance drops only about 1 percentage point relative to the
  English-only baseline.
- Lowest bilingual English mean: **MFCC + LCNN-trim-pad + P2S-Grad, 3.64%**.
- Near-parity on English plus near-zero on Spanish reads as bilingual
  generalization — subject to the corpus-imbalance caveat (57.7% Spanish,
  see `03_methodology.md`).

## Quick comparison across conditions (for a single summary slide)

| Condition | Best config | Best mean EER |
|---|---|---|
| English -> English | LFCC / LCNN-LSTM-sum / P2S-Grad | 2.67% |
| English -> Spanish (cross-lingual) | any of 48 | >= 34.4% (all fail) |
| Spanish -> Spanish | LFCC / LCNN-Attention / P2S-Grad | 0.14% |
| Bilingual -> Spanish | MFCC / LCNN-LSTM-sum / AM-Softmax | 0.087% |
| Bilingual -> English | MFCC / LCNN-trim-pad / P2S-Grad | 3.64% |
