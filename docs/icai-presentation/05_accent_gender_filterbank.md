# The Two Secondary Studies: Accent/Gender, and Filter-Bank Scale

These are contributions 3 and 4 of the paper. Given the ICAI accent-focused
framing of the title, the accent/gender result is likely to be the one the
audience remembers, or pushes back on, so it is written out in full detail
here.

## Accent and gender analysis (Section `accent-analysis`, Table
`accent-gender`)

Setup: using the **bilingually trained** models (MFCC front-end; LFCC gives
qualitatively similar patterns), break down Spanish evaluation EER by
Latin-American accent and by speaker gender, aggregated over 3 back-ends, 4
losses, and 6 seeds.

| Country | Overall EER% | Female EER% | Male EER% |
|---|---|---|---|
| Chile | 0.032 | 0.059 | 0.005 |
| Peru | 0.053 | 0.085 | 0.021 |
| Argentina | 0.140 | 0.177 | 0.103 |
| Venezuela | 0.168 | 0.065 | 0.271 |
| Colombia | 0.287 | 0.029 | 0.544 |

Three observations, as the paper states them:

1. **Differences between accents are small.** Chile (0.032%) and Peru
   (0.053%) are best-modeled, Colombia (0.287%) worst, but all five medians
   are zero and the spread is below half a percentage point. Accent alone is
   a weak predictor of residual error.
2. **The back-end ranking does not vary by accent.** LCNN-LSTM-sum attains
   the lowest mean EER for all five countries. The paper reads this as a
   negative result that argues against accent-conditioned model selection.
3. **Speaker gender carries the dominant residual signal, and its direction
   reverses between countries.** Colombia and Venezuela are markedly worse
   on male speakers (Colombia: 0.544% male vs 0.029% female; Venezuela:
   0.271% male vs 0.065% female). Argentina, Chile and Peru are worse on
   female speakers. The Colombian gap alone accounts for essentially all of
   that country's overall disadvantage. Because the direction flips by
   country, the paper argues this cannot reflect a single global bias of the
   front-end — it points to an interaction between the speaker population
   and the specific attack generators used for each subset, not to a
   country-level acoustic property.

**Framing note for the talk:** this is a paper whose title promises an
accent-based evaluation, and its own strongest finding on that axis is that
accent is *not* the dominant factor once bilingual training is applied —
gender is, and inconsistently so. That's worth stating plainly rather than
softening; it is a more interesting result than "accent matters," and the
paper is confident enough in it to lead with it as contribution 3.

## Filter-bank scale study (Section `filter-bank`, Table `scale-aggregate`)

Setup: three scale modes for the cepstral filter banks — ascendant (gain
rises with frequency), constant (flat), descendant (gain falls with
frequency, i.e. low-frequency emphasis) — evaluated across the full grid (3
back-ends x 4 losses x 6 seeds = 72 runs per scale per front-end family),
English evaluation.

| Family | Scale | Median EER% | Mean EER% | Wins (out of 12) |
|---|---|---|---|---|
| Linear (LFCC) | Ascendant | 3.069 | 3.292 | 3 |
| Linear (LFCC) | Constant | 3.162 | 3.294 | 6 |
| Linear (LFCC) | Descendant | 3.190 | 3.383 | 3 |
| Mel (MFCC) | Ascendant | 3.793 | 4.368 | 3 |
| Mel (MFCC) | Constant | 3.855 | 4.385 | 1 |
| Mel (MFCC) | Descendant | **3.475** | **3.925** | **8** |

- **The benefit of descendant (low-frequency-emphasis) scaling is specific
  to the Mel family.** For Mel it wins 8 of 12 configurations and has the
  best median and mean. For linear-scale banks, descendant is the *worst*
  of the three by median, winning only 3 of 12 — constant wins there (6 of
  12).
- Reporting either family alone would invert the conclusion. The common
  assumption "low-frequency emphasis is universally preferable" does not
  survive testing both families.
- A single configuration doesn't settle it either: for LFCC with
  LCNN-LSTM-sum/P2S-Grad specifically, the three scales differ by less than
  0.16 percentage points of median EER — well inside seed-to-seed spread.
- **Physiological grounding, stated carefully (not overclaimed):** the
  first formant (F1) does sit below roughly 1 kHz for most vowels, but the
  second formant (F2) spans roughly 600-2500 Hz. So "formants are low"
  only partially justifies descendant weighting. The Mel scale already
  compresses high frequencies; descendant weighting compounds that
  compression. A linear bank has no such compression to begin with, so
  suppressing its high-frequency resolution actively discards information
  the classifier was using.
- **Practical takeaway offered by the paper:** the linear family is more
  repeatable (roughly half the across-configuration standard deviation of
  the Mel family, under every scale). Prefer a linear bank for run-to-run
  stability; prefer a Mel bank with descendant weighting if you want the
  single lowest expected EER and can tolerate more variance.
