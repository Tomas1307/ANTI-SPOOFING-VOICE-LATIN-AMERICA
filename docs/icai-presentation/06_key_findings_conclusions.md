# Key Findings / Conclusions (Five Statements)

This is the paper's own Conclusions section, condensed to five statements
you can each turn into one slide. Order matches the paper.

## 1. Cepstral front-ends win, but there is no single best configuration

LFCC and MFCC place consistently ahead of CQCC and Spectrogram. Within that
leading group, individual configurations cannot be separated: on English,
none of the seven runners-up differs significantly from the best
configuration; under bilingual training, the four leading Spanish-side
configurations have overlapping confidence intervals. Six seeds do not
support naming a single winner, and the paper does not pretend otherwise.

## 2. The cross-language collapse is an attack-family effect, not a uniform one

Models trained only on English fall to chance level on Spanish, confirming
Tamayo et al. Decomposing by attack shows voice conversion accounts for
almost all of the degradation (EER 43-66%), while text-to-speech attacks
remain detectable and reach 1.79% EER for the best English-trained model.
Vocoder artifacts appear to survive the language change; conversion
artifacts, entangled with speaker and accent characteristics, do not.

## 3. Bilingual training reaches comparable performance in both languages

Merging English and Spanish training data drives Spanish EER close to zero
and costs less than one percentage point on English. Caveat stated in the
paper itself: the merged corpus is not language-balanced (57.7% Spanish),
so the magnitude of both effects remains confounded with corpus
composition. This is not disproven, just not yet cleanly isolated —
correctly flagged as future work.

## 4. Residual error tracks speaker gender, not regional accent

Differences between the five Latin-American varieties stay below half a
percentage point once bilingual training is applied, and the back-end
ranking is identical across all five. The larger and less expected effect
is a gender gap whose direction reverses from one country to another. This
points toward demographic balancing as a more productive line of work than
accent-adaptive architecture selection.

## 5. Descendant filter-bank scaling helps Mel banks and harms linear ones

Across the full configuration grid, descendant weighting gets the lowest
median EER for Mel-scale front-ends (wins 8 of 12 configurations) but is
the worst of the three scalings for linear-scale front-ends. Emphasizing
low frequencies is therefore not universally advantageous. The usual
justification (principal formants sit low in the spectrum) holds for F1 and
fails for F2.

## One-line versions, if you need a single sentence per point

1. Cepstral beats CQCC/Spectrogram, but the top group is a statistical tie.
2. Cross-lingual failure is a voice-conversion problem, not a TTS problem.
3. Bilingual training nearly erases the language gap (with a data-balance
   caveat).
4. Gender, not accent, is what's left once you train bilingually — and its
   direction flips by country.
5. "Emphasize low frequencies" is true for Mel filter banks and false for
   linear ones.
