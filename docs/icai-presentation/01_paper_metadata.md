# Paper Metadata

**Title:** Accent-Based Evaluation of Speech Anti-spoofing Countermeasures
Across Multiple Languages

**Authors:** Jaime Arturo Hurtado Romero (corresponding), Tomás Acosta
Bernal, Rubén Francisco Manrique.

**Affiliation:** FLAG Lab, Department of Systems and Computing Engineering,
Universidad de los Andes, Bogotá, Colombia.

**Venue:** ICAI 2026, Springer CCIS. Confirmed as paper 16, currently at the
camera-ready revision stage (per `scripts/ICAI/habla_v1/README.md`).

**Class/format:** `llncs` (Springer LNCS proceedings paper), 12-16 pages.

## One-paragraph summary

The paper trains and evaluates 48 anti-spoofing architectures (4 front-end
feature extractors times 3 LCNN back-end variants times 4 loss functions)
across three data conditions: English (ASVspoof 2019), Spanish (HABLA), and
a combined bilingual corpus. Every configuration is trained six times with
different seeds, and results are reported with medians, observed ranges, and
paired non-parametric statistical tests rather than a single point estimate,
because the leading configurations turn out not to be statistically
separable. The headline results are that language mismatch collapses
detection performance, that this collapse is driven almost entirely by voice
conversion attacks rather than text-to-speech attacks, that bilingual
training all but closes the gap, that accent contributes far less residual
error than speaker gender does, and that a common assumption about
filter-bank design (low-frequency emphasis is universally good) does not
survive being tested on both Mel-scale and linear-scale front-ends.

## The four contributions, as the paper states them

1. A systematic comparison of 48 architecture combinations, each trained and
   evaluated across three dataset conditions and reported with
   distribution-free statistics over six random seeds, separating
   architectural differences from seed variation.
2. A per-attack decomposition of the cross-language degradation reported by
   Tamayo et al., showing the collapse is driven almost entirely by voice
   conversion rather than text-to-speech — a distinction not previously
   reported.
3. An accent- and gender-resolved analysis over five Spanish regional
   varieties (Argentina, Chile, Colombia, Peru, Venezuela), identifying
   speaker gender as a residual error source whose direction reverses
   between countries.
4. A filter-bank scale analysis showing descendant (low-frequency-emphasis)
   scaling benefits Mel-scale front-ends but not linear-scale ones.

## Why this matters for the ICAI talk

Contribution 2 is the most citable single finding: it overturns a common
implicit assumption (cross-lingual failure = TTS is the hard case) by
showing TTS actually survives the language switch and voice conversion is
what collapses. Contribution 3 is the most surprising one for an audience
expecting an accent-focused paper: accent barely matters once bilingual
training is in place, gender does. Lead with whichever of the two better
fits your allotted time; both are strong enough to be the single takeaway
slide if time is short.
