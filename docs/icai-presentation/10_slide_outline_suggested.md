# Suggested Slide Outline

Sized for a standard 15-18 minute conference talk plus Q&A (typical ICAI
slot). Each bullet is one slide unless marked otherwise. Source doc for each
slide is noted so you can go pull the exact numbers/wording.

1. **Title slide.** Title, authors, affiliation, venue. (`01_paper_metadata.md`)
2. **The problem.** ASV under spoofing attack; TTS vs VC in one sentence
   each; why this matters (fraud, identity theft). (`02_motivation_related_work.md`)
3. **The gap.** ASVspoof is English-only; Spanish CMs degrade to >30% EER
   even for strong baselines (cite Tamayo et al.); nobody had systematically
   asked *which* architectures degrade how much, or *why*. (`02_motivation_related_work.md`)
4. **Contributions, up front.** The four numbered contributions, one line
   each. (`01_paper_metadata.md`)
5. **The experimental grid.** 4 front-ends x 3 back-ends x 4 losses = 48
   architectures, 3 data conditions, 6 seeds each. Use the redrawn "recipe"
   graphic, not the raw table. (`03_methodology.md`, `09_tables_figures_inventory.md`)
6. **Datasets.** English (ASVspoof2019) vs Spanish (HABLA) vs Combined,
   partition sizes table. (`03_methodology.md`)
7. **The statistical discipline (short).** Why medians/ranges/Wilcoxon
   instead of a single best number; six seeds is deliberately treated as
   underpowered for small effects. One slide, don't over-invest time here,
   but it inoculates against "why no single winner" questions later.
   (`03_methodology.md`)
8. **Result 1: English-only baseline works, but no single winner.** Best
   mean 2.67%, indistinguishable group of eight. (`04_results_by_condition.md`, condition 1)
9. **Result 2: cross-lingual collapse — the headline finding.** All 48
   configs fail (>30% EER) when English-trained models see Spanish. Grouped
   bar chart from `table:per-attack`. (`04_results_by_condition.md`, condition 2;
   `09_tables_figures_inventory.md`)
10. **Result 2, continued: it's voice conversion, not TTS.** 1.79% EER on
    Spanish TTS vs 43-66% on voice conversion, same English-trained model.
    This is the single slide most worth rehearsing. (`04_results_by_condition.md`, condition 2)
11. **Result 3: Spanish-only training works well, and the architecture
    ranking flips.** Best 0.14% EER; LSTM-sum wins for English, Attention
    wins for Spanish. (`04_results_by_condition.md`, condition 3)
12. **Result 4: bilingual training closes the gap.** Near-zero Spanish EER,
    <1pp cost on English; name the corpus-imbalance caveat in the same
    breath, don't let it surface only in Q&A. (`04_results_by_condition.md`,
    condition 4)
13. **Accent vs gender.** Grouped bar chart from `table:accent-gender`.
    Lead with the surprise: accent spread <0.5pp, gender gap reverses by
    country. (`05_accent_gender_filterbank.md`)
14. **Filter-bank scale.** Descendant scaling helps Mel, hurts linear — the
    common assumption doesn't survive testing both. Short physiological
    aside (F1 vs F2) if time allows. (`05_accent_gender_filterbank.md`)
15. **Five conclusions, one slide.** The condensed one-line versions.
    (`06_key_findings_conclusions.md`)
16. **Limitations, stated by us before anyone asks.** Attack coverage (only
    4 HABLA attack types), target-identity leakage caveat, corpus
    imbalance. (`07_limitations_future_work.md`)
17. **What's next: MARSA.** Six TTS systems, partial-spoof splicing tier,
    acoustic augmentation, frame-level scoring need. This is your bridge
    into your own thesis and a natural closing slide. (`07_limitations_future_work.md`)
18. **Thank you / questions.**

## Timing guide (if 15 minutes)

- Slides 1-7 (setup): ~4 minutes.
- Slides 8-14 (results): ~8 minutes — this is where the content lives,
  don't rush 9-10 or 13.
- Slides 15-18 (wrap-up): ~3 minutes.

## Anticipated hard questions and where the honest answer already lives

- *"Isn't 0.14% Spanish EER too good to be true?"* → target-identity
  leakage caveat, `03_methodology.md`. Say it before they ask if you have
  the slide real estate.
- *"Why only Wilcoxon and not just report the single best model?"* → the
  statistical-power section, `03_methodology.md`. The honest answer is that
  six seeds cannot support naming a single winner, and the paper says so.
- *"Doesn't 'accent doesn't matter' contradict your paper's own title?"* →
  `05_accent_gender_filterbank.md` framing note. Answer directly: the
  paper's contribution is precisely that the intuitive story (accent
  drives error) turns out to be wrong, and gender is the real signal — say
  that plainly, don't hedge it.
- *"Is the bilingual result just because Spanish dominates the merged
  corpus?"* → acknowledged directly as unresolved, `07_limitations_future_work.md`.
  Don't claim it's ruled out; say it's explicit future work.
