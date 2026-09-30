# Tables and Figures Inventory

Every table and figure in `jaime.tex`, in document order, with what it
contains and a suggestion for how to turn it into a slide visual (the
original LaTeX tables are dense and not directly presentable as-is).

| Label | Content | Slide suggestion |
|---|---|---|
| `table:datasets-taxonomy` | 6 anti-spoofing datasets by language and attack type | Keep as a simple comparison table; useful on the motivation slide |
| `table:antispoofing-modules` | The 3-axis grid: front-ends x back-ends x losses | Redraw as a 3-column "recipe" graphic (4 x 3 x 4 = 48), more visual than tabular |
| `table:lcnn-architecture` | Full LCNN stem layer-by-layer (filters, strides, output shapes) | Too dense for a slide as-is; replace with a simplified block diagram (Input -> Conv/MFM blocks -> pooled embedding -> loss head) |
| `table:datasets-partitions` | Train/dev/test utterance counts for English, Spanish, Combined | Keep as a table, it's already slide-sized |
| `table:mean-train-en-eval-en` | Full 48-config EER grid, English train/eval | Do not reproduce in full; extract only the "leading group of 8" statement and the best-config number |
| `table:per-attack` | EER by attack family (CycleGAN/DiffVC/StarGAN/TTS), English- vs Spanish-trained, 3 back-ends | **This is the single most important table in the paper.** Reproduce as a bar chart: x-axis = attack family, grouped bars = EN-trained vs ES-trained, one small multiple per back-end or just P2S-Grad-equivalent aggregate |
| `fig:det` | DET curves on Spanish eval, EN- vs ES-trained models, EER marked | Reuse the actual figure (`figures/det_curves`) if available; it's already presentation-ready |
| `table:mean-train-es-eval-es` | Full 48-config EER grid, Spanish train/eval | Extract only the best-config number and the Spectrogram-exception note |
| `table:mean-train-un-eval-es` | Full 48-config EER grid, combined train / Spanish eval | Extract only the leading-family-of-4 statement |
| `table:mean-train-un-eval-en` | Full 48-config EER grid, combined train / English eval | Extract only the best-config number and the ~1pp-cost statement |
| `table:accent-gender` | EER by country x gender, 5 countries | **Second most important table.** Reproduce as a grouped bar chart (country on x-axis, female/male bars), this is the visual that carries contribution 3 |
| `table:scale-aggregate` | Filter-bank scale (ascendant/constant/descendant) x family (linear/Mel), median/mean/wins | Reproduce as a small grouped bar chart or a 2x3 heatmap-style table, this carries contribution 4 |

## Figures referenced but not embedded as tables

- `figures/det_curves` — the only true figure (not table) in the paper.
  Check whether the actual PDF/image file exists in the IEEE build folder
  before assuming you can reuse it directly; if not, it needs to be
  regenerated from the same evaluation run described in `sec:cross-language`.

## Recommended reproduction priority for slides

If you only have time to faithfully reproduce two visuals, make them:
1. `table:per-attack` as a grouped bar chart (carries the paper's single
   biggest finding: VC collapses cross-lingually, TTS doesn't).
2. `table:accent-gender` as a grouped bar chart (carries the
   accent-vs-gender finding, which is the one that will surprise an
   audience expecting an "accent matters" paper).

Everything else can be stated as a number in a sentence rather than
reproduced as a full table — the 48-configuration grids in particular are
not legible at slide resolution and the paper's own point about them is
statistical indistinguishability, which a giant table actively undercuts
visually (it invites the audience to eyeball a "winner" the paper argues
against).
