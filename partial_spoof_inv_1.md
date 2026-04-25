# Word-level splicing mechanics for partial-spoof corpora

## A) Executive summary

Four canonical partial-spoof corpora (PartialSpoof, LlamaPartialSpoof, HAD, HQ-MPSD) collectively publish **far less waveform-level splicing detail than their prominence would suggest**. After a strict source-text and repository audit, only one of them — LlamaPartialSpoof — names any concrete waveform parameter beyond "loudness normalization" (it uses a uniformly-drawn 30–80 ms overlap and one of five fade shapes). **No public pipeline documents time-stretching, F0 smoothing, formant matching, LPC/MFCC continuity enforcement, or any ablation of overlap duration versus detector EER.** None publishes a MOS-vs-EER scatter on its own construction choices. None releases the dataset-construction source code for the splicer itself: PartialSpoof's `00data-prepare` folder is flagged "to be released," LlamaPartialSpoof's repo is metadata-only, HAD's Zenodo release is audio-only, and HQ-MPSD (arXiv Dec 2025) has no discoverable code artifact.

Two third-party papers now anchor most of the rigorous evidence: **Negroni et al. (arXiv:2408.13784, 2024)** shows a simple artifact analyzer hits 6.16 % / 7.36 % EER on PartialSpoof/HAD with no learned model — proof that naïve splicing bleeds detectable signal — and **Huang et al. (SLT 2024, arXiv:2501.03805)** demonstrates that neural infilling fools humans but not SSL detectors. Together they bound the perceptual-vs-detection gap. Everything else listed in the user's brief is a gap the user must resolve empirically.

## B) Comparison table — 4 pipelines × 7 questions

Legend: **EXPLICIT** (verbatim paper/repo), **AMBIG** (mentioned but underspecified), **NSP** (not specified in paper), **CNR** (code not publicly released). Two important premise corrections are flagged with ⚠.

| Question | PartialSpoof (Zhang et al. 2021 / TASLP 2023) | LlamaPartialSpoof (Luong et al., ICASSP 2025, arXiv 2409.14743) | HAD (Yi et al., Interspeech 2021, arXiv 2104.03617) | HQ-MPSD (Li et al., arXiv 2512.13012, Dec 2025) |
|---|---|---|---|---|
| Q1 Zero-gap handling | **Structurally N/A**: VAD-segment swap assumes non-speech margin around each segment (TASLP §III-B step 3). Behavior if that margin is absent: **NSP**. | **NSP / AMBIG**. Paper only gives 30–80 ms crossfade target, silent on what happens when adjacent MFA word has <30 ms bona-fide margin. §II-B. | **N/A**: paper restricts to "one replacing operation" per utterance (§2.2). | **NSP / AMBIG**. Paper notes "a limited number of segments are substituted per utterance," silent on adjacent replacements. |
| Q2 Duration mismatch | **Not applicable**: selects a bona-fide and a spoof segment of "similar duration" (TASLP §III-B step 2, cond. 3); no TTS-insertion step at all. ⚠ user premise | **NSP**. No time-stretch, global shift, or silence-compression discussed. | **NSP**. Only "volume normalization" mentioned (§2.4 step 3). | **NSP**. Only "loudness and spectral-characteristic alignment" in pre-normalization. |
| Q3 F0 discontinuity | **NSP**. No F0 smoothing or ΔF0 statistic reported. | **NSP**. No F0 handling anywhere in construction or dataset analysis. | **NSP**. | **NSP** in retrieved excerpts. |
| Q4 Spectral envelope | **NSP** beyond **ITU-T SV56 amplitude norm to −26 dBov** (TASLP §III-B step 1). No LPC/MFCC/formant ops. | **NSP** beyond loudness normalization. | **NSP** beyond volume normalization. | **EXPLICIT**: "pre-normalization aligns loudness and spectral characteristics." Algorithmic detail **NSP**. ⚠ "Adaptive pre-emphasis" phrasing in user brief was **not recovered verbatim** from retrieved text. |
| Q5 Sub-word placement | **EXPLICIT**: cuts at **VAD boundaries** (majority vote of 3 VADs). Explicitly not word- or phone-aware — listed as Limitation (TASLP §III-D). | **EXPLICIT**: **MFA word boundaries**. Snapping to zero-crossing / silence / sub-phonetic / GCI: **NSP**. | **EXPLICIT**: forced-aligned **character-level** timestamps. Silence-region restriction: **NSP** ⚠ user brief's "cuts at silence regions" premise is **not attested** in the HAD paper. | **EXPLICIT**: "**midpoints between aligned word pairs**." Whether midpoint is by duration or energy: **AMBIG**. |
| Q6 Implementation code | **CNR**. Repo `nii-yamagishilab/PartialSpoof` is CM-training only; the `00data-prepare` folder is listed in the README as "(To be released)"; companion repo `PartialSpoof_database` "(TBA)" is not public. | **CNR**. Repo `hieuthi/LlamaPartialSpoof` contains only `README.md`, `split/`, `transcripts/`. No splicing Python. | **CNR**. Zenodo (DOI 10.5281/zenodo.10377492) releases `HAD.zip` audio only; construction uses **jiaaro/pydub** per paper but no splicing script is published. | **CNR** (as of retrieval). Zenodo URL truncated in retrieved excerpts; no GitHub repo indexed. |
| Q7 Perceptual vs detection | No MOS/PESQ/NISQA reported. TASLP §VI-D Hypothesis 2 / Fig. 5 shows detector EER worsens when spoof has *fewer* concatenation boundaries — implicitly confirms detectors key on concat artifacts. | No MOS/PESQ/NISQA. Table V(b) **does ablate concatenation method** (crossfade / cut-paste / OLA) vs. EER per training set — the one published "insertion-technique vs. detector" table in this set. | Only EER / segment P-R-F1 (Tables 3–7). No perceptual metric, no ablation vs. insertion technique, no listening test. | **DNSMOS 3.58** reported on partial deepfakes (stated as highest among compared datasets). Transfer degrades detectors by >80 %. Per-model EER tables and fade-shape ablation **NSP in retrieved excerpts**. |

## C) Per-question analysis

### C1 — Margin acquisition / zero-gap (the adjacency problem)

None of the four pipelines explicitly addresses the zero-gap problem. **PartialSpoof sidesteps it by design**: concatenation is always performed inside the non-speech margin that surrounds each VAD-selected segment, with "50 % of the non-speech part in the head and tail" retained (Interspeech 2021 §2 step 3; TASLP 2023 §III-B step 3). The authors use **time-domain cross-correlation to locate the best concatenation point within that silent margin**, then do a waveform overlap-add *within the silence*. This is elegant when a VAD margin exists; it is simply undefined when two cloned segments abut. HAD dodges the issue by construction — **exactly one replacement per utterance** (§2.2), so adjacency between cloned words is structurally excluded. LlamaPartialSpoof allows multi-segment replacements (Fig. 2 shows utterances with up to dozens of fake words), and its paper text states crossfade overlap is drawn uniformly from 30–80 ms (§II-B). What the paper does **not** say, and what the metadata-only GitHub release does not let us verify, is how overlap is realized when adjacent MFA word-boundaries leave fewer than 30 ms of bona-fide audio between two cloned words. The five candidate strategies in the user's brief — (a) butt-splice at zero-crossing, (b) silence insertion, (c) micro-shift to steal margin, (d) multi-word TTS re-generation, (e) cluster-external overlap only — are all plausible, **none is attested in any retrieved paper or repo**. HQ-MPSD uses a fixed 30 ms cosine overlap-add; because cuts land at word midpoints, splicing into bona-fide material on both sides is normally guaranteed, but the adjacent-replacement case is again silent in the text. **This is the single largest gap in the literature for the user's pipeline.** The only useful transferable technique is PartialSpoof's cross-correlation best-join search *inside the available margin*, which could be adapted by the user even if the ultimate zero-gap policy must be engineered from scratch.

### C2 — Duration mismatch (cloned word length ≠ bonafide slot)

This question is only meaningful for the two TTS-insertion pipelines (LlamaPartialSpoof, HQ-MPSD). **PartialSpoof does not perform TTS insertion at all** — a critical correction to the brief: it swaps VAD-segments of similar duration between a bona-fide and a *pre-existing* ASVspoof-2019-LA spoofed utterance of the same speaker, and explicitly requires similar duration (TASLP §III-B step 2). HAD replaces a keyword span with the same keyword cut out of a fully-synthesized utterance, but does not describe any duration reconciliation beyond accepting the Δt in the output. LlamaPartialSpoof and HQ-MPSD are both completely silent on time-stretching, global shifting, or silence compression; both appear to accept the mismatch and let the global utterance timeline drift. There is **no published evidence** (Task A of the cross-cutting search) of the specific curve mapping time-scale modification percentage to MOSNet / NISQA / PESQ degradation in an anti-spoofing context, nor of the point at which RawNet2 / AASIST / wav2vec2-XLSR detectors begin keying on WSOLA/PSOLA artifacts. The closest anchor is the TSM Subjective Quality Dataset (Roberts 2020, IEEE DataPort; arXiv:2006.00848), which carries 42,529 subjective MOS at 20 time-scale ratios β ∈ [0.22, 2.2] — sufficient for the user to derive a conservative envelope (subjective MOS typically falls steeply outside roughly β ∈ [0.85, 1.20] in that data) but not cited as a single anti-spoofing threshold. Chakravarty & Dua (2023), cited in the Li et al. anti-spoofing survey (arXiv:2404.13914), use TSM as augmentation, implying detectors can *internalize* TSM patterns when exposed to them — but this cuts both ways for dataset builders. **The user should treat duration reconciliation as an explicit engineering decision, not a solved-by-prior-art problem.**

### C3 — F0 discontinuity at the splice boundary

Not a single paper among the four target pipelines reports a measured |ΔF0| distribution at splice boundaries, and none applies F0 smoothing post-splice. The problem is real: because zero-shot TTS generates each word without prosodic context of the preceding bona-fide speech, the F0 contour entering and exiting the splice will be discontinuous. The cross-cutting search confirmed that **no partial-spoof paper (PartialSpoof, LlamaPartialSpoof, HAD, HQ-MPSD, Psynd, LAV-DF family, ADD 2022/2023) publishes ΔF0 statistics**. The toolkit for smoothing is well-established outside anti-spoofing: **Moulines & Charpentier (1990), Speech Communication 9(5-6):453–467** introduced TD-PSOLA / FD-PSOLA for pitch-period-synchronous F0 smoothing at concatenation joins; **Stylianou (2001), IEEE TSAP 9(1):21–29** provided HNM-based phase- and F0-coherent concatenation; Taylor's *Text-to-Speech Synthesis* (2009) discusses F0 mismatch as a unit-selection join cost. **None is documented as deployed in any partial-spoof construction script.** A suggestive finding from the adjacent literature: Bořil & Skarnitzl show that formant discontinuities (not F0) drive perceived vowel-duration artifacts in Czech synthetic speech — implying the user's marginal payoff from formant matching (Q4) may exceed that from F0 smoothing for *perceptual* realism, although the reverse could hold for *detection* robustness. This remains the user's call: the literature provides no empirical priors on how far bona-fide→TTS F0 typically jumps at MFA word boundaries, nor on the detection-EER payoff of smoothing it.

### C4 — Spectral envelope and formant alignment

Each pipeline does the minimum: PartialSpoof normalizes to **−26 dBov via ITU-T SV56** (TASLP §III-B step 1); LlamaPartialSpoof and HAD perform unspecified loudness normalization; HQ-MPSD states it "aligns loudness and spectral characteristics between bonafide and synthetic speech" in pre-normalization but gives no algorithmic detail. **Two important corrections to the user's brief:** (a) the "adaptive pre-emphasis" formulation attributed to HQ-MPSD is **not verified verbatim** in the retrievable v1 text of arXiv:2512.13012 — only the generic phrase "spectral-characteristic alignment" appears in the accessible excerpts; (b) the "OLA-Hanning" formulation attributed to HAD is **not in the HAD paper at all** — that recipe originates from Negroni et al.'s *external* analysis (arXiv:2408.13784), which applied OLA-Hanning to its own experimental splicing on ASVspoof, not to HAD. No pipeline performs LPC-coefficient interpolation in the overlap region, MFCC-trajectory smoothing across the join, or explicit formant-frequency matching. No pipeline publishes an ablation of spectral-matching variants against perceptual or detection metrics. The concatenative-synthesis literature has the relevant machinery — unit-selection join costs explicitly penalize spectral-envelope mismatch (Hunt & Black 1996; Taylor 2009) — but, once again, none of this is documented as deployed for partial-spoof construction. The strongest cross-cutting signal is **Negroni et al. (2024, arXiv:2408.13784)**, which shows that splice-boundary artifacts in PartialSpoof and HAD are detectable (6.16 % / 7.36 % EER) *without training any detector*, purely from spectral-dynamic-range analysis of the join — implying the loudness-only policy common to all four pipelines is empirically insufficient for a corpus that is supposed to force detectors to learn deeper cues.

### C5 — Sub-word placement strategy

Three policies are in play, all different. **PartialSpoof cuts at VAD-segment boundaries** that are indifferent to word or phone structure — explicitly flagged as a limitation in TASLP §III-D: "variable-length speech segments found by the VADs are replaced without considering the meaning of sentences and words as well as the phonemes before and after the segments." **LlamaPartialSpoof cuts at MFA word boundaries** (§II-B) with no documented snapping to zero-crossings, silence, sub-phonetic landmarks, or glottal closure instants. **HAD cuts at forced-aligned character-level timestamps** — and critically, the paper *does not* constrain these to silence regions despite the common misreading; in Mandarin, character-aligned boundaries frequently land mid-syllable on voiced material. **HQ-MPSD is the most explicit**: "replacement boundaries are placed at midpoints between aligned word pairs to avoid cutting across phones or prosodic transitions." The motivation is sound (midword cuts are in phonetically stable material) but whether "midpoint" is computed in time or in energy is not specified. Outside anti-spoofing, the foundations for phonetic-aware placement are mature: **Moulines & Charpentier (1990)** established pitch-synchronous cutting aligned to GCIs; **Naylor et al. (2007, IEEE TASLP 15(1):34–43)** validated DYPSA-based GCI detection at 95.7 % identification rate; **Drugman et al. (arXiv:2001.00473)** reviews SEDREAMS, DPI, and zero-frequency-filtering GCI detectors that work under noise. Zero-crossing snapping is universal folk wisdom in audio editing but is not quantified against detector EER in any paper retrieved. **No 2015–2026 anti-spoofing paper in our search measures detection-EER deltas attributable to GCI-aligned versus voiced-unvoiced-boundary versus vowel-steady-state versus arbitrary cuts**, leaving this as a tractable research contribution for the user.

### C6 — Implementation specifics from open-source repos

The verdict on code release is uniformly grim and must be reported honestly. **`github.com/nii-yamagishilab/PartialSpoof`** contains only countermeasure training and evaluation infrastructure (`03multireso/`, `config_ps/`, `metric/`, `modules/`, `project-NN-Pytorch-scripts.202102/`, `01_download_database.sh`); its README's folder table lists `00data-prepare` as "(To be released)" and the linked `PartialSpoof_database` repo as "(TBA)" — the latter does not resolve publicly. **`github.com/hieuthi/LlamaPartialSpoof`** contains `README.md`, `split/` (speaker partitions), and `transcripts/` only; there is no `splice.py`, `crossfade.py`, or forced-alignment / TTS driver; the Zenodo record 14214149 ships audio + metadata CSVs including `metadata_crossfade.csv`, which records the fade-shape label per utterance but not the per-utterance overlap length. **HAD's Zenodo record 10377492 ships `HAD.zip` (~8.1 GB) and nothing else**; the paper names `jiaaro/pydub` (a slice/concatenate wrapper around `ffmpeg`) as the manipulation library, but no pipeline script is public. **HQ-MPSD** (arXiv:2512.13012, Dec 2025) references a Zenodo artifact whose URL was truncated in retrieved text; no GitHub index entry was located. Consequently, for all four sub-sub-questions — (i) overlap-window-length logic, (ii) zero-gap resolution, (iii) time-stretch call site, (iv) F0 smoothing call site — the answer across all four pipelines is **CNR**, with only LlamaPartialSpoof's prose giving a numeric anchor ("randomly assigned the length of the overlap between 30 and 80 ms," §II-B). The uniform default implied by paper prose is: loudness normalize, cut at the aligner's stated boundary, crossfade (or not) with the stated shape, write the file — i.e., no time-stretch and no F0 adjustment. Users requiring engineering-grade behavior must build it themselves.

### C7 — Perceptual versus detection asymmetry

This is the best-attested cross-cutting finding, but the evidence comes from *outside* the four target pipelines. Among the four, **only LlamaPartialSpoof publishes an insertion-technique ablation against detector EER** (Table V(b), contrasting crossfade vs. cut-and-paste vs. overlap-add across five training sets), and **only HQ-MPSD publishes a per-dataset perceptual number** (DNSMOS 3.58). Neither pairs both sides in a scatter. The gap is filled by two third-party papers. **Huang et al. (SLT 2024, arXiv:2501.03805)** built SINE — a dataset that uses Voicebox neural infilling in addition to cut-and-paste — and ran a 17-listener × 20-sample × 3-condition subjective test: infilling edits are **significantly harder for humans** to distinguish from bona-fide than CaP edits, yet an SSL-based detector (wav2vec2 + linear) detects infilling **as robustly as or more robustly than CaP**. This is the canonical demonstration that perceptual and detection optima diverge. **Negroni et al. (2024, arXiv:2408.13784)** demonstrates the symmetric case: naïve CaP splices that humans plausibly miss are detectable at 6.16 % / 7.36 % EER on PartialSpoof / HAD by a *hand-coded* spectral-dynamic-range threshold — no training needed. Together, these bound the problem: sophisticated neural infilling moves the perceptual needle but barely the detection needle; simple splicing without continuity enforcement barely moves the perceptual needle but moves the detection needle massively. For a partial-spoof corpus to pressure detectors in realistic ways it must push *both* — concretely, high perceptual quality *and* splice-boundary continuity that survives hand-coded artifact tests. No published work in the four target pipelines pursues this explicitly. **No paper in any retrieved source publishes a MOS-versus-EER scatter across insertion techniques.**

## D) Synthesized best-practice splicing procedure (user synthesis, not published canon)

The following pseudocode integrates the safest operations from the literature — PartialSpoof's cross-correlation best-join search within available margin, LlamaPartialSpoof's documented 30–80 ms overlap range, HQ-MPSD's word-midpoint cut strategy, classical PSOLA-style F0 smoothing (Moulines & Charpentier 1990), and Negroni et al.'s implicit recommendation that spectral-envelope continuity matters. **Every step below is the user's synthesis; no single pipeline in the literature does all of this.**

```
INPUT:  bonafide_audio (16 kHz mono), MFA word alignments,
        cloned_word_waveforms[], replacement_mask[]  # which words to replace
OUTPUT: partially-spoofed utterance

# ---------- Stage 1: cluster adjacent replacements ----------
clusters = group_consecutive_replacements(replacement_mask)
# each cluster = contiguous run of cloned words; two-sided bona-fide
# margin is guaranteed to exist at cluster boundaries

# ---------- Stage 2: per-cluster duration policy ----------
for C in clusters:
    bona_dur = MFA_end(C.last) - MFA_start(C.first)
    tts_dur  = sum(len(w) for w in C.cloned_words)
    ratio    = tts_dur / bona_dur
    if 0.90 <= ratio <= 1.10:
        strategy = "accept_mismatch"          # cosmetic only
    elif 0.80 <= ratio < 0.90 or 1.10 < ratio <= 1.25:
        strategy = "global_shift_downstream"  # shift later audio by Δt
    else:
        strategy = "wsola_stretch_cluster"    # bound 0.80-1.25 per
                                              # TSMDB subjective data
# NOTE: thresholds 0.90/1.10/0.80/1.25 are engineering choices;
#       the anti-spoofing literature gives no attested thresholds.

# ---------- Stage 3: intra-cluster concatenation (no bona-fide gap) ----------
for C in clusters:
    # Option (e) from the user's menu: concatenate within cluster
    # with ZERO crossfade; reserve crossfade for cluster-boundary only.
    cluster_wav = concat_zero_overlap_at_zero_crossings(C.cloned_words)
    # snap each intra-cluster join to the nearest zero crossing
    # within ±2 ms; optional GCI snap (DYPSA, Naylor et al. 2007)
    # when both sides are voiced.

# ---------- Stage 4: margin acquisition at cluster boundaries ----------
for C in clusters:
    left_margin  = bonafide_segment_before(C, max_ms=80)
    right_margin = bonafide_segment_after (C, max_ms=80)
    # cross-correlation best-join point inside each margin,
    # à la PartialSpoof TASLP §III-B step 3
    L_cut = xcorr_best_join(left_margin,  cluster_wav[:80ms])
    R_cut = xcorr_best_join(cluster_wav[-80ms:], right_margin)

# ---------- Stage 5: spectral and F0 continuity at cluster boundaries ----------
for each cluster_boundary B:
    # Loudness match (all four pipelines agree on this minimum)
    rms_match(cluster_wav, neighbour_bonafide)

    # F0 smoothing (not published in partial-spoof, borrowed from
    # Moulines & Charpentier 1990 TD-PSOLA); apply only if both
    # sides voiced within ±40 ms of the join.
    if voiced_both_sides(B):
        psola_f0_ramp(B, width_ms=40, target="linear_interp")

    # Spectral-envelope taper via short LPC interpolation
    # (Stylianou 2001); optional — cost/benefit unpublished.
    lpc_blend(B, order=16, taper_ms=20)

# ---------- Stage 6: crossfade ----------
for each cluster_boundary B:
    # LlamaPartialSpoof range; user may vary for own ablation.
    overlap_ms = draw_uniform(30, 80)
    apply_crossfade(B, shape=COSINE, overlap_ms=overlap_ms)

# ---------- Stage 7: post-processing ----------
peak_normalize_random(-0.01_dBFS, -10_dBFS)  # LlamaPartialSpoof recipe
write_16kHz_mono(output_wav)
```

**Explicit user-decision points flagged inline:** the 0.90/1.10 duration-acceptance window, the 0.80/1.25 stretch-vs-shift boundary, the choice to apply F0 smoothing only on voiced-voiced joins, the LPC order and taper width, and the overlap distribution shape (uniform vs. fixed) are all **engineering settings with no published priors** in anti-spoofing literature.

## E) Open questions the user must resolve

The following list is exhaustive for the set of questions in the brief. For each, the literature is silent or the evidence is merely suggestive from adjacent domains.

**Zero-gap boundary policy.** No published partial-spoof pipeline specifies behavior when two cloned segments abut with <30 ms of bona-fide margin. The five strategies (butt-splice at zero-crossing, silence insertion, micro-shift, multi-word TTS, cluster-external overlap) are all open for empirical comparison. This is a defensible thesis contribution on its own.

**Time-scale modification thresholds.** No anti-spoofing paper retrieved reports the time-stretch percentage at which MOSNet/NISQA/PESQ measurably degrade, nor the percentage at which RawNet2 / AASIST / SSL detectors begin discriminating. TSMDB (Roberts 2020) provides subjective MOS at 20 ratios but not detector-keyed data. The user would need to generate both curves from their own corpus.

**Per-pipeline F0 discontinuity statistics.** No paper reports the empirical |ΔF0| distribution across splice boundaries produced by a given TTS-plus-splicer combination. Publishing a histogram of |ΔF0| at MFA word boundaries for Qwen3-TTS and Fish Speech 4B outputs against Spanish bona-fide audio would be a novel contribution.

**F0 smoothing payoff.** No published evidence quantifies the EER or MOS change attributable to TD-PSOLA / HNM F0 smoothing at partial-spoof splice joins. Whether smoothing *helps* detection robustness (hides artifacts → harder target for the detector to memorize → better cross-dataset generalization) or *hurts* it (removes a genuine discriminator) is a completely open question.

**Spectral-envelope continuity payoff.** No published ablation contrasts "loudness normalization only" (all four pipelines) against LPC-interpolated or HNM-blended joins. Negroni et al. (2024) strongly suggests the loudness-only default is leaving a ~6 % EER floor on the table for any trained detector.

**Sub-word cut placement.** No anti-spoofing paper measures EER deltas for zero-crossing vs. GCI-aligned vs. voiced-unvoiced-boundary vs. vowel-steady-state vs. arbitrary cuts. HQ-MPSD's word-midpoint policy is the closest prior, and its motivation ("avoid cutting across phones or prosodic transitions") is phenomenological, not empirical.

**Crossfade overlap duration versus detection EER.** LlamaPartialSpoof's 30–80 ms range is a *de facto* anchor but was not ablated. No paper in the retrieved set maps overlap length to EER. This is a tractable and publishable ablation.

**Perceptual-vs-detection Pareto frontier for *word-level* edits.** Huang et al. (SLT 2024) establishes the phenomenon for *sentence-level* Voicebox infilling vs. CaP; Negroni et al. (2024) establishes the reverse phenomenon for simple CaP splicing. No paper plots a full MOS-vs-EER scatter across insertion techniques *for word-level edits with MFA boundaries and zero-shot voice cloning* — which is exactly the HABLA 2.0 setting.

**Duration-mismatch absorption strategy.** No published pipeline discriminates between (b) global-shift, (c) silence-compression, and (d) accept-mismatch. Each has different implications for MFA realignment, for downstream prosody, and for detectors that use duration features.

**Loudness-matching algorithm.** All four pipelines say "loudness normalization" without specifying the metric (ITU-R BS.1770 integrated LUFS? Per-segment RMS? Per-segment dBFS peak? SV56 active-speech?). PartialSpoof is the only one that names its algorithm (SV56). The others are black boxes at this level.

**Two premise corrections for the thesis text.** (i) HAD's construction paper (Interspeech 2021) does not use OLA-Hanning — that attribution belongs to Negroni et al. (2024), who apply OLA-Hanning in their *own* experimental splicing. (ii) The HQ-MPSD v1 paper's accessible text describes "loudness and spectral-characteristic alignment," not a named "adaptive pre-emphasis" operation; the user's citation should be softened unless the camera-ready or appendix uses that term explicitly.

## F) Bibliography

**Target pipelines.**
Zhang, Y., Wang, X., Cooper, E., Yamagishi, J. "An Initial Investigation for Detecting Partially Spoofed Audio." Interspeech 2021. arXiv:2104.02518.
Zhang, L., Wang, X., Cooper, E., Evans, N., Yamagishi, J. "The PartialSpoof Database and Countermeasures for the Detection of Short Fake Speech Segments Embedded in an Utterance." IEEE/ACM TASLP 31:813–825 (2023). arXiv:2204.05177.
Repo: github.com/nii-yamagishilab/PartialSpoof (detector code only; `00data-prepare` listed "to be released").
Yi, J., Bai, Y., Tao, J., Ma, H., Tian, Z., Wang, C., Wang, T., Fu, R. "Half-Truth: A Partially Fake Audio Detection Dataset." Interspeech 2021. arXiv:2104.03617. Dataset: zenodo.org/records/10377492.
Luong, H.-T., Li, H., Zhang, L., Lee, K.A., Chng, E.S. "LlamaPartialSpoof: An LLM-Driven Fake Speech Dataset Simulating Disinformation Generation." ICASSP 2025. arXiv:2409.14743. Repo: github.com/hieuthi/LlamaPartialSpoof (metadata only). Dataset: zenodo.org/records/14214149.
Li, M., Alber, M., Asgarianamiri, R., Zhao, L., Zhang, X.-P. "HQ-MPSD: A Multilingual Artifact-Controlled Benchmark for Partial Deepfake Speech Detection." arXiv:2512.13012 (Dec 2025).

**Third-party analyses (critical cross-cutting).**
Negroni, A., Salvi, D., Bestagini, P., Tubaro, S. "Analyzing the Impact of Splicing Artifacts in Partially Fake Speech Signals." ASVspoof Workshop 2024. arXiv:2408.13784. *(6.16 % / 7.36 % EER on PartialSpoof/HAD with no trained detector.)*
Huang, W.-C., Kuo, C.-C., Chen, Y.-F., Yang, Y.-C., Yang, C.-H., Tsao, Y., Wang, H.-M., Lee, H., Fu, S.-W. "Detecting the Undetectable: Assessing the Efficacy of Current Spoof Detection Methods Against Seamless Speech Edits." SLT 2024. arXiv:2501.03805. *(Perceptual-vs-detection asymmetry for infilling.)*

**Adjacent / cross-reference datasets.**
Cai, Z., et al. "Do You Really Mean That? / Glitch in the Matrix (LAV-DF)." DICTA 2022 / CVIU 236:103818 (2023).
Cai, Z., et al. "AV-Deepfake1M." ACM MM 2024. arXiv:2311.15308.
Cai, Z., et al. "AV-Deepfake1M++." arXiv:2507.20579.
Zhang, C., Sim, K.C. "Localizing Fake Segments in Speech." ICPR 2022. doi:10.1109/ICPR56361.2022.9956134. *(Psynd.)*
Yi, J., et al. "ADD 2022: the First Audio Deep Synthesis Detection Challenge." ICASSP 2022. arXiv:2202.08433.
Yi, J., et al. "ADD 2023." arXiv:2305.13774.
Zhang, Y., Tian, X., Zhang, L., Duan, Z. "PartialEdit: Identifying Partial Deepfakes in the Era of Neural Speech Editing." Interspeech 2025.
Yan, et al. "Speech-Forensics." IJCAI 2024. arXiv:2412.09032.

**Time-scale modification / perceptual quality.**
Roberts, T. "A Time-Scale Modification Dataset with Subjective Quality Labels." arXiv:2006.00848; IEEE DataPort.
Roberts, T., Paliwal, K. "An Objective Measure of Quality for Time-Scale Modification of Audio." arXiv:2006.06153.
Roberts, T., Nicolson, A., Paliwal, K. "Deep Learning-Based Single-Ended Objective Quality Measures for Time-Scale Modified Audio." arXiv:2009.02940.

**Foundations (concatenative synthesis, GCI, PSOLA).**
Moulines, E., Charpentier, F. "Pitch-synchronous waveform processing techniques for text-to-speech synthesis using diphones." Speech Communication 9(5-6):453–467 (1990).
Stylianou, Y. "Applying the harmonic plus noise model in concatenative speech synthesis." IEEE TSAP 9(1):21–29 (2001).
Naylor, P., Kounoudes, A., Gudnason, J., Brookes, M. "Estimation of Glottal Closure Instants in Voiced Speech Using the DYPSA Algorithm." IEEE TASLP 15(1):34–43 (2007).
Drugman, T., Thomas, M., Gudnason, J., Naylor, P., Dutoit, T. "Detection of Glottal Closure Instants from Speech Signals: A Quantitative Review." arXiv:2001.00473.
Taylor, P. *Text-to-Speech Synthesis.* Cambridge University Press, 2009.

**Surveys and supporting work.**
Jung, J., et al. "AASIST: Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks." ICASSP 2022. arXiv:2110.01200.
Zhang, Y., Yi, J., Tao, J., Wang, C., Zhang, X., Zhao, Y. "Audio Deepfake Detection: A Survey." arXiv:2308.14970.
Li, M., Yi, J., Tao, J., Wang, C., Zhang, X., Zhao, Y. "Audio Anti-Spoofing Detection: A Survey." arXiv:2404.13914.
Cooper, E., Huang, W.-C., Tsao, Y., Wang, H.-M., Toda, T., Yamagishi, J. "A review on subjective and objective evaluation of synthetic speech." Acoust. Sci. & Tech. 45(4) (2024).