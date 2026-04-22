We're continuing work on the partial spoof pipeline for HABLA 2.0 (Latin American Spanish anti-spoofing thesis). Last session we:

1. **Validated partial spoof end-to-end** on ml-server03 (7/7 passed, NISQA=4.72, SIM=0.789)
2. **Fixed the splice engine** with text-matching (not positional), Spanish accent normalization (NFKD), smart retry with regeneration loop
3. **Found a crossfade bug**: the 20ms crossfade was EATING 40ms from the cloned word, truncating it ("nicament" instead of "unicamente")
4. **Researched SOTA splicing**: 6 papers (PartialSpoof, HAD, LlamaPartialSpoof, HQ-MPSD, Analyzing Artifacts, survey). Details in `investigation.md` Section 8.

**What needs to be done now:**

A. **Implement 7 splicing techniques** in `app/pipeline/partial_spoof/utils/crossfade.py` and `splice_engine.py`:
   - Direct cut-paste (10%), OLA Hanning (20%), Linear fade (15%), Cosine fade (20%), Half-sine (15%), Log fade (10%), Inverted parabola (10%)
   - Random overlap 30-80ms per splice, zero-crossing alignment, RMS energy normalization
   - The cloned word must be inserted at FULL natural duration (no compression/truncation)
   - The crossfade margin must come from OUTSIDE the word boundaries (gap between adjacent cloned words), NOT eat the word itself

B. **Fix the margin issue**: when adjacent cloned words have 0ms gap, the crossfade cannot grab margin without eating into the neighboring word. Need a fallback strategy.

C. **Test locally** using `app/tests/test_splice_real_audio.py` which uses real bonafide+cloned audio from `data/attacks/qwen_partial_spoof/`

D. **Check production runs**: Chatterbox and OuteTTS should be done or close to done on ml-server03 GPU 2.

E. **Deep research results**: I may have results from a Claude deep research query about splicing techniques. If so, incorporate findings.

Key files:
- `app/pipeline/partial_spoof/utils/splice_engine.py` — core splicing logic
- `app/pipeline/partial_spoof/utils/crossfade.py` — fade functions
- `app/pipeline/partial_spoof/settings.py` — CROSSFADE_MS parameter
- `investigation.md` Section 8 — literature review with all 7 techniques
- `app/tests/test_splice_real_audio.py` — local test with real audio
- `splice_debug.html` — waveform visualization tool

Production status:
- FishGram: DONE (34,197 passed, 95.2%)
- Qwen3-TTS: DONE (31,568 passed, 87.9%)
- OpenVoice: DONE (29,626 passed, 83.4%)
- Chatterbox: RUNNING on GPU 2 (~40hrs)
- OuteTTS: RUNNING on GPU 2 (~12-15 days, slow RTF)
- CosyVoice: DROPPED (no Spanish)
- Partial Spoof: TESTING — splice algorithm being improved

Presentation: component-based in `presentation/slides/`, run `python presentation/build.py` to assemble.
