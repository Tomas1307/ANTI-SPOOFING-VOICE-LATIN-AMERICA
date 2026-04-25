# Change Log

Chronological record of wiki updates and research discoveries.

---

- **2026-04-25** — Wiki created. Initial structure: 5 state-of-art, 5 methodology, 3 experiments, 1 decisions page.
- **2026-04-25** — Ingested `investigation.md` partial spoof section into wiki. Added 7 splice technique pages (13c–13j) to presentation.
- **2026-04-25** — Key discovery: all 7 crossfade techniques sound identical in listening tests. The real problem is word boundary alignment (TTS generates fluid speech, no energy dip at "Dame"→"únicamente") and duration mismatch (480ms clone in 640ms slot shifts rhythm 160ms).
- **2026-04-25** — Implemented valley-score word selection: `score = min_rms / avg_rms` in ±100ms window, threshold 0.65. Words with no energy valley (score > 0.65) are ineligible. "el" (idx 2) scores 0.165 — deep valley. "únicamente" scores 0.700 — fluid boundary, rejected.
- **2026-04-25** — Implemented duration-preserving splice: time-stretch cloned word to fit exact bonafide slot, overwrite `result[b_start:b_end]` in-place. Total audio length never changes.
- **2026-04-25** — Implemented clone similarity gate: ECAPA-TDNN cosine SIM ≥ 0.60 between bonafide and clone. Runs between Step 2 and Step 3. Known rates: FishGram avg 0.602 (~50% pass at 0.60), Qwen avg 0.720 (most pass), OpenVoice avg 0.394 (most fail).
- **2026-04-25** — Production status updated: FishGram DONE (95.2%), Qwen DONE (87.9%), OpenVoice DONE (83.4%), Chatterbox 41% running (ETA May 13), OuteTTS 66% running (ETA May 1).
- **2026-04-25** — Added slides 06b (Chatterbox progress), 06c (OuteTTS progress), 13a–13l (splice quality problem + solution). Total: 34 slides.
- **2026-04-25** — v2 pipeline pending validation on ml-server03: 5 speakers (arf_00295, arf_00610, arf_01523, arm_00412, arm_00780), ≥10 audios each, using fishgram_env + GPU 1.
