# Production Runs

**Status:** Active
**Last updated:** 2026-04-25
**Source:** ml-server03 logs, pipeline output metadata

---

## Summary Table

| Pipeline | Status | Samples | Passed | Pass Rate | WER | NISQA | SIM | RTF |
|----------|--------|---------|--------|-----------|-----|-------|-----|-----|
| FishGram | DONE | 35,927 | 34,197 | 95.2% | 2.17% | 4.57 | 0.602 | 2-3x |
| Qwen3-TTS | DONE | 35,927 | 31,568 | 87.9% | 1.46% | 4.37 | 0.720 | 3-5x |
| OpenVoice | DONE | 35,544 | 29,626 | 83.4% | 1.50% | 4.41 | 0.394 | 0.07-0.10x |
| Chatterbox | RUNNING | ~14,818 | TBD | TBD | TBD | TBD | TBD | 31-45x |
| OuteTTS | RUNNING | ~23,561 | TBD | TBD | TBD | TBD | TBD | ~5.6x |
| CosyVoice | DROPPED | - | - | - | - | - | - | - |

**Hardware:** ml-server03, NVIDIA A40 (46GB VRAM), CUDA 12.6
**Bonafide corpus:** HABLA v2, 1,567 speakers, 7 Latin American accents, ~35,927 samples

## Per-Pipeline Notes

### FishGram (Fish Speech / OpenAudio-S1)
- Best pass rate (95.2%). Highest NISQA (4.57).
- Moderate speaker similarity (0.602) — voice quality is excellent but timbre transfer is imperfect.
- Runs as HTTP API server on a separate port. RTF 2-3x.
- Completed on GPU 1.

### Qwen3-TTS
- Highest speaker similarity (0.720). Lowest WER (1.46%).
- x_vector_only_mode=True required (ref_text mismatch with concatenated reference).
- All sampling params needed: do_sample=True, temperature, top_k, top_p, repetition_penalty.
- Completed on GPU 3.

### OpenVoice V2
- Fastest pipeline (RTF 0.07-0.10x, 10-14x real-time).
- Lowest speaker similarity (0.394) — MeloTTS base voice bleeds through ToneColorConverter.
- Lowest CER (0.45%). Very consistent (lowest std on NISQA).
- Completed on GPU 1.

### Chatterbox (Resemble.ai)
- Started April 12 on GPU 2. Currently at ~14,818/35,927 (41%).
- RTF 31-45x — dramatically slower than all other pipelines.
- GPT-style autoregressive with CFG. EnCodec tokens + Vocos vocoder.
- ETA: ~May 13 (18 more days).

### OuteTTS
- Started April 13 on GPU 2. Currently at ~23,561/35,927 (66%).
- RTF ~5.6x (consistent). Llama 3.1-based, WavTokenizer codes.
- Known issue: PyLoudNorm clipping warnings.
- ETA: ~May 1 (6 more days).

### CosyVoice 3.0
- Dropped. Generates Chinese output for Spanish input text. No Spanish support despite multilingual claims.
