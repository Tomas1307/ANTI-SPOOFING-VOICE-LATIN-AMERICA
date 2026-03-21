# Research Notes - Tutor Requested Tasks

Date: 2026-03-21

---

## 1. Parakeet TDT 0.6b-v3 for Spanish Validation

**Source**: https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3

### Key Finding: This model SUPPORTS SPANISH and resolves our Step 4 tech debt.

| Property | parakeet-tdt-1.1b (current) | parakeet-tdt-0.6b-v3 (replacement) |
|---|---|---|
| Parameters | 1.1B | 0.6B |
| Languages | English ONLY | 25 European languages incl. Spanish |
| Tokenizer vocab | 1,024 tokens | 8,192 tokens (unified SentencePiece) |
| LibriSpeech clean WER | 1.39% | 1.93% |
| Spanish WER (FLEURS) | N/A | **3.45%** |
| Spanish WER (MLS) | N/A | **4.39%** |
| Spanish WER (CoVoST2) | N/A | **3.41%** |
| Language detection | N/A | Automatic (no parameter needed) |
| License | CC-BY-4.0 | CC-BY-4.0 |

### Spanish Benchmarks

- FLEURS: 3.45% WER (one of the best-performing languages in the model)
- MLS: 4.39% WER
- CoVoST2: 3.41% WER
- Multilingual average: 11.97% WER (Spanish is well below average)

### Architecture

- FastConformer encoder with 8x depthwise-separable convolutional downsampling
- TDT (Token-and-Duration Transducer) decoder: decouples token identity from duration prediction
- Unified SentencePiece tokenizer with 8,192 tokens for 25 languages
- Input: 16kHz mono WAV/FLAC
- Output: Text with punctuation and capitalization

### Usage Code (for Step 4 replacement)

```python
import nemo.collections.asr as nemo_asr

asr_model = nemo_asr.models.ASRModel.from_pretrained(
    model_name="nvidia/parakeet-tdt-0.6b-v3"
)

output = asr_model.transcribe(["path/to/audio.wav"])
transcript = output[0].text
```

### Caveats for Our Thesis

1. **Latin American Spanish**: Benchmarks are on European/neutral Spanish corpora. HABLA dataset has Colombian, Mexican, Argentinian, Chilean, Peruvian, Puerto Rican, and Venezuelan accents. WER may be higher on these dialects.
2. **No language forcing**: Unlike Whisper (language="es"), Parakeet auto-detects language. If it misclassifies an accent, there is no override.
3. **No CER published**: Only WER figures available. We compute CER ourselves.
4. **VRAM**: ~3-4 GB on GPU. Trivial on A40 (46GB).

### Action Items

- Replace `nvidia/parakeet-tdt-1.1b` with `nvidia/parakeet-tdt-0.6b-v3` in all pipeline settings
- pip install in each env: `pip install -U "nemo_toolkit[asr]"`
- Run empirical validation on HABLA bonafide samples to measure actual WER on Latin American accents
- Re-enable Step 4 across all pipelines

---

## 2. WER / CER Metrics for Validation

### Definitions

- **WER (Word Error Rate)**: Measures the edit distance at the word level between the reference transcript and the ASR hypothesis.
  - Formula: `WER = (S + D + I) / N`
  - S = substitutions, D = deletions, I = insertions, N = total words in reference
  - Perfect transcription: WER = 0%

- **CER (Character Error Rate)**: Same as WER but at the character level.
  - Formula: `CER = (S + D + I) / C`
  - C = total characters in reference
  - More granular than WER; a single misspelled word has lower CER than WER impact

### Why Both Matter

- WER captures semantic accuracy (did it get the right words?)
- CER captures phonetic accuracy (did it get close even if not exact?)
- For anti-spoofing validation: if WER=0% and CER=0%, the TTS system perfectly reproduced the intended text, meaning the synthetic sample is a valid attack

### Our Target

- **Ideal**: WER=0%, CER=0% (perfect transcription match)
- **Maximum acceptable**: WER <= 15%, CER <= 10% (current settings)
- Samples exceeding thresholds are rejected as failed generations

### Post-Processing for Word Matching

Before computing WER/CER, standard normalization should be applied:
- Lowercase both reference and hypothesis
- Remove punctuation (periods, commas, question marks, etc.)
- Normalize Unicode (accented characters: e.g., a vs a)
- Collapse multiple whitespace to single space
- Optionally: number-to-word conversion (e.g., "3" -> "tres")

Library: `jiwer` (already installed in all envs)

```python
from jiwer import wer, cer
import unicodedata
import re

def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = unicodedata.normalize("NFKD", text)
    text = re.sub(r"[^\w\s]", "", text)  # remove punctuation
    text = re.sub(r"\s+", " ", text)      # collapse whitespace
    return text

ref = normalize_text(reference_text)
hyp = normalize_text(asr_transcript)
word_error_rate = wer(ref, hyp)
char_error_rate = cer(ref, hyp)
```

---

## 3. MDPI Paper: "Enhancing Voice Cloning Quality through Data Selection and Alignment-Based Metrics"

**Source**: https://www.mdpi.com/2076-3417/13/14/8049
**Authors**: Ander Gonzalez-Docasal, Aitor Alvarez (Fundacion Vicomtech, BRTA)
**Year**: 2023, Applied Sciences, vol. 13, no. 14
**PDF**: `docs/applsci-13-08049.pdf`

### Paper Overview

The paper investigates the impact of data selection techniques on voice cloning quality using Tacotron-2 TTS. They train 20 models across 3 corpora (XRey=Spanish dictator Franco's voice, Tux=Spanish LibriVox, Hi-Fi TTS 92=English LibriVox) and evaluate using objective metrics during training (not just at the end).

Key contribution: a novel algorithm that calculates the **fraction of aligned input characters** by exploiting the Tacotron-2 attention matrix diagonal.

### Section 5.2 - Quality Measurement (Metrics Definition)

The paper uses two categories of objective evaluation metrics:

#### A. MOS Estimation Models (Speech Quality, Section 5.2.1)

1. **NISQA** (Non-Intrusive Speech Quality Assessment) [Mittag et al., Interspeech 2021]
   - Deep CNN-Self-Attention model for multidimensional speech quality prediction
   - Range: 1.0 - 5.0 (higher = better)
   - Non-intrusive: does NOT require a reference signal
   - Trained on crowdsourced datasets
   - GitHub: `gabrielmittag/NISQA`

2. **MOSnet** [Lo et al., Interspeech 2019]
   - Deep learning-based objective assessment originally designed for voice conversion
   - Range: 1.0 - 5.0
   - Non-intrusive: does NOT require a reference signal
   - GitHub: `lochenchou/MOSNet`

#### B. Alignment-Based Metrics (Intelligibility, Section 5.2.2)

3. **Fraction of Aligned Sentences (MFA-based)**
   - Uses Montreal Forced Aligner (MFA) to match generated audio with input text
   - Binary: a sentence is either successfully aligned or not
   - Limitation: reaches ~100% very early in training, so it is NOT suitable for comparing model quality over time
   - Useful only as a coarse filter (reject completely unintelligible outputs)

4. **Fraction of Aligned Characters (Novel Algorithm)**
   - Novel metric introduced in this paper (Algorithm 1)
   - Exploits the attention matrix A of Tacotron-2 (dimensions E x D, where E=input length, D=spectrogram length)
   - Checks whether the attention matrix shows a diagonal pattern (indicator of successful synthesis)
   - Uses sliding rectangular windows (w=150 spectrogram frames, h=8 characters) moving along the diagonal
   - A character i is "aligned" if any attention score a_ij > threshold (theta=0.7) inside the current window
   - Window slides to the position of the last aligned character after each step
   - Margin of 1/3 w to the left for search flexibility
   - Stops when: no aligned character found, exceeds input length, or exceeds spectrogram length
   - Rarely reaches 100% (spaces, punctuation don't directly map to audio frames)
   - Shows rising trend during training (unlike aligned sentences which saturate early)
   - **Key advantage over aligned sentences**: provides granular, continuous metric that differentiates model quality during training

### Section 6 - Evaluation Results and Discussion (COMPLETE)

20 voice cloning models were trained: 3 speakers (XRey, Tux, Hi-Fi TTS 92), each with complete or 3h partition, each with/without ablations (SNR filtering, utterance speed filtering, both).

Evaluation set: 207 Spanish sentences (from Tux "other" subset), 221 English sentences (from Hi-Fi TTS 92 long utterance substrings). Metrics computed every 5,000 training steps.

#### 6.1 XRey (Low-Quality Spanish Corpus, ~3h, Fine-Tuned)

| Metric | All Data | SNR >= 20dB | Utt. Speed Filtered | SNR + Speed |
|---|---|---|---|---|
| NISQA (final) | ~2.8 | ~3.0 | ~3.1 | ~3.3 |
| MOSnet (final) | ~2.9 | ~3.1 | ~3.2 | ~3.4 |
| Aligned Chars | ~0.4 | ~0.6 | ~0.7 | ~0.8 |
| Aligned Sents | ~1.0 | ~1.0 | ~1.0 | ~1.0 |

Key findings:
- **Substantial improvement across ALL metrics** when removing high-variability audio files
- Even the harshest ablation (removing 50% of data) showed better results than using all data
- MOSnet was more generous than NISQA for this low-quality speaker (opposite of HQ corpora)
- Aligned sentences reached ~100% very early = not useful for quality comparison
- Aligned characters showed steady improvement during training = useful continuous metric
- **Conclusion**: For low-quality data, reducing variability beats having more data

#### 6.2 HQ Speakers Trained on 3h Partitions (Fine-Tuned)

| Metric | Tux (Spanish) | Hi-Fi TTS 92 (English) |
|---|---|---|
| NISQA range | 3.0 - 3.5 | 3.0 - 3.5 |
| MOSnet range | 2.3 - 3.2 | 2.3 - 3.2 |
| Aligned Chars (trend) | Slight improvement with ablations | Slight improvement with ablations |

Key findings:
- **No significant MOS change** from ablations on HQ data (already clean enough)
- NISQA more generous than MOSnet for HQ speakers (opposite of XRey)
- Aligned sentences again saturated at ~100% early = not useful
- Removing both low-SNR + variable-speed files had slightly better impact than SNR-only
- **Conclusion**: For HQ data, ablations don't hurt but don't dramatically help either

#### 6.3 HQ Speakers Trained on Whole Corpora (From Scratch)

| Metric | Tux (53h Spanish) | Hi-Fi TTS 92 (27h English) |
|---|---|---|
| NISQA (final) | ~3.0 | ~3.3 |
| MOSnet (final) | ~2.5 | ~3.1 |
| Aligned Chars | Improved with ablations | **Hurt** by ablations |

Key findings:
- MOS values at advanced training stages were **remarkably similar** between 3h fine-tuned and full-corpus from-scratch models
- **Critical insight**: 3h fine-tuned from pre-trained model achieves comparable quality to 27-53h from scratch
- For Tux (53h): ablations improved aligned characters even from scratch
- For Hi-Fi TTS 92 (27h): ablations **hurt** aligned characters (removing data from smaller corpus was detrimental)
- **Conclusion**: Data volume matters more for from-scratch training; pre-training compensates for limited data

### Section 7 - Key Conclusions

1. **Data quality > data quantity** for low-quality corpora: removing noisy/variable data improves cloning quality even when 50% of data is discarded
2. **3h fine-tuned ~ full-corpus from scratch**: A 3h corpus fine-tuned from a pre-trained model (even cross-language) achieves comparable quality to much larger from-scratch training
3. **Aligned characters > aligned sentences** as a training monitoring metric: sentences saturate early; characters provide continuous, informative signal
4. **NISQA vs MOSnet**: NISQA tends to be more generous for HQ speakers; MOSnet more generous for low-quality speakers. Using both provides complementary views.
5. **SNR + utterance speed filtering combined** yields the best results on low-quality data

### Relevance to Our Thesis

| Aspect | This Paper | Our Work |
|---|---|---|
| TTS system | Tacotron-2 (attention-based) | Fish Speech, Chatterbox, OpenVoice, Qwen3-TTS (modern flow-matching/autoregressive) |
| Voice cloning approach | Fine-tune entire TTS on target speaker | Zero-shot cloning (reference audio only, no fine-tuning) |
| Alignment metric | Tacotron-2 attention matrix | Not directly applicable (our TTS systems don't expose attention matrices) |
| MOS estimation | NISQA + MOSnet | Could adopt NISQA for quality scoring (Step 5 enhancement) |
| Language | Spanish (XRey, Tux) + English | Spanish (Latin American, 7 accents) |
| Data selection | SNR + utterance speed filtering | We select reference audio by duration; text by word count |
| Evaluation | During training (checkpoints) | Post-generation (WER, CER, speaker similarity) |

**Actionable takeaways**:
- **NISQA** is a strong candidate for our Step 5 quality metrics (replace hardcoded DNSMOS placeholder)
- The paper validates that **objective MOS estimators** (NISQA, MOSnet) correlate with quality improvements, supporting their use as automated evaluation metrics
- Their alignment metric is Tacotron-2 specific and not applicable to our zero-shot TTS systems, but the principle (measuring text-audio alignment) maps to our WER/CER approach
- Their finding that **data quality matters more than quantity** supports our careful reference audio selection in Step 1

### Additional Metrics Common in Voice Cloning Evaluation (from literature)

- **Speaker Similarity (SIM)**: Cosine similarity between speaker embeddings of reference and cloned audio. Computed using speaker verification models (ECAPA-TDNN, WavLM, etc.). Typical threshold: > 0.7 for acceptable cloning.
- **PESQ (Perceptual Evaluation of Speech Quality)**: ITU-T P.862. Compares degraded audio to clean reference. Range: -0.5 to 4.5.
- **STOI (Short-Time Objective Intelligibility)**: Measures intelligibility. Range: 0-1.
- **F0 RMSE**: Root mean square error of fundamental frequency contour between reference and synthetic. Lower = better prosody matching.

---

## 4. LRLSpoof: Low-Resource Language Spoofing Corpus

**Paper**: "When Spoof Detectors Travel: Evaluation Across 66 Languages in the Low-Resource Language Spoofing Corpus"
**Authors**: Kirill Borodin, Vasiliy Kudryavtsev, Maxim Maslov, Mikhail Gorodnichev, Grach Mkrtchian
**Source**: https://arxiv.org/abs/2603.02364
**Venue**: Submitted to Interspeech 2026
**License**: CC-BY-4.0

### Dataset Overview

- **2,732 hours** of synthetic speech
- **66 languages** (45 low-resource)
- **24 TTS systems** (classical to generative)
- **Only synthetic audio** (no bonafide - uses threshold transfer for evaluation)
- Available on HuggingFace and ModelScope

### TTS Systems Used (24 total)

| Category | Systems |
|---|---|
| Classical | eSpeak NG, RHVoice, AhoTTS |
| Neural Supervised | Silero, SpeechT5, FastPitch, Matcha-TTS, Parler-TTS, Piper, MeloTTS |
| Multilingual/Low-Resource | MMS-TTS, Indic-TTS, TurkicTTS, IMS Toucan, QirimtatarTTS |
| Generative/Voice Cloning | XTTS, XTTS2, OuteTTS, **Chatterbox**, **F5-TTS**, CosyVoice, Zonos, **Fish-Speech**, Kokoro |

### Spanish-Specific Data

- **23.14 hours** of synthetic speech
- **11 TTS systems** used for Spanish generation

### Spanish Spoof Rejection Rates (SRR) by Countermeasure

| Countermeasure | Spanish SRR |
|---|---|
| w2v2_300 | **100.00%** |
| aasist3 | **99.83%** |
| df_arena_500 | **99.99%** |
| rescapsguard | **99.98%** |
| df_arena_1b | 31.64% |
| res2tcn | 26.28% |
| tcm_add | 2.96% |
| sls | 1.36% |
| ssl_aasist | 0.61% |
| w2v2_1b | 0.04% |
| nes2net | 0.00% |

### Key Findings for Our Thesis

1. **Massive cross-model disparity**: Some detectors (w2v2_300, aasist3) achieve near-perfect Spanish detection while others (nes2net, w2v2_1b) completely fail. This proves that detector choice matters enormously.

2. **Language as domain shift**: Even well-performing English detectors can fail catastrophically on other languages, including Spanish. This supports our thesis motivation (Latin American Spanish is underserved).

3. **Overlap with our TTS systems**: LRLSpoof uses **Fish-Speech** and **Chatterbox** (2 of our 4 attack systems). They also use MeloTTS (used by our OpenVoice pipeline). Direct comparison is possible.

4. **Missing from LRLSpoof**: They do NOT use Qwen3-TTS (our 4th attack). Our work adds this as a novel contribution.

5. **No Latin American focus**: LRLSpoof treats Spanish as a single language. Our thesis differentiates 7 Latin American accents.

### Comparison: Our HABLA Dataset vs LRLSpoof Spanish

| Aspect | HABLA (ours) | LRLSpoof |
|---|---|---|
| Spanish hours | TBD (162 speakers) | 23.14 hours |
| Accents | 7 Latin American | Undifferentiated |
| TTS systems | 4 (Fish, Qwen, OpenVoice, Chatterbox) | 11 |
| Bonafide data | Yes (real speakers) | No (synthetic only) |
| Evaluation | Full pipeline (WER, CER, speaker sim) | Threshold transfer (SRR) |
| Focus | Voice cloning attacks | General TTS spoofing |

---

## 5. SpeechFake-MD (Multilingual Dataset)

**Paper**: "SpeechFake: A Large-Scale Multilingual Speech Deepfake Dataset Incorporating Cutting-Edge Generation Methods"
**Authors**: Wen Huang, Yanmei Gu, Zhiming Wang, Huijia Zhu, Yanmin Qian
**Source**: https://arxiv.org/abs/2507.21463
**Venue**: ACL 2025 (Long Papers)
**GitHub**: YMLLG/SpeechFake

### Dataset Overview

- **3+ million deepfake samples**, 3,000+ hours of audio
- **40 different speech synthesis tools** (TTS, VC, neural vocoder)
- **46 languages** (multilingual portion)
- Two parts: BD (Bilingual: English + Chinese) and **MD (Multilingual: 46 languages)**

### Structure

- BD (Bilingual Dataset): English + Chinese, used for training + evaluation
- MD (Multilingual Dataset): 46 languages, evaluation-focused
  - 9 primary languages with larger volumes: en, zh, **es**, fr, hi, ja, ko, fa, it
  - 37 additional languages with smaller volumes
  - ~5,000 clips per language in test sets

### Multilingual TTS Systems

- **6 multilingual speech generation tools** for MD portion
- EdgeTTS covers the widest range of languages
- Others cover subsets based on capabilities

### Bonafide Data Source

- Real speech from **Mozilla CommonVoice** (same as our pipeline)

### Spanish Detection Results

Using W2V+AASIST (50 epochs):
- Spanish EER: **0.12-0.42%** (moderate performance range)
- English EER: 0.15% (best)
- Hindi EER: 0.98% (worst)

### Key Finding

"Language content does affect detection performance, even when the generation methods are seen during training. However, prior exposure to a language through multilingual pretraining can help mitigate this effect to some extent."

### Relevance to Our Thesis

1. SpeechFake uses CommonVoice for bonafide (same as our text source, though we use HABLA for bonafide audio)
2. Spanish is a primary language in their dataset, enabling direct comparison
3. Their 40 TTS tools provide broader coverage, but our 4 systems are more targeted for voice cloning specifically
4. Their low Spanish EER (0.12-0.42%) suggests Spanish spoofing is detectable with proper multilingual training

---

## 6. Action Items Summary

### Immediate (code changes)

1. Replace Parakeet TDT 1.1b with 0.6b-v3 in all pipeline settings files
2. Re-enable Step 4 validation across all 4 pipelines
3. Add text normalization post-processing before WER/CER computation
4. Run empirical Parakeet 0.6b-v3 validation on HABLA bonafide samples

### Research (for thesis writing)

1. Complete MDPI Section 6 analysis (need manual paper access)
2. Compare our Spanish results against LRLSpoof's 23.14 hours
3. Compare our attack systems against SpeechFake-MD's 6 multilingual tools
4. Highlight our novel contributions: Qwen3-TTS attacks, Latin American accent focus, HABLA bonafide data
