# Mozilla Common Voice Speaker Selection Pipeline

This pipeline selects 20,000 acoustically diverse speakers from Mozilla Common Voice Spanish dataset that are **distinct from HABLA dataset speakers**, using ECAPA-TDNN speaker embeddings and cosine similarity filtering.

## Overview

The pipeline ensures speaker-independent data augmentation by:
1. Extracting ECAPA-TDNN embeddings from 162 HABLA speakers
2. Extracting embeddings from ~3,000 Common Voice Spanish speakers (LatAm + Spain)
3. Filtering CV speakers by cosine similarity threshold (0.75)
4. Stratified sampling of 20K samples balanced by accent, gender, and age

## Requirements

- Python 3.11+
- CUDA-capable GPU (recommended)
- ~100 GB disk space for extracted Common Voice data
- Dependencies:
  ```bash
  pip install torch torchaudio speechbrain numpy tqdm
  ```

## Input Data

- `data/cv-corpus-24.0-2025-12-05-es.tar.gz` (Common Voice archive)
- `data/cv_metadata/validated.tsv` (436K validated samples)
- `data/bonafide_dataset_by_speaker/` (162 HABLA speakers)

## Pipeline Steps

### Step 1: Extract HABLA Embeddings
```bash
python -m pipeline.select_mozilla_speakers.01_extract_habla_embeddings
```
- Loads ECAPA-TDNN model (SpeechBrain `spkrec-ecapa-voxceleb`)
- Processes 162 HABLA speakers from `data/bonafide_dataset_by_speaker/`
- Averages embeddings across training samples per speaker
- **Output:**
  - `data/mozilla_speaker_selection/habla_embeddings.npy` (162, 512)
  - `data/mozilla_speaker_selection/habla_speaker_ids.json`

### Step 2: Extract Common Voice Embeddings
```bash
python -m pipeline.select_mozilla_speakers.02_extract_cv_embeddings
```
- Extracts `data/cv-corpus-24.0-2025-12-05-es.tar.gz` → `data/cv_extracted/`
- Filters validated.tsv by accent (LatAm + Spain only, excludes "Not Provided")
- Groups samples by speaker (`client_id`)
- Extracts one representative ECAPA-TDNN embedding per speaker
- **Output:**
  - `data/mozilla_speaker_selection/cv_speaker_embeddings.npy` (N, 512)
  - `data/mozilla_speaker_selection/cv_speaker_metadata.json`

### Step 3: Filter by Similarity
```bash
python -m pipeline.select_mozilla_speakers.03_filter_by_similarity
```
- Computes cosine similarity: `CV_embeddings @ HABLA_embeddings.T`
- For each CV speaker, finds max similarity to any HABLA speaker
- **Filters out speakers with `max_similarity >= 0.75`**
- **Output:**
  - `data/mozilla_speaker_selection/filtered_speakers.json`

### Step 4: Balanced Sampling
```bash
python -m pipeline.select_mozilla_speakers.04_balanced_sampling
```
- Loads filtered speaker pool
- Stratifies by: **accent** (Mexico, Andino-Pacifico, Spain, etc.), **gender**, **age**
- Samples 20,000 samples with proportional representation
- **Output:**
  - `data/mozilla_speaker_selection/selected_20k.tsv` (subset of validated.tsv)
  - `data/mozilla_speaker_selection/selection_stats.json` (demographics)

## Running the Full Pipeline

### Option A: Master Script (Recommended)
```bash
python pipeline/select_mozilla_speakers/run_pipeline.py
```

This runs all 4 steps sequentially with timing and error handling.

### Option B: Step-by-Step
```bash
python pipeline/select_mozilla_speakers/01_extract_habla_embeddings.py
python pipeline/select_mozilla_speakers/02_extract_cv_embeddings.py
python pipeline/select_mozilla_speakers/03_filter_by_similarity.py
python pipeline/select_mozilla_speakers/04_balanced_sampling.py
```

## Output Files

```
data/mozilla_speaker_selection/
├── habla_embeddings.npy           # (162, 512) HABLA embeddings
├── habla_speaker_ids.json         # List of 162 HABLA speaker IDs
├── cv_speaker_embeddings.npy      # (N, 512) CV embeddings
├── cv_speaker_metadata.json       # CV speaker metadata (accent, gender, age)
├── filtered_speakers.json         # CV speakers passing similarity filter
├── selected_20k.tsv               # Final 20K sample selection
└── selection_stats.json           # Demographics breakdown
```

## Configuration

- **Similarity threshold:** 0.75 (edit in `03_filter_by_similarity.py`)
- **Target samples:** 20,000 (edit in `04_balanced_sampling.py`)
- **Random seed:** 42 (for reproducibility)

## Accent Categories

The pipeline normalizes Common Voice accents into these categories:
- **Mexico**
- **Andino-Pacifico** (Colombia, Peru, Ecuador)
- **Rioplatense** (Argentina, Uruguay, Paraguay)
- **Caribe** (Cuba, Venezuela, Puerto Rico, Dominican Republic)
- **Chile**
- **America Central** (Guatemala, Costa Rica)
- **Spain** (all Peninsular Spanish dialects)

Each category is treated equally in stratified sampling.

## Performance Notes

- **Step 1:** ~5-10 minutes (162 speakers, GPU)
- **Step 2:** ~30-60 minutes (~3,000 speakers, GPU)
- **Step 3:** ~1 minute (vectorized similarity computation)
- **Step 4:** ~2 minutes (sampling)

**Total runtime:** ~40-75 minutes on ml-server03 with A40 GPU

## Troubleshooting

### "No such file or directory: clips"
The extracted archive structure may vary. Update `clips_dir` path in `02_extract_cv_embeddings.py` line 139.

### Out of memory
Reduce batch size or use smaller embedding model. ECAPA-TDNN is already lightweight.

### Too few speakers passing filter
Lower similarity threshold in `03_filter_by_similarity.py` (e.g., 0.70 instead of 0.75).

## References

- **ECAPA-TDNN:** Desplanques et al., "ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in TDNN Based Speaker Verification" (Interspeech 2020)
- **SpeechBrain:** Ravanelli et al., "SpeechBrain: A General-Purpose Speech Toolkit" (2021)
- **Mozilla Common Voice:** Ardila et al., "Common Voice: A Massively-Multilingual Speech Corpus" (LREC 2020)
