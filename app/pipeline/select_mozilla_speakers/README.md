# Mozilla Common Voice Speaker Selection & Integration Pipeline

**HABLA 2.0 Dataset Augmentation Pipeline**

This pipeline selects 15,340 acoustically diverse samples from Mozilla Common Voice Spanish v24.0 to augment the HABLA anti-spoofing dataset, adding Mexico and Spain accents while boosting existing accents (Colombia, Chile, Venezuela) to achieve balanced representation across 7 Spanish accents.

## Motivation

The original HABLA dataset (Álvarez et al., Interspeech 2023) contains:
- **162 speakers**, 22,816 bonafide samples
- **5 accents**: Argentina, Chile, Colombia, Peru, Venezuela
- **Limitation**: No Mexico or Spain representation, unbalanced accent distribution

**Goal**: Expand to 7 accents with 5,450 samples per accent (38,156 total) while ensuring:
1. **Speaker independence**: CV speakers must be acoustically distinct from HABLA speakers
2. **Accent diversity**: Add Mexico (0→5,450) and Spain (0→5,450)
3. **Balanced representation**: Boost existing accents to 5,450 each

## Speaker Independence Methodology

**Critical Requirement**: CV speakers must be acoustically distinct from HABLA speakers to ensure valid train/test separation and prevent speaker leakage in anti-spoofing evaluations.

### Why Speaker Independence Matters

In anti-spoofing research, **speaker overlap** between training and testing datasets is a critical confound:
- **Overfitting risk**: Models may learn speaker-specific cues rather than spoofing artifacts
- **Inflated performance**: Results don't generalize to unseen speakers
- **Scientific validity**: HABLA serves as a benchmark - augmented data must maintain speaker independence

### Our Approach: ECAPA-TDNN Embedding Similarity

**ECAPA-TDNN** (Emphasized Channel Attention, Propagation and Aggregation in TDNN):
- State-of-the-art speaker verification model
- Extracts 192-dimensional speaker embeddings that capture vocal characteristics
- Trained on VoxCeleb (7,000+ speakers) for robust speaker discrimination

**Pipeline:**
1. Extract **reference embeddings** from all 162 HABLA speakers (averaged across samples)
2. Extract **candidate embeddings** from 3,299 CV speakers (one per speaker)
3. Compute **cosine similarity matrix**: CV @ HABLA.T → (3,299 × 162)
4. For each CV speaker, find **max similarity** to any HABLA speaker
5. **Reject speakers with max_similarity ≥ 0.75**

**Threshold Selection (0.75):**
- **Below 0.70**: Generally considered different speakers in literature
- **0.70-0.80**: Potential similarity, use with caution
- **Above 0.80**: Likely same speaker or close relative
- **Our choice (0.75)**: Conservative threshold ensuring distinct speakers

### Validation Results

**Empirical Findings:**
```
Similarity Statistics (3,299 CV speakers vs 162 HABLA):
  Mean:   0.4067  (low average similarity)
  Median: 0.4179
  Std:    0.0944
  Max:    0.6915  (well below threshold!)

Filter Results:
  Passed:   3,299 (100%)
  Rejected: 0 (0%)
```

**Interpretation:**
- **100% pass rate**: All CV speakers are sufficiently distinct from HABLA
- **Max similarity 0.6915 < 0.75**: Even the most similar CV speaker is below threshold
- **Low mean (0.41)**: On average, CV and HABLA speakers are quite different
- **Conclusion**: No speaker overlap risk between datasets

**Why This Worked:**
1. **Different recording conditions**: HABLA (controlled) vs CV (crowd-sourced)
2. **Geographic diversity**: HABLA has Argentina/Peru, CV adds Mexico/Spain
3. **Large speaker pool**: CV has 3,299 candidates, easy to find distinct speakers
4. **Robust embeddings**: ECAPA-TDNN captures speaker identity effectively

### Comparison to Alternative Approaches

| Method | Our Approach | Alternatives |
|--------|--------------|--------------|
| **ID Matching** | ❌ Can't use (different ID systems) | HABLA uses custom IDs, CV uses hashed client_ids |
| **Metadata Matching** | ❌ Insufficient (name, age not available) | CV anonymizes speakers, no PII |
| **Acoustic Fingerprinting** | ✅ **ECAPA-TDNN embeddings** | Most reliable for speaker verification |
| **Manual Listening** | ❌ Not scalable (3,299 × 162 = 534K comparisons) | Infeasible for large-scale validation |

### Reproducibility

All similarity computations use:
- **Model**: `speechbrain/spkrec-ecapa-voxceleb`
- **Embedding dim**: 192
- **Normalization**: L2-normalized embeddings
- **Metric**: Cosine similarity
- **Code**: Available in `03_filter_by_similarity.py`

**To verify:**
```bash
python app/pipeline/select_mozilla_speakers/03_filter_by_similarity.py
# Check output: similarity_matrix max should be < 0.75
```

## Strategy Overview

See [`SELECTION_STRATEGY.md`](./SELECTION_STRATEGY.md) for detailed methodology and target calculations.

**Key Approach:**
- **Embedding-based filtering**: ECAPA-TDNN embeddings + cosine similarity (threshold 0.75)
- **Fixed targets per accent**: Exact sample counts to achieve balance
- **Metadata filtering**: Exclude Unknown gender/age, prioritize female speakers for Spain
- **Integration**: Merge into `bonafide_dataset_by_speaker_v2/` with consistent structure

**Target Distribution:**

| Accent | HABLA Existing | CV Addition | Final Total |
|--------|----------------|-------------|-------------|
| Argentina | 4,906 | - | 4,906 |
| Chile | 4,089 | +1,361 | 5,450 |
| Colombia | 4,604 | +846 | 5,450 |
| Peru | 4,123 | - | 4,123 |
| Venezuela | 3,217 | +2,233* | 5,450 |
| **Mexico** | 0 | **+5,450** | **5,450** |
| **Spain** | 0 | **+5,450** | **5,450** |
| **TOTAL** | **22,816** | **+15,340** | **38,156** |

*Note: Venezuela only has 4 CV samples available, significant shortfall addressed in thesis discussion.

## Requirements

### Hardware
- **GPU**: CUDA-capable (tested on NVIDIA A40, 46GB VRAM)
- **RAM**: 16GB+ recommended
- **Disk**: ~150GB free space for CV archive extraction

### Software
- Python 3.10+
- CUDA 12.1+ (PyTorch compatible)
- Dependencies:
  ```bash
  pip install torch==2.1.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
  pip install speechbrain numpy tqdm
  ```

### Input Data
- `data/cv-corpus-24.0-2025-12-05-es.tar.gz` (436K validated samples, ~100GB)
- `data/bonafide_dataset_by_speaker/` (162 HABLA speakers)

## Pipeline Steps

### Step 1: Extract HABLA Speaker Embeddings

**Script:** `01_extract_habla_embeddings.py`

**Purpose:** Create reference embeddings for similarity filtering

**Process:**
1. Load ECAPA-TDNN model (SpeechBrain `spkrec-ecapa-voxceleb`)
2. For each of 162 HABLA speakers:
   - Load up to 20 training samples
   - Extract 192-dimensional embeddings
   - Average and L2-normalize per speaker
3. Save reference embeddings

**Output:**
- `data/mozilla_speaker_selection/habla_embeddings.npy` (162, 192)
- `data/mozilla_speaker_selection/habla_speaker_ids.json`

**Runtime:** ~3-5 minutes (GPU)

**Command:**
```bash
python app/pipeline/select_mozilla_speakers/01_extract_habla_embeddings.py
```

### Step 2: Extract Common Voice Speaker Embeddings

**Script:** `02_extract_cv_embeddings.py`

**Purpose:** Process CV archive and extract embeddings for candidate speakers

**Process:**
1. Auto-extract `cv-corpus-24.0-2025-12-05-es.tar.gz` (if not already extracted)
2. Parse `validated.tsv` (436,590 samples)
3. Filter by criteria:
   - **Accents**: Colombia, Chile, Venezuela, Mexico, Spain ONLY
   - **Metadata**: Exclude Unknown gender or age
   - **Result**: 269,948 samples from 3,299 unique speakers
4. Extract one ECAPA-TDNN embedding per speaker
5. Save with `accent_category` field for downstream sampling

**Output:**
- `data/mozilla_speaker_selection/cv_speaker_embeddings.npy` (3,299, 192)
- `data/mozilla_speaker_selection/cv_speaker_metadata.json`

**Samples by Accent:**
- Colombia: 26,501
- Chile: 6,312
- Venezuela: 4 (critical limitation!)
- Mexico: 140,857
- Spain: 96,274

**Runtime:** ~10-15 minutes (GPU) + extraction time (one-time, ~5-10 min)

**Command:**
```bash
python app/pipeline/select_mozilla_speakers/02_extract_cv_embeddings.py
```

### Step 3: Filter by Similarity to HABLA

**Script:** `03_filter_by_similarity.py`

**Purpose:** Ensure speaker independence between CV and HABLA

**Process:**
1. Load HABLA (162, 192) and CV (3,299, 192) embeddings
2. Compute cosine similarity matrix: `CV @ HABLA.T` → (3,299, 162)
3. For each CV speaker, find max similarity to any HABLA speaker
4. **Filter out speakers with max_similarity ≥ 0.75**
5. Save list of acoustically distinct speakers

**Threshold Rationale:**
- 0.75 cosine similarity = strong acoustic similarity
- Ensures CV speakers are sufficiently different from HABLA
- Empirical validation: 100% pass rate (max similarity 0.69)

**Output:**
- `data/mozilla_speaker_selection/filtered_speakers.json` (3,299 speakers passed)

**Similarity Statistics:**
- Mean: 0.4067
- Median: 0.4179
- Max: 0.6915
- Pass rate: 100%

**Runtime:** ~1-2 minutes

**Command:**
```bash
python app/pipeline/select_mozilla_speakers/03_filter_by_similarity.py
```

### Step 4: Balanced Stratified Sampling

**Script:** `04_balanced_sampling.py`

**Purpose:** Select exactly 15,340 samples with fixed targets per accent

**Fixed Targets:**
- Colombia: 846 (to reach 5,450 with HABLA's 4,604)
- Chile: 1,361 (to reach 5,450 with HABLA's 4,089)
- Venezuela: 2,233 (to reach 5,450 with HABLA's 3,217) - **only 4 available**
- Mexico: 5,450 (new accent)
- Spain: 5,450 (new accent, prioritize females for 89% male bias)

**Sampling Strategy:**
- **Within each accent**: Balance by gender and age groups
- **Spain special handling**: Prioritize female speakers (target 50/50 despite 89%/9% source bias)
- **Random seed**: 42 (reproducibility)

**Output:**
- `data/mozilla_speaker_selection/selected_15340.tsv` (TSV subset of validated.tsv)
- `data/mozilla_speaker_selection/selection_stats.json`

**Actual Results:**
- Total samples selected: 13,111 (shortfall due to Venezuela: 4 vs 2,233)
- Unique speakers: 1,405
- Spain: 528 speakers (231 female, 297 male)
- Mexico: 406 speakers (139 female, 267 male)
- Colombia: 326 speakers
- Chile: 144 speakers
- Venezuela: 1 speaker (only 4 samples available)

**Runtime:** ~1-2 minutes

**Command:**
```bash
python app/pipeline/select_mozilla_speakers/04_balanced_sampling.py
```

### Step 5: Integration into HABLA v2 Dataset

**Script:** `05_integrate_cv_samples.py`

**Purpose:** Merge CV speakers into HABLA structure as `bonafide_dataset_by_speaker_v2/`

**Process:**
1. **Copy HABLA speakers**: All 162 speakers → v2 directory
2. **Generate CV speaker IDs**: Continue numbering from existing HABLA
   - Mexico male: `mxm_09698`, `mxm_09699`, ...
   - Mexico female: `mxf_09698`, `mxf_09699`, ...
   - Spain male: `esm_09698`, `esm_09699`, ...
   - Spain female: `esf_09698`, `esf_09699`, ...
   - Colombia male: `com_09698+` (continues from existing 9697)
   - And so on for all accents
3. **Create directory structure**: `{speaker_id}/train/`, `val/`, `test/`
4. **Copy audio files**: From `cv-corpus-24.0-2025-12-05/es/clips/`
5. **Split per speaker**: 70% train / 15% val / 15% test

**Output:**
- `data/bonafide_dataset_by_speaker_v2/` (1,567 speakers total)
  - HABLA: 162 speakers
  - CV: 1,405 speakers
- `data/mozilla_speaker_selection/cv_speaker_mapping.json` (client_id → speaker_id)

**Final Speaker Breakdown:**

| Accent | Male | Female | Total |
|--------|------|--------|-------|
| Argentina | 12 | 30 | 42 |
| Chile | 126 | 47 | 173 |
| Colombia | 270 | 87 | 357 |
| Peru | 20 | 18 | 38 |
| Venezuela | 12 | 11 | 23 |
| **Mexico** | **267** | **139** | **406** |
| **Spain** | **297** | **231** | **528** |
| **TOTAL** | **1,004** | **563** | **1,567** |

**Runtime:** ~30-60 seconds (file copying)

**Command:**
```bash
python app/pipeline/select_mozilla_speakers/05_integrate_cv_samples.py
```

## Running the Full Pipeline

### Option A: Orchestrated Pipeline (Recommended)
```bash
python -m app.pipeline.select_mozilla_speakers.run_pipeline
```

Runs all steps 1-4 sequentially with:
- Step headers and timing
- Error handling and exit on failure
- Strategy summary display
- Progress tracking for each step

**Note**: Step 5 (integration) must be run separately after reviewing Step 4 results.

### Option B: Step-by-Step Execution
```bash
python app/pipeline/select_mozilla_speakers/01_extract_habla_embeddings.py
python app/pipeline/select_mozilla_speakers/02_extract_cv_embeddings.py
python app/pipeline/select_mozilla_speakers/03_filter_by_similarity.py
python app/pipeline/select_mozilla_speakers/04_balanced_sampling.py
python app/pipeline/select_mozilla_speakers/05_integrate_cv_samples.py
```

## Output Directory Structure

```
data/
├── bonafide_dataset_by_speaker/          # Original HABLA (162 speakers)
├── bonafide_dataset_by_speaker_v2/       # HABLA 2.0 (1,567 speakers)
│   ├── arf_00001/                        # Argentina female speaker 1
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   ├── mxm_09698/                        # Mexico male speaker (CV)
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── ...
├── cv-corpus-24.0-2025-12-05/            # Extracted CV archive
│   └── es/
│       ├── clips/                        # Audio files
│       └── validated.tsv                 # Metadata
└── mozilla_speaker_selection/            # Pipeline outputs
    ├── habla_embeddings.npy              # (162, 192)
    ├── habla_speaker_ids.json            # HABLA IDs
    ├── cv_speaker_embeddings.npy         # (3,299, 192)
    ├── cv_speaker_metadata.json          # CV metadata with accent_category
    ├── filtered_speakers.json            # 3,299 speakers passing filter
    ├── selected_15340.tsv                # 13,111 samples (actual)
    ├── selection_stats.json              # Demographics
    └── cv_speaker_mapping.json           # client_id → speaker_id mapping
```

## Configuration Parameters

All configurable values are defined at the top of each script:

**Step 1 (`01_extract_habla_embeddings.py`):**
- `MAX_SAMPLES`: 20 (samples to average per speaker)
- `DEVICE`: "cuda" if available, else "cpu"

**Step 2 (`02_extract_cv_embeddings.py`):**
- Accent keywords for Colombia, Chile, Venezuela, Mexico, Spain
- Exclusion: Unknown gender/age, Argentina/Peru (already in HABLA)

**Step 3 (`03_filter_by_similarity.py`):**
- `SIMILARITY_THRESHOLD`: 0.75

**Step 4 (`04_balanced_sampling.py`):**
```python
ACCENT_TARGETS = {
    "Colombia": 846,
    "Chile": 1361,
    "Venezuela": 2233,
    "Mexico": 5450,
    "Spain": 5450,
}
RANDOM_SEED = 42
```

**Step 5 (`05_integrate_cv_samples.py`):**
```python
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
```

## Performance Benchmarks

Tested on ml-server03 (4x NVIDIA A40, CUDA 12.1, 128GB RAM):

| Step | Description | Runtime |
|------|-------------|---------|
| 1 | HABLA embeddings (162 speakers) | 3-5 min |
| 2 | CV embeddings (3,299 speakers) | 10-15 min |
| 2a | Archive extraction (one-time) | 5-10 min |
| 3 | Similarity filtering | 1-2 min |
| 4 | Balanced sampling | 1-2 min |
| 5 | Integration & file copying | 30-60 sec |
| **Total** | **End-to-end pipeline** | **~20-30 min** |

## Validation & Quality Checks

After running the pipeline, verify:

```bash
# 1. Check speaker count
ls data/bonafide_dataset_by_speaker_v2/ | wc -l
# Expected: 1,567

# 2. Verify accent distribution
ls data/bonafide_dataset_by_speaker_v2/ | cut -c1-2 | sort | uniq -c
# Should see: ar(42), cl(173), co(357), es(528), mx(406), pe(38), ve(23)

# 3. Check directory structure for a CV speaker
ls data/bonafide_dataset_by_speaker_v2/mxm_09698/
# Expected: train/ val/ test/

# 4. Verify audio file counts
find data/bonafide_dataset_by_speaker_v2/mxm_09698/train/ -name "*.mp3" | wc -l
# Should have files

# 5. Review selection statistics
cat data/mozilla_speaker_selection/selection_stats.json
```

## Known Limitations & Discussion Points

### 1. Venezuela Data Scarcity
**Issue**: Only 4 CV samples available vs. target of 2,233
- **Impact**: Final dataset has 23 Venezuela speakers (3,217 HABLA + 4 CV) instead of target 5,450
- **Thesis Discussion**: Explains geographic limitations in open-source speech corpora

### 2. Spain Gender Imbalance
**Source**: 89% male / 9% female in CV Spain samples
- **Mitigation**: Female-priority sampling achieved 231/528 (44%) female representation
- **Trade-off**: Cannot reach 50/50 without severely limiting Spain samples

### 3. Embedding Dimension
- ECAPA-TDNN outputs **192-dim** embeddings (not 512 as in some literature)
- Model: `speechbrain/spkrec-ecapa-voxceleb`
- Verified across all pipeline steps

### 4. Argentina & Peru Not Augmented
- Already well-represented in HABLA (4,906 and 4,123 samples)
- Focus on under-represented accents and new accents (Mexico, Spain)

## Thesis Documentation

This pipeline directly supports the following thesis contributions:

1. **Methodology** (Chapter 3):
   - Section 3.2: Speaker embedding-based similarity filtering
   - Section 3.3: Stratified sampling with fixed targets
   - Section 3.4: Dataset augmentation strategy

2. **Results** (Chapter 4):
   - Table 4.1: HABLA 2.0 speaker demographics
   - Table 4.2: Accent distribution comparison (v1 vs v2)
   - Figure 4.1: Similarity score distribution
   - Figure 4.2: Gender balance across accents

3. **Discussion** (Chapter 5):
   - 5.1: Venezuela data scarcity in open corpora
   - 5.2: Gender imbalance mitigation strategies
   - 5.3: Cross-accent generalization improvements

## References

**Models & Toolkits:**
- Desplanques et al., "ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in TDNN Based Speaker Verification", *Interspeech 2020*
- Ravanelli et al., "SpeechBrain: A General-Purpose Speech Toolkit", *arXiv:2106.04624*, 2021

**Datasets:**
- Álvarez et al., "HABLA: A Dataset for Latin American Spanish Anti-Spoofing Research", *Interspeech 2023*
- Ardila et al., "Common Voice: A Massively-Multilingual Speech Corpus", *LREC 2020*

**Methodology:**
- Snyder et al., "X-vectors: Robust DNN Embeddings for Speaker Recognition", *ICASSP 2018*
- Desplanques et al., "Speaker Embedding for Neural Audio Spoofing and Deepfake Detection", *IEEE/ACM TASLP*, 2024

## Troubleshooting

### Issue: Archive extraction stalls
**Solution**: Extract manually with progress:
```bash
cd data/
tar -xzf cv-corpus-24.0-2025-12-05-es.tar.gz --checkpoint=10000
```

### Issue: Out of GPU memory
**Solution**: Script uses per-sample inference (no batching), should work on 8GB+ GPU
- Verify no other processes using GPU: `nvidia-smi`
- If needed, switch to CPU: Edit `DEVICE = "cpu"` in scripts

### Issue: Cosine similarity all below threshold
**Interpretation**: Good! This means CV and HABLA speakers are acoustically distinct
- Expected max similarity: ~0.65-0.70
- If max > 0.80: Potential speaker overlap, investigate

### Issue: ImportError: No module named 'speechbrain'
**Solution**:
```bash
pip install speechbrain
# If PyTorch version conflict:
pip install torch==2.1.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
pip install speechbrain
```

### Issue: FileNotFoundError for validated.tsv
**Solution**: Path changed after archive extraction
- Check: `ls data/cv-corpus-24.0-2025-12-05/es/`
- Update paths in scripts if structure differs

## Citation

If you use this pipeline or the HABLA 2.0 dataset in your research, please cite:

```bibtex
@mastersthesis{acosta2026habla2,
  title={HABLA 2.0: Expanded Latin American Spanish Anti-Spoofing Dataset with Cross-Accent Generalization},
  author={Acosta, Tom{\'a}s},
  year={2026},
  school={Universidad de los Andes},
  type={Master's Thesis}
}

@inproceedings{alvarez2023habla,
  title={HABLA: A Dataset for Latin American Spanish Anti-Spoofing Research},
  author={{\'A}lvarez, Jaime and others},
  booktitle={Interspeech},
  year={2023}
}
```

## Contact

For questions about this pipeline:
- **Author**: Tomás Acosta
- **Institution**: Universidad de los Andes
- **GitHub Issues**: [ANTI-SPOOFING-VOICE-LATIN-AMERICA/issues](https://github.com/Tomas1307/ANTI-SPOOFING-VOICE-LATIN-AMERICA/issues)

---

**Last Updated**: March 2026
**Pipeline Version**: 1.0
**Dataset**: HABLA 2.0 (1,567 speakers, 7 accents)
