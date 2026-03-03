# Mozilla Common Voice Speaker Selection Strategy

**Date:** February 23, 2026
**Author:** Tomas Acosta
**Advisor:** Ruben Manrique Piramanrique

---

## Objective

Augment the HABLA dataset with acoustically diverse speakers from Mozilla Common Voice Spanish to create a balanced 7-accent corpus for voice anti-spoofing research.

---

## Current HABLA Dataset Analysis

**Total HABLA samples:** 22,816
**Total HABLA speakers:** 162
**Accents:** 5 (Argentina, Chile, Colombia, Peru, Venezuela)

### HABLA Distribution by Accent

| Accent     | Samples | Percentage | Male  | Female | Total |
|------------|---------|------------|-------|--------|-------|
| Argentina  | 5,460   | 23.93%     | 1,670 | 3,790  | 5,460 |
| Peru       | 5,446   | 23.87%     | 2,917 | 2,529  | 5,446 |
| Colombia   | 4,604   | 20.18%     | 2,534 | 2,070  | 4,604 |
| Chile      | 4,089   | 17.92%     | 2,487 | 1,602  | 4,089 |
| Venezuela  | 3,217   | 14.10%     | 1,754 | 1,463  | 3,217 |

**Observations:**
- Argentina and Peru have the most samples (~5,450 each)
- Venezuela has the fewest samples (3,217)
- Gender distribution is relatively balanced across accents

---

## Common Voice Available Data

### Mexico
- **Total samples:** 152,013
- **Unique speakers:** 1,523
- **Gender:** 57% Male / 42% Female (balanced ✓)
- **Age:** 72% twenties, 12% thirties, 10% teens (young bias)

### Spain
- **Total samples:** 98,736
- **Unique speakers:** 1,201
- **Gender:** 89% Male / 9% Female (SEVERE imbalance ✗)
- **Age:** 37% sixties, 24% fifties, 13% thirties (old bias)

---

## Selection Strategy

### Target Distribution

**Goal:** Balance all 7 accents to ~5,450 samples each (matching Argentina/Peru)

**Final Dataset:**
- **7 accents × 5,450 samples = 38,150 total samples**

### Samples Needed from Common Voice

| Accent     | Current HABLA | Target | CV Boost Needed |
|------------|---------------|--------|-----------------|
| Argentina  | 5,460         | 5,450  | 0 (at target)   |
| Peru       | 5,446         | 5,450  | 0 (at target)   |
| Colombia   | 4,604         | 5,450  | **+846**        |
| Chile      | 4,089         | 5,450  | **+1,361**      |
| Venezuela  | 3,217         | 5,450  | **+2,233**      |
| Mexico     | 0             | 5,450  | **+5,450**      |
| Spain      | 0             | 5,450  | **+5,450**      |

**Total CV samples needed:** 15,340

---

## Quality Criteria

### 1. Speaker Independence
- **Method:** ECAPA-TDNN speaker embeddings + cosine similarity filtering
- **Threshold:** 0.75 (speakers with similarity ≥0.75 to any HABLA speaker are excluded)
- **Rationale:** Ensure no acoustic overlap between HABLA and CV speakers

### 2. Metadata Filtering
- **Exclude "Unknown" gender and age** from all CV selections
- **Rationale:** Ensure demographic transparency and reproducibility

### 3. Gender Balance (per accent)
- **Target:** 50% Male / 50% Female where possible
- **Spain exception:** Maximize female representation (limited to 9% of available data)
- **Rationale:** Reduce gender bias in anti-spoofing models

### 4. Age Diversity (per accent)
- **Strategy:** Proportional sampling across available age groups
- **Age groups:** teens, twenties, thirties, fourties, fifties, sixties
- **Rationale:** Maintain age diversity to improve model robustness

---

## Pipeline Implementation

### Step 1: Extract HABLA Embeddings
- Load 162 HABLA speakers from `data/bonafide_dataset_by_speaker/`
- Extract ECAPA-TDNN embeddings (512-dim) from training samples
- Average embeddings per speaker
- **Output:** `habla_embeddings.npy` (162, 512)

### Step 2: Extract CV Embeddings
- Filter Common Voice `validated.tsv` for:
  - **Accents:** Colombia, Chile, Venezuela, Mexico, Spain ONLY
  - **Exclude:** Unknown gender, Unknown age
- Group samples by speaker (`client_id`)
- Extract one representative ECAPA-TDNN embedding per speaker
- **Output:** `cv_speaker_embeddings.npy` (N, 512)

### Step 3: Filter by Similarity
- Compute cosine similarity: `CV_embeddings @ HABLA_embeddings.T`
- For each CV speaker, find max similarity to any HABLA speaker
- **Reject speakers with max_similarity ≥ 0.75**
- **Output:** `filtered_speakers.json`

### Step 4: Balanced Sampling
- From filtered pool, sample per accent:
  - Colombia: 846 samples
  - Chile: 1,361 samples
  - Venezuela: 2,233 samples
  - Mexico: 5,450 samples
  - Spain: 5,450 samples (prioritize females)
- Within each accent, balance by gender and age
- **Output:** `selected_15340.tsv`, `selection_stats.json`

---

## Expected Outcomes

### Final HABLA 2.0 Dataset

| Accent     | Original HABLA | CV Addition | Total  | Percentage |
|------------|----------------|-------------|--------|------------|
| Argentina  | 5,460          | 0           | 5,460  | 14.31%     |
| Peru       | 5,446          | 0           | 5,446  | 14.27%     |
| Colombia   | 4,604          | 846         | 5,450  | 14.28%     |
| Chile      | 4,089          | 1,361       | 5,450  | 14.28%     |
| Venezuela  | 3,217          | 2,233       | 5,450  | 14.28%     |
| Mexico     | 0              | 5,450       | 5,450  | 14.28%     |
| Spain      | 0              | 5,450       | 5,450  | 14.28%     |
| **TOTAL**  | **22,816**     | **15,340**  | **38,156** | **100%** |

### Benefits

1. **Accent Balance:** All 7 accents equally represented (~14% each)
2. **Geographic Coverage:** Expands from 5 to 7 Spanish dialects
3. **Speaker Diversity:** Adds 1,500+ new speakers acoustically distinct from HABLA
4. **Gender Balance:** Improved representation across all accents
5. **Age Diversity:** Broader age range (teens to sixties)
6. **Speaker Independence:** Zero acoustic overlap with original HABLA

---

## Risks and Mitigations

### Risk 1: Spain Gender Imbalance
- **Issue:** Spain has only 9% female speakers
- **Mitigation:** Prioritize female selection to maximize representation
- **Expected:** ~50% female in Mexico, ~20% female in Spain

### Risk 2: Age Bias
- **Issue:** Mexico skews young (72% twenties), Spain skews old (37% sixties)
- **Mitigation:** Proportional sampling across available age groups
- **Expected:** Age distribution reflects real-world Common Voice demographics

### Risk 3: Similarity Filter Too Strict
- **Issue:** Threshold 0.75 may reject too many speakers
- **Mitigation:** Monitor pass rate in Step 3; adjust threshold if needed
- **Expected:** >80% of CV speakers should pass filter

---

## Validation

After pipeline execution, verify:
1. **Accent counts:** Each accent has ~5,450 samples (±50)
2. **Gender distribution:** Mexico ~50/50, Spain maximize female
3. **Age diversity:** At least 3 age groups per accent
4. **Speaker independence:** No speaker overlap with HABLA
5. **Total samples:** 38,156 total (22,816 HABLA + 15,340 CV)

---

## References

- **HABLA Dataset:** Tamayo-Florez et al., "HABLA: A dataset of Latin American Spanish accents for voice anti-spoofing," Interspeech 2023
- **Common Voice:** Mozilla Common Voice Spanish v24.0 (December 2025)
- **ECAPA-TDNN:** Desplanques et al., "ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in TDNN Based Speaker Verification," Interspeech 2020
- **Speaker Verification:** SpeechBrain `spkrec-ecapa-voxceleb` pre-trained model

---

**Last Updated:** February 23, 2026
