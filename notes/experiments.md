# Experiments Log

Training runs, scores, findings, and decisions. Updated after every experiment or meaningful technical decision.

---

## Format

Each entry should include:
- **Date**
- **What was run** (model, dataset size, hyperparameters)
- **Result** (metrics, observations)
- **Decision** (what to do next based on the result)
- **Checkpoint** (path to saved model, if applicable)

---

## Entries

### 2026-03-21 — Notebook 01 Executed — Data Acquired & EDA Complete

**What:** Ran `notebooks/01_data_acquisition_eda.ipynb` on Colab T4. Full data acquisition + EDA pipeline.

**Dataset:**
- 10 categories × 5K reviews = 49,989 reviews (11 dropped for empty text)
- Sampling: top-reviewed products per category (not random) — ensures dense review coverage
- Saved to `data/raw/reviews_50k.parquet` on Google Drive (9.2 MB)
- Loading method: direct JSONL via `hf://datasets/McAuley-Lab/Amazon-Reviews-2023/raw/review_categories/{cat}.jsonl` (original `load_dataset` approach broke due to `trust_remote_code` deprecation)

**EDA results:**
- Label distribution: positive 84.5% (42,253) / negative 9.1% (4,526) / neutral 6.4% (3,210)
- Imbalance ratio: ~13:1 (positive vs. neutral)
- Product coverage: 4,173 unique products, mean 12.0 reviews/product, median 7.0
- Products with 10+ reviews (usable for ranking): 1,357 — covering 32,247 reviews (64.5%)
- Max reviews on a single product: 1,518

**Key decisions:**
- Class weighting is required during training — without it, model will default to predicting "positive"
- Top-product sampling strategy confirmed as correct — product distribution is dramatically better than random sampling (previous run had 92.6% single-review products)
- Train/test split must be product-aware — many reviews per product means naive random split would leak product-specific language into test set

**Next:** Build TF-IDF + Logistic Regression baseline (Notebook 02). Use `class_weight='balanced'`. Implement product-aware split.
