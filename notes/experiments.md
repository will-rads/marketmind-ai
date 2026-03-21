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

---

### 2026-03-21 — Notebook 02 Executed — TF-IDF + LR Baseline Complete

**What:** Ran `notebooks/02_tfidf_baseline.ipynb` on Colab T4. TF-IDF + Logistic Regression with `class_weight='balanced'`.

**Setup:**
- Product-aware train/test split via `GroupShuffleSplit` on `parent_asin` — 0 product overlap
- Train: 39,484 reviews (3,338 products) / Test: 10,505 reviews (835 products)
- Train/test sentiment distributions nearly identical (~84.7% / ~83.9% positive)
- TF-IDF: 50K max features, unigrams + bigrams, `sublinear_tf=True`, `min_df=3`, `max_df=0.95`
- Logistic Regression: `class_weight='balanced'`, `C=1.0`, `solver='lbfgs'`, `max_iter=1000`
- Training time: 12.6 seconds

**Results:**
- Accuracy: **0.8310**
- Macro F1: **0.6107**
- Positive — precision 0.959, recall 0.882, F1 **0.919**
- Negative — precision 0.552, recall 0.660, F1 **0.601**
- Neutral — precision 0.246, recall 0.427, F1 **0.312**
- Misclassified: 1,775 / 10,505 (16.9%)

**Error analysis:**
- Top error: 697 positive → neutral (hedged praise, e.g., "it's okay, only 4 stars because...")
- 341 positive → negative (negative language with positive ratings, e.g., sarcasm, backhanded compliments)
- 215 negative → neutral (mild complaints without strong negative language)
- 207 neutral → positive (3-star reviews with mostly positive text but one caveat)
- Neutral class is the weakest — F1 0.31, precision only 0.25. Confirms 3-star ambiguity hypothesis.

**Saved:**
- Model: `models/tfidf_lr_baseline.joblib` (1.2 MB)
- Vectorizer: `models/tfidf_vectorizer.joblib` (2.0 MB)

**Key takeaways:**
- Baseline is solid — 83% accuracy with honest product-aware split and class weighting
- Positive class is effectively solved at this level (F1 0.92)
- Neutral class is where DistilBERT needs to improve — the model struggles with hedged, mixed, or ambiguous language that TF-IDF can't capture contextually
- The error patterns are exactly what we predicted and will make strong analysis content in the report

**Next:** DistilBERT fine-tuning (Notebook 03). Target: beat macro F1 0.61, especially on neutral class.
