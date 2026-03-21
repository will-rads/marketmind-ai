# MarketMind AI

**Review Intelligence for Marketing Generation**

A marketing decision-support platform that transforms public product review data into actionable campaign assets. A trained sentiment classifier processes reviews at scale, an algorithmic ranking layer surfaces the best products to promote for a given campaign context, and an agentic Gemini pipeline generates grounded ad copy and promotional images.

**Author:** Will — MS Applied AI, Lebanese American University
**Deadline:** Beginning of May 2026
**Repo:** https://github.com/will-rads/marketmind-ai.git
**Status:** Notebook 01 complete — data acquired, EDA done, building TF-IDF baseline next

---

## Why This Exists

People launching product-based businesses online — whether dropshipping, private label, or small e-commerce brands — face the same problem when it comes to marketing. They pick a product, skim a handful of reviews, and write ad copy based on whatever sounds convincing. The actual patterns buried across thousands of reviews — what buyers consistently love, what frustrates them, what almost made them return it — none of that gets read at scale. There is too much of it, and nobody has time.

MarketMind AI bridges that gap. A trained model classifies review sentiment at scale, an algorithmic layer ranks products by promotional fit for a given campaign context, structured analysis extracts recurring themes, and Gemini generates campaign assets grounded in what actual buyers say. The result: marketing copy that speaks to real strengths and addresses real objections, not guesswork.

---

## Architecture

### Stage 1 — Trained Model: 3-Class Sentiment Classifier (Preprocessing)

**Task:** Classify product reviews into three sentiment classes by collapsing star ratings:
- **Positive** (4-5 stars)
- **Neutral** (3 stars)
- **Negative** (1-2 stars)

Star ratings are used as labels. This is a known noisy proxy — a 3-star review saying "love the product, terrible shipping" is mixed sentiment labeled as neutral. This limitation is stated honestly and explored in the evaluation.

**This stage runs as batch preprocessing, not at demo time.** The trained model classifies all reviews in the dataset. Each product then gets a precomputed sentiment profile (% positive, % negative, % neutral, review volume) that downstream stages use. This keeps the demo fast and the model's work reusable.

**Dataset:** Amazon Reviews 2023 (McAuley Lab, UCSD)
- Source: https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023
- **10 broad categories:** Beauty & Personal Care, Electronics, Home & Kitchen, Sports & Outdoors, Toys & Games, Pet Supplies, Health & Household, Baby Products, Office Products, Tools & Home Improvement
- **Sampling strategy:** Top-reviewed products per category. For each category, reviews are taken from the most-reviewed products first (not randomly), ensuring dense review coverage per product. This is critical for the ranking layer — products need enough reviews to build reliable sentiment profiles.
- Equal representation: 5K reviews per category (50K MVP) / 10K per category (100K final)
- Sample size: **50K reviews for MVP, 100K for final model** — mixed across all categories
- Labels: Star ratings (1-5) are pre-existing in the dataset — collapsed to 3 classes
- **Class imbalance (confirmed by EDA):** Positive 84.5%, negative 9.1%, neutral 6.4%. Imbalance ratio ~13:1. Class weighting will be applied during training to prevent the model from defaulting to "positive" for everything. Both TF-IDF + LR (via `class_weight='balanced'`) and DistilBERT (via weighted loss function) will use this.

**Models:**

| Model | Role | Training Time (Colab T4) |
|-------|------|--------------------------|
| TF-IDF + Logistic Regression | Baseline — fast, interpretable | Under 2 minutes |
| DistilBERT (fine-tuned) | Main model — stronger on nuance | 20–40 min (50K) / 40–90 min (100K) |

**Fine-tuning approach:**
- DistilBERT sourced pretrained from Hugging Face (`distilbert-base-uncased`)
- Replace classification head for 3-class output
- Fine-tune 3–5 epochs, batch size 16–32, learning rate ~2e-5
- Early debugging runs: 1 epoch sanity checks (~3-5 min each)
- Expect 3–5 experimental runs total (learning rate, class weighting, epoch count)

**Colab free-tier session management:**
- Sessions last ~4–6 hours before disconnection
- Longest single training run is ~90 minutes — well within session limits
- Save checkpoints after each epoch
- Store dataset on Google Drive — no re-downloading per session
- Use CPU sessions for data exploration; reserve GPU sessions for training

### Stage 2 — Product Ranking Layer (Algorithmic, Not a Trained Model)

**Goal:** When the user enters campaign constraints, surface a ranked list of the best products to promote — based on the sentiment profiles computed in Stage 1.

**How it works:**
1. Each product in the dataset has a precomputed sentiment profile from the classifier:
   - % positive / % neutral / % negative reviews
   - Total review volume
   - Recurring strengths, complaints, and friction points (from theme extraction — see below)
2. User enters campaign constraints: category, campaign goal, target audience
3. Backend logic ranks products using the sentiment profiles. The ranking criteria adapt to the campaign goal:
   - **"Highlight strengths"** → rank by high positive ratio + high review volume (proven winners with lots of evidence)
   - **"Address objections"** → rank by products with high volume but mixed sentiment (products people buy but complain about — opportunity to reframe)
   - **"Differentiate"** → rank by products where positive reviews mention distinctive attributes (unique strengths the model surfaced)
4. User sees a ranked list and selects a product to build a campaign around

**This is algorithmic ranking, not a second trained model.** The ranking uses the trained classifier's output as its input. Without the classifier, the ranking would have to rely on raw star averages, which are noisy and lose the nuance of review text. The classifier is what makes the ranking meaningful.

### Stage 3 — Theme Extraction

**Not a second model.** A structured, reproducible analysis step using the classifier's output.

**Pipeline:**
1. For the selected product, take all its reviews classified by the trained model
2. Group reviews by predicted class (positive / neutral / negative)
3. Run TF-IDF on each group relative to the others — surfaces distinguishing terms and phrases per sentiment class
4. Top positive terms → **recurring strengths** (e.g., "long-lasting," "gentle on skin," "great scent")
5. Top negative terms → **recurring complaints** (e.g., "broke out," "misleading size," "smells cheap")
6. Top neutral terms → **friction points** (e.g., "okay for the price," "works but nothing special")
7. Pull 2–3 example review snippets per theme as evidence

**Output:** A structured insight object per product:
```json
{
  "product_id": "B00XYZ123",
  "product_name": "Example Moisturizer",
  "sentiment_profile": {"positive": 0.72, "neutral": 0.15, "negative": 0.13},
  "review_count": 1847,
  "strengths": ["long-lasting", "gentle formula", "good value"],
  "complaints": ["misleading product size", "irritated skin", "cheap packaging"],
  "friction_points": ["works but nothing special", "fine for the price"],
  "evidence": {
    "strengths": ["This moisturizer lasts all day and doesn't irritate my sensitive skin..."],
    "complaints": ["The bottle is tiny for the price. Felt misled by the listing photo..."],
    "friction_points": ["It's okay. Does what it says but I wouldn't repurchase..."]
  }
}
```

**Note:** Theme extraction can also run at the category level (aggregating across all products) for the ranking stage, and at the product level for the campaign generation stage. Both use the same method.

### Stage 4 — Agentic Layer: Gemini Campaign Generator

**Goal:** Use the structured review intelligence to generate marketing assets that are grounded in real buyer language, not generic AI copy.

**Meta-prompt architecture:** This stage is built around a structured meta-prompt designed using prompt engineering techniques to get the best possible output from Gemini. The meta-prompt is not generic. It is built specifically around the outputs of the sentiment classifier and TF-IDF theme extractor, so when those results come in (sentiment profile, recurring strengths, complaints, friction points, evidence snippets), they slot directly into the prompt as structured input. This ensures every piece of review intelligence feeds Gemini in a way that produces grounded, specific creative output rather than vague marketing language.

**Pipeline:**
1. Receives the selected product's structured insight object + the user's campaign brief + optional brand context
2. Gemini generates:
   - **Campaign angle** — a strategic recommendation grounded in the review data
   - **Ad copy** — headline + body, referencing real strengths and preemptively addressing complaints, tailored to brand voice if provided
   - **Promotional image** — generated by Gemini's multimodal capabilities, styled to match the campaign tone and brand preferences if provided

**Prompt engineering:** The Gemini prompt templates are documented, disclosed, and reproducible. This is a deliberate contribution — showing that structured prompts grounded in trained model output produce better results than generic prompting alone.

**Why Gemini needs the trained model:** Without the classifier, Gemini is writing copy based on generic category assumptions. With it, Gemini writes copy grounded in statistically recurring themes from thousands of real reviews. The model provides the "what to say" — Gemini provides the "how to say it."

**Optional brand context input.** Users can optionally provide brand context to further personalize the output. Three paths:
- **Upload files:** A product image (e.g., an empty perfume bottle from their manufacturer, or a style they like from another brand) and/or brand guidelines (logo, colors, font rules, tone doc). Gemini incorporates these into the generated image and copy.
- **Describe it:** Briefly describe the look, feel, tone, product packaging shape, colors, or any details they want Gemini to account for.
- **Skip it entirely:** Leave it blank and let Gemini work from the review intelligence and campaign brief alone. The meta-prompt handles this gracefully — brand context improves the output but is not required.

This is handled entirely by Gemini's multimodal generation and does not affect the trained model or the core pipeline.

### Stage 5 — Demo Application

**Interface (4 screens):**

**Screen 1 — Campaign Brief:**
- Select product category (dropdown)
- Select campaign goal (dropdown: highlight strengths, address objections, differentiate)
- Select target audience / tone (dropdown)
- (Optional) Brand context: upload a product image and/or brand guidelines, or briefly describe the look/feel/tone, or leave blank
- "Find Products" button

**Screen 2 — Product Recommendations:**
- Ranked list of recommended products for the given constraints
- Each product shows: name, sentiment distribution bar, review count, top strength, top complaint
- User selects a product to build a campaign around

**Screen 3 — Review Intelligence Card:**
- Full sentiment distribution for the selected product
- Top 3 recurring strengths with example review quotes
- Top 3 recurring complaints with example review quotes
- Key friction points from the neutral zone
- User sees the evidence *before* the creative — the system shows its reasoning

**Screen 4 — Generated Campaign:**
- Campaign angle
- Ad headline + body copy
- Promotional image
- "Why this works" — a note explaining which review insights drove the copy

**Deployment:** Frontend designed to be Vercel-compatible. The trained model and theme extraction run as a preprocessing step (or a lightweight API endpoint), not inside Vercel functions. Gemini API calls can run from serverless functions. The heavy ML work is done offline; the demo serves precomputed sentiment profiles + live Gemini generation.

---

## Evaluation Plan

### Academic Core

**Baseline:** TF-IDF + Logistic Regression
**Main model:** DistilBERT (fine-tuned)

**Metrics:**
- Accuracy
- Macro F1
- Per-class F1 (positive, neutral, negative)
- Confusion matrix

**Analysis:**
- **Neutral/3-star ambiguity:** The 3-star class is inherently noisy — people use it to mean "mediocre," "mixed feelings," and "good product, bad experience." The confusion matrix will show this class bleeding into positive and negative. The report analyzes *why* specific reviews are misclassified (mixed sentiment, conditional praise, sarcasm).
- **Per-category comparison:** If a second category is added, compare model performance across categories. Do beauty reviews behave differently from electronics reviews? Why?
- **Error analysis:** Sample misclassified reviews and categorize failure modes.

**Ablations:**
- **Baseline vs. main model:** TF-IDF + LR vs. DistilBERT — quantify what the deeper model buys you and whether it changes the downstream themes.
- **Generic vs. structured Gemini prompting:** Gemini generating copy with just "write an ad for [product]" vs. Gemini generating copy with the full structured insight object. Qualitative side-by-side demonstrating the model's downstream value.

**Honest limitations to report:**
- Star ratings are noisy proxies for sentiment — a 3-star review with mixed sentiment is labeled "neutral" regardless of content.
- Theme extraction via class-separated TF-IDF produces recurring terms, not nuanced aspect-level analysis. These are called "recurring strengths/complaints/friction points," not "emotional triggers."
- The product ranking layer is algorithmic, not a trained recommender. It uses the classifier's output effectively but does not learn ranking preferences from data.
- The system does not have access to business performance data (sales, conversions). Campaign goals are framed in terms of review signal, not revenue metrics.

---

## Training Plan & Time Estimates

| Phase | What | Estimated Time |
|-------|------|----------------|
| Data loading + exploration | Download Amazon Reviews, inspect categories, pick primary category, clean, split | 1 session (~2–3 hours) |
| Baseline | TF-IDF + Logistic Regression, full evaluation | Under 1 hour including analysis |
| DistilBERT experiments | 3–5 runs varying LR, epochs, class weighting | 2–4 hours total across sessions |
| Evaluation + analysis | Confusion matrix, per-class F1, error analysis, theme extraction | 1 session (~2–3 hours) |
| Product ranking logic | Implement algorithmic ranking using precomputed sentiment profiles | 1 session (~2–3 hours) |
| Gemini integration | Prompt design, structured input, copy/image generation | 1 session (~2–3 hours) |
| Demo build | Wire up full pipeline into working app (4 screens) | 1–2 sessions |

**Total: ~5–7 working sessions across 2–3 weeks for the core model + ranking + evaluation. Then ~2 weeks for Gemini integration, demo, and polish.**

---

## Timeline

| Week | Dates (approx.) | Milestone |
|------|-----------------|-----------|
| 1 | Mar 17–23 | Direction finalized. Architecture refined. README and project structure locked. |
| 2 | Mar 24–30 | Download Amazon Reviews, explore categories, pick primary category, build preprocessing pipeline, train TF-IDF baseline. |
| 3 | Mar 31–Apr 6 | DistilBERT fine-tuning experiments. Compare against baseline. Select best model. Run evaluation. |
| 4 | Apr 7–13 | Batch-classify all reviews. Build product sentiment profiles. Implement ranking logic. Theme extraction pipeline. |
| 5 | Apr 14–20 | Gemini prompt engineering. Build agentic layer. Demo application (4 screens). |
| 6 | Apr 21–27 | Evaluation write-up. Ablations. Limitations section. End-to-end testing. |
| 7 | Apr 28–May 4 | Polish + submit. Final demo, code cleanup, documentation, presentation. |

---

## Next Steps

1. ~~**Download Amazon Reviews 2023** — explore available categories, inspect review quality/volume/label distribution.~~ ✅ Done (Notebook 01)
2. ~~**Set up Colab notebook** — environment, dependencies, data loading from Google Drive.~~ ✅ Done (Notebook 01)
3. **Train TF-IDF + LR baseline** — Notebook 02. Validate the 3-class task works, get initial metrics, establish performance floor. Use `class_weight='balanced'` to handle imbalance.
4. **DistilBERT fine-tuning** — Notebook 03. Fine-tune with weighted loss, compare against baseline.
5. **Test Gemini API** — verify image generation and copy generation quality with a manual structured prompt.

---

## Key Resources

**Dataset:**
- Amazon Reviews 2023: https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023

**Models:**
- DistilBERT: `distilbert-base-uncased` via Hugging Face Transformers
- TF-IDF + Logistic Regression: scikit-learn

**APIs:**
- Gemini API (text generation + image generation)

**Repo:**
- https://github.com/will-rads/marketmind-ai.git

---

## Security & Secrets

No API key, token, or secret should ever appear anywhere in this repository — not in source files, notebooks, config files, logs, screenshots, or documentation.

**Local development:** Secrets are stored in a `.env` file at the project root. This file is `.gitignore`d and never committed. A `.env.example` file with placeholder variable names (no real values) is committed to document which keys the project requires.

**Deployment (Vercel):** Environment variables and secrets are configured in the Vercel dashboard, not stored in the repo. The deployment flow pushes application code only — never files containing real credentials.

**Enforcement:** `.gitignore` includes `.env`, `*.key`, and any other secret-bearing patterns. If a secret is accidentally committed, it must be rotated immediately — removing it from git history alone is not sufficient.

---

## Project Documentation

| Document | Purpose |
|----------|---------|
| `README.md` | Project direction, architecture, dataset, evaluation plan, current status |
| `notes/experiments.md` | Training runs, scores, findings, decisions — the lab notebook |
| Git commits | Code-level implementation history |

The README stays high-level. Detailed experiment results, hyperparameters, and per-run scores go in `notes/experiments.md`. Git commits track code changes. Together these three layers provide full project continuity.

---

## Changelog / Progress Log

All substantial changes logged in reverse chronological order. The most recent entry reflects the current project state.

**Policy:** Every meaningful change must be logged here — direction changes, architecture changes, new files, dataset decisions, experiment milestones, model results, demo updates, deployment-related decisions, and major prompt/pipeline changes. This log exists so that a fresh session (or a fresh pair of eyes) can recover full project context without reading git history.

| Date | Entry |
|------|-------|
| 2026-03-21 | **Notebook 01 complete — data acquired and validated.** `notebooks/01_data_acquisition_eda.ipynb` executed successfully on Colab T4. 50K reviews (5K per category × 10 categories) downloaded from Amazon Reviews 2023 via HuggingFace, saved to Google Drive as `data/raw/reviews_50k.parquet` (13 MB). Sampling strategy changed from random to **top-reviewed products per category** — ensures dense review coverage for the ranking layer. EDA results: 84.5% positive / 9.1% negative / 6.4% neutral (13:1 imbalance, class weighting needed). 4,173 unique products, 1,357 with 10+ reviews covering 64.5% of data. Mean 12 reviews/product, max 1,518. Dataset loading fix: switched from `load_dataset("McAuley-Lab/...")` to direct JSONL loading via `hf://` paths (resolved `trust_remote_code` deprecation). Next: TF-IDF + LR baseline (Notebook 02). |
| 2026-03-21 | **Multi-category dataset + first notebook created.** Dataset now spans 10 broad product categories (equal share from each) instead of single-category. Created `notebooks/01_data_acquisition_eda.ipynb` — streams reviews from HuggingFace, samples 50K (5K per category), collapses star ratings to 3 classes, saves to Google Drive as parquet, runs full EDA. Drive path set to `LAU FINAL PROJECT`. |
| 2026-03-21 | **Stage 4 refined + target audience defined.** Added meta-prompt architecture to Stage 4 — structured prompt engineered specifically around classifier and theme extractor outputs, not generic templates. Brand context input reworked: no longer post-MVP optional. Now three paths: upload product image + brand guidelines, briefly describe look/feel/tone, or skip entirely. Gemini handles all three gracefully. Updated "Why This Exists" to specify target audience: dropshippers, private label sellers, and small e-commerce brands launching product-based businesses online. Updated Screen 1 in demo to reflect brand input options. Created project proposal document (MarketMind_AI_Proposal.docx). |
| 2026-03-20 | **Architecture refined.** Added product ranking layer (Stage 2) — algorithmic, not a trained model. Classifier runs as batch preprocessing; each product gets a precomputed sentiment profile. When user enters campaign constraints, backend logic ranks products by promotional fit using those profiles. User selects from the ranked list, then theme extraction + Gemini run on the selected product. Demo updated to 4 screens (brief → product recommendations → review intelligence → campaign). Removed category lock — primary category chosen after data exploration. Updated timeline to include ranking logic build in Week 4. |
| 2026-03-18 | Added changelog policy (every meaningful change must be logged). Added security/secrets section — no keys in repo, .env for local secrets, Vercel dashboard for deployment secrets. Created .gitignore and .env.example. |
| 2026-03-18 | **Major pivot.** Replaced interior design classifier with review intelligence for marketing generation. New direction: 3-class sentiment classifier (positive/neutral/negative from collapsed star ratings) on Amazon Reviews 2023. Pipeline: trained model → theme extraction (TF-IDF on grouped reviews) → Gemini campaign generation (copy + image). Models: TF-IDF + LR baseline, DistilBERT main. Sample size: 50K MVP, 100K final. Evaluation: accuracy, macro F1, per-class F1, confusion matrix, neutral-class ambiguity analysis, ablation (baseline vs. DistilBERT, generic vs. structured Gemini prompting). Restructured documentation: README for direction, notes/experiments.md for runs/scores, git for code. Added Vercel compatibility requirement. Added optional brand guidelines upload (post-MVP, Gemini-handled). |
| 2026-03-18 | Added model training details for interior design project (now superseded by pivot above). |
| 2026-03-17 | Added buyer persona selector to Stage 2 of interior design pipeline (now superseded). |
| 2026-03-17 | Direction locked: Interior Design Style Classifier + AI Redesign Agent (now superseded). |
| 2026-03-17 | Confirmed key decisions: flexible modality, public data only, demo-heavy submission, no domain lock. |
| 2026-03-17 | Created project folder and initial README.md. Defined constraints, candidate directions, open questions, and timeline. |

---

## Project Instructions for Future Chats

**Your role:** You are assisting Will (MS Applied AI student at LAU) with his capstone project — MarketMind AI, a review intelligence platform for marketing generation. You help with architecture decisions, code, training, evaluation, and writing. Will is also Head of Marketing, so the project has a real-world marketing framing.

**On every new session:**
1. Read this README first — it is the single source of truth for direction and architecture.
2. Check `notes/experiments.md` for the latest training runs, scores, and technical decisions.
3. Check the changelog (bottom of this file) for the latest state — especially the most recent entry.
4. Read and internalize the **Security & Secrets** section. No API key, token, or credential should ever be committed, hardcoded, or appear in any file that reaches GitHub. This is a hard rule, not a suggestion.
5. Briefly confirm to Will what the project is, what stage it's at, and what the next steps are. Then pick up where the last session ended.

**On every change:**
- **Always update the changelog** whenever any substantial change is made — scope shifts, new files, decisions, direction changes, experiments, training runs, model scores, or meaningful additions.
- **Training runs must be logged in `notes/experiments.md`:** When a training run completes (or is interrupted by a Colab timeout), log the checkpoint saved, epoch reached, best metric achieved, and any observations. This is critical for resuming work across sessions.
- **Update any affected sections** of this README (next steps, timeline, etc.) to keep it current.
- The changelog is reverse chronological. The most recent entry should always reflect the current state.

**Development environment:** Colab via VS Code extension, T4 free-tier GPU, data stored on Google Drive. Colab sessions may disconnect after ~4–6 hours — checkpointing and experiment logging are essential for continuity.

**Deployment target:** Vercel-compatible frontend. Trained model runs offline or via lightweight API — not inside Vercel functions.
