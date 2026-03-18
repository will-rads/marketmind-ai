# MarketMind AI

**Review Intelligence for Marketing Generation**

A marketing decision-support platform that transforms public product review data into actionable campaign assets. A trained sentiment classifier extracts structured insight from customer reviews — recurring strengths, complaints, and friction points — which an agentic Gemini pipeline uses to generate grounded ad copy and promotional images.

**Author:** Will — MS Applied AI, Lebanese American University
**Deadline:** Beginning of May 2026
**Repo:** https://github.com/will-rads/marketmind-ai.git
**Status:** Direction locked — building review intelligence pipeline

---

## Why This Exists

Marketing teams write ad copy and design campaigns based on intuition about what customers care about. That intuition is often wrong, generic, or based on a handful of anecdotal reviews. Meanwhile, thousands of reviews contain real signal about what drives satisfaction and frustration — but no one reads them systematically.

MarketMind AI bridges that gap. A trained model classifies review sentiment at scale, structured analysis extracts the recurring themes, and Gemini generates campaign assets grounded in what actual buyers say. The result: marketing copy that speaks to real strengths and addresses real objections, not guesswork.

---

## Architecture

### Stage 1 — Trained Model: 3-Class Sentiment Classifier

**Task:** Classify product reviews into three sentiment classes by collapsing star ratings:
- **Positive** (4-5 stars)
- **Neutral** (3 stars)
- **Negative** (1-2 stars)

Star ratings are used as labels. This is a known noisy proxy — a 3-star review saying "love the product, terrible shipping" is mixed sentiment labeled as neutral. This limitation is stated honestly and explored in the evaluation.

**Dataset:** Amazon Reviews 2023 (McAuley Lab, UCSD)
- Source: https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023
- Primary category: **All Beauty** — produces lifestyle-oriented marketing copy and strong demo visuals
- Second category (for generalizability testing): TBD, likely Home & Kitchen or Electronics
- Sample size: **50K reviews for MVP, 100K for final model**
- Labels: Star ratings (1-5) are pre-existing in the dataset — collapsed to 3 classes

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

### Stage 2 — Theme Extraction

**Not a second model.** A structured, reproducible analysis step using the classifier's output.

**Pipeline:**
1. Classify all reviews in a category using the trained model
2. Group reviews by predicted class (positive / neutral / negative)
3. Run TF-IDF on each group relative to the others — surfaces distinguishing terms and phrases per sentiment class
4. Top positive terms → **recurring strengths** (e.g., "long-lasting," "gentle on skin," "great scent")
5. Top negative terms → **recurring complaints** (e.g., "broke out," "misleading size," "smells cheap")
6. Top neutral terms → **friction points** (e.g., "okay for the price," "works but nothing special")
7. Pull 2–3 example review snippets per theme as evidence

**Output:** A structured insight object per category:
```json
{
  "category": "All Beauty",
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

### Stage 3 — Agentic Layer: Gemini Campaign Generator

**Goal:** Use the structured review intelligence to generate marketing assets that are grounded in real buyer language, not generic AI copy.

**Pipeline:**
1. User inputs a campaign brief:
   - Product category (dropdown)
   - Campaign goal (e.g., highlight strengths, address objections, differentiate from competitors)
   - Target audience / tone (e.g., budget-conscious, premium, gift buyers)
2. System surfaces the review intelligence card for that category
3. Structured insight object + campaign brief → Gemini
4. Gemini generates:
   - **Campaign angle** — a strategic recommendation grounded in the review data
   - **Ad copy** — headline + body, referencing real strengths and preemptively addressing complaints
   - **Promotional image** — generated by Gemini's multimodal capabilities, styled to match the campaign tone

**Prompt engineering:** The Gemini prompt templates are documented, disclosed, and reproducible. This is a deliberate contribution — showing that structured prompts grounded in trained model output produce better results than generic prompting alone.

**Why Gemini needs the trained model:** Without the classifier, Gemini is writing copy based on generic category assumptions. With it, Gemini writes copy grounded in statistically recurring themes from thousands of real reviews. The model provides the "what to say" — Gemini provides the "how to say it."

**Optional (post-MVP): Brand guidelines upload.** User uploads a blank product image and/or brand guidelines (logo, colors, font rules, tone doc). Gemini incorporates these into the generated image and copy. This is handled entirely by Gemini's multimodal generation and does not affect the trained model or the core pipeline.

### Stage 4 — Demo Application

**Interface (3 screens):**

**Screen 1 — Campaign Brief:**
- Select product category (dropdown)
- Select campaign goal (dropdown)
- Select target audience / tone (dropdown)
- (Optional) Upload brand guidelines
- "Generate Campaign" button

**Screen 2 — Review Intelligence Card:**
- Sentiment distribution bar (X% positive, Y% neutral, Z% negative)
- Top 3 recurring strengths with example review quotes
- Top 3 recurring complaints with example review quotes
- Key friction points from the neutral zone
- User sees the evidence *before* the creative — the system shows its reasoning

**Screen 3 — Generated Campaign:**
- Campaign angle
- Ad headline + body copy
- Promotional image
- "Why this works" — a note explaining which review insights drove the copy

**Deployment:** Frontend designed to be Vercel-compatible. The trained model and theme extraction run as a preprocessing step (or a lightweight API endpoint), not inside Vercel functions. Gemini API calls can run from serverless functions. The heavy ML work is done offline; the demo serves precomputed insights + live Gemini generation.

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
- **Generic vs. structured Gemini prompting:** Gemini generating copy with just "write an ad for beauty products" vs. Gemini generating copy with the full structured insight object. Qualitative side-by-side demonstrating the model's downstream value.

**Honest limitations to report:**
- Star ratings are noisy proxies for sentiment — a 3-star review with mixed sentiment is labeled "neutral" regardless of content.
- Theme extraction via class-separated TF-IDF produces recurring terms, not nuanced aspect-level analysis. These are called "recurring strengths/complaints/friction points," not "emotional triggers."
- The system does not have access to business performance data (sales, conversions). Campaign goals like "boost underperformer" are framed in terms of review signal, not sales metrics.

---

## Training Plan & Time Estimates

| Phase | What | Estimated Time |
|-------|------|----------------|
| Data loading + exploration | Download Amazon Reviews (All Beauty), inspect, clean, split | 1 session (~2–3 hours) |
| Baseline | TF-IDF + Logistic Regression, full evaluation | Under 1 hour including analysis |
| DistilBERT experiments | 3–5 runs varying LR, epochs, class weighting | 2–4 hours total across sessions |
| Evaluation + analysis | Confusion matrix, per-class F1, error analysis, theme extraction | 1 session (~2–3 hours) |
| Gemini integration | Prompt design, structured input, copy/image generation | 1 session (~2–3 hours) |
| Demo build | Wire up full pipeline into working app | 1–2 sessions |

**Total: ~4–6 working sessions across 2–3 weeks for the core model + evaluation. Then ~2 weeks for Gemini integration, demo, and polish.**

---

## Timeline

| Week | Dates (approx.) | Milestone |
|------|-----------------|-----------|
| 1 | Mar 17–23 | Direction finalized. Dataset selected. README and project structure locked. |
| 2 | Mar 24–30 | Download Amazon Reviews (All Beauty), explore data, build preprocessing pipeline, train TF-IDF baseline. |
| 3 | Mar 31–Apr 6 | DistilBERT fine-tuning experiments. Compare against baseline. Select best model. Run evaluation. |
| 4 | Apr 7–13 | Theme extraction pipeline. Gemini prompt engineering. Build agentic layer. |
| 5 | Apr 14–20 | Demo application. Wire up full pipeline. End-to-end testing. |
| 6 | Apr 21–27 | Evaluation write-up. Ablations. Limitations section. Prepare presentation. |
| 7 | Apr 28–May 4 | Polish + submit. Final demo, code cleanup, documentation. |

---

## Next Steps

1. **Download Amazon Reviews 2023** (All Beauty category) — inspect label distribution, review lengths, data quality.
2. **Set up Colab notebook** — environment, dependencies, data loading from Google Drive.
3. **Train TF-IDF + LR baseline** — validate the 3-class task works and get initial metrics.
4. **Test Gemini API** — verify image generation and copy generation quality with a manual structured prompt.

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
| 2026-03-18 | Added changelog policy (every meaningful change must be logged). Added security/secrets section — no keys in repo, .env for local secrets, Vercel dashboard for deployment secrets. Created .gitignore and .env.example. |
| 2026-03-18 | **Major pivot.** Replaced interior design classifier with review intelligence for marketing generation. New direction: 3-class sentiment classifier (positive/neutral/negative from collapsed star ratings) on Amazon Reviews 2023, starting with All Beauty category. Pipeline: trained model → theme extraction (TF-IDF on grouped reviews) → Gemini campaign generation (copy + image). Models: TF-IDF + LR baseline, DistilBERT main. Sample size: 50K MVP, 100K final. Evaluation: accuracy, macro F1, per-class F1, confusion matrix, neutral-class ambiguity analysis, ablation (baseline vs. DistilBERT, generic vs. structured Gemini prompting). Restructured documentation: README for direction, notes/experiments.md for runs/scores, git for code. Added Vercel compatibility requirement. Added optional brand guidelines upload (post-MVP, Gemini-handled). |
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
4. Briefly confirm to Will what the project is, what stage it's at, and what the next steps are. Then pick up where the last session ended.

**On every change:**
- **Always update the changelog** whenever any substantial change is made — scope shifts, new files, decisions, direction changes, experiments, training runs, model scores, or meaningful additions.
- **Training runs must be logged in `notes/experiments.md`:** When a training run completes (or is interrupted by a Colab timeout), log the checkpoint saved, epoch reached, best metric achieved, and any observations. This is critical for resuming work across sessions.
- **Update any affected sections** of this README (next steps, timeline, etc.) to keep it current.
- The changelog is reverse chronological. The most recent entry should always reflect the current state.

**Development environment:** Colab via VS Code extension, T4 free-tier GPU, data stored on Google Drive. Colab sessions may disconnect after ~4–6 hours — checkpointing and experiment logging are essential for continuity.

**Deployment target:** Vercel-compatible frontend. Trained model runs offline or via lightweight API — not inside Vercel functions.
