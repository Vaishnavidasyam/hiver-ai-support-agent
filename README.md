# Evidence-Grounded AI Support Agent (`@AmazonHelp`)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com)
[![FAISS](https://img.shields.io/badge/Vector%20Search-FAISS-purple.svg)](https://github.com/facebookresearch/faiss)
[![Tests](https://img.shields.io/badge/pytest-10%2F10%20passed-brightgreen.svg)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Hiver SDE Intern — Take-Home Assignment Submission**  
> **Candidate:** Vaishnavi Dasyam ([@Vaishnavidasyam](https://github.com/Vaishnavidasyam))  
> **Repository:** [https://github.com/Vaishnavidasyam/hiver-ai-support-agent](https://github.com/Vaishnavidasyam/hiver-ai-support-agent)  
> **Core Philosophy:** *"Turn a messy real-world dataset into a working AI system and prove it works. The proof is worth more than the system."*

---

## 📑 Table of Contents
1. [⚡ 15-Minute Headline Reproduction Guide](#-15-minute-headline-reproduction-guide)
2. [📊 Headline Benchmark Results](#-headline-benchmark-results)
3. [⚠️ Mandatory Section: What is Misleading About My Headline Number?](#️-mandatory-section-what-is-misleading-about-my-headline-number)
4. [🎯 Problem Framing: What "Good" Means for @AmazonHelp](#-problem-framing-what-good-means-for-amazonhelp)
5. [🧪 200 Hand-Labelled Golden Evaluation Set](#-200-hand-labelled-golden-evaluation-set)
6. [⚖️ Evaluation Harness & LLM-as-Judge Validation](#️-evaluation-harness--llm-as-judge-validation)
7. [🔍 Top 5 Failure Modes & Hypotheses](#-top-5-failure-modes--hypotheses)
8. [📝 Decision Log (15 Non-Obvious Engineering Decisions)](#-decision-log-15-non-obvious-engineering-decisions)
9. [🚀 One-Week Roadmap (What We'd Do Next)](#-one-week-roadmap-what-wed-do-next)
10. [📚 Citations & Borrowed Components](#-citations--borrowed-components)
11. [💻 Interactive Triage Console & Web UI](#-interactive-triage-console--web-ui)
12. [📂 Repository Structure](#-repository-structure)

---

## ⚡ 15-Minute Headline Reproduction Guide

Reproduce headline benchmark results and run the system locally in under **2 minutes**:

### 1. Clone & Setup Environment
```bash
git clone https://github.com/Vaishnavidasyam/hiver-ai-support-agent.git
cd hiver-ai-support-agent

# Create and activate virtual environment (optional but recommended)
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Benchmark Harness (< 15 seconds)
```bash
python -m evaluation.run_evaluation
```
This executes:
- **Baseline 1 (Majority Class Intent)**
- **Baseline 2 (TF-IDF + Logistic Regression)**
- **Proposed Evidence-Grounded Agent**
- Evaluates **200 hand-labelled Golden Set examples** (Intent Macro F1, Recall@5, Reply Rubric, False Auto-Handle Rate)
- Validates the **LLM-as-Judge vs. Human Agreement** (Spearman correlation & Cohen's Kappa on 60 human ratings)
- Saves full benchmark outputs to `reports/results.json`.

### 3. Launch Interactive Web Console & API
```bash
python -m scripts.run_agent
```
Open **[http://localhost:8000](http://localhost:8000)** in your browser to test incoming tweets, inspect the 3 independent gates, and view failure post-mortems and decision logs.

### 4. Run Automated Test Suite (10/10 tests pass)
```bash
pytest tests/
```
Runs leakage-safety assertions (`tests/test_leakage.py`) and pipeline integration tests (`tests/test_pipeline.py`).

---

## 📊 Headline Benchmark Results

| System / Model | Intent Macro F1 | Recall@5 | Reply Score (0–12) | Reply Score % | False Auto-Handle Rate | Automation Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline 1 (Majority Class)** | 0.0261 | N/A | 9.8 / 12 | 81.4% | **100.0%** | UNSAFE |
| **Baseline 2 (TF-IDF + LR)** | **0.7813** | N/A | 10.4 / 12 | 87.1% | **22.6%** | RISKY |
| **Proposed Support Agent** | 0.5913 | **0.6950** | 9.0 / 12 | 75.2% | **3.8%** | **SAFETY-GATED / PRODUCTION-READY** |

### Key Safety & Performance Metrics:
- **Escalation Safety Recall**: **96.2%** (51 / 53 sensitive queries safely intercepted and escalated to humans).
- **False Auto-Handle Rate**: **3.8%** (Only 2 out of 53 escalated queries mistakenly automated vs. 22.6% in Baseline 2).
- **Retrieval Quality**: `Recall@3`: **0.6250** | `Recall@5`: **0.6950** | `MRR`: **0.5443**.
- **LLM-as-Judge vs. Human Agreement**: Spearman Correlation $\rho = \mathbf{0.7632}$ ($p < 0.0001$), Close Agreement = **81.7%**, Weighted Cohen's Kappa $\kappa = \mathbf{0.6575}$.
- **Average Inference Latency**: **18.4 ms** per query.

---

## ⚠️ Mandatory Section: What is Misleading About My Headline Number?

> **"Baseline 2 achieved a Macro F1 of 0.7813, whereas our Proposed Agent achieved 0.5913. Looking solely at the headline Macro F1, one might assume Baseline 2 is superior. That conclusion would be catastrophic in production."**
>
> *"The proposed system deliberately trades some closed-set classification performance for safer handling of uncertain and high-risk queries. Macro F1 alone is an unsafe metric to evaluate automation suitability."*

### Why the Headline Number is Misleading:

1. **Classification Accuracy $\neq$ Operational Safety**:
   - Baseline 2 achieved high Macro F1 by overfitting to common lexical patterns. However, **Baseline 2 mistakenly auto-handled 22.6% of critical escalation queries**, including unauthorized payment deductions and hacked account takeovers.
   - In contrast, our Proposed Agent achieved a **False Auto-Handle Rate of only 3.8%**, prioritizing customer account protection over raw intent score.

2. **Rejection Thresholds Artificially Depress Multi-Class F1**:
   - Our semantic classifier enforces an intentional confidence rejection threshold (`0.58`). Terse, noisy, or ambiguous tweets fall back to `unknown_ambiguous` to trigger safe human escalation.
   - In standard multi-class evaluation, this deliberate safety fallback is penalized as a misclassification, mechanically lowering Macro F1 while vastly improving production safety.

3. **Intent Identification $\neq$ Issue Resolution**:
   - Correctly labelling a tweet as `delivery_delay_tracking` does not resolve the customer's issue. If a courier lost the package or misdelivered it to a neighbor, auto-replying with a generic tracking link infuriates the customer.
   - Evidence sufficiency and historical precedent alignment matter infinitely more than classification accuracy alone.

---

## 🎯 Problem Framing: What "Good" Means for @AmazonHelp

### Brand Selected: `@AmazonHelp`
We chose `@AmazonHelp` from the 3M Kaggle Twitter dataset because e-commerce customer support involves **high-stakes operational interactions** (money, account security, lost deliveries) rather than low-friction social chatter.

### What "Good" Means:
1. **Never Make Unverified Promises**: The agent must never fabricate refund guarantees, delivery date commitments, or tracking numbers.
2. **Prioritize Safety Over Automation Volume**: An auto-reply that gives the wrong answer on an account compromise causes catastrophic trust loss; escalating to a human takes 5 seconds and protects the customer.
3. **Evidence-Grounded Resolutions**: Drafted replies must mirror proven historical customer support precedents from the `@AmazonHelp` team.

### What We Deliberately Chose NOT to Build:
- **No Autonomous Financial Actions**: The agent drafts refund navigation guidance; it does NOT execute refunds autonomously via billing APIs.
- **No Hallucinated Courier Tracking**: If the customer asks "where is my driver?", the agent routes them to the official tracking portal rather than fabricating a real-time driver ETA.
- **No Generative Free-Wheeling**: Every response is constrained by verified historical precedents or structured escalation templates.

---

## 🧪 200 Hand-Labelled Golden Evaluation Set

Located at [`data/golden/golden_set.json`](data/golden/golden_set.json).

### Sampling & Labelling Methodology:
- **Zero-Contamination Split**: Sampled strictly from held-out conversations (`data/processed/test_pool.parquet`) with **0% overlap** against the 8,000 FAISS training precedents (`tests/test_leakage.py` asserts this).
- **Stratified Distribution (10 Intent Classes)**:
  1. `delivery_delay_tracking`: 30 cases (15.0%)
  2. `refund_return_status`: 35 cases (17.5%)
  3. `damaged_defective_wrong_item`: 25 cases (12.5%)
  4. `payment_billing_issue`: 21 cases (10.5%) *(Hard Escalation)*
  5. `cancellation_modification`: 20 cases (10.0%)
  6. `account_security_login`: 18 cases (9.0%) *(Hard Escalation)*
  7. `prime_membership_benefits`: 15 cases (7.5%)
  8. `unknown_ambiguous`: 14 cases (7.0%) *(Auto Escalation)*
  9. `product_stock_inquiry`: 12 cases (6.0%)
  10. `feedback_complaint`: 10 cases (5.0%)

### Schema per Golden Example:
- `example_id`, `conversation_id`, `customer_message`, `context`
- `gold_intent`, `gold_decision` (`AUTO_HANDLE` vs. `ESCALATE`)
- `gold_reason_code` (`STRONG_EVIDENCE`, `SENSITIVE_ACCOUNT_SECURITY`, `FINANCIAL_DISPUTE`, `AMBIGUOUS_QUERY`)
- `reply_guidelines` (must-include procedural elements, prohibited claims)

---

## ⚖️ Evaluation Harness & LLM-as-Judge Validation

### Three-Tier Evaluation:
1. **Automated Classification & Routing Metrics** ([`evaluation/metrics.py`](evaluation/metrics.py)): Macro F1, Per-Class F1, Escalation Recall, False Auto-Handle Rate.
2. **Retrieval Benchmark**: Recall@3, Recall@5, Recall@10, Mean Reciprocal Rank (MRR) against gold precedents.
3. **LLM-as-Judge Reply Quality Rubric (0–12 Points)** ([`evaluation/llm_judge.py`](evaluation/llm_judge.py)):
   - **Grounding (0–3)**: Is the reply supported by historical brand precedents without hallucinations?
   - **Safety (0–3)**: Does it avoid making unauthorized financial or delivery promises?
   - **Actionability (0–3)**: Does it provide an explicit next step (portal URL, account check)?
   - **Tone & Persona (0–3)**: Does it match the polite, concise `@AmazonHelp` voice with proper sign-off?

### Human vs. Judge Validation Study:
Evaluated on **60 blind-annotated human labels** ([`evaluation/human_validation.py`](evaluation/human_validation.py)):
- **Spearman Rank Correlation**: $\mathbf{\rho = 0.7632}$ ($p = 1.84 \times 10^{-12}$)
- **Weighted Cohen's Kappa**: $\mathbf{\kappa = 0.6575}$ (Substantial agreement)
- **Close Agreement Rate**: **81.7%** (Scores within $\pm 1$ point on a 12-point scale)

---

## 🔍 Top 5 Failure Modes & Hypotheses

Detailed post-mortems in [`reports/failure_analysis.md`](reports/failure_analysis.md) and live in the Web UI:

1. **Multi-Intent Overload**:
   - *Example*: *"Where is my package? Also you charged me twice on my Visa!"*
   - *Failure*: Intent classifier latched onto tracking tokens, diluting the financial dispute signal and risking an improper auto-reply.
   - *Fix*: Multi-label classification head with safety union logic.
2. **Promotional Hashtag Bias**:
   - *Example*: *"So much for two-day shipping! Still waiting for my order #AmazonPrime"*
   - *Failure*: `#AmazonPrime` biased embedding toward Prime subscription perks instead of delivery logistics.
   - *Fix*: Preprocessing regex to strip marketing hashtags before semantic embedding.
3. **Entangled Cross-Domain Inquiries**:
   - *Example*: *"Can I return an opened Prime Video digital purchase?"*
   - *Failure*: Boundary ambiguity between retail returns and Prime digital content licensing.
   - *Fix*: Composite dual-intent routing.
4. **Terse Financial Complaints**:
   - *Example*: *"What is going on with my transaction??"*
   - *Failure*: Query similarity fell slightly below confidence threshold (0.547 < 0.58) and was categorized as `unknown_ambiguous`.
   - *Analysis*: Safely escalated, but the reason code was coarse rather than specific.
5. **Historical Policy Drift (Knowledge Drift)**:
   - *Example*: 2017 historical precedents instructed customers to print physical shipping labels, whereas current Amazon policy uses QR codes at drop-off hubs.
   - *Fix*: Metadata timestamp filtering and procedural override layers.

---

## 📝 Decision Log (15 Non-Obvious Engineering Decisions)

Documented in [`reports/decision_log.md`](reports/decision_log.md):

1. **Brand Choice (`@AmazonHelp`)**: Selected for high interaction volume (81k pairs) and grounded retail logistics vs. airline weather volatility.
2. **Conversation-Level Splitting**: Strictly partitioned at the conversation thread level to prevent test-to-train retrieval leakage.
3. **Custom 10-Class Taxonomy**: Derived empirically from real Twitter e-commerce queries rather than forcing the fintech-specific Banking77 dataset.
4. **Explicit `UNKNOWN_AMBIGUOUS` Intent**: Created a dedicated intent sink to prevent nearest-neighbor hallucination on short/vague tweets.
5. **Local FAISS-CPU Search**: Chose in-memory FAISS over cloud vector databases to eliminate API keys, latency, and rate limits during evaluation.
6. **Subsampling to 8,000 High-Quality Precedents**: Filtered for signal-to-noise, keeping RAM < 300MB and search latency < 10ms.
7. **Three Independent Architectural Gates**: Decoupled Intent Confidence $\neq$ Evidence Sufficiency $\neq$ Risk Escalation so no single module can bypass safety.
8. **Hard Escalation Policies**: Programmatically forced human review for payment disputes and hacked accounts regardless of classifier confidence.
9. **Optimizing False Auto-Handle Rate (3.8%)**: Chose to prioritize zero unsafe automations over maximizing raw automation volume.
10. **Macro F1 Metric Selection**: Used unweighted Macro F1 to penalize models that collapse on rare or critical classes.
11. **Human Validation of LLM Judge**: Conducted human correlation study on 60 samples to establish statistical validity of the automated judge.
12. **Deterministic Response Generation**: Used precedent-grounded templating with LLM synthesis to guarantee zero API quota failures during 15-minute reproduction runs.
13. **200 Curated Golden Examples**: Chose 200 cases to achieve statistical power ($\\pm 5.5\\%$ margin of error) with meticulous manual verification.
14. **Dark Editorial Web UI**: Built a clean Linear/Vercel-inspired dashboard to visually inspect every stage of the pipeline.
15. **Alternating Preset Suite**: Designed test presets that strictly alternate between `AUTO_HANDLE` and `ESCALATE` with live outcome badges.

---

## 🚀 One-Week Roadmap (What We'd Do Next)

1. **Multi-Label Intent Architecture**: Deploy multi-head binary classifiers to detect secondary billing/security complaints in compound queries.
2. **Temporal Precedent Weighting**: Incorporate exponential decay into FAISS scores to prioritize newer policy precedents over older ones.
3. **Cross-Encoder Reranker**: Add a lightweight `ms-marco-MiniLM-L-6-v2` cross-encoder to re-rank top-15 FAISS candidates for nuanced intent alignment.
4. **Scale Golden Set to 500 Examples**: Use active learning to sample queries where model confidence is lowest.
5. **Cross-Brand Portability Benchmark**: Evaluate the framework on `@AppleSupport` and `@Uber_Support` to test zero-shot transferability.

---

## 📚 Citations & Borrowed Components

| Component | What Was Borrowed | Source / Citation | Custom Logic Built on Top |
|---|---|---|---|
| **Raw Dataset** | Twitter Customer Support (~3M tweets) | *Customer Support on Twitter*, Kaggle (`thoughtvector/customer-support-on-twitter`) | Parsed turn graph, filtered for `@AmazonHelp`, masked PII, sanitized URLs, isolated 0% leakage splits. |
| **Embeddings** | `all-MiniLM-L6-v2` | Sentence-Transformers (Reimers & Gurevych, 2019) | Custom prototype aggregation per intent, cosine similarity matrix, calibrated rejection threshold. |
| **Vector Store** | FAISS CPU (`IndexFlatIP`) | Facebook AI Research (Johnson et al., 2017) | In-memory top-5 nearest neighbor retrieval over 8,000 precedents with resolution metadata extraction. |
| **Baseline 2** | TF-IDF + Logistic Regression | `scikit-learn` (Pedregosa et al., 2011) | Calibrated n-gram multi-class baseline model. |
| **API & UI** | FastAPI & Tailwind CSS | `tiangolo/fastapi` & Tailwind Play CDN / Lucide | 4-tab interactive dashboard (Triage Console, Failure Post-Mortem, Decision Log, Benchmark Proof). |

---

## 💻 Interactive Triage Console & Web UI

Launch locally via `python -m scripts.run_agent` and visit **`http://localhost:8000`**:

- **Tab 1: Triage Console**: Test customer tweets live. Displays the 6-stage linear pipeline, the 3 independent gates, inference latency, and the 6 alternating presets (`Auto` ➔ `Escalate`).
- **Tab 2: Failure Post-Mortem**: Interactive case studies of the top 5 failure modes with root-cause hypotheses and architectural fixes.
- **Tab 3: Decision Log**: 15 Architecture Decision Records (ADRs) formatted in clean Linear/Vercel style.
- **Tab 4: Benchmark & Proof**: Quantitative scorecard comparing the proposed system against Baseline 1 and Baseline 2.

---

## 📂 Repository Structure

```text
hiver-ai-support-agent/
├── data/
│   ├── raw/                  # Source Kaggle data (excluded from Git via .gitignore)
│   ├── processed/            # train_conversations.parquet & test_pool.parquet
│   └── golden/
│       └── golden_set.json   # 200 hand-labelled evaluation cases
├── backend/
│   ├── main.py               # FastAPI application & API endpoints
│   ├── pipeline.py           # Unified inference & triage engine
│   └── schemas.py            # Pydantic data contracts
├── src/
│   ├── config.py             # Hyperparameters, taxonomy, and thresholds
│   ├── preprocessor.py       # Tweet cleaning, PII masking, turn reconstruction
│   ├── embeddings.py         # Dense transformer embedding wrappers
│   ├── intent_classifier.py  # Majority, TF-IDF+LR, and Semantic Prototype classifiers
│   ├── retriever.py          # FAISS CPU vector index over 8,000 precedents
│   ├── evidence_layer.py     # Sufficiency & consensus assessment
│   ├── reply_generator.py    # Grounded response drafting (^AH sign-off)
│   └── escalation_engine.py  # Deterministic safety rules & reason codes
├── evaluation/
│   ├── metrics.py            # Automated classification, retrieval & safety metrics
│   ├── llm_judge.py          # 12-point reply quality rubric
│   ├── human_validation.py   # Human vs Judge Spearman correlation & Cohen's Kappa
│   └── run_evaluation.py     # 15-minute reproduction harness
├── reports/
│   ├── final_report.md       # Complete 6-page technical report
│   ├── failure_analysis.md   # Detailed failure post-mortems
│   ├── decision_log.md       # 15 Architecture Decision Records
│   └── results.json          # Cached benchmark output metrics
├── frontend/
│   └── index.html            # Dark Editorial Web Triage Console (Tailwind + Lucide)
├── scripts/
│   ├── prepare_data.py       # Turn reconstruction & golden set sampler
│   ├── build_index.py        # FAISS vector database builder
│   ├── fit_models.py         # Prototype & baseline model fitting
│   └── run_agent.py          # Web server launcher (FastAPI / Uvicorn)
├── tests/
│   ├── test_pipeline.py      # End-to-end pipeline unit tests
│   └── test_leakage.py       # Zero-contamination data leakage tests
├── .gitignore                # Excludes large raw archives (>100MB)
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation & benchmark guide
```

---

## 👤 Author & Contact
- **Candidate**: Vaishnavi Dasyam
- **GitHub**: [@Vaishnavidasyam](https://github.com/Vaishnavidasyam)
- **Repository**: [https://github.com/Vaishnavidasyam/hiver-ai-support-agent](https://github.com/Vaishnavidasyam/hiver-ai-support-agent)
- **Submission To**: `anurag@hiverhq.com`
