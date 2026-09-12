# Evidence-Grounded AI Support Agent — @AmazonHelp

> **Hiver SDE Intern — Take-Home Assignment**
>
> **Turn messy real-world customer-support data into a working AI system — and prove it works.**

<p align="center">
  <strong>Understand → Retrieve → Verify → Decide → Reply</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI">
  <img src="https://img.shields.io/badge/FAISS-Vector_Search-7B1FA2?style=for-the-badge" alt="FAISS">
  <img src="https://img.shields.io/badge/Evaluation-200_Golden_Cases-2EA44F?style=for-the-badge" alt="Evaluation">
  <img src="https://img.shields.io/badge/Tests-10%2F10_Passing-2EA44F?style=for-the-badge" alt="Tests">
</p>

---

## ✦ Overview

**Evidence-Grounded AI Support Agent** is an AI customer-support system built for the **Hiver SDE Intern Take-Home Assignment**.

The system uses real-world customer-support conversations from Twitter/X and focuses on one brand:

**`@AmazonHelp`**

For every incoming customer message, the agent:

1. **Understands** the customer's intent.
2. **Retrieves** relevant historical support precedents.
3. **Verifies** whether the available evidence is sufficient.
4. **Assesses** the risk of automation.
5. **Decides** whether to auto-handle or escalate to a human.
6. **Drafts** a response grounded in the retrieved historical evidence.

The core principle is:

> **A confident prediction is not enough to justify automation.**

---

# 01 — Product Goal

Customer-support conversations are messy.

Real conversations can contain:

- incomplete messages
- very short queries
- ambiguous intent
- multiple issues in one message
- informal language
- sarcasm
- noisy hashtags
- conflicting historical examples
- outdated support procedures
- financially or security-sensitive requests

A generic LLM can still produce a fluent response in these cases.

The objective of this project is different:

> **Determine whether the system has enough evidence to safely respond — and escalate when it does not.**

The resulting workflow is:

```text
Customer Message
       ↓
Intent Understanding
       ↓
Historical Evidence Retrieval
       ↓
Evidence Sufficiency
       ↓
Risk Assessment
       ↓
Auto-Handle / Escalate
       ↓
Grounded Reply

02 — What the Agent Does
2.1 Understand

The system classifies each incoming message into a compact taxonomy of 10 empirically defined intents.

Example:

Customer:
"Where is my package? It was supposed to arrive yesterday."

Intent:
delivery_delay_tracking

Confidence:
0.78

The intent classifier is one part of the pipeline — not the final automation decision.

2.2 Retrieve

The system searches historical @AmazonHelp support conversations using FAISS.

Example:

Historical Precedent

Case #91C7C2D2
Intent: delivery_delay_tracking
Similarity: 0.77

Case #A3FB0274
Intent: delivery_delay_tracking
Similarity: 0.74

The retrieved precedents provide context for the reply-generation and decision stages.

2.3 Verify Evidence

The agent evaluates whether retrieved historical evidence is strong enough to support a response.

The system deliberately separates:

Intent Confidence
        ≠
Evidence Sufficiency
        ≠
Risk Assessment
        ≠
Automation Decision

This prevents high classifier confidence from becoming an automatic permission to act.

2.4 Decide

The system can return:

AUTO_HANDLE

or:

ESCALATE

along with a reason code.

Example:

Decision:
AUTO_HANDLE

Reason:
STRONG_EVIDENCE

Explanation:
High-confidence intent and consistent historical precedents support
safe automated handling.

For sensitive cases:

Decision:
ESCALATE

Reason:
SENSITIVE_ACCOUNT_SECURITY
03 — Target Brand
@AmazonHelp

The project focuses on @AmazonHelp because the selected portion of the source data provides strong coverage of retail customer-support scenarios such as:

delivery
tracking
returns
refunds
billing
account access
Prime-related support
product availability

The taxonomy is derived from the target brand's support data rather than copied directly from an unrelated benchmark.

04 — Intent Taxonomy

The agent uses the following 10-class support taxonomy:

Intent	Description
delivery_delay_tracking	Delivery delays, missing tracking updates and shipment status
refund_return_status	Refund and return-related support
damaged_defective_wrong_item	Damaged, defective or incorrect products
payment_billing_issue	Charges, payment and billing disputes
cancellation_modification	Order cancellation or modification
account_security_login	Account access and security-related problems
prime_membership_benefits	Prime membership and associated benefits
unknown_ambiguous	Unclear, conflicting or insufficiently specified queries
product_stock_inquiry	Product availability and stock questions
feedback_complaint	General complaints and customer feedback
Why unknown_ambiguous?

A support system should not be forced to choose a known class when it does not have enough information.

Instead:

Low confidence
      ↓
unknown_ambiguous
      ↓
Human escalation

This is a deliberate safety mechanism.

05 — Safety Architecture

The most important system-level distinction is:

                INTENT
                  │
                  ▼
             CONFIDENCE
                  │
                  ▼
               EVIDENCE
                  │
                  ▼
            SUFFICIENCY
                  │
                  ▼
                RISK
                  │
                  ▼
             DECISION
            /         \
     AUTO-HANDLE     ESCALATE

Sensitive domains such as:

account security
unauthorized charges
financial disputes

are handled conservatively through deterministic escalation rules.

The system is intentionally optimized for safer automation, not maximum automation volume.

06 — Evaluation

The assignment emphasizes that the system must not only work, but that the implementation should prove that it works.

The evaluation harness therefore measures multiple dimensions.

Intent Classification
Macro F1
precision / recall
confusion matrix
Retrieval
Recall@3
Recall@5
Mean Reciprocal Rank (MRR)
Reply Quality

A structured 12-point LLM-as-judge rubric.

Escalation Safety
false auto-handle rate
escalation safety recall
Human Validation

LLM-as-judge scores are compared against human annotations.

07 — Headline Results

Evaluation is performed on a 200-example hand-labelled Golden Set.

System	Macro F1	Recall@5	Reply Score	False Auto-Handle
Majority Class Baseline	0.0261	N/A	9.8 / 12	100.0%
TF-IDF + Logistic Regression	0.7813	N/A	10.4 / 12	22.6%
Proposed Support Agent	0.5913	0.6950	9.0 / 12	3.8%

Additional evaluation results:

Metric	Result
Escalation Safety Recall	96.2%
Recall@3	0.6250
Recall@5	0.6950
MRR	0.5443
LLM Judge ↔ Human Spearman	0.7632
Close Agreement	81.7%
Weighted Cohen's Kappa	0.6575
Average Inference Latency	18.4 ms/query

These are experimental results on the project's curated evaluation setup and should not be interpreted as evidence of production readiness.

08 — What Is Misleading About My Headline Number?

This is a deliberate part of the evaluation.

The proposed agent achieves:

Macro F1 = 0.5913

while the simpler TF-IDF + Logistic Regression baseline achieves:

Macro F1 = 0.7813

If Macro F1 were the only metric considered, the baseline would appear superior.

However, this does not capture the complete support-automation problem.

The proposed system deliberately rejects uncertain and high-risk cases.

For example:

Uncertain Query
      ↓
unknown_ambiguous
      ↓
Human Escalation

This behavior can reduce closed-set classification metrics while improving operational safety.

The key comparison is:

TF-IDF + LR
False Auto-Handle = 22.6%

Proposed Agent
False Auto-Handle = 3.8%

Therefore:

Macro F1 measures intent-classification quality. It does not, by itself, measure whether an AI support agent is safe to automate.

A complete assessment needs:

Classification
+
Retrieval
+
Grounding
+
Reply Quality
+
Escalation Safety
+
Human Validation
09 — Golden Evaluation Set

The project contains:

200 hand-labelled Golden Set examples

Location:

data/golden/golden_set.json

The examples are sampled from a held-out evaluation pool using conversation-level separation from the historical retrieval corpus.

This reduces the risk of evaluation leakage.

Distribution
Intent	Cases
delivery_delay_tracking	30
refund_return_status	35
damaged_defective_wrong_item	25
payment_billing_issue	21
cancellation_modification	20
account_security_login	18
prime_membership_benefits	15
unknown_ambiguous	14
product_stock_inquiry	12
feedback_complaint	10
Total	200

Each evaluation example contains structured ground truth such as:

example_id
conversation_id
customer_message
context
gold_intent
gold_decision
gold_reason_code
reply guidelines
10 — Leakage Prevention

Retrieval-based systems can accidentally make evaluation look better by retrieving examples that belong to the test set.

This project therefore follows:

Training / Retrieval Corpus
          ≠
Held-Out Golden Set

The split is performed at the conversation level, rather than treating individual tweets as independent observations.

The evaluation conversations are kept out of the historical retrieval corpus used for the benchmark.

11 — Retrieval Evaluation

The historical precedent layer is evaluated independently.

Current results:

Recall@3    0.6250
Recall@5    0.6950
MRR         0.5443

The retrieval evaluation answers:

"Can the system find useful historical precedents before it drafts the response?"

This prevents response quality from hiding weaknesses in retrieval.

12 — LLM-as-Judge

Generated replies are evaluated using a 12-point rubric.

Dimension	Score
Intent Correctness	0–2
Query Relevance	0–2
Historical Precedent Grounding	0–2
Actionable Next Step	0–2
Brand Tone & Sign-off	0–2
Customer Privacy & Safety	0–2

Total:

12 points

The judge itself is validated against human annotations rather than being treated as ground truth.

13 — Human Validation

The LLM judge was compared against a human-annotated subset.

Reported agreement:

Spearman correlation      0.7632
Close agreement           81.7%
Weighted Cohen's Kappa    0.6575

This provides evidence that the automated evaluation rubric has meaningful alignment with human judgement.

14 — Failure Analysis

Real-world customer-support data revealed five major failure modes.

01 — Multi-Intent Overload

A customer can combine a delivery problem with an unauthorized financial charge.

A single-label classifier may focus on the delivery problem and fail to capture the higher-risk financial issue.

Hypothesis: single-label intent classification loses secondary intent information.

Potential improvement: multi-label intent detection.

02 — Sarcasm & Hashtag Bias

Example pattern:

"so well for amazon prime! Still waiting for my order #AmazonPrime"

The promotional hashtag can bias semantic retrieval toward Prime-related cases rather than delivery-related cases.

Hypothesis: hashtags can distort semantic similarity.

Potential improvement: contextual hashtag normalization.

03 — Cross-Domain Intent Entanglement

Some messages sit between support categories, such as Prime benefits and delivery expectations.

A single-label taxonomy can force an artificial choice.

Potential improvement: composite or multi-intent tagging.

04 — Terse Financial Queries

Example:

"What is going on with my recent transaction??"

The message may fall below the configured confidence threshold and become:

unknown_ambiguous

The intent is coarse, but the system safely escalates instead of attempting an unsupported automated resolution.

Potential improvement: stronger domain-specific financial signals combined with conservative escalation.

05 — Historical Policy Drift

Historical support precedents may contain procedures that are no longer current.

Therefore:

Historical Precedent
        ≠
Current Policy

This is an important limitation of historical-data-grounded support.

Potential improvement:

temporal weighting
recency decay
policy override layers
current-policy retrieval
15 — Baselines

The proposed system is compared against two baselines.

Baseline 1 — Majority Class

Always predicts the most frequent intent.

Purpose:

Establish a trivial lower bound.

Baseline 2 — TF-IDF + Logistic Regression
Customer Message
      ↓
TF-IDF
      ↓
Logistic Regression
      ↓
Intent

Purpose:

Establish a strong and simple lexical classification baseline.

The proposed system is not required to outperform the simple classifier on every metric.

Instead, the comparison exposes the trade-off between:

classification performance

and

safe automation behavior.

16 — Product Interface

The application has four primary views.

Triage Console

The customer-support experience.

Shows:

incoming customer message
intent
intent confidence
historical evidence
evidence sufficiency
risk
decision
decision reason
grounded reply
Benchmark & Proof

The evaluation interface.

Shows:

benchmark metrics
baseline comparison
retrieval results
false auto-handle rate
reply-quality evaluation
human / LLM judge validation
misleading headline-number analysis
Failure Post-Mortem

Documents:

real failure examples
observed behavior
expected behavior
root-cause hypotheses
mitigation directions
Decision Log

Documents the project's non-obvious engineering decisions and their rationale.

17 — Technology Stack
Layer	Technology
Language	Python 3.10+
API	FastAPI
Machine Learning	scikit-learn
Semantic Embeddings	Transformer-based embeddings
Vector Search	FAISS
Data Processing	pandas / NumPy
Structured Schemas	Pydantic
Evaluation	Python evaluation harness
LLM Evaluation	LLM-as-Judge
Human Validation	Statistical agreement analysis
Frontend	HTML / CSS / JavaScript
Data Formats	Parquet / JSON
Testing	pytest

The architecture deliberately avoids unnecessary infrastructure so that the experiment remains reproducible and easy to inspect.

18 — Architecture
                         CUSTOMER MESSAGE
                                │
                                ▼
                       ┌──────────────────┐
                       │  PREPROCESSING   │
                       │ Normalization +  │
                       │ Context Parsing  │
                       └────────┬─────────┘
                                │
                 ┌──────────────┴──────────────┐
                 │                             │
                 ▼                             ▼
        ┌──────────────────┐         ┌──────────────────┐
        │ Intent Classifier│         │  FAISS Retriever │
        │                  │         │                  │
        │ Intent +         │         │ Historical       │
        │ Confidence       │         │ Precedents      │
        └────────┬─────────┘         └────────┬─────────┘
                 │                            │
                 └─────────────┬──────────────┘
                               ▼
                    ┌───────────────────────┐
                    │   EVIDENCE LAYER      │
                    │ Sufficiency +         │
                    │ Precedent Consensus   │
                    └───────────┬───────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │   RISK / ESCALATION   │
                    │        ENGINE         │
                    └───────────┬───────────┘
                                │
                     ┌──────────┴──────────┐
                     ▼                     ▼
               ┌────────────┐        ┌────────────┐
               │ AUTO-HANDLE│        │  ESCALATE  │
               └──────┬─────┘        └────────────┘
                      │
                      ▼
             ┌──────────────────┐
             │  GROUNDED REPLY  │
             │    GENERATION    │
             └────────┬─────────┘
                      │
                      ▼
             ┌──────────────────┐
             │  FINAL RESPONSE  │
             └──────────────────┘
19 — Repository Structure
hiver-ai-support-agent/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── golden/
│       └── golden_set.json
│
├── src/
│   ├── config.py
│   ├── preprocessor.py
│   ├── embeddings.py
│   ├── intent_classifier.py
│   ├── retriever.py
│   ├── evidence_layer.py
│   ├── reply_generator.py
│   ├── escalation_engine.py
│   └── pipeline.py
│
├── backend/
│   ├── main.py
│   └── schemas.py
│
├── evaluation/
│   ├── metrics.py
│   ├── llm_judge.py
│   ├── human_validation.py
│   └── run_evaluation.py
│
├── reports/
│   ├── final_report.md
│   ├── failure_analysis.md
│   ├── decision_log.md
│   └── results.json
│
├── frontend/
│   └── index.html
│
├── scripts/
│   ├── prepare_data.py
│   ├── build_index.py
│   ├── fit_models.py
│   └── run_agent.py
│
├── tests/
│   ├── test_pipeline.py
│   └── test_leakage.py
│
├── requirements.txt
└── README.md
20 — Reproducibility

The repository is designed for fast local evaluation on the curated dataset artifacts rather than requiring full processing of the complete source dataset.

Requirements
Python 3.10+
pip
Step 1 — Clone
git clone https://github.com/Vaishnavidasyam/hiver-ai-support-agent.git
cd hiver-ai-support-agent
Step 2 — Install Dependencies
pip install -r requirements.txt
Step 3 — Run the Evaluation
python -m evaluation.run_evaluation

This runs:

Majority Class baseline
TF-IDF + Logistic Regression baseline
Proposed Support Agent
Intent metrics
Retrieval metrics
Reply evaluation
Escalation evaluation
LLM-judge / human agreement
Step 4 — Launch the Support Console
python -m scripts.run_agent

Open:

http://localhost:8000
Step 5 — Run Tests
pytest tests/
21 — Engineering Decisions

The project contains 15 non-obvious engineering decisions.

A representative subset includes:

Selected @AmazonHelp as the target brand.
Used conversation-level splitting.
Built an empirical support taxonomy.
Kept the taxonomy compact at 10 classes.
Added unknown_ambiguous.
Used local FAISS retrieval.
Used a bounded historical precedent corpus.
Added an independent evidence-sufficiency layer.
Added deterministic escalation for sensitive domains.
Prioritized low false auto-handle rate.
Used Macro F1 for class-balanced evaluation.
Validated the LLM judge against humans.
Used grounded reply synthesis.
Curated a 200-example Golden Set.
Built an audit-oriented product UI.

The full rationale is available in:

reports/decision_log.md
22 — What I Chose Not to Build

To keep the project focused on the assignment, this implementation intentionally does not include:

Twitter/X API integration
autonomous refunds
autonomous account modifications
autonomous financial resolution
full customer-support ticketing
multi-brand production deployment
cloud-scale vector infrastructure
complete processing of the full source dataset
autonomous handling of sensitive security cases

The project focuses on:

Evidence-grounded support decisions with explicit escalation boundaries.

23 — Limitations

This is an assignment-scale prototype, not a production support deployment.

Important limitations include:

evaluation uses a curated Golden Set
historical support data can contain outdated policies
the current intent model is single-label
ambiguous messages remain difficult
retrieval quality is limited by the historical corpus
LLM-as-judge evaluation is not equivalent to perfect human judgement
the model and rules are evaluated primarily on @AmazonHelp
no live customer account or transaction systems are connected

The most important limitation is:

Past Support Resolution
        ≠
Current Company Policy

A production system would require authoritative current-policy sources, monitoring, stronger privacy controls, access control, and additional safety validation.

24 — One-Week Improvement Roadmap

With one additional week, I would prioritize:

Multi-Label Intent Architecture

Allow simultaneous issues such as:

delivery problem
+
unauthorized charge

to be represented together.

Temporal Retrieval Weighting

Give more weight to recent precedents to reduce historical policy drift.

Hybrid Retrieval + Reranking

Rerank retrieved candidates with a cross-encoder after initial FAISS retrieval.

Expanded Golden Set

Increase the evaluation set from 200 to approximately 500 examples using uncertainty-based sampling.

Cross-Brand Benchmark

Evaluate the same architecture on additional brands to test portability.

25 — Assignment Alignment
Hiver Requirement	Implementation
Classify incoming messages	10-class intent taxonomy
Define intents from data	Brand-derived support taxonomy
Ground replies in historical resolutions	FAISS precedent retrieval
Auto-handle vs escalate	Risk-aware escalation engine
Stated escalation reason	Explicit reason codes
Golden evaluation set	200 hand-labelled cases
Trivial baseline	Majority Class
Simple baseline	TF-IDF + Logistic Regression
Automated evaluation	Evaluation harness
LLM-as-judge	12-point reply rubric
Human validation	Human vs judge agreement
Failure analysis	Five documented failure modes
Misleading headline number	Dedicated analysis
Decision log	15 engineering decisions
Reproducibility	Local evaluation and test harness
26 — Responsible Use

This prototype is intended for evaluation and research purposes.

It should not be used to autonomously:

resolve financial disputes
modify customer accounts
issue refunds
make security-sensitive decisions
make unsupported policy claims

When evidence is weak or the request is sensitive, the intended behavior is:

STOP
↓
ESCALATE
↓
HUMAN REVIEW
27 — Key Takeaway

The central lesson from this project is:

A customer-support AI should not be judged only by whether it predicts the correct intent.

A trustworthy agent must answer:

What is the customer asking?
            ↓
What historical evidence supports the response?
            ↓
Is the evidence sufficient?
            ↓
Is the situation safe to automate?
            ↓
Should the AI respond or escalate?

That is why this project evaluates:

classification + retrieval + grounding + reply quality + escalation safety + human validation

as parts of one system.

Author

Vaishnavi Dasyam

Hiver SDE Intern — Take-Home Assignment

Target Brand: @AmazonHelp







