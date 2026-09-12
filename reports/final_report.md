# Evidence-Grounded AI Customer Support Agent for Twitter Support (`@AmazonHelp`)

**Hiver SDE Intern Take-Home Project Report**  
**Author:** AI Engineering Candidate  
**Date:** September 2026  
**Brand Studied:** `@AmazonHelp` (Customer Support on Twitter Dataset)  
**Evaluation Set:** 200 Hand-Labelled Golden Examples  

---

## 1. Problem Framing: What "Good" Means for `@AmazonHelp`

Social media customer support on Twitter/X is structurally distinct from traditional email or ticket portals. Incoming tweets are character-constrained, colloquial, laden with typos, frequently missing account identifiers, and often sent during moments of peak customer frustration.

For `@AmazonHelp`, **"good" customer support does not mean generating eloquent, ungrounded conversational prose**. It means:
1. **Accurately Discerning Underlying Intent**: Identifying whether the customer has a logistics delay, a refund delay, a damaged delivery, an account security breach, or a vague complaint.
2. **Strict Grounding in Established Precedents**: Responding only with verified procedures (e.g. directing to the secure customer service portal or checking the "Your Orders" tracking page), without hallucinating delivery timelines, refund promises, or compensation amounts.
3. **Conservative, Auditable Escalation**: Identifying when an automated reply is safe (`AUTO_HANDLE`) versus when human intervention is mandatory (`ESCALATE`), backed by an explicit machine-readable reason code.

### What We Intentionally Chose NOT to Build
To ensure reliability, auditability, and zero hallucination risk, the following were deliberately excluded:
- **Autonomous Account Operations**: The agent does not execute database writes, issue refunds, or reset customer passwords autonomously.
- **Unconstrained Generative Chatbots**: We explicitly avoided open-ended LLM loops that can hallucinate non-existent Amazon policies or commit the company to unauthorized financial compensations.
- **Multi-Brand Bloat**: We concentrated deeply on one high-volume brand (`@AmazonHelp`) to understand its real domain nuances rather than building a shallow multi-tenant prototype.

---

## 2. System Architecture & Leakage-Free Design

The pipeline follows an evidence-first architecture:

```text
Incoming Customer Tweet
          │
          ▼
   [ Preprocessor & Normalizer ] (HTML unescape, URL replacement, handle normalization)
          │
          ├──► Step 1: Semantic Intent Classification (Sentence-Transformers Centroids)
          │            └─► (intent, calibrated confidence)
          │
          ├──► Step 2: Historical Precedent Retrieval (Dense Embeddings + FAISS FlatIP)
          │            └─► Top 5 similar resolved conversations with similarity scores
          │
          ├──► Step 3: Evidence Sufficiency Assessment
          │            └─► Evaluates top similarity (>=0.55), consensus ratio (>=0.60), and domain sensitivity
          │
          ├──► Step 4: Risk & Escalation Decision Engine
          │            └─► AUTO_HANDLE vs. ESCALATE with explicit reason code
          │
          └──► Step 5: Grounded Reply Generation
                       └─► Adapts verified precedent resolution while adhering to brand sign-off (^AH)
```

### Strict Data Leakage Prevention
- **Conversation-Level Splitting**: Individual tweets were never randomly partitioned. All turns within any given `conversation_id` were grouped together.
- **Index Isolation**: The FAISS vector database was constructed strictly from the 8,000 training conversations. The 200 Golden Set examples were sampled from a held-out test pool (`test_pool.parquet`), guaranteeing that no test query or its historical response was ever indexed or seen during retrieval.

---

## 3. Experimental Results vs. Baselines

All systems were evaluated on the exact same **200 hand-labelled Golden Set**.

### Headline Results Table

| System / Model | Macro F1 | Recall@5 | Reply Quality (0–12) | Reply Score % | False Auto-Handle Rate | Automation Rate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline 1 (Majority Class)** | 0.0261 | N/A | 9.8 / 12 | 81.4% | **100.0% (UNSAFE)** | 100.0% |
| **Baseline 2 (TF-IDF + LR)** | **0.7813** | N/A | 10.4 / 12 | 87.1% | **22.6% (RISKY)** | 73.5% |
| **Proposed Support Agent** | 0.5913 | **0.6950** | 9.0 / 12 | 75.2% | **3.8% (SAFETY-GATED / NOT PRODUCTION-READY)** | 36.5% |

### Detailed Metric Breakdown for Proposed System
- **Retrieval Performance**:
  - `Recall@3`: **0.6250**
  - `Recall@5`: **0.6950**
  - `Recall@10`: **0.6950**
  - `Mean Reciprocal Rank (MRR)`: **0.5443**
- **Escalation Safety**:
  - `Escalation Precision`: **0.4016**
  - `Escalation Recall`: **0.9623** (Captured 51 out of 53 cases requiring human review)
  - `Escalation F1`: **0.5667**
  - `False Auto-Handle Rate`: **3.8%** (Only 2 out of 53 cases mistakenly automated)
- **Average Inference Latency**: **18.4 ms** per message.

---

## 4. LLM-as-Judge & Human Agreement Validation

To avoid reliance on n-gram overlap metrics (BLEU/ROUGE) which penalize valid semantic variations in support replies, we implemented a **6-dimension, 12-point rubric** assessing:
1. Intent correctness (0–2)
2. Semantic relevance (0–2)
3. Historical precedent grounding (0–2)
4. Actionable helpfulness & next step (0–2)
5. Brand tone & sign-off consistency (0–2)
6. Security and customer safety (0–2)

### Human-Judge Agreement on 60 Sampled Golden Cases
To evaluate the judge itself, 60 examples were independently evaluated by a human annotator following the identical rubric:
- **Spearman Rank Correlation**: **0.7632** ($p < 0.0001$), demonstrating strong monotonic correlation between the judge's scoring and human quality assessments.
- **Close Agreement Rate (within 1 point)**: **81.7%**.
- **Weighted Cohen's Kappa**: **0.6575** (Substantial inter-rater reliability across discretized tiers).

---

## 5. Failure Analysis: Top 5 Real-World Failure Modes

1. **Multi-Intent & Emotional Overload**:
   - *Example*: Customer complaining about an India order being double-charged while also complaining about a previous failed return pickup and rude courier.
   - *Failure*: Order number pattern and delivery tokens skewed prototype similarity toward `delivery_delay_tracking`, causing an unsafe auto-handle.
   - *Hypothesis*: Single-label semantic centroids average all tokens, diluting urgent secondary billing cues.
2. **Sarcasm and Promotional Hashtag Pull**:
   - *Example*: *"so well for amazon prime! Still waiting for my order. Disappointed #AmazonPrime"*
   - *Failure*: Classified as `prime_membership_benefits` (`AUTO_HANDLE`) instead of a delayed delivery grievance.
   - *Hypothesis*: The hashtag `#AmazonPrime` strongly pulled the embedding toward subscription benefits.
3. **Entangled Cross-Domain Queries**:
   - *Example*: *"@AmazonHelp since when is Prime a week for delivery? Hmmm...."*
   - *Failure*: Predicted Prime benefits instead of delivery delay tracking.
   - *Hypothesis*: Query straddles Prime SLA terms and delivery execution; single-label taxonomy forces an artificial winner.
4. **Terse Financial Inquiries**:
   - *Example*: *"What is going on with my recent transaction??"*
   - *Failure*: Classified as `unknown_ambiguous` (Confidence: 0.547 < 0.58) rather than `payment_billing_issue`.
   - *Hypothesis*: Brevity reduced centroid similarity, though the system safely escalated.
5. **Historical Policy & Workflow Drift (Knowledge Drift)**:
   - *Example*: *"How do I return this third party seller item without a printer?"*
   - *Failure*: Historical precedent from 2017 advised DMing address so Amazon could email a printable return label, whereas modern policy provides QR-code label-less drop-off at UPS Stores/Whole Foods.
   - *Hypothesis*: *Historical precedent $\neq$ current policy*. Precedent-grounded systems are temporally bounded by the training window. Without policy override layers or recency decay, grounded agents risk confidently reproducing obsolete company procedures.

---

## 6. Mandatory Section: "What is Misleading About My Headline Number?"

> **"Our Baseline 2 achieved an Intent Macro F1 of 0.7813, whereas our Proposed Agent achieved a Macro F1 of 0.5913. Looking solely at Macro F1, one might conclude that Baseline 2 is superior. That conclusion would be catastrophic in production."**

> **"The proposed system trades some closed-set classification performance for safer handling of uncertain and high-risk queries. Therefore Macro F1 alone is insufficient to evaluate automation suitability."**

Here is why relying on the headline Macro F1 number is deeply misleading:

1. **Classification Accuracy $\neq$ Customer Safety**:
   Baseline 2 achieved a high Macro F1 because its TF-IDF keyword matching aggressively memorized vocabulary associations. However, when tested on real escalation decisions, **Baseline 2 mistakenly auto-handled 22.6% of sensitive customer issues** (including unauthorized charges and account compromises). In contrast, our Proposed Agent achieved a **False Auto-Handle Rate of only 3.8%**, prioritizing customer safety above raw intent scores.
2. **Intent Identification $\neq$ Issue Resolution**:
   Correctly categorizing a tweet as `delivery_delay_tracking` does not mean the system resolved the issue. If the customer's carrier lost the shipment, an automated tweet saying "Check your tracking link!" frustrates the customer further. Grounding and precedent relevance matter far more than classification accuracy.
3. **Rejection Thresholds Artificially Lower Intent Scores**:
   Our proposed model incorporates a calibrated rejection threshold: when an incoming message is ambiguous, terse, or conflicting, it intentionally falls back to `unknown_ambiguous` to trigger safe human escalation. In standard multi-class evaluation, this is penalized as an intent error, artificially reducing the Macro F1 score while substantially improving real-world operational safety.
4. **Golden Set Scope & Offline Limitations**:
   The golden set contains 200 manually labelled examples from `@AmazonHelp`. While carefully sampled and leakage-isolated, it represents an offline benchmark. A single headline metric cannot capture dynamic policy shifts, seasonal spikes, or live conversational degradation.

---

## 7. What We Would Do With One More Week

1. **Multi-Label Intent Architecture**: Replace single-label centroid classification with a binary relevance or multi-head classification layer to cleanly detect secondary billing or security complaints in composite tweets.
2. **Temporal Precedent Weighting**: Incorporate recency decay into FAISS retrieval scoring to prioritize resolutions from the last 6 months, avoiding obsolete legacy return URLs.
3. **Hybrid Cross-Encoder Reranking**: Deploy a cross-encoder (`ms-marco-MiniLM-L-6-v2`) on top-15 retrieved FAISS candidates to rerank precedents for precise semantic alignment before passing to the evidence layer.
4. **Active Learning Golden Set Expansion**: Expand the golden set from 200 to 500 examples using uncertainty sampling on production edge cases.
5. **Cross-Brand Portability Benchmark**: Test the exact pipeline on `@AppleSupport` and `@Uber_Support` to empirically measure transferability across domains.

---

## 8. Summary & Conclusion

In customer support AI, **controlled uncertainty is infinitely superior to confident hallucination**. By combining semantic prototype classification, dense historical retrieval over 8,000 precedents, an independent evidence sufficiency layer, and strict safety rules, our system reduced unsafe auto-handling to 3.8% while maintaining sub-20ms latency and 100% reproducible evaluation in under 15 minutes.
