# Failure Analysis Report: Top 5 Real-World Failure Modes

This document provides an honest, granular post-mortem analysis of the top failure modes discovered during the evaluation of our Evidence-Grounded AI Support Agent for `@AmazonHelp` over the 200 hand-labelled Golden Set.

---

## Failure Mode 1: Multi-Intent and Emotional Complaint Overload
- **Example Customer Message**:
  > *"@customer what a pathetic service Amazon is turning out in India. Your Amazon.in Order # 404-3072127-5113917 double charged and no one is able to help. They just keep passing the buck. In the order before, they didn't collect returns and now this. Pathetic and horrible"*
- **Ground Truth**:
  - Gold Intent: `payment_billing_issue`
  - Gold Decision: `ESCALATE` (Financial dispute + multiple unresolved issues)
- **Model Behavior**:
  - Predicted Intent: `delivery_delay_tracking` (Confidence: 0.72)
  - Decision: `AUTO_HANDLE` (False Auto-Handle Error)
- **Root-Cause Hypothesis**:
  - Customer messages expressing intense frustration often combine multiple separate issues (double charge, failed pickup return, prior unresolved ticket, Order # string).
  - The presence of the dominant token pattern `"Order # 404-..."` and general delivery order tokens overwhelmed the semantic embedding, biasing the prototype similarity toward order/delivery tracking rather than the critical financial double charge.
- **Potential Fix**:
  - Implement a **Multi-Label Intent Detector** or a high-priority regex rule for financial trigger tokens (`double charged`, `unauthorized fee`, `deducted twice`) that guarantees escalation before passing to general semantic centroid matching.

---

## Failure Mode 2: Sarcasm and Hashtag-Induced Distraction
- **Example Customer Message**:
  > *"so well for amazon prime! Still waiting for my order. Disappointed #AmazonPrime"*
- **Ground Truth**:
  - Gold Intent: `unknown_ambiguous` / `delivery_delay_tracking`
  - Gold Decision: `ESCALATE` (Short sarcastic expression lacking order reference)
- **Model Behavior**:
  - Predicted Intent: `prime_membership_benefits` (Confidence: 0.81)
  - Decision: `AUTO_HANDLE`
- **Root-Cause Hypothesis**:
  - Sarcastic expressions ("so well for amazon prime!") invert sentiment. The explicit token `"amazon prime"` and hashtag `"#AmazonPrime"` heavily pulled the sentence embedding vector toward the `prime_membership_benefits` cluster, ignoring the operational complaint ("Still waiting for my order").
- **Potential Fix**:
  - Strip promotional hashtags during pre-processing or apply contextual discourse-marker weighting that downweights branded hashtags when action verbs like "waiting for", "late", or "where is" appear in the predicate.

---

## Failure Mode 3: Composite Cross-Domain Inquiries (Entangled Intents)
- **Example Customer Message**:
  > *"@AmazonHelp since when is Prime a week for delivery? Hmmm...."*
- **Ground Truth**:
  - Gold Intent: `delivery_delay_tracking` (Customer complaining about shipping duration)
- **Model Behavior**:
  - Predicted Intent: `prime_membership_benefits` (Confidence: 0.99)
  - Precedents Retrieved: Prime membership perks and delivery duration FAQs.
- **Root-Cause Hypothesis**:
  - This query sits at the intersection of two valid taxonomy categories: Prime delivery SLA perks and actual shipment tracking.
  - A single-label taxonomy forces an artificial winner. The presence of "Prime" had a higher cosine similarity with the Prime prototype centroid than "week for delivery" had with the delivery delay centroid.
- **Potential Fix**:
  - Support multi-label tagging (e.g. `[delivery_delay_tracking, prime_membership_benefits]`) and generate a hybrid response that explains Prime guaranteed transit times while offering to track the specific package via DM.

---

## Failure Mode 4: Vague Financial Discrepancies Falling Below Confidence
- **Example Customer Message**:
  > *"What is going on with my recent transaction??"*
- **Ground Truth**:
  - Gold Intent: `payment_billing_issue`
  - Gold Decision: `ESCALATE` (Reason: `FINANCIAL_DISPUTE`)
- **Model Behavior**:
  - Predicted Intent: `unknown_ambiguous` (Confidence: 0.547)
  - Decision: `ESCALATE` (Reason: `AMBIGUOUS_QUERY`)
- **Root-Cause Hypothesis**:
  - Although the system made the correct safety decision (`ESCALATE`), the intent classification misclassified the category as ambiguous rather than billing. The isolated word "transaction" was too brief in the 384-dimensional space to cross the 0.58 confidence threshold for `payment_billing_issue`.
- **Potential Fix**:
  - Add single-word financial and transaction lexicons to elevate the prototype similarity for queries containing banking/card/transaction terms even when brevity lowers the overall sentence similarity.

---

## Failure Mode 5: Historical Policy & Workflow Drift (Knowledge Drift)
- **Example Customer Message**:
  > *"How do I return this third party seller item without a printer?"*
- **Ground Truth**:
  - Gold Intent: `refund_return_status`
  - Gold Decision: `AUTO_HANDLE` / `ESCALATE` (Direct to modern QR-code drop-off options)
- **Model Behavior**:
  - Retrieved Precedents: 2017 tweets instructing customer to DM their address so Amazon customer care can email a printable return label.
  - Drafted Reply: Prompts customer for address or promises an emailed return label PDF.
- **Root-Cause Hypothesis**:
  - *Historical precedent $\neq$ current policy*. Twitter support datasets represent a fixed snapshot in time (e.g. 2017).
  - Amazon's return infrastructure has since shifted to QR-code, label-free drop-offs at UPS Stores, Kohl's, and Whole Foods.
  - A system grounded purely in historical precedents faithfully reproduces obsolete operational workflows that are no longer accurate today.
- **Potential Fix**:
  - Implement **Temporal Policy Overrides** and metadata-based recency decay filters in the FAISS retrieval layer. Precedents older than a policy-change cut-off date should either be pruned or dynamically mapped to active standard operating procedure (SOP) snippets.

---

### Bonus Failure Mode: Pure Grievance Sentiment vs. Ambiguous Operational Intent
- **Example Customer Message**: *"Worst experience ever. Never using your service again."*
- **Ground Truth**: Gold Intent: `unknown_ambiguous` / Gold Decision: `ESCALATE` (Requires triage).
- **Model Behavior**: Predicted `feedback_complaint` (Confidence: 0.802) -> `AUTO_HANDLE`.
- **Root-Cause**: Semantically, grievance phrases align with complaints, but unassisted automated apologies alienate high-churn-risk customers.
- **Fix**: Route high-negative-polarity feedback to human retention specialists.
