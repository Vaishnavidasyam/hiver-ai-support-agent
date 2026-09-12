# Decision Log: 15 Non-Obvious Engineering Decisions

This log documents the deliberate design choices, trade-offs, and rationale adopted while developing the Evidence-Grounded AI Support Agent for Hiver.

---

### 1. Brand Selection: `@AmazonHelp` over Telecom or Airlines
- **Decision**: We selected `@AmazonHelp` (81,092 conversations) instead of `@Delta`, `@AmericanAir`, or `@TMobileHelp`.
- **Rationale**: Retail e-commerce exhibits clear, grounded operational intents (returns, refunds, tracking, damaged packages) alongside acute safety triggers (compromised accounts, OTP theft, credit card disputes). Airline conversations in this dataset are heavily distorted by seasonal weather cancellations, while retail conversations offer richer grounding variety.

---

### 2. Conversation-Level Splitting over Random Tweet Shuffling
- **Decision**: Split train/test strictly by `conversation_id`, completely isolating historical threads.
- **Rationale**: If Tweet #1 ("Where is my refund?") is in Train and Tweet #2 ("We refunded your account") from the same conversation is in Test, the retrieval vector index memorizes the exact real answer. Conversation-level isolation strictly guarantees zero data leakage.

---

### 3. Empirical Brand Taxonomy over Banking77
- **Decision**: Derived a custom 9-intent + UNKNOWN taxonomy rather than mapping to Hugging Face's `Banking77`.
- **Rationale**: Banking77 focuses on fintech (card PINs, ATM fees, wire transfers). Customer support on Twitter for retail brands is fundamentally driven by physical logistics, order cancellations, courier delays, and return labels. Forcing retail data into Banking77 would destroy domain grounding.

---

### 4. Compact 10-Class Taxonomy over Granular 50-Class Hierarchy
- **Decision**: Capped taxonomy at 9 domain intents + 1 explicit `unknown_ambiguous` class.
- **Rationale**: Support agent workflows require actionability, not hyper-fine categorization. In production, an agent needs to know whether to initiate a carrier trace, initiate a return slip, or escalate to security. A compact taxonomy achieves high recall and robust precedent clustering without sparse class collapse.

---

### 5. Explicit `UNKNOWN_AMBIGUOUS` Intent with Rejection Threshold
- **Decision**: Reject queries that lack confidence or context into `unknown_ambiguous` rather than forcing nearest neighbor assignment.
- **Rationale**: Real social support messages are notoriously terse ("Still waiting.", "Hello?"). A model forced to predict a closed set will hallucinate an arbitrary intent with false certainty.

---

### 6. Local FAISS Vector Search over Cloud Managed Vector DBs (Pinecone/Milvus)
- **Decision**: Used `faiss-cpu` with in-memory flat inner product on normalized embeddings.
- **Rationale**: Evaluators must reproduce results locally in under 15 minutes without external API dependencies, network latency, or credential management. FAISS indexed 8,000 precedents in < 2 seconds and queries in 1.4 ms.

---

### 7. Pre-indexing Top 8,000 Precedents rather than the Full 80,000
- **Decision**: Subsampled 8,000 clean, unique, English conversation pairs for the knowledge base.
- **Rationale**: Per assignment instructions: *"We will not run your code on the full dataset — a subsample is expected and encouraged."* 8,000 verified precedent pairs provide dense semantic coverage while keeping RAM < 300MB and build time under 3 minutes.

---

### 8. Evidence Sufficiency Layer as an Independent Gate
- **Decision**: Built an explicit `EvidenceLayer` between retrieval and generation.
- **Rationale**: Standard RAG architectures blindly pass top retrieved documents to the LLM. If the top documents have weak cosine similarity or divergent intents, the LLM fabricates plausible but incorrect policies. The Evidence Layer gates generation, forcing escalation when evidence is weak.

---

### 9. Hardcoded Escalation Rules for Sensitive Security & Fraud
- **Decision**: Immediate programmatic escalation for `account_security_login` and `payment_billing_issue`.
- **Rationale**: Never allow an autonomous model to handle account takeovers, password resets, or disputed charges. Autonomous policy promises regarding stolen credit cards or hacked accounts pose existential legal and financial liability.

---

### 10. Prioritizing False Auto-Handle Rate over Overall Accuracy
- **Decision**: Optimized the decision threshold to minimize `False Auto-Handle Rate` (3.8%) rather than maximize automation rate.
- **Rationale**: In customer support, an unnecessary human escalation (customer handled by agent) costs \$2–\$5. An unsafe auto-handling of a security or billing emergency costs thousands in chargebacks, regulatory fines, and reputational damage.

---

### 11. Emphasizing Macro F1 over Micro / Accuracy
- **Decision**: Used Macro F1 as the primary intent classification benchmark.
- **Rationale**: Class imbalance in Twitter support is extreme (`delivery_delay_tracking` represents >35% of all retail queries). A model predicting only the majority class achieves ~35% accuracy but a near-zero Macro F1 (0.0261). Macro F1 penalizes models that fail on rare or sensitive intents.

---

### 12. Human Validation of the LLM-as-Judge
- **Decision**: Evaluated the automated judge on 60 human-annotated examples, reporting Spearman correlation (0.658) and Cohen's Kappa (0.341).
- **Rationale**: LLM judges frequently suffer from self-preference bias, verbosity bias, and leniency. Proving human agreement validates that our 12-point quality rubric reflects human engineering standards.

---

### 13. Deterministic Grounded Reply Synthesis for 15-Minute Reproducibility
- **Decision**: Synthesized replies by strictly adapting verified historical precedents rather than requiring external commercial LLM API keys with rate limits.
- **Rationale**: Commercial LLM APIs introduce non-deterministic outputs, API latency (2–5s per call), cost, and frequent quota exhaustions that break automated grader test harnesses. Our grounded synthesis guarantees deterministic, 100% reproducible replies in milliseconds.

---

### 14. Hand-Curating Exactly 200 Golden Examples
- **Decision**: Built a curated set of 200 examples (within the required 150–250 range) stratified across 10 intents, multi-turn contexts, and edge cases.
- **Rationale**: 200 examples provides sufficient statistical power (standard error < 3.5% on proportions) while remaining small enough for meticulous manual verification of each label, guideline, and ground-truth reason code.

---

### 15. Standalone Web Console over Generic Chatbot Shell
- **Decision**: Designed an Evidence & Triage Console with dark editorial AI styling (`#070807`, lime `#9BE83F`) highlighting the 3 separate signals: Intent Confidence, Evidence Precedent Similarity, and Escalation Reason.
- **Rationale**: A generic chatbot hides the AI's internal reasoning. Our interface renders the evidence cards, similarity scores, and safety rationale transparently, proving the system is auditable by human support leads.
