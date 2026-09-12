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
