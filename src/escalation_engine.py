"""Risk assessment, safety policy enforcement, and escalation decision engine."""

from typing import Tuple
from src.config import (
    AUTO_HANDLE_DECISION, ESCALATE_DECISION, INTENT_TAXONOMY,
    INTENT_CONFIDENCE_THRESHOLD, REASON_CODES
)
from backend.schemas import IntentResult, EvidenceSufficiencyResult


class EscalationEngine:
    """Evaluates whether an incoming customer interaction is safe for automated resolution."""

    def decide(
        self,
        intent_result: IntentResult,
        evidence_result: EvidenceSufficiencyResult
    ) -> Tuple[str, str, str]:
        """Computes escalation decision, machine-readable reason code, and human-readable explanation."""
        intent = intent_result.label
        confidence = intent_result.confidence
        is_sufficient = evidence_result.is_sufficient

        # Rule E1: Sensitive Security / Login Issue
        if intent == "account_security_login":
            return (
                ESCALATE_DECISION,
                "SENSITIVE_ACCOUNT_SECURITY",
                REASON_CODES["SENSITIVE_ACCOUNT_SECURITY"]
            )

        # Rule E2: Financial / Payment Dispute
        if intent == "payment_billing_issue":
            return (
                ESCALATE_DECISION,
                "FINANCIAL_DISPUTE",
                REASON_CODES["FINANCIAL_DISPUTE"]
            )

        # Rule E3: Unknown / Ambiguous Intent
        if intent == "unknown_ambiguous":
            return (
                ESCALATE_DECISION,
                "AMBIGUOUS_QUERY",
                REASON_CODES["AMBIGUOUS_QUERY"]
            )

        # Rule E4: Low Intent Confidence
        if confidence < INTENT_CONFIDENCE_THRESHOLD:
            return (
                ESCALATE_DECISION,
                "LOW_INTENT_CONFIDENCE",
                f"{REASON_CODES['LOW_INTENT_CONFIDENCE']} (confidence: {confidence:.2f} < {INTENT_CONFIDENCE_THRESHOLD})."
            )

        # Rule E5: Insufficient or Conflicting Historical Evidence
        if not is_sufficient:
            if evidence_result.top_similarity < 0.55:
                return (
                    ESCALATE_DECISION,
                    "INSUFFICIENT_EVIDENCE",
                    f"{REASON_CODES['INSUFFICIENT_EVIDENCE']} (top similarity: {evidence_result.top_similarity:.2f})."
                )
            else:
                return (
                    ESCALATE_DECISION,
                    "CONFLICTING_EVIDENCE",
                    f"{REASON_CODES['CONFLICTING_EVIDENCE']} ({evidence_result.reason})"
                )

        # Rule E6: High-confidence, Strong Evidence, Low-risk operational case
        return (
            AUTO_HANDLE_DECISION,
            "STRONG_EVIDENCE",
            f"{REASON_CODES['STRONG_EVIDENCE']} (confidence: {confidence:.2f}, similarity: {evidence_result.top_similarity:.2f})."
        )
