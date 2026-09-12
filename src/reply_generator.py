"""Evidence-grounded reply generation adhering to brand precedents without hallucination."""

import re
from typing import List, Optional
from backend.schemas import EvidenceCase


class GroundedReplyGenerator:
    """Generates customer replies grounded strictly in historical brand precedents."""

    def __init__(self, brand_tag: str = "^AH"):
        # Standard canonical brand team sign-off for @AmazonHelp
        self.brand_tag = brand_tag

    def generate_reply(
        self,
        customer_message: str,
        predicted_intent: str,
        evidence: List[EvidenceCase],
        is_sufficient: bool,
        decision: str
    ) -> str:
        """Generates reply. If escalated, returns a safe intake or triage acknowledgment.

        If auto-handled, adapts the closest historical precedent resolution while enforcing
        grounding constraints and removing dataset/Twitter artifacts.
        """
        # Case 1: Sensitive or Escalated queries
        if decision == "ESCALATE":
            return self._generate_escalation_reply(customer_message, predicted_intent)

        # Case 2: Auto-handle with strong precedent
        if evidence and is_sufficient:
            return self._synthesize_from_precedents(customer_message, predicted_intent, evidence)

        # Case 3: Fallback safe acknowledgment
        return "Hi there, we'd like to look into this for you. Please send us a Direct Message with your order details so we can assist."

    def _generate_escalation_reply(self, query: str, intent: str) -> str:
        """Generates safety-first escalation responses for sensitive, financial, or security issues."""
        if intent == "account_security_login":
            return (
                "We take account security very seriously. Please do not share any passwords, OTPs, or personal details in public messages. "
                "Please reach out directly through our secure Help portal at amazon.com/help or send us a private Direct Message so our security team can assist immediately."
            )
        elif intent == "payment_billing_issue":
            return (
                "We understand your concern regarding this charge. To protect your financial security, please do not post card numbers publicly. "
                "Please send us a Direct Message with the charge date and amount or contact our billing team at amazon.com/help to investigate."
            )
        elif intent == "unknown_ambiguous":
            return (
                "Hello, we would be glad to help! Could you please share a few more details or context regarding your issue so we can connect you with the appropriate support specialist?"
            )
        else:
            return (
                "Thank you for reaching out. A customer support specialist is reviewing your case details and will follow up with you shortly via Direct Message."
            )

    def _synthesize_from_precedents(self, query: str, intent: str, evidence: List[EvidenceCase]) -> str:
        """Synthesizes grounded reply from the top matching historical brand precedent,

        thoroughly removing historical customer names, Twitter handles, and raw placeholders.
        """
        top_precedent = evidence[0].historical_response

        # 1. Strip historical agent signatures (^DA, ^KA, ^AH, ^SM, etc.)
        clean_precedent = re.sub(r"\^[A-Z]{2,}\b", "", top_precedent).strip()

        # 2. Strip Twitter handles (@customer, @AmazonHelp, @115858, etc.)
        clean_precedent = re.sub(r"@\w+\s*", "", clean_precedent).strip()

        # 3. Replace raw URL placeholders with standard official portal destination
        clean_precedent = re.sub(r"\[URL\]", "amazon.com/orders", clean_precedent)
        clean_precedent = re.sub(r"https?://\S+", "amazon.com/orders", clean_precedent)

        # 4. Remove historical customer names from greetings & apologies
        # E.g. "I'm sorry for the inconvenience, Declan!" -> "I'm sorry for the inconvenience!"
        clean_precedent = re.sub(
            r"\b(inconvenience|trouble|delay),\s+[A-Z][a-z]+[!,\.]?",
            r"\1!",
            clean_precedent,
            flags=re.IGNORECASE
        )
        # E.g. "Hi Declan, please DM us" -> "Hi there, please DM us"
        clean_precedent = re.sub(
            r"\b(hi|hello|hey)\s+[A-Z][a-z]+[,!]?",
            "Hi there,",
            clean_precedent,
            flags=re.IGNORECASE
        )
        # E.g. "That is unusual, Dhrumil." -> "That is unusual."
        clean_precedent = re.sub(
            r"\b(unusual|unfortunate|concerning),\s+[A-Z][a-z]+[!,\.]?",
            r"\1.",
            clean_precedent,
            flags=re.IGNORECASE
        )

        # 5. Grounding safety check: Mask any specific fabricated order numbers or dollar amounts
        clean_precedent = re.sub(r"\b\d{3}-\d{7}-\d{7}\b", "[Order Number]", clean_precedent)
        clean_precedent = re.sub(r"\$\d+(\.\d{2})?", "[Amount]", clean_precedent)

        # 6. Normalize whitespace and clean dangling punctuation
        clean_precedent = re.sub(r"\s+", " ", clean_precedent).strip()
        clean_precedent = re.sub(r"\s+([,\.!\?])", r"\1", clean_precedent)

        # 7. Ensure initial capitalization
        if clean_precedent and clean_precedent[0].islower():
            clean_precedent = clean_precedent[0].upper() + clean_precedent[1:]

        # 8. Append canonical brand sign-off if configured
        if self.brand_tag and not clean_precedent.endswith(self.brand_tag):
            clean_precedent = f"{clean_precedent} {self.brand_tag}"

        return clean_precedent
