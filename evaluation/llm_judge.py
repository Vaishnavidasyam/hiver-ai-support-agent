"""LLM-as-Judge evaluation module using a multi-dimensional 12-point rubric for reply quality."""

import re
from typing import Dict, Any, List


class ReplyQualityJudge:
    """Evaluator assessing customer support reply quality across 6 standardized dimensions (0-2 each, max 12)."""

    RUBRIC_DIMENSIONS = [
        "intent_correctness",
        "relevance",
        "grounding",
        "helpfulness",
        "brand_consistency",
        "safety"
    ]

    def evaluate_reply(
        self,
        customer_message: str,
        gold_intent: str,
        pred_intent: str,
        generated_reply: str,
        gold_decision: str,
        pred_decision: str,
        gold_guidelines: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Scores generated reply from 0 to 2 on each rubric dimension."""
        reply_lower = generated_reply.lower()
        
        # 1. Intent Correctness (0 - 2)
        if pred_intent == gold_intent:
            score_intent = 2
        elif pred_intent in ["unknown_ambiguous"] and gold_decision == "ESCALATE":
            score_intent = 2
        else:
            score_intent = 0

        # 2. Relevance (0 - 2)
        has_question = any(q in customer_message.lower() for q in ["where", "how", "why", "when", "can", "what"])
        addresses_issue = any(k in reply_lower for k in ["order", "dm", "message", "help", "account", "assist", "details", "link", "tracking"])
        if addresses_issue and len(generated_reply.split()) >= 8:
            score_relevance = 2
        elif addresses_issue:
            score_relevance = 1
        else:
            score_relevance = 0

        # 3. Grounding (0 - 2)
        # Check if reply introduces fabricated order numbers or claims
        has_fabricated_order = bool(re.search(r"\b\d{3}-\d{7}-\d{7}\b", generated_reply))
        has_fabricated_dollar = bool(re.search(r"\$\d{2,}", generated_reply))
        if has_fabricated_order or has_fabricated_dollar:
            score_grounding = 0
        elif "[URL]" in generated_reply or "amazon.com" in reply_lower or "dm" in reply_lower or "portal" in reply_lower or "[order number]" in reply_lower:
            score_grounding = 2
        else:
            score_grounding = 1

        # 4. Helpfulness (0 - 2)
        has_next_step = any(k in reply_lower for k in ["dm", "direct message", "link", "reach out", "portal", "track", "check", "details"])
        if has_next_step:
            score_helpfulness = 2
        else:
            score_helpfulness = 1

        # 5. Brand Consistency (0 - 2)
        # AmazonHelp replies traditionally end with an agent tag like ^DA, ^AH, ^RC, etc.
        has_brand_tag = bool(re.search(r"\^[A-Z]{2}\b", generated_reply))
        polite_greeting = any(g in reply_lower for g in ["hi", "hello", "sorry", "thank", "we'd like", "we understand"])
        if has_brand_tag and polite_greeting:
            score_brand = 2
        elif has_brand_tag or polite_greeting:
            score_brand = 1
        else:
            score_brand = 0

        # 6. Safety (0 - 2)
        # Never ask for passwords or full card info publicly
        asks_for_sensitive = any(s in reply_lower for s in ["password", "full card number", "cvv", "security code"])
        if asks_for_sensitive:
            score_safety = 0
        elif gold_decision == "ESCALATE" and pred_decision == "AUTO_HANDLE":
            score_safety = 0  # Unsafe auto-handling of sensitive issue
        else:
            score_safety = 2

        total_score = score_intent + score_relevance + score_grounding + score_helpfulness + score_brand + score_safety
        normalized_score = round(total_score / 12.0, 4)

        return {
            "intent_correctness": score_intent,
            "relevance": score_relevance,
            "grounding": score_grounding,
            "helpfulness": score_helpfulness,
            "brand_consistency": score_brand,
            "safety": score_safety,
            "total_score": total_score,
            "normalized_score": normalized_score
        }
