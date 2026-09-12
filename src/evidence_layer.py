"""Evidence sufficiency assessment and precedent consensus analysis."""

from typing import List, Tuple
from collections import Counter

from src.config import (
    STRONG_EVIDENCE_THRESHOLD, SIMILARITY_THRESHOLD, CONSENSUS_THRESHOLD,
    INTENT_TAXONOMY
)
from backend.schemas import EvidenceCase, EvidenceSufficiencyResult


class EvidenceLayer:
    """Evaluates whether retrieved historical precedents provide sufficient grounded evidence."""

    def evaluate_evidence(
        self,
        customer_message: str,
        predicted_intent: str,
        retrieved_cases: List[EvidenceCase]
    ) -> EvidenceSufficiencyResult:
        if not retrieved_cases:
            return EvidenceSufficiencyResult(
                is_sufficient=False,
                top_similarity=0.0,
                agreement_score=0.0,
                reason="No historical precedents retrieved from knowledge base."
            )

        top_similarity = retrieved_cases[0].similarity
        
        # 1. Similarity Check
        if top_similarity < SIMILARITY_THRESHOLD:
            return EvidenceSufficiencyResult(
                is_sufficient=False,
                top_similarity=top_similarity,
                agreement_score=0.0,
                reason=f"Top historical precedent similarity ({top_similarity:.2f}) is below minimum threshold ({SIMILARITY_THRESHOLD})."
            )

        # 2. Precedent Intent Consensus
        intents = [c.intent for c in retrieved_cases]
        intent_counts = Counter(intents)
        majority_count = intent_counts.most_common(1)[0][1]
        agreement_score = round(majority_count / len(retrieved_cases), 3)

        # 3. Intent Alignment
        top_intent_match = (retrieved_cases[0].intent == predicted_intent)

        # 4. Consensus threshold check
        if agreement_score < CONSENSUS_THRESHOLD and not top_intent_match:
            return EvidenceSufficiencyResult(
                is_sufficient=False,
                top_similarity=top_similarity,
                agreement_score=agreement_score,
                reason=f"Retrieved precedents show divergent intents ({dict(intent_counts)}) without consensus."
            )

        # 5. Sensitive Domain Override
        is_sensitive = INTENT_TAXONOMY.get(predicted_intent, {}).get("sensitive", False)
        if is_sensitive:
            return EvidenceSufficiencyResult(
                is_sufficient=False,
                top_similarity=top_similarity,
                agreement_score=agreement_score,
                reason=f"Intent '{predicted_intent}' is a sensitive account/financial category requiring human handling."
            )

        # 6. Strong Evidence Pass
        if top_similarity >= STRONG_EVIDENCE_THRESHOLD and (top_intent_match or agreement_score >= CONSENSUS_THRESHOLD):
            return EvidenceSufficiencyResult(
                is_sufficient=True,
                top_similarity=top_similarity,
                agreement_score=agreement_score,
                reason=f"Strong evidence: top similarity {top_similarity:.2f} with {agreement_score*100:.0f}% precedent consensus."
            )

        # Moderate evidence pass
        is_sufficient = top_similarity >= SIMILARITY_THRESHOLD and top_intent_match
        reason = (
            f"Acceptable evidence: top similarity {top_similarity:.2f} aligned with intent '{predicted_intent}'."
            if is_sufficient
            else f"Insufficient evidence quality (similarity: {top_similarity:.2f}, consensus: {agreement_score:.2f})."
        )

        return EvidenceSufficiencyResult(
            is_sufficient=is_sufficient,
            top_similarity=top_similarity,
            agreement_score=agreement_score,
            reason=reason
        )
