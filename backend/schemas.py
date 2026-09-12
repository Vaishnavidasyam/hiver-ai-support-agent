"""Pydantic data schemas for request, response, internal structures, and evaluation."""

from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field


class CustomerQueryRequest(BaseModel):
    message: str = Field(..., description="Incoming customer message or tweet", example="My package is 3 days late. Where is it?")
    conversation_context: Optional[List[str]] = Field(default_factory=list, description="Prior conversation turns if available", example=[])


class IntentResult(BaseModel):
    label: str = Field(..., description="Predicted intent label")
    confidence: float = Field(..., description="Calibrated confidence score [0.0 - 1.0]")
    all_scores: Optional[Dict[str, float]] = Field(default=None, description="Confidence distribution across candidate intents")


class EvidenceCase(BaseModel):
    case_id: str = Field(..., description="Unique conversation or precedent ID")
    similarity: float = Field(..., description="Cosine similarity score [0.0 - 1.0]")
    customer_message: str = Field(..., description="Historical customer message")
    historical_response: str = Field(..., description="Historical brand resolution")
    intent: str = Field(..., description="Intent of historical case")


class EvidenceSufficiencyResult(BaseModel):
    is_sufficient: bool = Field(..., description="Whether evidence is sufficient for automated handling")
    top_similarity: float = Field(..., description="Similarity of top precedent")
    agreement_score: float = Field(..., description="Consensus ratio among top retrieved precedents")
    reason: str = Field(..., description="Explanation of evidence quality assessment")


class AgentResponse(BaseModel):
    intent: IntentResult
    evidence: List[EvidenceCase]
    evidence_sufficient: bool
    reply: str
    decision: str = Field(..., description="AUTO_HANDLE or ESCALATE")
    decision_reason_code: str
    decision_reason: str
    latency_ms: float = 0.0


class GoldenEvaluationItem(BaseModel):
    example_id: str
    conversation_id: str
    customer_message: str
    context: List[str]
    gold_intent: str
    gold_decision: str
    gold_reason_code: str
    gold_reason: str
    gold_reference_reply: str
    reply_guidelines: Dict[str, Any]


class MetricResult(BaseModel):
    metric_name: str
    value: float
    description: str


class EvaluationReport(BaseModel):
    model_name: str
    golden_set_size: int
    intent_metrics: Dict[str, float]
    retrieval_metrics: Dict[str, float]
    reply_quality_metrics: Dict[str, float]
    escalation_metrics: Dict[str, float]
    judge_agreement: Dict[str, float]
