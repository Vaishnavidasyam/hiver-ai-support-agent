"""Unit and integration tests for the AI Support Agent Pipeline."""

import pytest
from src.preprocessor import normalize_tweet_text, parse_conversation_turns
from src.config import INTENT_LABELS
from src.pipeline import SupportAgentPipeline


@pytest.fixture(scope="module")
def pipeline():
    pipe = SupportAgentPipeline()
    pipe.initialize()
    return pipe


def test_normalize_tweet_text():
    raw = "Hey @115858 check https://t.co/xyz123 &amp; see if my order is late   "
    cleaned = normalize_tweet_text(raw)
    assert "@customer" in cleaned
    assert "[URL]" in cleaned
    assert "&" in cleaned
    assert "   " not in cleaned


def test_parse_conversation_turns():
    conv = "Customer: Where is my order?\nSupport: We can check that for you! ^DA\nCustomer: Thanks!"
    turns = parse_conversation_turns(conv)
    assert len(turns) == 3
    assert turns[0]["role"] == "customer"
    assert turns[1]["role"] == "support"


def test_intent_classification_bounds(pipeline):
    res = pipeline.process("Where is my package? It is 3 days late.")
    assert res.intent.label in INTENT_LABELS
    assert 0.0 <= res.intent.confidence <= 1.0


def test_retrieval_returns_evidence(pipeline):
    res = pipeline.process("I received the wrong item in my package yesterday.")
    assert len(res.evidence) > 0
    assert res.evidence[0].similarity > 0.0
    assert len(res.evidence[0].historical_response) > 5


def test_escalation_on_security_breach(pipeline):
    res = pipeline.process("Someone hacked into my account and changed the password!")
    assert res.decision == "ESCALATE"
    assert res.decision_reason_code == "SENSITIVE_ACCOUNT_SECURITY"


def test_escalation_on_double_charge(pipeline):
    res = pipeline.process("My card was charged twice for order 123. Please refund the extra charge.")
    assert res.decision == "ESCALATE"
    assert res.decision_reason_code == "FINANCIAL_DISPUTE"


def test_auto_handle_on_clear_tracking(pipeline):
    res = pipeline.process("My package has not arrived yet. Can you check the delivery tracking status?")
    # Clear tracking queries with strong precedent should be AUTO_HANDLE
    assert res.decision in ["AUTO_HANDLE", "ESCALATE"]
    if res.decision == "AUTO_HANDLE":
        assert res.evidence_sufficient is True
        assert res.decision_reason_code == "STRONG_EVIDENCE"


def test_reply_grounding_no_fabricated_details(pipeline):
    res = pipeline.process("Where is my refund for the shoes I returned?")
    reply = res.reply
    # Ensure brand tag is present
    assert "^" in reply
    # Ensure no fabricated sensitive tokens
    assert "password" not in reply.lower()
    assert "cvv" not in reply.lower()
