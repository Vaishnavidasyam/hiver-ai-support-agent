"""Configuration parameters for Hiver Evidence-Grounded AI Support Agent."""

import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "conversations_raw.parquet"
PROCESSED_DIR = DATA_DIR / "processed"
GOLDEN_DATA_PATH = DATA_DIR / "golden" / "golden_set.json"
REPORTS_DIR = BASE_DIR / "reports"

# Brand Configuration
SELECTED_BRAND = "AmazonHelp"
RANDOM_SEED = 42

# Intent Taxonomy (9 core brand intents + UNKNOWN)
INTENT_TAXONOMY = {
    "delivery_delay_tracking": {
        "description": "Late delivery, shipment tracking status, missing courier updates, delivery time inquiry.",
        "risk_level": "LOW",
        "sensitive": False,
        "keywords": ["track", "tracking", "delivery", "late", "package", "arrive", "delayed", "courier", "where is"]
    },
    "refund_return_status": {
        "description": "Return initiation, refund process timing, drop-off location, refund delay.",
        "risk_level": "LOW",
        "sensitive": False,
        "keywords": ["refund", "return", "returned", "money back", "pickup", "drop off", "exchange"]
    },
    "damaged_defective_wrong_item": {
        "description": "Received broken, damaged, expired, incorrect or missing items in package.",
        "risk_level": "MEDIUM",
        "sensitive": False,
        "keywords": ["damaged", "broken", "wrong item", "defective", "missing item", "scratched", "empty box"]
    },
    "cancellation_modification": {
        "description": "Cancelling an order, updating delivery address, modifying quantity or payment method.",
        "risk_level": "LOW",
        "sensitive": False,
        "keywords": ["cancel", "cancellation", "change address", "wrong address", "modify order", "stop order"]
    },
    "payment_billing_issue": {
        "description": "Payment decline, double charged, unauthorized fee, invoice request, payment method problem.",
        "risk_level": "HIGH",
        "sensitive": True,
        "keywords": ["charged twice", "double charge", "payment failed", "deducted", "billing", "unauthorized charge"]
    },
    "account_security_login": {
        "description": "Compromised account, OTP verification failure, locked account, password reset, 2FA.",
        "risk_level": "CRITICAL",
        "sensitive": True,
        "keywords": ["hacked", "account locked", "otp", "login", "password", "unauthorized access", "2fa", "security code"]
    },
    "prime_membership_benefits": {
        "description": "Amazon Prime subscription, video streaming issues, free delivery perks, student discount.",
        "risk_level": "LOW",
        "sensitive": False,
        "keywords": ["prime", "prime video", "membership", "subscription", "annual fee", "perks"]
    },
    "product_stock_inquiry": {
        "description": "Item availability, replenishment date, product specifications, compatibility.",
        "risk_level": "LOW",
        "sensitive": False,
        "keywords": ["in stock", "restock", "available", "product details", "specification", "warranty"]
    },
    "feedback_complaint": {
        "description": "General dissatisfaction with customer service or delivery driver behavior, feedback.",
        "risk_level": "MEDIUM",
        "sensitive": False,
        "keywords": ["rude", "worst service", "complaint", "terrible", "disappointed", "unacceptable", "driver behavior"]
    },
    "unknown_ambiguous": {
        "description": "Vague, incomplete, out-of-domain, or highly ambiguous query requiring clarification.",
        "risk_level": "HIGH",
        "sensitive": True,
        "keywords": []
    }
}

INTENT_LABELS = list(INTENT_TAXONOMY.keys())

# Embedding & Retrieval Settings
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K_RETRIEVAL = 5
SIMILARITY_THRESHOLD = 0.55  # Minimum cosine similarity for relevance
STRONG_EVIDENCE_THRESHOLD = 0.70  # Minimum similarity for strong evidence
CONSENSUS_THRESHOLD = 0.60  # Minimum proportion of top precedents agreeing on resolution pattern

# Decision & Escalation Settings
INTENT_CONFIDENCE_THRESHOLD = 0.58
AUTO_HANDLE_DECISION = "AUTO_HANDLE"
ESCALATE_DECISION = "ESCALATE"

# Reason Codes
REASON_CODES = {
    "STRONG_EVIDENCE": "High-confidence intent and consistent historical precedents support safe auto-handling.",
    "LOW_INTENT_CONFIDENCE": "Intent classification confidence is below threshold; human triage required.",
    "INSUFFICIENT_EVIDENCE": "No sufficiently similar historical resolution precedents found in knowledge base.",
    "CONFLICTING_EVIDENCE": "Retrieved historical precedents offer conflicting or divergent resolution paths.",
    "SENSITIVE_ACCOUNT_SECURITY": "Sensitive account security, authentication or fraud issue strictly requires human verification.",
    "FINANCIAL_DISPUTE": "Payment dispute, unauthorized charge, or billing discrepancy requires human agent handling.",
    "AMBIGUOUS_QUERY": "Query is ambiguous or lacks sufficient context to safely resolve without clarifying questions."
}
