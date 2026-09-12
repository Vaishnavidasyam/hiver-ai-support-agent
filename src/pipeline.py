"""Unified end-to-end Support Agent Pipeline coordinating classification, retrieval, evidence assessment, reply drafting, and escalation."""

import time
import pickle
from typing import List, Optional
import pandas as pd

from src.config import PROCESSED_DIR, SELECTED_BRAND
from src.preprocessor import normalize_tweet_text
from src.intent_classifier import (
    SemanticIntentClassifier, MajorityIntentBaseline, TfidfLogisticBaseline
)
from src.retriever import HistoricalRetriever
from src.evidence_layer import EvidenceLayer
from src.reply_generator import GroundedReplyGenerator
from src.escalation_engine import EscalationEngine
from backend.schemas import (
    AgentResponse, IntentResult, EvidenceCase, EvidenceSufficiencyResult
)


class SupportAgentPipeline:
    """Production pipeline for the Evidence-Grounded AI Support Agent."""

    def __init__(self, brand_name: str = SELECTED_BRAND):
        self.brand_name = brand_name
        self.intent_classifier = SemanticIntentClassifier()
        self.retriever = HistoricalRetriever()
        self.evidence_layer = EvidenceLayer()
        self.reply_generator = GroundedReplyGenerator()
        self.escalation_engine = EscalationEngine()

        # Baselines
        self.majority_baseline = MajorityIntentBaseline()
        self.tfidf_baseline = TfidfLogisticBaseline()

        self.is_initialized = False

    def initialize(self):
        """Loads cached models and FAISS retrieval index from disk."""
        if self.is_initialized:
            return

        maj_file = PROCESSED_DIR / "majority_model.pkl"
        tfidf_file = PROCESSED_DIR / "tfidf_model.pkl"
        train_path = PROCESSED_DIR / "train_conversations.parquet"

        if maj_file.exists() and tfidf_file.exists() and self.intent_classifier.load_prototypes():
            with open(maj_file, "rb") as f:
                self.majority_baseline = pickle.load(f)
            with open(tfidf_file, "rb") as f:
                self.tfidf_baseline = pickle.load(f)
        else:
            if not train_path.exists():
                raise FileNotFoundError(f"Training data not found at {train_path}. Run prepare_data.py first.")
            train_df = pd.read_parquet(train_path)
            texts = train_df["customer_message"].tolist()
            intents = train_df["intent"].tolist()
            self.majority_baseline.fit(texts, intents)
            self.tfidf_baseline.fit(texts, intents)
            self.intent_classifier.fit(texts, intents)

        # Load FAISS index
        if not self.retriever.load_index():
            if not train_path.exists():
                raise FileNotFoundError(f"Training data not found at {train_path}.")
            train_df = pd.read_parquet(train_path)
            self.retriever.build_index(train_df)

        self.is_initialized = True

    def process(self, message: str, context: Optional[List[str]] = None) -> AgentResponse:
        """Processes an incoming customer message through the 5-step evidence-grounded pipeline."""
        start_time = time.perf_counter()
        if not self.is_initialized:
            self.initialize()

        cleaned_query = normalize_tweet_text(message)

        # Step 1: Semantic Intent Classification
        pred_intent, conf, scores = self.intent_classifier.predict(cleaned_query)
        intent_res = IntentResult(label=pred_intent, confidence=conf, all_scores=scores)

        # Step 2: Historical Precedent Retrieval (Top-K)
        evidence = self.retriever.retrieve(cleaned_query, top_k=5)

        # Step 3: Evidence Sufficiency Assessment
        ev_result = self.evidence_layer.evaluate_evidence(cleaned_query, pred_intent, evidence)

        # Step 4: Risk & Escalation Decision
        decision, reason_code, reason_desc = self.escalation_engine.decide(intent_res, ev_result)

        # Step 5: Grounded Reply Generation
        reply = self.reply_generator.generate_reply(
            customer_message=cleaned_query,
            predicted_intent=pred_intent,
            evidence=evidence,
            is_sufficient=ev_result.is_sufficient,
            decision=decision
        )

        latency = (time.perf_counter() - start_time) * 1000.0

        return AgentResponse(
            intent=intent_res,
            evidence=evidence,
            evidence_sufficient=ev_result.is_sufficient,
            reply=reply,
            decision=decision,
            decision_reason_code=reason_code,
            decision_reason=reason_desc,
            latency_ms=round(latency, 2)
        )
