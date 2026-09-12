"""Intent classification models: Majority baseline, TF-IDF + Logistic Regression, and Semantic Classifier."""

import pickle
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, List, Optional
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src.config import (
    INTENT_LABELS, INTENT_TAXONOMY, INTENT_CONFIDENCE_THRESHOLD,
    PROCESSED_DIR
)
from src.embeddings import DenseEmbedder
from src.preprocessor import normalize_tweet_text


class MajorityIntentBaseline:
    """Baseline 1: Always predicts the most frequent training intent."""

    def __init__(self):
        self.majority_intent: str = "delivery_delay_tracking"
        self.confidence: float = 0.50

    def fit(self, texts: List[str], labels: List[str]):
        counts = pd.Series(labels).value_counts()
        self.majority_intent = counts.index[0]
        self.confidence = float(counts.iloc[0] / len(labels))

    def predict(self, text: str) -> Tuple[str, float, Dict[str, float]]:
        scores = {intent: (self.confidence if intent == self.majority_intent else 0.0) for intent in INTENT_LABELS}
        return self.majority_intent, self.confidence, scores


class TfidfLogisticBaseline:
    """Baseline 2: Traditional TF-IDF n-grams with Logistic Regression."""

    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words="english")
        self.model = LogisticRegression(max_iter=500, class_weight="balanced", random_state=42)
        self.is_fitted = False

    def fit(self, texts: List[str], labels: List[str]):
        cleaned_texts = [normalize_tweet_text(t) for t in texts]
        X = self.vectorizer.fit_transform(cleaned_texts)
        self.model.fit(X, labels)
        self.is_fitted = True

    def predict(self, text: str) -> Tuple[str, float, Dict[str, float]]:
        if not self.is_fitted:
            return "unknown_ambiguous", 0.0, {}

        cleaned = normalize_tweet_text(text)
        X = self.vectorizer.transform([cleaned])
        probs = self.model.predict_proba(X)[0]
        classes = self.model.classes_

        scores = {cls: float(probs[i]) for i, cls in enumerate(classes)}
        for intent in INTENT_LABELS:
            scores.setdefault(intent, 0.0)

        best_intent = max(scores.items(), key=lambda x: x[1])[0]
        best_conf = float(scores[best_intent])

        return best_intent, best_conf, scores


class SemanticIntentClassifier:
    """Proposed: Dense embedding representation with precomputed prototypes and calibrated confidence."""

    def __init__(self):
        self._embedder: Optional[DenseEmbedder] = None
        self.intent_prototypes: Dict[str, np.ndarray] = {}
        self.prototype_file = PROCESSED_DIR / "intent_prototypes.pkl"
        self.is_fitted = False

    @property
    def embedder(self) -> DenseEmbedder:
        if self._embedder is None:
            self._embedder = DenseEmbedder()
        return self._embedder

    def load_prototypes(self) -> bool:
        if self.prototype_file.exists():
            with open(self.prototype_file, "rb") as f:
                self.intent_prototypes = pickle.load(f)
            self.is_fitted = True
            return True
        return False

    def fit(self, texts: List[str], labels: List[str]):
        """Computes dense centroid embeddings for each intent and caches to disk."""
        if self.load_prototypes():
            return

        df = pd.DataFrame({"text": texts, "label": labels})
        
        for intent in INTENT_LABELS:
            examples = df[df["label"] == intent]["text"].tolist()
            desc = INTENT_TAXONOMY[intent]["description"]
            kws = " ".join(INTENT_TAXONOMY[intent].get("keywords", []))
            canonical = f"{desc} {kws}"
            
            all_texts = [canonical] + examples[:80]
            embs = self.embedder.encode(all_texts, normalize_embeddings=True)
            
            weights = np.ones(len(all_texts))
            weights[0] = 3.0
            weights /= weights.sum()
            prototype = np.dot(weights, embs)
            prototype /= np.linalg.norm(prototype)
            self.intent_prototypes[intent] = prototype

        with open(self.prototype_file, "wb") as f:
            pickle.dump(self.intent_prototypes, f)

        self.is_fitted = True

    def predict(self, text: str) -> Tuple[str, float, Dict[str, float]]:
        if not self.is_fitted:
            if not self.load_prototypes():
                raise RuntimeError("Semantic intent classifier not fitted or loaded.")

        cleaned = normalize_tweet_text(text)
        if len(cleaned.split()) < 3:
            return "unknown_ambiguous", 0.90, {k: (0.90 if k == "unknown_ambiguous" else 0.01) for k in INTENT_LABELS}

        emb = self.embedder.encode([cleaned], normalize_embeddings=True)[0]

        similarities = {}
        for intent, proto in self.intent_prototypes.items():
            sim = float(np.dot(emb, proto))
            similarities[intent] = sim

        best_intent = max(similarities.items(), key=lambda x: x[1])[0]
        raw_top_sim = similarities[best_intent]

        # Calibrated confidence mapped to [0.0, 1.0]
        # In 384-dim space, similarity >= 0.50 is strong semantic alignment
        calibrated_conf = float(np.clip((raw_top_sim - 0.20) / 0.45, 0.05, 0.99))

        # Temperature-scaled probability distribution across all intents
        temp = 0.05
        exp_sims = {k: np.exp((v - raw_top_sim) / temp) for k, v in similarities.items()}
        total_exp = sum(exp_sims.values())
        probs = {k: round(float(v / total_exp), 4) for k, v in exp_sims.items()}

        # Uncertainty threshold: if similarity is too low or intent is ambiguous
        if raw_top_sim < 0.38 or calibrated_conf < INTENT_CONFIDENCE_THRESHOLD:
            return "unknown_ambiguous", round(calibrated_conf, 3), probs

        return best_intent, round(calibrated_conf, 3), probs
