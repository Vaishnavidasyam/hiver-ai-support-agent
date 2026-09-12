"""Precompute prototypes and fit baseline models so pipeline starts up in milliseconds."""

import pickle
import pandas as pd
from src.config import PROCESSED_DIR
from src.intent_classifier import MajorityIntentBaseline, TfidfLogisticBaseline, SemanticIntentClassifier

def main():
    train_path = PROCESSED_DIR / "train_conversations.parquet"
    train_df = pd.read_parquet(train_path)
    texts = train_df["customer_message"].tolist()
    labels = train_df["intent"].tolist()

    print("Fitting Majority baseline...")
    maj = MajorityIntentBaseline()
    maj.fit(texts, labels)
    with open(PROCESSED_DIR / "majority_model.pkl", "wb") as f:
        pickle.dump(maj, f)

    print("Fitting TF-IDF Logistic baseline...")
    tfidf = TfidfLogisticBaseline()
    tfidf.fit(texts, labels)
    with open(PROCESSED_DIR / "tfidf_model.pkl", "wb") as f:
        pickle.dump(tfidf, f)

    print("Computing and caching Semantic Intent Prototypes...")
    semantic = SemanticIntentClassifier()
    semantic.fit(texts, labels)
    print("All models successfully fitted and cached!")

if __name__ == "__main__":
    main()
