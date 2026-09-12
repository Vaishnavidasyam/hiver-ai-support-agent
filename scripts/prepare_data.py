"""Data preparation, conversation extraction, leakage-free splitting, and golden set generation.

Supports loading from:
1. Reconstructed Parquet (data/raw/conversations_raw.parquet)
2. Raw Kaggle twcs.csv or archive.zip (directly from archive folder or Downloads)
"""

import os
import sys
import argparse
import re
import json
import logging
from pathlib import Path
import pandas as pd
import numpy as np

from src.config import (
    RAW_DATA_PATH, PROCESSED_DIR, GOLDEN_DATA_PATH,
    RANDOM_SEED, INTENT_TAXONOMY, INTENT_LABELS
)
from src.preprocessor import extract_customer_support_pair, normalize_tweet_text
from src.conversation_builder import extract_amazon_conversations_from_twcs

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

ENGLISH_WORD_PATTERN = re.compile(
    r"\b(the|is|to|and|my|you|we|for|in|on|have|order|delivery|help|package|refund|return|tracking|prime)\b",
    re.IGNORECASE
)


def heuristic_intent(text: str) -> str:
    text_lower = text.lower()
    
    if any(k in text_lower for k in ["hacked", "account locked", "unauthorized access", "otp", "login issue", "password reset", "two factor", "2fa", "someone logged into"]):
        return "account_security_login"
    
    if any(k in text_lower for k in ["charged twice", "double charge", "unknown charge", "deducted twice", "unauthorized charge", "credit card charged"]):
        return "payment_billing_issue"
    
    if any(k in text_lower for k in ["damaged", "broken", "defective", "wrong item", "missing parts", "shattered", "opened package", "received empty"]):
        return "damaged_defective_wrong_item"
    
    if any(k in text_lower for k in ["refund", "return", "returned", "send back", "drop off", "exchange", "money back", "pickup"]):
        return "refund_return_status"
    
    if any(k in text_lower for k in ["cancel order", "cancellation", "change address", "wrong address", "modify order", "stop shipment"]):
        return "cancellation_modification"
    
    if any(k in text_lower for k in ["prime membership", "prime video", "prime delivery", "prime student", "charged for prime"]):
        return "prime_membership_benefits"
    
    if any(k in text_lower for k in ["tracking", "track", "delivery", "delivered", "carrier", "courier", "late", "delay", "where is my package", "where's my order", "hasn't arrived", "not arrived"]):
        return "delivery_delay_tracking"
    
    if any(k in text_lower for k in ["in stock", "restock", "available", "product details", "warranty", "specifications", "when will you have"]):
        return "product_stock_inquiry"
    
    if any(k in text_lower for k in ["rude", "worst customer service", "terrible service", "horrible driver", "unacceptable behavior", "disgusted"]):
        return "feedback_complaint"
    
    if len(text.split()) < 4 or any(k in text_lower for k in ["help me", "still waiting", "hello?", "anyone there", "dm sent", "check dm"]):
        return "unknown_ambiguous"
        
    return "delivery_delay_tracking"


def find_archive_dataset() -> Optional[str]:
    """Checks standard locations for archive.zip or twcs.csv."""
    candidates = [
        Path(r"C:\Users\D Vaishnavi\Downloads\archive.zip"),
        Path(r"data\raw\archive.zip"),
        Path(r"data\raw\twcs.csv"),
        Path(r"data\raw\twcs\twcs.csv"),
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    return None


def main():
    parser = argparse.ArgumentParser(description="Prepare data from archive.zip, twcs.csv, or parquet.")
    parser.add_argument("--source", type=str, default=None, help="Path to archive.zip or twcs.csv")
    args = parser.parse_args()

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    GOLDEN_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    archive_src = args.source or find_archive_dataset()

    if archive_src and Path(archive_src).exists() and (archive_src.endswith(".zip") or archive_src.endswith(".csv")):
        logger.info(f"Found archive dataset at: {archive_src}. Reconstructing from raw twcs...")
        df_clean = extract_amazon_conversations_from_twcs(archive_src, max_pairs=15000)
        df_clean["intent"] = df_clean["customer_message"].apply(heuristic_intent)
    elif RAW_DATA_PATH.exists():
        logger.info(f"Loading raw dataset from {RAW_DATA_PATH}...")
        df_raw = pd.read_parquet(RAW_DATA_PATH)
        amz_df = df_raw[df_raw["company"] == "AmazonHelp"].copy()

        records = []
        for _, row in amz_df.iterrows():
            conv_id = str(row["conversation_id"])
            conv_text = str(row["conversation"])

            pair = extract_customer_support_pair(conv_text)
            if not pair:
                continue

            cust_msg, supp_reply, context = pair
            if not ENGLISH_WORD_PATTERN.search(cust_msg) or not ENGLISH_WORD_PATTERN.search(supp_reply):
                continue

            if len(cust_msg.split()) < 3 or len(supp_reply.split()) < 4:
                continue

            intent = heuristic_intent(cust_msg)
            records.append({
                "conversation_id": conv_id,
                "customer_message": cust_msg,
                "brand_response": supp_reply,
                "context": context,
                "intent": intent,
            })
        df_clean = pd.DataFrame(records)
    else:
        raise FileNotFoundError(f"Neither archive.zip nor {RAW_DATA_PATH} found.")

    # Deduplicate
    df_clean = df_clean.drop_duplicates(subset=["customer_message"]).reset_index(drop=True)
    logger.info(f"Total unique conversation pairs available: {len(df_clean)}")

    # Leakage-Free Conversation-Level Split
    np.random.seed(RANDOM_SEED)
    shuffled_indices = np.random.permutation(len(df_clean))
    train_size = min(8000, int(len(df_clean) * 0.85))
    train_indices = shuffled_indices[:train_size]
    test_indices = shuffled_indices[train_size:]

    train_df = df_clean.iloc[train_indices].copy().reset_index(drop=True)
    test_pool_df = df_clean.iloc[test_indices].copy().reset_index(drop=True)

    logger.info(f"Split completed: Train (Knowledge Base) = {len(train_df)}, Test Pool = {len(test_pool_df)}")

    train_path = PROCESSED_DIR / "train_conversations.parquet"
    test_path = PROCESSED_DIR / "test_pool.parquet"

    train_df.to_parquet(train_path, index=False)
    test_pool_df.to_parquet(test_path, index=False)
    logger.info(f"Saved processed datasets to {train_path} and {test_path}.")

    if not GOLDEN_DATA_PATH.exists():
        logger.info("Building Golden Evaluation Set (200 examples)...")
        # Build golden set
        from scripts.prepare_data import build_golden_set
        golden_examples = build_golden_set(test_pool_df)
        with open(GOLDEN_DATA_PATH, "w", encoding="utf-8") as f:
            json.dump(golden_examples, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved 200 Golden Evaluation examples to {GOLDEN_DATA_PATH}")


if __name__ == "__main__":
    main()
