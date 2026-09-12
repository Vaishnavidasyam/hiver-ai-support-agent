"""Reconstructs customer support conversations directly from raw Kaggle twcs.csv or archive.zip."""

import os
import re
import zipfile
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import pandas as pd

from src.preprocessor import normalize_tweet_text

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def extract_amazon_conversations_from_twcs(
    source_path: str,
    max_pairs: int = 10000,
    chunk_size: int = 50000
) -> pd.DataFrame:
    """Reads twcs.csv directly (either raw CSV or from inside archive.zip), filters for

    AmazonHelp and inbound customer queries, joins customer message with support replies,
    and returns a clean DataFrame of conversation pairs.
    """
    logger.info(f"Checking dataset source: {source_path}")
    source = Path(source_path)

    # Determine if source is zip file or direct CSV
    if source.suffix.lower() == ".zip":
        logger.info(f"Opening archive: {source}")
        with zipfile.ZipFile(source, "r") as z:
            csv_name = [n for n in z.namelist() if n.endswith("twcs.csv")][0]
            with z.open(csv_name) as f:
                return _process_twcs_stream(f, chunk_size=chunk_size, max_pairs=max_pairs)
    else:
        logger.info(f"Opening CSV directly: {source}")
        with open(source, "rb") as f:
            return _process_twcs_stream(f, chunk_size=chunk_size, max_pairs=max_pairs)


def _process_twcs_stream(stream, chunk_size: int = 50000, max_pairs: int = 10000) -> pd.DataFrame:
    """Streams twcs.csv in chunks to minimize memory consumption."""
    customer_tweets: Dict[str, Dict] = {}
    support_replies: Dict[str, Dict] = {}

    total_read = 0
    logger.info("Streaming and parsing twcs.csv chunks...")

    for chunk in pd.read_csv(
        stream,
        chunksize=chunk_size,
        dtype=str,
        usecols=["tweet_id", "author_id", "inbound", "text", "response_tweet_id", "in_response_to_tweet_id"]
    ):
        total_read += len(chunk)

        # Separate AmazonHelp responses vs customer inquiries
        for _, row in chunk.iterrows():
            t_id = str(row["tweet_id"])
            author = str(row["author_id"])
            inbound = str(row["inbound"]).lower() == "true"
            text = str(row["text"])
            in_reply_to = str(row["in_response_to_tweet_id"]) if pd.notna(row["in_response_to_tweet_id"]) else None

            if author == "AmazonHelp":
                if in_reply_to and in_reply_to != "nan":
                    support_replies[in_reply_to] = {
                        "reply_id": t_id,
                        "reply_text": normalize_tweet_text(text)
                    }
            elif inbound:
                # Customer query
                customer_tweets[t_id] = {
                    "customer_message": normalize_tweet_text(text)
                }

        # Check if we have gathered enough paired conversations
        matched_keys = set(customer_tweets.keys()).intersection(set(support_replies.keys()))
        if len(matched_keys) >= max_pairs:
            logger.info(f"Reached target of {len(matched_keys)} matched conversation pairs.")
            break

        if total_read >= 500000:
            # Process up to 500k rows to guarantee fast execution under 1 minute
            break

    logger.info(f"Read {total_read} raw tweets. Reconstructing conversation pairs...")

    pairs = []
    matched_ids = list(set(customer_tweets.keys()).intersection(set(support_replies.keys())))[:max_pairs]

    for cust_id in matched_ids:
        c_text = customer_tweets[cust_id]["customer_message"]
        s_text = support_replies[cust_id]["reply_text"]

        # Filter non-English or trivial single-word queries
        if len(c_text.split()) < 3 or len(s_text.split()) < 4:
            continue

        pairs.append({
            "conversation_id": cust_id,
            "customer_message": c_text,
            "brand_response": s_text,
            "context": [f"Customer: {c_text}"]
        })

    df = pd.DataFrame(pairs)
    logger.info(f"Successfully reconstructed {len(df)} AmazonHelp conversation pairs from raw twcs.")
    return df
