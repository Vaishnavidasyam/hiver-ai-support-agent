"""Test ensuring strict conversation-level isolation and zero data leakage."""

import json
import pandas as pd
from src.config import GOLDEN_DATA_PATH, PROCESSED_DIR


def test_zero_data_leakage_between_train_and_golden():
    """Asserts that no conversation in the Golden Evaluation Set appears in the Training Knowledge Base."""
    assert GOLDEN_DATA_PATH.exists(), "Golden set must exist."
    train_path = PROCESSED_DIR / "train_conversations.parquet"
    assert train_path.exists(), "Train dataset must exist."

    with open(GOLDEN_DATA_PATH, "r", encoding="utf-8") as f:
        golden_set = json.load(f)

    golden_conv_ids = set(item["conversation_id"] for item in golden_set)
    assert len(golden_conv_ids) > 0

    train_df = pd.read_parquet(train_path)
    train_conv_ids = set(train_df["conversation_id"].tolist())

    # Critical Assertion: Zero overlap
    overlap = golden_conv_ids.intersection(train_conv_ids)
    assert len(overlap) == 0, f"Critical Data Leakage Detected! {len(overlap)} conversation IDs appeared in both train and golden set: {overlap}"


def test_zero_data_leakage_in_faiss_index_metadata():
    """Asserts that the FAISS retrieval index metadata contains zero golden test conversations."""
    index_meta_path = PROCESSED_DIR / "index_metadata.parquet"
    assert index_meta_path.exists(), "FAISS metadata must exist."

    with open(GOLDEN_DATA_PATH, "r", encoding="utf-8") as f:
        golden_set = json.load(f)

    golden_conv_ids = set(item["conversation_id"] for item in golden_set)

    meta_df = pd.read_parquet(index_meta_path)
    indexed_conv_ids = set(meta_df["conversation_id"].tolist())

    overlap = golden_conv_ids.intersection(indexed_conv_ids)
    assert len(overlap) == 0, f"FAISS Index Contaminated! {len(overlap)} golden conversations found in retrieval index: {overlap}"
