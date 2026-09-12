"""Script to build the historical FAISS retrieval index from training conversations."""

import logging
import pandas as pd
from src.config import PROCESSED_DIR
from src.retriever import HistoricalRetriever

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def main():
    train_path = PROCESSED_DIR / "train_conversations.parquet"
    if not train_path.exists():
        raise FileNotFoundError(f"Processed training data not found at {train_path}. Run prepare_data.py first.")

    logger.info(f"Loading training conversations from {train_path}...")
    df_train = pd.read_parquet(train_path)
    logger.info(f"Loaded {len(df_train)} conversations.")

    retriever = HistoricalRetriever()
    logger.info("Building FAISS index (encoding embeddings)...")
    retriever.build_index(df_train, batch_size=256)
    logger.info(f"Successfully built and saved FAISS index to {retriever.index_file} and {retriever.metadata_file}")

if __name__ == "__main__":
    main()
