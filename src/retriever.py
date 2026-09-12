"""Dense historical retrieval using FAISS and Transformer embeddings."""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
import faiss

from src.config import (
    TOP_K_RETRIEVAL, PROCESSED_DIR, SIMILARITY_THRESHOLD
)
from src.embeddings import DenseEmbedder
from src.preprocessor import normalize_tweet_text
from backend.schemas import EvidenceCase


class HistoricalRetriever:
    """FAISS-based historical support precedent retrieval system."""

    def __init__(self):
        self._embedder: Optional[DenseEmbedder] = None
        self.index: Optional[faiss.IndexFlatIP] = None
        self.metadata: Optional[pd.DataFrame] = None
        self.index_file = PROCESSED_DIR / "faiss_index.bin"
        self.metadata_file = PROCESSED_DIR / "index_metadata.parquet"

    @property
    def embedder(self) -> DenseEmbedder:
        if self._embedder is None:
            self._embedder = DenseEmbedder()
        return self._embedder

    def build_index(self, train_df: pd.DataFrame, batch_size: int = 256):
        """Encodes customer messages from training conversations and builds FAISS index."""
        texts = [normalize_tweet_text(t) for t in train_df["customer_message"].tolist()]
        
        embeddings = self.embedder.encode(
            texts,
            normalize_embeddings=True,
            batch_size=batch_size
        ).astype(np.float32)

        dimension = embeddings.shape[1]
        # Inner product on L2-normalized vectors is exactly Cosine Similarity
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings)

        self.metadata = train_df[["conversation_id", "customer_message", "brand_response", "intent"]].copy()

        # Save index and metadata
        faiss.write_index(self.index, str(self.index_file))
        self.metadata.to_parquet(self.metadata_file, index=False)

    def load_index(self) -> bool:
        """Loads existing FAISS index and metadata from disk if available."""
        if self.index_file.exists() and self.metadata_file.exists():
            self.index = faiss.read_index(str(self.index_file))
            self.metadata = pd.read_parquet(self.metadata_file)
            return True
        return False

    def retrieve(self, query: str, top_k: int = TOP_K_RETRIEVAL) -> List[EvidenceCase]:
        """Retrieves top-K most semantically similar historical precedents."""
        if self.index is None or self.metadata is None:
            if not self.load_index():
                raise RuntimeError("FAISS index not built or loaded.")

        cleaned_query = normalize_tweet_text(query)
        q_emb = self.embedder.encode([cleaned_query], normalize_embeddings=True).astype(np.float32)

        distances, indices = self.index.search(q_emb, top_k)
        
        results: List[EvidenceCase] = []
        for rank, (score, idx) in enumerate(zip(distances[0], indices[0])):
            if idx == -1:
                continue
            row = self.metadata.iloc[int(idx)]
            results.append(
                EvidenceCase(
                    case_id=str(row["conversation_id"]),
                    similarity=round(float(score), 4),
                    customer_message=str(row["customer_message"]),
                    historical_response=str(row["brand_response"]),
                    intent=str(row["intent"])
                )
            )

        return results
