"""Robust, dependency-free dense embedding model using PyTorch and HuggingFace Transformers."""

import os
from typing import List, Union
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

from src.config import EMBEDDING_MODEL_NAME


class DenseEmbedder:
    """Wrapper around HuggingFace Transformers for generating L2-normalized sentence embeddings with mean pooling."""

    def __init__(self, model_name: str = f"sentence-transformers/{EMBEDDING_MODEL_NAME}"):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._tokenizer = None
        self._model = None

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        return self._tokenizer

    @property
    def model(self):
        if self._model is None:
            self._model = AutoModel.from_pretrained(self.model_name)
            self._model.to(self.device)
            self._model.eval()
        return self._model

    def encode(
        self,
        sentences: Union[str, List[str]],
        batch_size: int = 128,
        normalize_embeddings: bool = True,
        show_progress_bar: bool = False
    ) -> np.ndarray:
        if isinstance(sentences, str):
            sentences = [sentences]

        all_embeddings = []

        for i in range(0, len(sentences), batch_size):
            batch = sentences[i : i + batch_size]
            encoded_input = self.tokenizer(
                batch, padding=True, truncation=True, max_length=128, return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                model_output = self.model(**encoded_input)
                # Mean Pooling - Take attention mask into account for correct averaging
                token_embeddings = model_output[0]  # First element contains hidden state
                input_mask_expanded = (
                    encoded_input["attention_mask"].unsqueeze(-1).expand(token_embeddings.size()).float()
                )
                sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
                sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                batch_embeddings = sum_embeddings / sum_mask

                if normalize_embeddings:
                    batch_embeddings = torch.nn.functional.normalize(batch_embeddings, p=2, dim=1)

                all_embeddings.append(batch_embeddings.cpu().numpy())

        return np.vstack(all_embeddings).astype(np.float32)
