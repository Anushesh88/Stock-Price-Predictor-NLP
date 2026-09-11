import os
import hashlib
from typing import List, Optional, Union
import numpy as np
import torch
from tqdm import tqdm
from src.utils.logger import get_logger

logger = get_logger("embeddings")

class FinBERTEmbedder:
    """
    Extracts dense [CLS] financial NLP embeddings using ProsusAI/finbert with MD5-keyed disk caching.
    """
    def __init__(
        self,
        model_name: str = "ProsusAI/finbert",
        cache_dir: str = "data/processed/embedding_cache",
        device: Optional[str] = None
    ):
        self.model_name = model_name
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._tokenizer = None
        self._model = None

    def _load_model(self):
        if self._model is None or self._tokenizer is None:
            logger.info(f"Loading tokenizer & transformer model from '{self.model_name}' on device '{self.device}'...")
            from transformers import AutoTokenizer, AutoModel
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModel.from_pretrained(self.model_name).to(self.device)
            self._model.eval()

    def get_text_embedding(self, text: Union[str, float]) -> np.ndarray:
        """
        Computes or loads the 768-d [CLS] embedding for a text string from disk cache.
        """
        if not isinstance(text, str) or not text.strip():
            text = "No major news or filings reported for HDFC Bank on this trading day."

        # Compute MD5 hash of text for cache key
        text_hash = hashlib.md5(text.encode("utf-8")).hexdigest()
        cache_file = os.path.join(self.cache_dir, f"{text_hash}.npy")

        if os.path.exists(cache_file):
            return np.load(cache_file)

        self._load_model()

        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding="max_length"
        ).to(self.device)

        with torch.no_grad():
            outputs = self._model(**inputs)
            # Extract [CLS] token representation (shape: [1, 768])
            cls_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy().squeeze(0)

        # Save to disk cache
        np.save(cache_file, cls_embedding)
        return cls_embedding

    def extract_partition_embeddings(
        self,
        texts: List[str],
        output_path: Optional[str] = None,
        desc: str = "Partition"
    ) -> np.ndarray:
        """
        Extracts embeddings for an entire partition with progress bar.
        """
        logger.info(f"Extracting embeddings for {desc} ({len(texts)} samples)...")
        embeddings = []
        for text in tqdm(texts, desc=f"FinBERT [{desc}]"):
            emb = self.get_text_embedding(text)
            embeddings.append(emb)

        emb_array = np.array(embeddings, dtype=np.float32)

        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            np.save(output_path, emb_array)
            logger.info(f"Saved {desc} embeddings with shape {emb_array.shape} to: {output_path}")

        return emb_array
