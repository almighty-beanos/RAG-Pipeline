from typing import List, Dict
import numpy as np
from utils import logger

def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> List[Dict]:
    """Chunk by characters. Returns list of dicts: {chunk_id, text, start, end}"""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    step = chunk_size - overlap
    chunks = []
    pos = 0
    chunk_id = 0
    while pos < len(text):
        end = pos + chunk_size
        chunk_text_ = text[pos:end]
        chunks.append({
            "id": f"chunk_{chunk_id}",
            "text": chunk_text_,
            "start": pos,
            "end": min(end, len(text))
        })
        chunk_id += 1
        pos += step
    logger.info(f"Created {len(chunks)} chunks (size={chunk_size}, overlap={overlap})")
    return chunks

class EmbeddingModel:
    def embed(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError


class SentenceTransformerEmbeddings(EmbeddingModel):
    """Local embedding model using sentence-transformers"""
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: List[str]) -> List[List[float]]:
        embs = self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
        return [e.tolist() for e in embs]