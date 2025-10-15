import faiss
import numpy as np
import os
import json
from typing import List, Dict, Tuple
from utils import logger

class FaissStore:
    def __init__(self, dim: int, index_path: str = "faiss.index", meta_path: str = "meta.json"):
        self.dim = dim
        self.index_path = index_path
        self.meta_path = meta_path
        self.index = faiss.IndexFlatIP(dim)  # inner-product on normalized vectors
        self.id_to_meta: Dict[str, Dict] = {}
        self._trained = True

    def add(self, ids: List[str], embeddings: List[List[float]], metadatas: List[Dict]):
        arr = np.array(embeddings, dtype=np.float32)
        # normalize for IP similarity
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        arr = arr / norms
        self.index.add(arr)
        # map index position to id & meta
        base_pos = len(self.id_to_meta)
        for i, _id in enumerate(ids):
            self.id_to_meta[str(base_pos + i)] = {"id": _id, "meta": metadatas[i]}

    def search(self, query_emb: List[float], top_k: int = 5) -> List[Tuple[Dict, float]]:
        import numpy as np
        q = np.array(query_emb, dtype=np.float32).reshape(1, -1)
        q = q / (np.linalg.norm(q) + 1e-12)
        D, I = self.index.search(q, top_k)
        results = []
        for dist, idx in zip(D[0], I[0]):
            if idx == -1:
                continue
            meta = self.id_to_meta.get(str(idx), None)
            if meta:
                results.append((meta, float(dist)))
        return results

    def save(self):
        faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(self.id_to_meta, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved FAISS index to {self.index_path} and meta to {self.meta_path}")

    def load(self):
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
        if os.path.exists(self.meta_path):
            with open(self.meta_path, "r", encoding="utf-8") as f:
                self.id_to_meta = json.load(f)
        logger.info("Loaded FAISS index and metadata")
