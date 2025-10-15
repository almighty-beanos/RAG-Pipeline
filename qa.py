from typing import List, Dict
import time
from utils import logger

class QASystem:
    def __init__(self, vector_store, embedder, generator_client=None):
        self.vector_store = vector_store
        self.embedder = embedder
        self.generator = generator_client

        if self.generator is None:
            from transformers import pipeline
            logger.info("Using local transformer pipeline for generation (BART)")
            self.local_generator = pipeline(
                "summarization",  # better for long contexts
                model="facebook/bart-large-cnn"
            )
        else:
            self.local_generator = None

    def ask(self, question: str, top_k: int = 5) -> Dict:
        start_time = time.time()

        # Embed the question
        q_emb = self.embedder.embed([question])[0]

        # Retrieve top-k relevant chunks
        retrieved = self.vector_store.search(q_emb, top_k=top_k)
        min_score_threshold = 0.3
        relevant_chunks = [r for r in retrieved if r[1] >= min_score_threshold]
        retrieval_time = int((time.time() - start_time) * 1000)

        if not relevant_chunks:
            return {
                "answer": "not enough information",
                "sources": [],
                "timings": {"retrieval_ms": retrieval_time, "generation_ms": 0, "total_ms": retrieval_time}
            }

        # Extract context from retrieved tuples
        context_texts = [meta["meta"]["chunk_text"][:1000] for meta, score in retrieved]
        context = "\n\n".join(context_texts)

        gen_start = time.time()

        if self.generator:
            # Use external generator if provided
            answer = self.generator(f"Context:\n{context}\n\nQuestion: {question}")
        else:
            # Use local BART summarization to answer
            summary = self.local_generator(context, max_length=200, min_length=50, do_sample=False)
            answer = summary[0]["summary_text"]

        gen_time = int((time.time() - gen_start) * 1000)
        total_time = int((time.time() - start_time) * 1000)

        # Build sources list
        sources = [
            {"url": meta["meta"]["url"], "snippet": meta["meta"]["chunk_text"][:400], "score": float(score)}
            for meta, score in retrieved
        ]

        return {
            "answer": answer.strip(),
            "sources": sources,
            "timings": {"retrieval_ms": retrieval_time, "generation_ms": gen_time, "total_ms": total_time}
        }