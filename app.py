from fastapi import FastAPI
from pydantic import BaseModel
import os
from typing import Optional
from crawler import PoliteCrawler
from extractor import extract_main_text
from indexer import chunk_text, SentenceTransformerEmbeddings
from vectorstore import FaissStore
from qa import QASystem
from utils import logger

app = FastAPI()

# In-memory stores
GLOBAL = {
    "pages": {},   # url -> {"title","text","html"}
    "chunks": [],  # list of {"id","text","start","end","url","page_title"}
    "vector_store": None,
    "embedder": None,
    "qa": None
}

# ----- Request Models -----
class CrawlRequest(BaseModel):
    start_url: str
    max_pages: int = 50
    max_depth: int = 3
    crawl_delay_ms: int = 500

class IndexRequest(BaseModel):
    chunk_size: int = 800
    chunk_overlap: int = 100
    embedding_model: Optional[str] = "sentence-transformers"

class AskRequest(BaseModel):
    question: str
    top_k: int = 5

# ----- Routes -----
@app.post("/crawl")
def crawl(req: CrawlRequest):
    crawler = PoliteCrawler(req.start_url)
    results, skipped = crawler.crawl(
        max_pages=req.max_pages,
        max_depth=req.max_depth,
        crawl_delay_ms=req.crawl_delay_ms
    )
    count = 0
    for r in results:
        url = r["url"]
        extracted = extract_main_text(r["html"], url)
        GLOBAL["pages"][url] = {
            "title": extracted["title"],
            "text": extracted["text"],
            "html": extracted["cleaned_html"]
        }
        count += 1
    return {
        "page_count": count,
        "skipped_count": skipped,
        "urls": list(GLOBAL["pages"].keys())
    }

@app.post("/index")
def index(req: IndexRequest):
    # Always use SentenceTransformer embeddings (no OpenAI)
    embedder = SentenceTransformerEmbeddings()
    GLOBAL["embedder"] = embedder

    # Chunk pages
    chunks = []
    for url, page in GLOBAL["pages"].items():
        text = page["text"]
        if not text:
            continue
        page_chunks = chunk_text(text, chunk_size=req.chunk_size, overlap=req.chunk_overlap)
        for c in page_chunks:
            cmeta = {
                "id": c["id"],
                "text": c["text"],
                "start": c["start"],
                "end": c["end"],
                "url": url,
                "page_title": page.get("title", "")
            }
            chunks.append(cmeta)
    GLOBAL["chunks"] = chunks

    # Embed chunks
    texts = [c["text"] for c in chunks]
    if len(texts) == 0:
        return {"vector_count": 0, "errors": ["no chunks to index"]}

    embeddings = embedder.embed(texts)
    dim = len(embeddings[0])

    vs = FaissStore(dim=dim, index_path="faiss.index", meta_path="meta.json")
    ids = [c["id"] for c in chunks]
    metas = [
        {
            "url": c["url"],
            "chunk_text": c["text"],
            "page_title": c["page_title"],
            "start": c["start"],
            "end": c["end"]
        } for c in chunks
    ]
    vs.add(ids, embeddings, metas)
    vs.save()

    GLOBAL["vector_store"] = vs
    GLOBAL["qa"] = QASystem(vector_store=vs, embedder=embedder)

    return {"vector_count": len(embeddings), "errors": []}

@app.post("/ask")
def ask(req: AskRequest):
    if GLOBAL["qa"] is None:
        return {"error": "index not built; call /index first"}
    resp = GLOBAL["qa"].ask(req.question, top_k=req.top_k)
    return resp

@app.get("/")
def home():
    return {"message": "RAG QA system is running"}