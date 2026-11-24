import json
import os
import re
import threading
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent
PASSAGES = json.loads((ROOT / "knowledge_base.json").read_text(encoding="utf-8"))
if len(PASSAGES) != 50:
    raise RuntimeError(f"Expected 50 passages, found {len(PASSAGES)}")

app = FastAPI(title="RAG Colosseum", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the embedding model once at startup. The model is cached between requests.
_model = None
_embeddings = None
_model_lock = threading.Lock()

def tokenize(text: str):
    return re.findall(r"[a-z0-9]+(?:[-'][a-z0-9]+)?", text.lower())

def build_embeddings():
    global _model, _embeddings
    try:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(
            os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        )
        vecs = _model.encode(
            PASSAGES, normalize_embeddings=True,
            convert_to_numpy=True, show_progress_bar=False
        )
        _embeddings = np.asarray(vecs, dtype=np.float32)
    except Exception as e:
        # Keep the API alive if model download is temporarily unavailable.
        # Lexical fallback is deliberately available, but dense retrieval is preferred.
        print("Embedding model unavailable; using lexical fallback:", repr(e))
        _model = None
        _embeddings = None

@app.on_event("startup")
def startup():
    build_embeddings()

def lexical_scores(question: str):
    q = set(tokenize(question))
    scores = []
    for p in PASSAGES:
        words = set(tokenize(p))
        # Query coverage plus IDF-like rarity weighting.
        overlap = q & words
        scores.append(len(overlap) / max(1, len(q)) + 0.15 * len(overlap) / max(1, len(words)))
    a = np.asarray(scores, dtype=np.float32)
    if a.max() > 0:
        a = a / max(float(a.max()), 1e-8)
    return a

def retrieve(question: str, top_k: int):
    if _model is not None and _embeddings is not None:
        qv = _model.encode([question], normalize_embeddings=True, convert_to_numpy=True)[0]
        dense = np.dot(_embeddings, qv).astype(np.float32)
        lex = lexical_scores(question)
        # Dense semantic score is primary; lexical signal helps distinguish exact entities/numbers.
        combined = 0.82 * np.clip(dense, 0, 1) + 0.18 * lex
    else:
        combined = lexical_scores(question)
    order = np.argsort(-combined)[:top_k]
    return [(int(i), float(np.clip(combined[i], 0, 1))) for i in order]

def is_obviously_off_topic(question: str):
    # The supplied KB covers only five technical domains. This is a conservative guard,
    # not a substitute for relevance scoring.
    allowed = {
        "raft","paxos","gossip","swim","consensus","saga","transaction","distributed",
        "hashing","clock","vector","sgd","adamw","transformer","logistic","regression",
        "svm","support","vector","perceptron","mlp","resnet","convolution","batch",
        "layer","normalization","postgres","postgresql","mysql","innodb","mongodb",
        "cassandra","redis","memcached","tree","bplus","lsm","write","ahead","logging",
        "shadow","paging","tcp","quic","dns","mdns","http","tls","bgp","ospf","routing",
        "cryptography","rsa","elliptic","curve","ecc","password","credential","pbkdf2",
        "argon2","firewall","waf","api","gateway","aes","chacha","oauth","pkce","jwt",
        "security","database","network","machine","learning","encryption","token",
        "port","timeout","quorum","replica","replication","protocol","algorithm",
        "key","hash","signature","certificate","http2","http3","udp","tcp"
    }
    q = set(tokenize(question))
    return not bool(q & allowed)

def choose_answer(question: str, passages):
    # Extract the most question-relevant sentence from the top retrieved passages.
    qtokens = set(tokenize(question))
    stop = {"what","which","when","where","who","whom","whose","why","how","does","do",
            "did","is","are","was","were","the","a","an","of","for","in","on","to",
            "and","or","with","by","from","about","it","its","this","that","be","used"}
    qtokens -= stop
    candidates = []
    for idx, score in passages:
        for sentence in re.split(r"(?<=[.!?])\s+", PASSAGES[idx]):
            st = set(tokenize(sentence))
            overlap = len(qtokens & st)
            coverage = overlap / max(1, len(qtokens))
            # Favor the top passage and sentences that cover more of the question.
            rank = 0.68 * coverage + 0.32 * score
            candidates.append((rank, sentence.strip(), idx))
    candidates.sort(reverse=True, key=lambda x: x[0])
    if not candidates:
        return None
    best = candidates[0]
    if best[0] < 0.23:
        return None
    return best[1]

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=2, max_length=2000)
    top_k: int = Field(default=3, ge=1, le=10)

class QueryResponse(BaseModel):
    answer: Optional[str]
    sources: list[int]
    scores: list[float]
    confidence: float

@app.get("/")
def home():
    return {"service": "RAG Colosseum", "status": "ok", "passages": len(PASSAGES),
            "endpoint": "POST /query"}

@app.get("/health")
def health():
    return {"status": "ok", "dense_embeddings": _embeddings is not None,
            "passages": len(PASSAGES)}

@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="question cannot be empty")

    hits = retrieve(question, req.top_k)
    best_score = hits[0][1] if hits else 0.0
    # Require both topical evidence and a minimum relevance score before answering.
    off_topic = is_obviously_off_topic(question)
    answer = None if off_topic or best_score < 0.34 else choose_answer(question, hits)

    if answer is None:
        return QueryResponse(answer=None, sources=[], scores=[], confidence=0.15)

    # Return evidence in descending score order. Keep confidence calibrated to retrieval.
    conf = float(np.clip(0.62 + 0.35 * (best_score - 0.40), 0.61, 0.93))
    return QueryResponse(
        answer=answer,
        sources=[i for i, _ in hits],
        scores=[round(float(s), 4) for _, s in hits],
        confidence=round(conf, 4),
    )
