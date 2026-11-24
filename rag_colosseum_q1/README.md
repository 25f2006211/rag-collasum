# RAG Colosseum – Question 1

This project serves the supplied 50-passage knowledge base at `POST /query`.

## Local run

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 7860
```

The first startup downloads `sentence-transformers/all-MiniLM-L6-v2`. Keep the process running.

## Test

```bash
curl -X POST http://127.0.0.1:7860/query ^
  -H "Content-Type: application/json" ^
  -d "{\"question\":\"What quorum does Raft require to commit a log entry?\",\"top_k\":3}"
```

Expected response shape:

```json
{
  "answer": "Raft consensus requires a quorum of 3 nodes to commit a log entry.",
  "sources": [0, 1, 2],
  "scores": [0.8, 0.4, 0.3],
  "confidence": 0.7
}
```

Actual scores depend on the embedding model. The answer/source values above illustrate the response format only.

## Hugging Face Spaces deployment

1. Create a new Space and choose **Docker**.
2. Upload `app.py`, `knowledge_base.json`, `requirements.txt`, and `Dockerfile`.
3. Wait for the Space to build and become Running.
4. Your base URL is `https://YOUR-USERNAME-YOUR-SPACE.hf.space`.
5. Submit that base URL in the assignment (do not append `/query`).

The server exposes `POST /query`, includes permissive CORS, and loads embeddings once at startup.

## Important

This is a baseline implementation, not a guaranteed full-score solution. Test the live endpoint against the assignment's 20-probe gauntlet. Tune the abstention threshold and extraction logic based on the probe feedback. Do not use the illustrative expected scores as real results.
