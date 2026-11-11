# ATLAS AI — Day 1

A runnable Python 3.12 modular-monolith enterprise RAG prototype: FastAPI + Streamlit, PDF parsing, Chroma persistence, BM25, semantic search, reciprocal rank fusion, cross-encoder reranking, cited LLM answers.

## Requirements
- Python 3.12; internet for first model downloads
- CPU is sufficient; first embedding/reranker downloads take time
- Text-based PDFs only on Day 1; OCR is planned for Day 5

## Setup
```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
```

On Intel macOS with older CPU/ABI constraints, the pinned `torch==2.2.2`, `numpy==1.26.4`, `transformers==4.44.2`, and `sentence-transformers==2.7.0` are intentional. If pip cannot install PyTorch for your specific OS, use a Python 3.12 Linux container or compatible machine.

Optionally set `OPENAI_API_KEY` in `.env` for generated answers. Without it, ATLAS returns clearly labeled retrieved excerpts. **Never commit `.env`.**

## Run (two terminals, from repository root)
```bash
source .venv/bin/activate
uvicorn atlas.api:app --reload
```
```bash
source .venv/bin/activate
streamlit run atlas/ui.py
```

FastAPI docs: http://127.0.0.1:8000/docs
Streamlit: http://localhost:8501

## API examples
```bash
curl -F 'file=@sample.pdf' http://127.0.0.1:8000/documents
curl -X POST http://127.0.0.1:8000/ask -H 'Content-Type: application/json' -d '{"question":"What does the policy say?","top_k":5}'
```

## Test
```bash
pytest -q
```

## Scope and limitations
- A single-process developer prototype, **not** production-ready. No authentication, per-tenant filtering, upload malware scanning, or concurrency-safe cross-process writes. Do not upload sensitive enterprise documents.
- BM25 loads all chunks into memory on each query; this is intentionally simple for Day 1 and must be replaced with an indexed keyword engine at scale.
- Generated citations are prompted, not yet independently validated; Day 4 will add citation faithfulness checks.
- Extractive fallback returns relevant passages rather than synthesizing answers.
- PDF tables may parse poorly; OCR and table handling are future work.
- Model downloads and installation were not validated on your machine.
