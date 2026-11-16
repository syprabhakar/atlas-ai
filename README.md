
# ATLAS AI — Agentic RAG Portfolio Project

ATLAS AI is a local development prototype combining document-grounded question answering with a LangGraph agent, calculator, and approval-gated mock ticket creation.

Built with Python 3.12, FastAPI, Streamlit, Chroma, BM25, SentenceTransformers, LangGraph, and OpenAI.

## Architecture

```text
                 Streamlit UI
                      |
                   FastAPI
                 /         \
          Document API    Agent API
                |             |
             AtlasRAG      LangGraph
                |         /    |    \
           PDF ingestion  RAG  Calc  Ticket
                |                     |
        Chroma + BM25              Approval
                |                     |
               RRF               SQLite state
                |
         Cross-Encoder
                |
          Cited answers
```

## Features

- PDF ingestion and persistent Chroma embeddings
- BM25 lexical search and semantic vector retrieval
- Reciprocal Rank Fusion (RRF)
- Cross-encoder reranking
- Grounded answers with source citations
- LangGraph agent with document search and calculator tools
- SQLite-backed conversation checkpoints
- Human approval before mock ticket creation
- Unified Streamlit interface
- Agent evaluation, retrieval evaluation, and execution tracing

## Quick Start

Requires Python 3.12.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Add your OpenAI API key to `.env`.

Start FastAPI:

```bash
uvicorn atlas.api:app --reload
```

In a second terminal, start Streamlit:

```bash
source .venv/bin/activate
streamlit run atlas/ui.py
```

Streamlit:
http://localhost:8501

API documentation:
http://127.0.0.1:8000/docs

On first run, embedding and reranking models may need to be downloaded.

## Demo Workflow

1. Upload a public, text-based PDF.
2. Ask a question about the document.
3. Inspect the answer and its citations.
4. Ask the agent to calculate `12500 * 18 / 100`.
5. Ask the agent to research WCAG conformance levels and prepare a compliance ticket.
6. Review the proposed ticket.
7. Approve or reject the action.

Ticket creation is mocked. No external Jira ticket is created.

## API Endpoints

| Endpoint | Purpose |
|---|---|
| GET /health | Health and indexed chunk count |
| POST /documents | Ingest a PDF |
| POST /ask | Document-grounded Q&A |
| POST /agent/chat | Agent conversation |
| POST /agent/approve | Resume approval |
| GET /agent/history/{thread_id} | Conversation history |

## Testing

Run offline tests:

```bash
TOKENIZERS_PARALLELISM=false PYTHONPATH=. \
python -m pytest tests/test_core.py tests/test_workflow_unit.py -v
```

Run live integration tests:

```bash
TOKENIZERS_PARALLELISM=false PYTHONPATH=. \
python -m pytest tests/test_agent_regression.py -v
```

Run evaluations:

```bash
PYTHONPATH=. python -m evals.run_agent_eval
PYTHONPATH=. python -m evals.run_rag_eval
PYTHONPATH=. python -m evals.run_retrieval_eval
PYTHONPATH=. python -m evals.trace_agent
```

Live evaluations may incur API costs.

## Historical Evaluation Results

Results from a small local test set:

- Agent evaluation: 5/5 cases passed
- Regression tests: 4/4 passed
- Retrieval Precision@5: 0.40 on two WCAG queries
- Required-fact coverage: 1.00 on two WCAG queries
- Calculator latency: approximately 4 seconds
- RAG agent latency: approximately 15.5 seconds

These are prototype observations, not production benchmarks.

The LLM-based evaluation scripts use custom judges rather than the official RAGAS implementation.

## Limitations

This is not a production-ready service.

Current limitations include:

- No authentication or tenant isolation
- No document-level access controls
- No production-grade retry and timeout strategy
- No durable external-action idempotency
- No multiworker-safe checkpoint storage
- No knowledge graph or long-term semantic memory
- No real ticketing-system integration

The research-to-ticket sequence is guided by agent instructions, not enforced through a deterministic workflow.

## Security

Do not upload confidential documents.

Never commit `.env`, local data, checkpoint databases, or generated traces containing sensitive content.

## Future Improvements

- Deterministic research-to-ticket workflow
- External ticketing integration
- Idempotent tool execution
- Long-term memory
- Knowledge graph retrieval
- Authentication and authorization
- CI/CD and deployment
- Larger evaluation datasets
- Distributed tracing and per-stage latency monitoring
