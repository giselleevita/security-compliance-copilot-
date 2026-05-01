# Contributing

Thank you for your interest in the Security & Compliance Copilot. This is a portfolio project showcasing production-oriented RAG design, security controls, and AI engineering practices.

## Setup

### Prerequisites
- Python 3.11+
- pip

### Local Development

1. **Clone the repository**:
```bash
git clone https://github.com/giselleevita/security-compliance-copilot.git
cd security-compliance-copilot
```

2. **Create a virtual environment**:
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**:
```bash
cp .env.example .env
```

Then add your Google Gemini API key to `.env`:
```
GEMINI_API_KEY=your-key-here
```

Get a free key at: https://aistudio.google.com/app/apikey

### Fetch and Ingest Data

The first time you run the app, it will auto-ingest the public corpus. To manually manage this:

```bash
# Fetch official NIST and CISA documents
python3 scripts/fetch_public_corpus.py

# Build the Chroma vector index
python3 scripts/ingest.py
```

This creates `data/chroma/` with embedded chunks (~1.4GB of documents).

## Running Locally

**Start the API server**:
```bash
uvicorn app.main:app --reload --port 8001
```

Open: http://127.0.0.1:8001/

**API endpoints**:
- `GET /health` — Vector store status and source coverage
- `POST /chat` — Answer security questions
- `POST /ingest` — Trigger manual re-ingestion

Example chat request:
```bash
curl -X POST http://127.0.0.1:8001/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What does the Govern function in NIST AI RMF cover?"}'
```

## Testing

**Run the test suite**:
```bash
pytest tests/ -v
```

**Test categories**:
- `test_chat_*` — Integration tests for the chat service
- `test_qa_*` — Quality assurance: guardrails, citations, retrieval
- `test_health_*` — Health check and vector store status
- `test_eval_*` — Offline evaluation suite

**Run evals**:
```bash
python3 evals/run_eval.py
```

Results are written to `evals/results.json`.

## Architecture Notes

The app follows a retrieval-first pipeline:

1. **Retrieval**: Query rewriting, vector search on NIST/CISA corpus
2. **Reranking**: Cross-encoder scoring of retrieved chunks
3. **Guardrails**: Check for unsafe patterns, insufficient context, proprietary requests
4. **Generation**: LLM answer from retrieved context only
5. **Audit**: JSONL log of every request in `logs/audit.jsonl`

Key modules:
- `app/retrieval/` — Search and query rewriting
- `app/ranking/` — Reranking logic
- `app/guardrails/` — Safety checks
- `app/generation/` — Answer generation and citation handling
- `app/core/audit.py` — Request logging

## Audit Logging

Every `/chat` request logs an audit event in `logs/audit.jsonl`:

```json
{
  "timestamp": "2026-05-01T12:34:56Z",
  "request_id": "...",
  "original_query": "...",
  "guardrail_status": "ok",
  "confidence": "high",
  "source_frameworks": ["NIST_AI_RMF", "CISA"],
  "refused_or_blocked": false
}
```

This supports compliance audits, debugging, and behavior analysis.

## Portfolio Use

This project demonstrates:

- **RAG Engineering**: Retrieval, reranking, context building, citation handling
- **Security**: Input validation, guardrails, output sanitization, audit trails
- **AI Engineering**: Prompt design, retrieval-grounded generation, evaluation
- **Production Practices**: Error handling, logging, configuration management, testing

It's designed to be understandable to hiring managers and suitable for security, AI, and cloud roles.

## Questions?

See `README.md` for architecture overview and `SECURITY_AND_COMPLIANCE.md` for safety details.
