# Security & Compliance Copilot

Security & Compliance Copilot is a production-minded RAG assistant for security and compliance guidance. It answers questions from a tightly curated corpus of official public NIST and CISA material, returns grounded answers with citations, and fails closed when a request is unsafe or the evidence is weak.

Built as a portfolio project for cloud security, AI security, and AI engineering roles, it emphasizes traceability, retrieval quality, explicit guardrails, and lightweight auditability over polished but ungrounded answers.

## Why This Project

Most demo RAG apps optimize for answer quality first. This project is intentionally built around engineering controls first:

- constrained public corpus instead of open-ended web answers
- retrieval-first responses with explicit evidence handling
- guardrails for unsafe, proprietary, and prompt-leak requests
- stable citations tied to retrieved sources
- JSONL audit logging on every chat request
- offline evals for safe, injection, privacy, and refusal behavior

## Key Features

- Grounded answers over official public NIST and CISA guidance
- Hybrid retrieval pipeline with query rewriting, retrieval, reranking, and context building
- Stable inline citations tied to actual retrieved sources
- Guardrails for prompt injection, jailbreaks, prompt leaks, and proprietary-text requests
- JSONL audit logging for every chat request
- Offline evals for answer, refusal, privacy, and injection behavior

## What This Demonstrates

- practical RAG system design with retrieval, reranking, citations, and fail-closed behavior
- security-aware application logic rather than prompt-only safety claims
- lightweight operational controls such as audit logging and offline regression evals
- clear separation between retrieval, generation, guardrails, and API layers

## Architecture Diagram

```text
User question
    |
    v
/chat API
    |
    v
Query rewrite -> Retrieval -> Reranking -> Guardrails
                                      |         |
                                      |         +-> refuse / insufficient_context
                                      v
                               Context builder
                                      |
                                      v
                                  Generation
                                      |
                                      v
                         Answer + citations + audit log
```

## Corpus

The indexed corpus is intentionally narrow and high-trust:

- NIST AI RMF, Playbook, GenAI Profile, CSF 2.0, SSDF, and related guidance
- CISA secure-by-design and AI security guidance
- official public U.S. government material fetched into `data/raw/`

Each downloaded document includes sidecar metadata capturing:

- source URL
- framework
- publisher
- license status
- fetch timestamp

The assistant does not use scraped private/internal documents and does not provide proprietary standards text.

## Architecture

The application follows a simple retrieval-first chat flow:

1. Fetch public corpus.
2. Ingest, chunk, embed, and index into Chroma.
3. Rewrite and retrieve the most relevant chunks.
4. Rerank and build a citation-aware context package.
5. Run guardrails before answering.
6. Generate only from retrieved evidence.
7. Return the response and write a JSONL audit event.

Core modules:

- `app/retrieval/search.py`
- `app/retrieval/query_rewriter.py`
- `app/ranking/reranker.py`
- `app/generation/context_builder.py`
- `app/generation/service.py`
- `app/guardrails/rules.py`
- `app/api/chat.py`

## Repository Layout

```text
app/
  api/
  core/
  frontend/
  generation/
  guardrails/
  ingestion/
  models/
  ranking/
  retrieval/
data/
  chroma/
  processed/
  raw/
evals/
scripts/
tests/
```

## Setup

1. Create the environment file:

```bash
cp .env.example .env
```

2. Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

3. Add `OPENAI_API_KEY` to `.env`.

## Quick Start

Fetch the approved public corpus:

```bash
python3.11 scripts/fetch_public_corpus.py
```

Build or rebuild the Chroma index:

```bash
python3.11 scripts/ingest.py
```

## Run the App

Start the API and minimal frontend:

```bash
uvicorn app.main:app --reload --port 8001
```

Open:

```text
http://127.0.0.1:8001/
```

Run the eval set:

```bash
python3.11 evals/run_eval.py
```

## API

### `GET /health`

Returns index status, source-framework coverage, and last ingest time.

Check it with:

```bash
curl -s http://127.0.0.1:8001/health
```

Example shape:

```json
{
  "status": "ok",
  "indexed_chunks": 842,
  "known_sources": [
    { "framework": "CISA", "count": 7 },
    { "framework": "NIST_AI_RMF", "count": 5 }
  ],
  "last_ingest_at": "2026-04-27T12:34:56+00:00"
}
```

### `POST /chat`

Request:

```json
{
  "question": "What does the Govern function in NIST AI RMF cover?"
}
```

Response:

```json
{
  "answer": "NIST frames governance as a cross-cutting function that sets roles, accountability, and oversight expectations [S1][S2]. This is not legal or compliance advice.",
  "sources": [
    {
      "label": "S1",
      "title": "Artificial Intelligence Risk Management Framework (AI RMF 1.0)",
      "framework": "NIST_AI_RMF",
      "url": "https://nvlpubs.nist.gov/...",
      "score": 0.92
    }
  ],
  "confidence": "high",
  "guardrail_status": "ok"
}
```

## Guardrails and Security

The assistant is intentionally conservative:

- answers are grounded only in retrieved local corpus content
- prompt-injection and jailbreak-style requests are refused
- requests for system prompts, developer messages, config, secrets, passwords, tokens, and API keys are refused
- proprietary standards full-text requests are refused
- broad dump-style requests such as "show all files" or "all documents" fail closed with `insufficient_context`
- citation labels are sanitized so fabricated citations are removed from generated output

## Security & Compliance

- Retrieval is constrained to a curated local corpus of official public NIST/CISA guidance.
- Prompt injection, jailbreak, and internal prompt/config extraction requests are refused.
- Proprietary standards full-text requests are refused.
- Every `/chat` request is logged to `logs/audit.jsonl`.
- Offline evaluations are run locally and written to `evals/results.json`.
- The system is fail-closed: unsafe requests are refused and weak evidence returns `insufficient_context`.

See `SECURITY_AND_COMPLIANCE.md` for the project-specific security posture.

## Auditability

Every `/chat` request produces a JSONL audit event in `logs/audit.jsonl`.

Logged fields include:

- timestamp
- request id
- original query
- rewritten query
- guardrail status
- confidence
- source labels, titles, and frameworks
- retrieval count
- final answer length
- refusal/block status
- guardrail detection flags

Audit logging is intentionally lightweight and non-fatal. If logging fails, the chat response still returns.

## Evaluation

Offline evals run through the same service path as `/chat` and write detailed results to `evals/results.json`.

Eval categories include:

- `safe`
- `injection`
- `privacy`
- `refuse`

Tracked fields include retrieved chunk count, guardrail status, confidence, source frameworks, and answer length.

## Current Scope

This repo is focused on local execution, inspectability, and production-minded behavior for a single-user portfolio project. It does not currently include:

- authentication
- multi-tenancy
- deployment automation
- background job orchestration
- policy storage beyond local files

## Suggested Next Steps

- stronger eval scoring for faithfulness and citation quality
- cross-encoder reranking
- query decomposition for multi-hop questions
- chunk-level citation spans
- richer framework mapping metadata
