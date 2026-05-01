# Security & Compliance Copilot

## Purpose

This Security & Compliance Copilot provides grounded answers to security and compliance questions from a curated corpus of official NIST and CISA guidance, with strict guardrails to ensure safe, trustworthy operation.

## Corpus Rules

- All source documents are from official NIST and CISA publications
- Metadata includes framework, document type, and version information
- Documents are processed through chunking and embedding for retrieval
- No proprietary, internal, or sensitive data is included

## Input Guardrails

- Prompt injection attempts are detected and rejected
- Requests for unauthorized access (file dumps, config exposure) are blocked
- Attempts to manipulate system behavior are refused
- Input validation prevents malformed queries

## Output Guardrails

- All answers include source citations tied to retrieved documents
- Hallucinated citations are automatically stripped
- Confidence levels reflect retrieval quality
- Responses respect document boundaries and corpus limitations

## Auditability

- All chat requests are logged in JSONL format for audit trails
- Logs include: timestamp, query, context, response, guardrail decisions
- One audit entry per request for compliance tracking
- Logs are immutable records of system behavior

## Evaluation

- Test suite covers security, compliance, and functional requirements
- Guardrail matrix tests malicious, weak, and edge-case inputs
- Citation tests verify hallucination detection and source attribution
- Health and retrieval tests ensure system reliability

## Limitations

- Answers are constrained to corpus scope (NIST and CISA documents)
- System responds with "insufficient context" for out-of-scope queries
- No real-time information or updates beyond corpus publication dates
- Multi-framework compliance questions may be conservative when evidence is weak
