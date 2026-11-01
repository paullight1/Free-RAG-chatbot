# RAG AI Engineering — job-to-project learning track (November 2026)

This guide turns **real AI engineer job descriptions** into demonstrable engineering work. Hiring criteria change, and job adverts may close; the links below are examples rather than application guarantees.

## Job descriptions used as the quality bar

| Example | Skills explicitly requested | What to build and prove |
| --- | --- | --- |
| [Capco — Mid Level AI Engineer](https://job-boards.greenhouse.io/capco/jobs/8178266) | Python APIs, LLMs, RAG, vector stores, cloud and CI/CD; agent skills as a bonus | FastAPI query/ingestion service, Qdrant integration tests, Docker and a deployment runbook |
| [Planso — Junior AI Engineer](https://apply.workable.com/planso/j/873BD4B8EF) | Python, embeddings, reranking, RAG, relevance/recall evaluation, document understanding, a GitHub portfolio | Repeatable five-query retrieval eval, document parsing and a failure-analysis notebook |
| [BPD — AI Engineer](https://job-boards.greenhouse.io/bpd/jobs/5386839008) | Python + SQL, hybrid search, tool use, guardrails, healthcare data controls, hallucination/evaluation checks | Retrieval metrics, citation verification, per-document permissions; expand with SQL and privacy testing |
| [Indicium AI — AI Engineer](https://job-boards.eu.greenhouse.io/indiciumai/jobs/4695065101) | Python, vector/graph search, LLM workflows, cloud, CI/CD, quality/latency/cost experiments | Add offline evaluations, traces and cost dashboards; document architectural tradeoffs |
| [RightShip — AI Engineer](https://job-boards.eu.greenhouse.io/rightship/jobs/4982444101) | Production Python, LangGraph agents, state/tool handling, retrieval, eval datasets and software testing | Extend the service with an agent that calls a structured policy SQL tool and a citation-safe retriever |

## Clone/study the existing open-source forks in this GitHub account

These forks **already existed before this lab**; they are reference implementations, not copied code inside this lab.

1. **paullight1/rag-from-scratch** (fork of langchain-ai/rag-from-scratch). Learn indexing, embedding, retrieval, and prompting. Follow notebook concepts in order; verify licensing before reusing code.
2. **paullight1/haystack** (fork of deepset-ai/haystack, Apache-2.0). Read component interfaces, evaluation and test conventions. Especially useful for framework-level Python skills.
3. **paullight1/ragflow** (fork of infiniflow/ragflow, Apache-2.0). Inspect real document parsing, datasets, citations, API, and container orchestration. Large system; start with one data path.
4. **paullight1/LightRAG** (fork of HKUDS/LightRAG, MIT). Compare entity/relationship graph retrieval with this lab's dense retrieval.
5. **paullight1/graphiti** (fork of getzep/graphiti, Apache-2.0). Study temporal knowledge graphs and changing facts. Build this later, after solid basic evaluation.

Study the original upstream repositories to check newer releases and bug fixes before deploying any existing fork.

## Four-week hands-on progression (quality > tutorials)

**Week 1 — Working backend.** Start this lab, index the three fictional documents, call /v1/search and /v1/ask, run the 8 code tests and the labelled retrieval eval. Draw the complete data path. Publish a screenshot showing a real cited result, plus actual measured metrics.

**Week 2 — Retrieval quality.** Implement full sparse/dense hybrid retrieval (not merely reranking dense results), benchmark chunk sizes, compare two embedders, add a cross-encoder reranker, and add at least 30 independently labelled queries. Track Recall@5 and MRR@5 before and after each change; keep bad cases in the dataset.

**Week 3 — Trust, agents and observability.** Add document-level ACL, request tracing, a PII-safe audit log, model prompt-injection test cases, and a LangGraph workflow combining retrieval with read-only SQL queries. Test unauthorized access with separate organizations and users; do not place real confidential material in the lab.

**Week 4 — Deployment and demonstration.** Add Docker integration tests with live Qdrant, a deployment runbook, basic latency/cost measurements, load-test notes, and a short 2–3 minute walkthrough. Deploy only with appropriate authentication and TLS. Write a technical case study describing architecture, metrics, failures, and tradeoffs.

## Evidence that matters in an interview

A candidate can point to: a public GitHub PR; clean unit and integration tests; a reproducible demo; the labelled evaluation set and real results; evidence of tenant isolation; decisions about embeddings, chunking and retrieval; failures that were found and fixed; and a small deployed service. **Do not claim production deployment, benchmark improvements, or precision percentages until actually measured.**

### Interview framing

**Problem:** Staff need answers grounded in internal policy documents, without cross-organization leakage.

**System:** FastAPI receives authorized uploads and questions; a page-aware parser extracts text; chunks are embedded locally; Qdrant searches within a server-authenticated tenant scope; the API returns snippets and citations, optionally synthesizing a clearly labelled unverified answer.

**Reliability and next improvements:** Currently tested with in-memory test doubles; integration against live Qdrant, access-control hardening, calibrated abstention, full hybrid retrieval, generation-faithfulness evaluation and operational monitoring remain.

**Why this is a strong portfolio:** It demonstrates engineering decisions and tests, rather than only a prompt chained to an LLM.
