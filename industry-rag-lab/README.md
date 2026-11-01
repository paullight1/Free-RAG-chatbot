# Industry RAG Engineering Lab

**Project type:** Multi-tenant, document-grounded enterprise knowledge assistant. This is an applied Python AI engineering portfolio project, **not** a production-approved SaaS.

Built to practice requirements commonly found in AI / ML engineer roles: **FastAPI, local embedding models, Qdrant vector database, PDF/text ingestion, tenant isolation, source citations, evaluation metrics, Docker, testing, and optional LLM generation**.

## Open-source foundation and references

- **Haystack** (Apache-2.0): https://github.com/deepset-ai/haystack — learn pipeline composition, retrieval and evaluation.
- **RAGFlow** (Apache-2.0): https://github.com/infiniflow/ragflow — inspect deep document parsing, citations and end-to-end deployments.
- **RAG From Scratch**: https://github.com/langchain-ai/rag-from-scratch — learning notebooks; inspect upstream licensing before reusing their code.
- The implementation in this directory is independently authored and does not copy the above repositories.

## Industry use case

A company uploads internal policy handbooks and incident response SLAs, and employees ask questions with document citations. Each organization has a different server-side API key, and retrieval is always filtered to its own documents. The included three example files contain **fictional training data**, not real company policy.

**Flow:** PDF / Markdown / TXT → page-aware extraction → overlapping chunks → local SentenceTransformers embeddings → Qdrant → tenant-filtered vector search → lexical candidate reranking → cited evidence → optional LLM → labelled retrieval evaluation.

### Start with Docker

~~~bash
cd industry-rag-lab
cp .env.example .env
# In .env, change TENANT_API_KEYS_JSON to your own unique, strong keys.
docker compose up --build -d
curl http://127.0.0.1:8000/health
~~~

The first embedding request downloads the small public SentenceTransformers model once; budget internet, disk, and RAM for it. The image installs ML packages and may be large. The demo binds the API to localhost and keeps Qdrant internal to Docker. **Do not publish it directly to the internet.**

### Index sample documents

Set RAG_API_KEY to the exact secret you entered for the demo tenant in .env. Keep it out of Git.

~~~bash
export RAG_API_KEY='replace-with-a-random-32-character-secret'

curl -sS -X POST http://127.0.0.1:8000/v1/documents \
  -H "X-API-Key: $RAG_API_KEY" \
  -F "file=@fixtures/incident_sla.md"

curl -sS -X POST http://127.0.0.1:8000/v1/documents \
  -H "X-API-Key: $RAG_API_KEY" \
  -F "file=@fixtures/employee_handbook.md"

curl -sS -X POST http://127.0.0.1:8000/v1/documents \
  -H "X-API-Key: $RAG_API_KEY" \
  -F "file=@fixtures/data_access_policy.md"
~~~

### Ask a grounded question

~~~bash
curl -sS -X POST http://127.0.0.1:8000/v1/ask \
  -H "X-API-Key: $RAG_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"question":"How fast must we respond to a P1 incident?","top_k":3}'
~~~

Without an LLM, the answer returns **retrieval_only** mode with cited excerpts. This intentionally avoids hallucinating that text generation is active.

To enable generation, configure **LLM_MODEL**, **LLM_BASE_URL** and **LLM_API_KEY** for an OpenAI-compatible endpoint. A local Ollama endpoint can work using an OpenAI-compatible /v1 URL if it is network-accessible to the container. Output mode **generated_unverified** means cited context was provided, NOT that answer claims have been validated.

### Evaluate retrieval, not just demos

After ingesting the three sample documents:

~~~bash
# In another shell, inside this directory, with httpx installed:
RAG_API_KEY="$RAG_API_KEY" python -m rag_lab.evaluate \
  --dataset evals/ground_truth.jsonl \
  --base-url http://127.0.0.1:8000 --top-k 3
~~~

The metrics are **recall_at_k** (fraction of questions where an expected source appeared) and **mrr_at_k** (rank-weighted score). Do not claim model accuracy or hallucination rates without labelled representative tests.

### What to inspect and learn

| Module | Job skill |
| --- | --- |
| **rag_lab/core.py** | Extraction, chunking, embeddings, Qdrant filtering, citations, lexical reranking and safe prompt boundaries |
| **rag_lab/api.py** | FastAPI, input constraints, tenant key checking |
| **rag_lab/settings.py** | Environment configuration and fail-closed authorization |
| **rag_lab/evaluate.py** | Evaluation set, retrieval recall@k and MRR |
| **tests/** | Python pytest, deterministic fakes, cross-tenant tests, idempotent uploads |
| **Dockerfile / docker-compose.yml** | Container deployment and persistent vector search |

### Next five industry-grade enhancements

1. Implement **full-corpus hybrid BM25 + dense retrieval**, plus a cross-encoder reranker. The current lexical step only reorders dense candidates.
2. Add **per-user document ACLs** from real content sources; tenant-scoped access alone is not sufficient.
3. Add **OCR, ingestion queues, retries, async workloads and malware scanning**.
4. Add **OpenTelemetry, latency and cost metrics, LLM faithfulness tests, citation validation and abstention calibration**.
5. Add **SSO/IAM, TLS, rate limits, secrets management, audits, DB integration tests, performance baselines, and model versioning**.

### Safety and limitations

A deliberate learning reference, **not production hardened**. API keys are static, and source texts remain untrusted. File uploads are length-bounded, but not malware-scanned; scanned PDFs need an OCR stage. Replacing documents deletes and then reindexes, so concurrent updates are not atomic. Citation excerpts are traceable to retrieved documents but generated claims can still be wrong. There is no calibrated relevance threshold, so an irrelevant query can retrieve unrelated passages. Do not upload real confidential or regulated data without an appropriate security review.

All example datasets are fictional. This project is original learning code; if you add upstream code later, preserve its license and attribution.
