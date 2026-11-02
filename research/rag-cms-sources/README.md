# RAG + CMS open-source source collection

This is a **source-code research collection of 26 separate upstream GitHub repositories** connected as Git submodules, rather than merged into one application. Each submodule pins a verified upstream commit as of **2026-11-02** and retains its own Git history, code and license. No upstream code has been modified or integrated.

## Clone everything

```bash
# Clone this branch of the parent repository:
git clone --branch feat/rag-cms-source-collection https://github.com/paullight1/Free-RAG-chatbot.git
cd Free-RAG-chatbot

# Check out the full source code for all 26 repositories:
git submodule update --init --recursive --jobs 4
```

**Large download:** The collection contains big projects (including Langflow, Dify, Payload, and Strapi). Plan for significant disk space and network usage. To save Git history downloads, try `git submodule update --init --depth 1 --jobs 4` (shallow fetches may occasionally require a deeper fetch for pinned commits).

To clone just one source:

```bash
git submodule update --init research/rag-cms-sources/rag-platforms/weknora
```

To see all sources and pins: [manifest.tsv](manifest.tsv). After upstream updates, review licenses and compatibility before using code in another product.

## Included repositories

| Category | Repository | Use case |
|---|---|---|
| rag-platforms | [Tencent/WeKnora](https://github.com/Tencent/WeKnora) | AI knowledge CMS and wiki |
| rag-platforms | [infiniflow/ragflow](https://github.com/infiniflow/ragflow) | Advanced document RAG |
| rag-platforms | [arc53/DocsGPT](https://github.com/arc53/DocsGPT) | Customer-facing RAG assistants |
| rag-platforms | [onyx-dot-app/onyx-foss](https://github.com/onyx-dot-app/onyx-foss) | Enterprise knowledge search |
| rag-platforms | [SciPhi-AI/R2R](https://github.com/SciPhi-AI/R2R) | API-first RAG backend |
| rag-platforms | [langflow-ai/openrag](https://github.com/langflow-ai/openrag) | All-in-one RAG application |
| rag-platforms | [Mintplex-Labs/anything-llm](https://github.com/Mintplex-Labs/anything-llm) | Private knowledge chat |
| rag-platforms | [langflow-ai/langflow](https://github.com/langflow-ai/langflow) | Visual RAG workflows |
| rag-platforms | [Cinnamon/kotaemon](https://github.com/Cinnamon/kotaemon) | Research PDFs and citations |
| rag-platforms | [1Panel-dev/MaxKB](https://github.com/1Panel-dev/MaxKB) | Knowledge-base agent platform |
| rag-platforms | [langgenius/dify](https://github.com/langgenius/dify) | AI application platform, licensing caution |
| rag-platforms | [The-Vibe-Company/quivr](https://github.com/The-Vibe-Company/quivr) | Continuous knowledge ingestion |
| cms | [payloadcms/payload](https://github.com/payloadcms/payload) | Full-featured Next.js CMS |
| cms | [techiejd/payloadcms-vectorize](https://github.com/techiejd/payloadcms-vectorize) | CMS embeddings extension |
| cms | [ashbuilds/payload-ai](https://github.com/ashbuilds/payload-ai) | Payload AI editor plugin |
| cms | [strapi/strapi](https://github.com/strapi/strapi) | Headless CMS |
| cms | [directus/directus](https://github.com/directus/directus) | Database-driven CMS, licensing caution |
| components | [docling-project/docling](https://github.com/docling-project/docling) | Document parsing |
| components | [qdrant/qdrant](https://github.com/qdrant/qdrant) | Vector database |
| components | [pgvector/pgvector](https://github.com/pgvector/pgvector) | Vector search for PostgreSQL |
| components | [HKUDS/LightRAG](https://github.com/HKUDS/LightRAG) | Graph-enhanced RAG |
| components | [run-llama/llama_index](https://github.com/run-llama/llama_index) | RAG indexing framework |
| components | [deepset-ai/haystack](https://github.com/deepset-ai/haystack) | RAG pipeline framework |
| components | [langfuse/langfuse](https://github.com/langfuse/langfuse) | LLM observability and evaluations |
| archive-extra | [FlowiseAI/Flowise](https://github.com/FlowiseAI/Flowise) | Archived visual agent workflow repository |
| archive-extra | [open-webui/open-webui](https://github.com/open-webui/open-webui) | AI chat interface, licensing caution |

## Commercial and maintenance cautions

- **FlowiseAI/Flowise is archived**. Included only because it was discussed, not recommended for new production use.
- **Dify, Directus and Open WebUI** have license/branding considerations; confirm terms before white-label or SaaS deployment.
- This collection is **not one runnable application**. The repositories use different languages, package managers and infrastructure. Do not install them all together or assume APIs are compatible.
- Never commit credentials, private documents or secrets into the collection. Review source licenses and software supply-chain risks.

## Suggested combinations (proposals, not prebuilt integrations)

- **Enterprise knowledge CMS:** WeKnora (standalone) or Onyx + your custom business interface.
- **RAG-backed content CMS:** Payload + payloadcms-vectorize + pgvector, with Docling for document extraction.
- **Document intelligence:** RAGFlow + a custom frontend; consider Langfuse for evaluation and monitoring.
- **API-first RAG backend:** R2R + Next.js + PostgreSQL; enforce permission-aware retrieval.
