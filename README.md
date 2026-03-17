<div align="center">

# 🧠 AI Codebase Intelligence Platform

**Query any GitHub repository using natural language.**
Powered by semantic search, dependency graphs, and a multi-provider LLM engine.

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Next.js](https://img.shields.io/badge/Next.js-14+-000000?style=flat&logo=next.js&logoColor=white)](https://nextjs.org)
[![Gemini](https://img.shields.io/badge/Gemini-2.0_Flash-4285F4?style=flat&logo=google&logoColor=white)](https://ai.google.dev)
[![DeepSeek](https://img.shields.io/badge/DeepSeek-Coder_V2-00B4D8?style=flat)](https://deepseek.com)
[![OpenRouter](https://img.shields.io/badge/OpenRouter-Multi--Model-FF6B35?style=flat)](https://openrouter.ai)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=flat&logo=docker&logoColor=white)](https://docker.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

---

## Table of Contents

1. [Overview](#1-overview)
2. [Core Capabilities](#2-core-capabilities)
3. [System Architecture](#3-system-architecture)
4. [LLM Provider Strategy](#4-llm-provider-strategy)
5. [Tech Stack](#5-tech-stack)
6. [Project Structure](#6-project-structure)
7. [Dataset Sources](#7-dataset-sources)
8. [Development Roadmap](#8-development-roadmap)
9. [API Reference](#9-api-reference)
10. [Getting Started](#10-getting-started)
11. [Contributing](#11-contributing)
12. [License](#12-license)

---

## 1. Overview

Modern software repositories contain thousands of files and deeply nested dependencies. Understanding how a system works — even for experienced engineers — can take days.

This platform solves that by:

- Ingesting any public GitHub repository via URL
- Parsing source files into Abstract Syntax Trees (ASTs)
- Building call graphs, module graphs, and class hierarchies
- Generating semantic embeddings using Google's `text-embedding-004`
- Enabling natural language Q&A using a smart multi-provider LLM router
- Visualizing architecture and dependency relationships interactively

**Example interaction**

```
User  →  "Why might login fail even when the password is correct?"

AI    →  "The issue likely originates in auth_service.py → verify_password().
          It calls bcrypt.checkpw() but does not handle the ValueError raised
          when the stored hash is malformed.

          Call trace:
          login_controller → auth_service → verify_password → bcrypt"
```

---

## 2. Core Capabilities

| # | Capability | Description |
|---|---|---|
| 1 | **Codebase Indexing** | Parses files into ASTs; extracts functions, classes, imports, docstrings |
| 2 | **Semantic Code Search** | Natural language → Gemini embeddings → vector search → ranked results |
| 3 | **Architecture Reconstruction** | Builds call graphs, module graphs, and class hierarchies |
| 4 | **Code Explanation** | Gemini 2.0 Flash / DeepSeek Coder explanations grounded in actual code |
| 5 | **Bug Localization** | Traces error paths across call graphs to pinpoint failure points |
| 6 | **Security Analysis** | Detects SQL injection, unsafe inputs, missing validation via LLM + static rules |

---

## 3. System Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                        User / Frontend                        │
│                    (Next.js + TypeScript)                     │
└────────────────────────────┬─────────────────────────────────┘
                             │  REST / WebSocket
┌────────────────────────────▼─────────────────────────────────┐
│                      FastAPI Backend                          │
│                                                               │
│   ┌─────────────┐   ┌─────────────┐   ┌───────────────────┐  │
│   │   Ingest    │   │    Query    │   │ Analysis/Security │  │
│   │   Service   │   │   Service   │   │     Service       │  │
│   └──────┬──────┘   └──────┬──────┘   └───────────────────┘  │
└──────────┼────────────────┼──────────────────────────────────┘
           │                │
┌──────────▼──────┐  ┌──────▼──────────────────────────────┐
│  Celery Worker  │  │         Retrieval Pipeline            │
│  (async tasks)  │  │  Embed Query (Gemini) →               │
│  Redis broker   │  │  Vector Search (Qdrant) →             │
└──────────┬──────┘  │  Graph Expand (Neo4j) →               │
           │         │  LLM Router → Response                │
           │         └──────────────────┬────────────────────┘
┌──────────▼──────────────────────────▼─────────────────────┐
│                        Data Layer                            │
│  ┌────────────┐   ┌────────────┐   ┌──────────────────────┐ │
│  │ PostgreSQL │   │   Qdrant   │   │        Neo4j          │ │
│  │ (metadata) │   │ (vectors)  │   │   (graphs / Cypher)   │ │
│  └────────────┘   └────────────┘   └──────────────────────┘ │
└──────────────────────────────────────────────────────────────┘
           │
┌──────────▼──────────────────────────────────────────────────┐
│                   Multi-Provider LLM Router                   │
│                                                               │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────────┐ │
│  │ Google Gemini│  │   DeepSeek   │  │    OpenRouter       │ │
│  │ 2.0 Flash /  │  │  Coder V2 /  │  │  (Fallback +        │ │
│  │ 1.5 Pro      │  │  V3 / R1     │  │   Model variety)    │ │
│  └──────────────┘  └──────────────┘  └────────────────────┘ │
└──────────────────────────────────────────────────────────────┘
```

**Ingestion pipeline**

```
GitHub URL → Clone (GitPython) → Scan Files → Parse AST (tree-sitter)
           → Build Graphs (Neo4j) → Chunk Code → Embed (Gemini text-embedding-004)
           → Store vectors (Qdrant)
```

**Query pipeline**

```
Natural Language → Embed (Gemini) → Vector Search (Qdrant)
               → Graph Expansion (Neo4j) → Context Assembly
               → LLM Router → Grounded Response
```

---

## 4. LLM Provider Strategy

The platform uses **three providers** with a smart routing strategy — each provider is used for what it does best.

### Provider Roles

| Provider | Models Used | Role | Why |
|---|---|---|---|
| **Google Gemini** | `gemini-2.0-flash`, `gemini-1.5-pro` | Primary reasoning + embeddings | 1M token context window, fast, free tier, best embeddings |
| **DeepSeek** | `deepseek-coder-v2`, `deepseek-chat`, `deepseek-reasoner` | Code-specific tasks | Best-in-class code understanding, very low cost |
| **OpenRouter** | Any model via single API | Fallback + experimentation | Route to 100+ models with one key, great for A/B testing |

### Routing Logic

```
Incoming task
     │
     ├─ Code explanation / bug trace → DeepSeek Coder V2
     │
     ├─ Architecture Q&A / broad reasoning → Gemini 2.0 Flash
     │
     ├─ Security analysis (deep reasoning) → DeepSeek R1 or Gemini 1.5 Pro
     │
     ├─ Embeddings (always) → Gemini text-embedding-004
     │
     └─ Any provider down → OpenRouter fallback
```

### Cost Breakdown (approximate)

| Provider | Model | Cost per 1M tokens (input) | Cost per 1M tokens (output) |
|---|---|---|---|
| Google Gemini | gemini-2.0-flash | $0.075 | $0.30 |
| Google Gemini | gemini-1.5-pro | $1.25 | $5.00 |
| Google Gemini | text-embedding-004 | $0.00 (free tier) | — |
| DeepSeek | deepseek-coder-v2 | $0.14 | $0.28 |
| DeepSeek | deepseek-chat (V3) | $0.07 | $1.10 |
| DeepSeek | deepseek-reasoner (R1) | $0.55 | $2.19 |
| OpenRouter | varies by model | from $0.00 | from $0.00 |

> For a typical 100-query session, your total spend will be under **$0.10** using this strategy.

### Getting API Keys

| Provider | Link | Free Tier |
|---|---|---|
| Google Gemini | [aistudio.google.com](https://aistudio.google.com/app/apikey) | Yes — 15 req/min free |
| DeepSeek | [platform.deepseek.com](https://platform.deepseek.com/api_keys) | $5 free credit |
| OpenRouter | [openrouter.ai/keys](https://openrouter.ai/keys) | Free credits on signup |

---

## 5. Tech Stack

### Backend

| Layer | Technology | Purpose |
|---|---|---|
| API Framework | FastAPI + Uvicorn | Async REST API, auto-generated OpenAPI docs |
| Task Queue | Celery + Redis | Non-blocking background ingestion jobs |
| Code Parsing | tree-sitter, Python `ast` | Multi-language AST extraction |
| Embeddings | Google `text-embedding-004` | Code-aware semantic vectors (free tier) |
| Embedding Fallback | BGE-code (local, HuggingFace) | Offline / no-API-key mode |
| Vector Database | Qdrant | Semantic search with metadata filtering |
| Graph Database | Neo4j + Cypher | Dependency graph storage and traversal |
| Relational DB | PostgreSQL + SQLAlchemy | Repo, file, and chunk metadata |
| Migrations | Alembic | Schema versioning |
| LLM — Primary | Google Gemini 2.0 Flash | Fast reasoning, 1M context, free tier |
| LLM — Code | DeepSeek Coder V2 | Best-in-class code Q&A and bug tracing |
| LLM — Fallback | OpenRouter | 100+ model access via single API key |
| Testing | Pytest | Unit and integration tests |

### Frontend

| Layer | Technology | Purpose |
|---|---|---|
| Framework | Next.js 14 + TypeScript | SSR, routing, typed API client |
| Graph Visualization | D3.js | Interactive force-directed dependency graphs |
| Code Display | Shiki | Syntax highlighting with VS Code themes |
| Styling | Tailwind CSS | Utility-first responsive design |

### Infrastructure

| Layer | Technology | Purpose |
|---|---|---|
| Containers | Docker + Docker Compose | Full reproducible environment |
| Reverse Proxy | Nginx | Routing and SSL termination |
| CI/CD | GitHub Actions | Automated test and lint pipeline |

---

## 6. Project Structure

```
ai-codebase-intelligence/
│
├── backend/                              # Python FastAPI application
│   ├── api/
│   │   ├── routes/
│   │   │   ├── ingest.py                 # POST /ingest
│   │   │   ├── query.py                  # POST /ask
│   │   │   ├── graph.py                  # GET  /graph/{repo_id}
│   │   │   ├── search.py                 # GET  /search
│   │   │   └── analysis.py               # POST /analyze
│   │   ├── schemas/
│   │   │   ├── ingest_schema.py          # Pydantic: IngestRequest / IngestResponse
│   │   │   ├── query_schema.py           # Pydantic: QueryRequest / QueryResponse
│   │   │   └── graph_schema.py           # Pydantic: GraphNode / GraphEdge
│   │   └── middleware.py                 # CORS, rate limiting, request logging
│   │
│   ├── ingestion/
│   │   ├── repo_loader.py                # Clone repo via GitPython
│   │   ├── file_scanner.py               # Walk dirs, filter by extension
│   │   ├── language_detector.py          # Detect language per file
│   │   └── chunker.py                    # Split files into logical code chunks
│   │
│   ├── parsing/
│   │   ├── base_parser.py                # Abstract parser interface
│   │   ├── tree_sitter_parser.py         # Multi-language AST (Python/JS/Go/Java)
│   │   ├── python_ast_parser.py          # Python-specific deep parsing
│   │   ├── js_parser.py                  # JavaScript / TypeScript parsing
│   │   └── metadata_extractor.py         # Extract functions, classes, imports
│   │
│   ├── graph/
│   │   ├── call_graph.py                 # Function-level call relationships
│   │   ├── module_graph.py               # Import / dependency relationships
│   │   ├── class_graph.py                # Class inheritance hierarchies
│   │   ├── graph_builder.py              # Orchestrates all graph types
│   │   ├── graph_store.py                # Persist and query via Neo4j
│   │   └── graph_utils.py                # Subgraph extraction, path finding
│   │
│   ├── embeddings/
│   │   ├── base_embedder.py              # Abstract embedder interface
│   │   ├── gemini_embedder.py            # Google text-embedding-004 (primary)
│   │   ├── local_embedder.py             # BGE-code local fallback (no API needed)
│   │   ├── vector_store.py               # Qdrant client: upsert and search
│   │   └── embedding_pipeline.py         # Orchestrate chunk → embed → store
│   │
│   ├── retrieval/
│   │   ├── retriever.py                  # Hybrid: vector search + graph expansion
│   │   ├── reranker.py                   # Cross-encoder reranking
│   │   ├── graph_expander.py             # Expand nodes via call graph
│   │   └── context_builder.py            # Assemble final LLM context window
│   │
│   ├── reasoning/
│   │   ├── llm_router.py                 # Smart provider routing logic
│   │   ├── gemini_client.py              # Google Gemini API client
│   │   ├── deepseek_client.py            # DeepSeek API client
│   │   ├── openrouter_client.py          # OpenRouter API client (fallback)
│   │   ├── prompt_templates.py           # All system and user prompt templates
│   │   ├── chain.py                      # RAG chain orchestration
│   │   └── response_parser.py            # Parse and structure LLM output
│   │
│   ├── analysis/
│   │   ├── bug_detector.py               # DeepSeek Coder + static analysis for bugs
│   │   ├── security_scanner.py           # Pattern + LLM security scanning
│   │   └── complexity_analyzer.py        # Cyclomatic complexity, depth metrics
│   │
│   ├── tasks/
│   │   ├── celery_app.py                 # Celery app + Redis broker config
│   │   ├── ingest_task.py                # Async ingestion task
│   │   └── embed_task.py                 # Async embedding task
│   │
│   ├── db/
│   │   ├── database.py                   # SQLAlchemy async engine + session
│   │   ├── models.py                     # ORM: Repository, File, CodeChunk
│   │   └── migrations/
│   │       └── versions/                 # Alembic migration files
│   │
│   ├── core/
│   │   ├── config.py                     # Pydantic Settings — reads .env
│   │   ├── logging.py                    # Structured JSON logging
│   │   └── exceptions.py                 # Custom exception classes
│   │
│   ├── tests/
│   │   ├── unit/
│   │   │   ├── test_repo_loader.py
│   │   │   ├── test_file_scanner.py
│   │   │   ├── test_ast_parser.py
│   │   │   ├── test_graph_builder.py
│   │   │   ├── test_gemini_embedder.py
│   │   │   ├── test_llm_router.py
│   │   │   └── test_retriever.py
│   │   ├── integration/
│   │   │   ├── test_ingest_endpoint.py
│   │   │   ├── test_query_endpoint.py
│   │   │   └── test_full_pipeline.py
│   │   └── fixtures/
│   │       └── sample_repo/
│   │
│   ├── main.py
│   └── requirements.txt
│
├── frontend/                             # Next.js application
│   ├── public/
│   └── src/
│       ├── components/
│       │   ├── Chat/
│       │   │   ├── ChatWindow.tsx
│       │   │   ├── MessageBubble.tsx
│       │   │   └── QueryInput.tsx
│       │   ├── Graph/
│       │   │   ├── DependencyGraph.tsx
│       │   │   ├── GraphControls.tsx
│       │   │   └── NodeTooltip.tsx
│       │   ├── CodeViewer/
│       │   │   ├── CodeBlock.tsx
│       │   │   └── FileTree.tsx
│       │   └── Layout/
│       │       ├── Header.tsx
│       │       └── Sidebar.tsx
│       ├── pages/
│       │   ├── index.tsx
│       │   ├── dashboard/[repoId].tsx
│       │   └── graph/[repoId].tsx
│       ├── hooks/
│       │   ├── useIngest.ts
│       │   ├── useQuery.ts
│       │   └── useGraph.ts
│       ├── lib/
│       │   ├── api.ts
│       │   └── types.ts
│       └── styles/globals.css
│
├── docker/
│   ├── Dockerfile.backend
│   ├── Dockerfile.frontend
│   ├── Dockerfile.worker
│   └── docker-compose.yml
│
├── infra/
│   ├── nginx/nginx.conf
│   └── scripts/
│       ├── setup.sh
│       └── seed_db.sh
│
├── notebooks/
│   ├── 01_ast_exploration.ipynb
│   ├── 02_gemini_embedding_experiments.ipynb
│   ├── 03_graph_analysis.ipynb
│   └── 04_llm_provider_comparison.ipynb
│
├── data/
│   ├── raw/                              # Cloned repositories (gitignored)
│   ├── processed/                        # Parsed metadata cache (gitignored)
│   └── benchmarks/                       # Evaluation query sets
│
├── .env.example
├── .gitignore
├── pyproject.toml
├── Makefile
└── README.md
```

---

## 7. Dataset Sources

### 7.1 Repository Input Data

| Source | Description | Access |
|---|---|---|
| GitHub Public Repos | Any public repository via URL | Direct `git clone` |
| GitHub REST API v3 | Repo metadata, file trees, commit history | `api.github.com` — 5,000 req/hr with auth |
| GH Archive | Bulk event data across millions of public repos | [gharchive.org](https://www.gharchive.org) |
| GHTorrent | Historical GitHub relational data dumps | [ghtorrent.org](http://ghtorrent.org) |

### 7.2 Code Corpora — Embedding and Fine-tuning

| Dataset | Languages | Size | Link |
|---|---|---|---|
| The Stack v2 (BigCode) | 600+ languages | 67 TB (filtered subsets) | [HuggingFace](https://huggingface.co/datasets/bigcode/the-stack-v2) |
| CodeSearchNet | Python, JS, Java, Go, PHP, Ruby | 2M function + docstring pairs | [HuggingFace](https://huggingface.co/datasets/code_search_net) |
| GitHub Code (CodeParrot) | Python | 50 GB | [HuggingFace](https://huggingface.co/datasets/codeparrot/github-code) |
| SWE-bench | Python (real GitHub issues) | 2,300 issue–fix pairs | [HuggingFace](https://huggingface.co/datasets/princeton-nlp/SWE-bench) |
| HumanEval | Python | 164 evaluation tasks | [GitHub](https://github.com/openai/human-eval) |

### 7.3 Semantic Search Benchmarks

| Dataset | Use | Link |
|---|---|---|
| CodeSearchNet Queries | Evaluate NL → code retrieval (MRR, NDCG) | Included in CodeSearchNet |
| CoSQA | 20K natural language + code pairs | [HuggingFace](https://huggingface.co/datasets/code_x_glue_tt_text_to_text) |
| CodeXGLUE | Full suite: search, summarization, repair | [GitHub](https://github.com/microsoft/CodeXGLUE) |

### 7.4 Security and Bug Detection

| Dataset | Description | Link |
|---|---|---|
| Devign | 27K C functions labeled vulnerable / safe | [HuggingFace](https://huggingface.co/datasets/google/code_x_glue_cc_defect_detection) |
| BigVul | 3,700 CVE-linked vulnerable C/C++ functions | [GitHub](https://github.com/ZeoVan/MSR_20_Code_vulnerability_CSV_Dataset) |
| DiverseVul | 18K vulnerable functions across 320 CWE types | [GitHub](https://github.com/wagner-group/diversevul) |
| NVD / CWE Database | Vulnerability pattern definitions | [nvd.nist.gov](https://nvd.nist.gov) |

### 7.5 QA Pairs — LLM Reasoning Evaluation

| Dataset | Description | Link |
|---|---|---|
| CodeQA | NL question-answer pairs grounded in real code | [GitHub](https://github.com/jadecxliu/CodeQA) |
| StaQC | 148K Python and SQL QA pairs from Stack Overflow | [GitHub](https://github.com/LittleYUYU/StackOverflow-Question-Code-Dataset) |

---

## 8. Development Roadmap

```
Weeks  1 –  2  │  Phase 1 — Foundation          │  Ingestion, DB, Celery
Weeks  3 –  5  │  Phase 2 — Parsing & Graphs    │  AST, call/module/class graphs
Weeks  6 –  7  │  Phase 3 — Embeddings & Search │  Gemini embeddings, Qdrant search
Weeks  8 – 10  │  Phase 4 — LLM Reasoning       │  Multi-provider RAG chain, /ask
Weeks 11 – 13  │  Phase 5 — Frontend            │  Next.js UI, chat, D3.js viz
Weeks 14 – 16  │  Phase 6 — Production          │  Docker stack, eval, docs
```

---

### Phase 1 — Foundation `Weeks 1–2`

| Week | Deliverables |
|---|---|
| Week 1 | Scaffold, Docker Compose, `.env` with all 3 API keys, PostgreSQL + Alembic, `POST /ingest`, GitPython cloning, file scanner |
| Week 2 | Celery + Redis task queue, chunker, async ingestion task, status polling, GitHub Actions CI |

**Exit criteria**
- [ ] `POST /ingest` returns `repo_id` + `task_id` immediately
- [ ] Celery worker clones repo and stores all source files in PostgreSQL
- [ ] 10+ unit tests pass; CI runs on push

---

### Phase 2 — Parsing & Graphs `Weeks 3–5`

| Week | Deliverables |
|---|---|
| Week 3 | tree-sitter multi-language parser, Python AST deep parser, metadata extractor |
| Week 4 | Call graph + module graph stored in Neo4j |
| Week 5 | Class hierarchy graph, `/graph/{repo_id}` endpoint, graph utilities |

**Exit criteria**
- [ ] All functions, classes, imports extracted from any Python repo
- [ ] Neo4j populated with call + import edges
- [ ] Graph traces `login_controller → auth_service → db_layer` correctly

---

### Phase 3 — Embeddings & Vector Search `Weeks 6–7`

| Week | Deliverables |
|---|---|
| Week 6 | Gemini `text-embedding-004` integration, Qdrant setup, batch embed pipeline with rate-limit handling |
| Week 7 | `GET /search` endpoint, hybrid search (Gemini vector + BM25), reranking, evaluated on CodeSearchNet |

**Exit criteria**
- [ ] Query `"password hashing"` returns `hash_password()` in top 3
- [ ] MRR@10 > 0.60 on internal test queries
- [ ] Gemini API rate limits handled with exponential backoff

---

### Phase 4 — LLM Reasoning & RAG `Weeks 8–10`

| Week | Deliverables |
|---|---|
| Week 8 | LLM router (Gemini primary, DeepSeek for code, OpenRouter fallback), prompt templates, basic RAG chain |
| Week 9 | Graph-aware context expansion, `POST /ask` with streaming, DeepSeek Coder routed for code-specific questions |
| Week 10 | Bug localization (DeepSeek R1), security analysis chain, provider failover tested, full RAG integration tests |

**Exit criteria**
- [ ] LLM router selects DeepSeek Coder for code tasks automatically
- [ ] `POST /ask` returns grounded answers with file/function citations
- [ ] If Gemini is down, OpenRouter fallback fires automatically

---

### Phase 5 — Frontend `Weeks 11–13`

| Week | Deliverables |
|---|---|
| Week 11 | Next.js + TypeScript setup, ingestion UI, API hooks |
| Week 12 | Chat interface with streaming, syntax highlighting via Shiki, file tree |
| Week 13 | D3.js force-directed graph, architecture view, mobile polish |

---

### Phase 6 — Production & Evaluation `Weeks 14–16`

| Week | Deliverables |
|---|---|
| Week 14 | Full Docker Compose stack, Nginx reverse proxy |
| Week 15 | End-to-end evaluation on 5 real repos, latency + cost benchmarks per LLM provider |
| Week 16 | README, OpenAPI docs, demo video, provider comparison report |

---

## 9. API Reference

### POST `/api/v1/ingest`

```json
Request:
{ "github_url": "https://github.com/tiangolo/fastapi", "branch": "master" }

Response 202:
{ "repo_id": "uuid...", "task_id": "celery-uuid...", "status": "queued" }
```

---

### GET `/api/v1/status/{task_id}`

```json
{ "task_id": "...", "status": "processing", "progress": 42, "message": "Embedding 1200/2847 chunks" }
```

Status values: `queued` · `cloning` · `scanning` · `parsing` · `embedding` · `completed` · `failed`

---

### POST `/api/v1/ask`

```json
Request:
{ "repo_id": "uuid...", "question": "How does authentication work?", "stream": true }

Response:
{
  "answer": "Authentication works in three stages...",
  "provider_used": "deepseek-coder-v2",
  "sources": [{ "file": "auth/service.py", "function": "verify_password", "lines": "45-67" }],
  "graph_path": ["login_controller", "auth_service", "verify_password"]
}
```

---

### GET `/api/v1/search`

```
GET /api/v1/search?q=password+hashing&repo_id=uuid...&top_k=5
```

```json
{ "results": [{ "function": "hash_password", "file": "utils/crypto.py", "score": 0.94 }] }
```

---

### GET `/api/v1/graph/{repo_id}`

```json
{
  "nodes": [{ "id": "auth_service.verify_password", "type": "function", "file": "auth/service.py" }],
  "edges": [{ "source": "login_controller.handle", "target": "auth_service.verify_password", "type": "calls" }]
}
```

---

## 10. Getting Started

### Prerequisites

| Tool | Version |
|---|---|
| Docker | 24.0+ |
| Docker Compose | 2.0+ |
| Python | 3.11+ |
| Node.js | 18+ |

### API Keys Required

| Key | Get It |
|---|---|
| `GEMINI_API_KEY` | [aistudio.google.com](https://aistudio.google.com/app/apikey) |
| `DEEPSEEK_API_KEY` | [platform.deepseek.com](https://platform.deepseek.com/api_keys) |
| `OPENROUTER_API_KEY` | [openrouter.ai/keys](https://openrouter.ai/keys) |

### Quick Start

```bash
git clone https://github.com/your-username/ai-codebase-intelligence
cd ai-codebase-intelligence

cp .env.example .env
# Open .env and fill in GEMINI_API_KEY, DEEPSEEK_API_KEY, OPENROUTER_API_KEY

docker compose -f docker/docker-compose.yml up --build
```

Services available after startup:

```
Frontend       →  http://localhost:3000
API docs       →  http://localhost:8000/docs
Qdrant UI      →  http://localhost:6333/dashboard
Neo4j browser  →  http://localhost:7474
```

### Local Backend Dev

```bash
cd backend
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
alembic upgrade head
uvicorn main:app --reload --port 8000

# Separate terminal
celery -A tasks.celery_app worker --loglevel=info
```

### Makefile Shortcuts

```bash
make dev        # Start full Docker stack
make test       # Run pytest
make lint       # Run ruff + mypy
make migrate    # Alembic upgrade head
make clean      # Stop and remove volumes
```

---

## 11. Contributing

1. Fork and create a branch: `git checkout -b feature/your-feature`
2. Make changes and verify: `make test && make lint`
3. Commit: `git commit -m "feat: add DeepSeek R1 for security analysis"`
4. Open a pull request

---

## 12. License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">
Built at IIIT SriCity · Data Team
</div>
