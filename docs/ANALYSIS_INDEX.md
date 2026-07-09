# SITREP Multi-Agent Swarm Analysis — Quick Reference

## Generated Documents

1. **Main Report:** [`SITREP_CODEBASE_ANALYSIS.md`](./SITREP_CODEBASE_ANALYSIS.md) — 13K+ words, comprehensive deep-dive
2. **Architecture Diagrams:** [`ARCHITECTURE_DIAGRAMS.md`](./ARCHITECTURE_DIAGRAMS.md) — 10 ASCII diagrams (see list below)

---

## ASCII Architecture Diagrams (10 Total)

See [`ARCHITECTURE_DIAGRAMS.md`](./ARCHITECTURE_DIAGRAMS.md) for full diagrams:

| # | Diagram | Purpose |
|---|---------|---------|
| 1 | **Overall System Architecture** | 5-layer clean architecture overview (Domain → Adapters → Application → Infrastructure) |
| 2 | **Data Model Relationships** | SQLite schema with foreign keys (Fact ← Passage, Decision, Feedback, etc.) |
| 3 | **Hybrid Retrieval Pipeline** | Complete flow: intent → dense search → sparse search → fusion → reranking → compression → explanation |
| 4 | **RL Training Loop (PPO)** | Compression agent training: rollout → advantage computation → policy gradient → checkpoint |
| 5 | **Ingest Pipeline** | End-to-end: parse → chunk → embed → extract → graph → atomic write → storage |
| 6 | **Dependency Graph** | Module dependencies (Application → Adapters → Infrastructure) with optional feature activation |
| 7 | **Fact Lifecycle** | State transitions: ACTIVE → UPDATE/FADE/MERGE/DELETE → ARCHIVED → RECOVER → ACTIVE |
| 8 | **Complete Query Flow** | Module interaction: classify → hybrid search → rerank → compress → explain → cache → feedback |
| 9 | **Error Recovery** | Transaction rollback across SQLite + KuzuDB + ChromaDB with consistency guarantees |
| 10 | **Performance Profile** | Latency breakdown: query (73-127 ms no LLM), ingest (455-3260 ms), scaling notes |

---

## Key Takeaways (30-second version)

**What is SITREP?**
- Self-Improving Token-Reduced Embeddable Pipeline
- Fully-local, privacy-first context-engineering system
- RL-optimized compression + hybrid retrieval + temporal reasoning

**Architecture:**
- **5-layer clean architecture** (Domain → Adapters → Application → Infrastructure → Utilities)
- **5 databases** (SQLite metadata, KuzuDB graph, ChromaDB vectors, DuckDB analytics, KV cache)
- **~12K LOC** core + 600 LOC scripts + 900 LOC tests

**Key Innovations:**
1. **PPO Compression Agent** — learns optimal compression ratios per passage (70–90% reduction)
2. **Hybrid Retrieval** — dense (semantic) + sparse (BM25) with learnable fusion weights
3. **Temporal Algebra** — 13 Allen interval relations for time-aware queries
4. **Causal Reasoning** — do-calculus support for counterfactual analysis
5. **Reversible Operations** — full audit trail + rollback capability

**Entry Points:**
- Web UI (`run_web.py`) → Gradio dashboard
- CLI (`query_cli.py`) → command-line queries
- Python API (`build_application()`) → programmatic use
- Claude Code Plugin (`plugin.py`) → integrated with Claude
- Batch Ingest (`ingest_batch.py`) → large document processing

---

## Navigation Guide

### For Understanding Architecture
→ Section 2: "Layered Architecture Deep Dive"
  - Domain layer (Fact, Passage, Episode, Decision schemas)
  - Infrastructure (Database clients, Retrieval, RL, LLM gateways)
  - Adapters (Repositories, Services, Business logic)
  - Application (Use cases: Query, Ingest, Train, Feedback, Versioning)
  - Presentation (Web UI, Plugin API)

### For Understanding Data Flow
→ Section 4: "Data Flow Diagrams" + Section 7: "Critical Data Flows"
  - Ingest pipeline (source → chunks → embeddings → storage)
  - Query pipeline (query → hybrid search → reranking → compression → explanation)
  - RL training loop (sample passages → rollout → compute advantages → PPO update)

### For Understanding APIs
→ Section 3: "Key APIs & Interfaces"
  - Composition root (`build_application()`)
  - Repository interface (CRUD + domain queries)
  - Service interfaces (extraction, compression, judgment, versioning)
  - Configuration schema (`sitrep.yaml`)

### For Understanding Persistence
→ Section 6: ".sitrep/ Directory Layout" + SQLite Schemas
  - Where data lives (metadata, graph, vectors, documents, agents, lineage, logs)
  - How data is organized (SQLite tables, KuzuDB nodes, ChromaDB vectors)
  - How transactions work (ACID + rollback across DBs)

### For Understanding Dependencies
→ Section 5: "Dependencies & Imports" + Section 11: "Dependency Graph"
  - Core deps (always installed): pydantic, numpy, rank-bm25, pyyaml
  - Optional extras: `[rag]`, `[graph]`, `[llm]`, `[rl]`, `[web]`, `[duckdb]`
  - Lazy import strategy (heavy deps only loaded if extra installed)
  - Full dependency tree (Application → Services → Infrastructure → DB clients)

### For Understanding Patterns
→ Section 10: "Critical Patterns & Anti-Patterns"
  - Clean Architecture (strict layer separation)
  - Repository Pattern (persistence abstraction)
  - Composition Root (single dependency wiring point)
  - Lazy Import (optional feature activation)
  - Atomic Writes (transaction consistency across DBs)
  - Strategy Pattern (compression strategies)

### For Understanding Testing
→ Section 9: "Testing Strategy"
  - Unit tests (domain + adapters)
  - Integration tests (end-to-end workflows)
  - Evaluation tests (compression ratio, retrieval recall, fusion weight learning)

---

## Critical Insights

### 1. Clean Architecture Enables Evolution
- Swap databases (SQLite ↔ PostgreSQL) without touching domain logic
- Replace retrieval strategy (hybrid → dense-only) via service injection
- Plug in different LLMs (Ollama ↔ Transformers ↔ Demo) via adapter pattern

### 2. RL Compression is Core Innovation
- PPO agent learns optimal compression per passage type
- Achieves 70–90% token reduction while preserving semantic meaning
- Reward model trains on downstream task accuracy (QA, NER, classification)

### 3. Hybrid Retrieval Adapts from Feedback
- Dense search (semantic), sparse search (keyword), entity graph ranking combined
- Fusion weights w_dense, w_sparse learned online from user feedback
- No retraining required — weights updated incrementally

### 4. Temporal Reasoning is Underutilized
- Allen algebra supports 13 temporal relations
- Enables queries like "Facts during 2025-06" or "X overlaps Y's timeline"
- Currently dormant; consider surface in query UI

### 5. Causal Reasoning Foundation Exists
- do-calculus support for "Did X cause Y?" questions
- Decision graph traces all transformations (INGEST, UPDATE, MERGE, DELETE)
- Could enable counterfactual analysis ("What if we deleted fact F?")

### 6. Privacy-First by Design
- Zero external services; everything runs local
- All data in `.sitrep/` directory (can be encrypted, backed up, deleted)
- No model downloads unless optional extras installed

---

## Quick File Lookup

**Main Logic**
- Domain schemas: `src/domain/schemas.py` (200 LOC)
- Use cases: `src/application/use_cases.py` (1200 LOC)
- Composition: `src/application/__init__.py` (457 LOC)

**Retrieval**
- Hybrid search: `src/infrastructure/retrieval.py` (814 LOC)

**Compression**
- SmartCrusher: `src/infrastructure/compression.py` (542 LOC)
- RL agent: `src/infrastructure/rl.py` (508 LOC)

**Database**
- Client wrappers: `src/infrastructure/database.py` (694 LOC)
- Repository impl: `src/adapters/repositories.py` (600 LOC)

**Entry Points**
- Web UI: `scripts/run_web.py` (120 LOC)
- CLI: `scripts/query_cli.py` (80 LOC)
- Plugin: `plugin.py` (131 LOC)

---

## Recommendations for Next Phase

### 🟢 Quick Wins (1–2 days)
1. **Surface temporal queries in UI** — "Facts active during date range" button
2. **Expose entity graph ranking** — toggle PPR in query settings
3. **Add compression metrics** — show original vs. compressed token count per fact

### 🟡 Medium Effort (1 week)
1. **Increase test coverage** — from 7% to 30% (unit + integration tests)
2. **Profile retrieval bottleneck** — identify slow paths (dense search? BM25? reranking?)
3. **Document configuration** — expand `.env.example` with all PRAGMA options, timeouts, thresholds
4. **Add multimodal image ingest** — web scraping + CLIP embeddings for image retrieval

### 🔴 Strategic (2+ weeks)
1. **Causal query integration** — wire do-calculus into main query flow ("What caused X?")
2. **Memory consolidation scheduler** — auto-run importance scoring + forgetting policy
3. **Distributed architecture** — consider multi-node graph store + vector DB for scale
4. **Offline-to-online learning** — train RL agent on inference data (user feedback) without interruption

---

## How to Use This Analysis

1. **Quick context refresh** → Read this file (5 min)
2. **Understand a specific layer** → Jump to Section 2 of main report
3. **Trace a data flow** → Jump to Section 4 or 7
4. **Understand API contracts** → Jump to Section 3
5. **Plan refactoring** → Read Section 10 (patterns) + Section 11 (dependency graph)
6. **Onboard new engineer** → Start with Section 1 (structure) + walk through Section 2 (layers)

---

**Report Generated:** 2026-07-09  
**Analysis Method:** 7-agent parallel swarm (Scout → Analyze → Synthesize)  
**Total Analysis Time:** ~7 minutes  
**Confidence Level:** High (all agents completed successfully, zero errors)
