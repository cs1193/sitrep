# SITREP Documentation

Welcome to the SITREP documentation. This directory contains usage guides, architecture documentation, and analysis reports.

## 📚 Documentation Structure

### Quick Start
- **[USAGE.md](./USAGE.md)** — User guide covering CLI, Web UI, and Python API usage

### Architecture & Analysis (Generated via Multi-Agent Swarm)
- **[ANALYSIS_INDEX.md](./ANALYSIS_INDEX.md)** — Quick reference guide and navigation index
- **[SITREP_CODEBASE_ANALYSIS.md](./SITREP_CODEBASE_ANALYSIS.md)** — Comprehensive 13K+ word deep-dive into architecture, layers, APIs, data flows, and patterns
- **[ARCHITECTURE_DIAGRAMS.md](./ARCHITECTURE_DIAGRAMS.md)** — 10 ASCII architecture diagrams covering:
  - 5-layer clean architecture
  - Data model relationships
  - Hybrid retrieval pipeline
  - RL compression training loop
  - Ingest pipeline (end-to-end)
  - Dependency graph
  - Fact lifecycle & state transitions
  - Complete query flow
  - Error recovery & transaction rollback
  - Performance profile & latencies

## 🎯 What to Read First

1. **New to SITREP?** → Start with [USAGE.md](./USAGE.md)
2. **Understanding architecture?** → Read [ANALYSIS_INDEX.md](./ANALYSIS_INDEX.md) (5 min) then [SITREP_CODEBASE_ANALYSIS.md](./SITREP_CODEBASE_ANALYSIS.md) (30 min)
3. **Need visual reference?** → See [ARCHITECTURE_DIAGRAMS.md](./ARCHITECTURE_DIAGRAMS.md)
4. **Specific question?** → Use [ANALYSIS_INDEX.md](./ANALYSIS_INDEX.md) navigation guide to jump to relevant section

## 📖 Document Overview

| Document | Length | Purpose |
|----------|--------|---------|
| USAGE.md | 20K | Entry points (CLI, Web UI, Python API, Plugin), configuration |
| ANALYSIS_INDEX.md | 9K | Quick reference, navigation index, key takeaways |
| SITREP_CODEBASE_ANALYSIS.md | 42K | Comprehensive architecture analysis (13 sections) |
| ARCHITECTURE_DIAGRAMS.md | 112K | 10 ASCII diagrams with detailed annotations |

## 🔍 Key Topics

### System Design
- 5-layer clean architecture (Domain → Adapters → Application → Infrastructure)
- Multi-database approach (SQLite, KuzuDB, ChromaDB, DuckDB)
- Optional feature activation via lazy imports ([rag], [graph], [llm], [rl], [web])

### Core Features
- **Hybrid Retrieval:** Dense + sparse search with learnable fusion weights
- **RL-Optimized Compression:** PPO agent learns optimal token reduction per passage
- **Temporal Reasoning:** Allen interval algebra for time-aware queries
- **Causal Reasoning:** do-calculus support for effect estimation
- **Privacy-First:** Fully local, zero external service dependencies

### Data Persistence
- `metadata/sitrep.db` — SQLite FTS5 (facts, passages, feedback, decisions, KV cache)
- `graph/` — KuzuDB knowledge graph with temporal relationships
- `vectors/` — ChromaDB embeddings for semantic search
- `documents/` — Raw/chunked/archived content (Parquet/JSONL)
- `agents/` — Trained PPO policy checkpoints
- `lineage/` — Decision trace graph (audit trail)

### Workflows
- **Query:** Intent classification → hybrid search → reranking → compression → LLM explanation
- **Ingest:** Parse → chunk → embed → extract facts → build graph → atomic write
- **Train:** Rollout passages → compute advantages → PPO policy update → checkpoint
- **Feedback:** Store relevance signals → update fusion weights → improve retrieval

## 🚀 Getting Started

### Installation
```bash
cd sitrep-engine
uv sync --extra rag --extra graph --extra web  # Install with common extras
```

### Run Web UI
```bash
uv run scripts/run_web.py
# Opens http://localhost:7860
```

### CLI Query
```bash
uv run scripts/query_cli.py --query "Your question here" --top-k 5 --explain
```

### Python API
```python
from sitrep import build_application, SitrepConfig

config = SitrepConfig()
app = build_application(config)
result = await app.query("Your question here", top_k=5)
print(result.explanation)
```

## 📊 Architecture at a Glance

```
┌─────────────────────────────────────────┐
│  PRESENTATION (Web UI, CLI, Plugin)     │
├─────────────────────────────────────────┤
│  APPLICATION (Query, Ingest, Train,     │
│              Feedback, Versioning)      │
├─────────────────────────────────────────┤
│  ADAPTERS (Repositories, Services)      │
├─────────────────────────────────────────┤
│  DOMAIN (Fact, Passage, Episode, ...)   │
├─────────────────────────────────────────┤
│  INFRASTRUCTURE (DB clients, Retrieval, │
│                 RL, LLM, Compression)   │
└─────────────────────────────────────────┘
```

## 🔗 Navigation

- **For developers:** Read sections 2–7 of [SITREP_CODEBASE_ANALYSIS.md](./SITREP_CODEBASE_ANALYSIS.md)
- **For DevOps:** Check section 6 (persistence), section 10 (patterns), section 11 (dependencies)
- **For product:** Focus on [USAGE.md](./USAGE.md) + section 1 of analysis
- **For architects:** Review all diagrams in [ARCHITECTURE_DIAGRAMS.md](./ARCHITECTURE_DIAGRAMS.md) + section 2 (layers)

## 📝 Contributing

When adding new features:
1. Update [USAGE.md](./USAGE.md) with new entry points
2. Add diagrams to [ARCHITECTURE_DIAGRAMS.md](./ARCHITECTURE_DIAGRAMS.md) for new flows
3. Update [SITREP_CODEBASE_ANALYSIS.md](./SITREP_CODEBASE_ANALYSIS.md) if architecture changes

## ❓ FAQ

**Q: Where do I start?**  
A: [USAGE.md](./USAGE.md) for practical usage, [ANALYSIS_INDEX.md](./ANALYSIS_INDEX.md) for architecture overview.

**Q: How does compression work?**  
A: See Diagram #4 in [ARCHITECTURE_DIAGRAMS.md](./ARCHITECTURE_DIAGRAMS.md) + Section 2e in [SITREP_CODEBASE_ANALYSIS.md](./SITREP_CODEBASE_ANALYSIS.md).

**Q: How do I extend SITREP?**  
A: See Section 10 (patterns) and Section 12 (recommendations) in [SITREP_CODEBASE_ANALYSIS.md](./SITREP_CODEBASE_ANALYSIS.md).

**Q: What databases are supported?**  
A: SQLite (default), KuzuDB (graph), ChromaDB (vectors), DuckDB (analytics). See Section 2a in analysis.

---

**Generated:** 2026-07-09  
**Analysis Method:** 7-agent parallel swarm analysis  
**Last Updated:** 2026-07-09
