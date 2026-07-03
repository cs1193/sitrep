# SITREP

**Self-Improving Token-Reduced Embeddable Pipeline** — a fully local,
privacy-first context-engineering system. It combines multi-agent knowledge
graph construction, RL-optimized adaptive compression, intelligent KV caching,
temporal memory, hybrid retrieval with learnable fusion weights, and a Gradio
web interface.

> No external services. Everything is embeddable: SQLite + FTS5, KuzuDB,
> ChromaDB, local LLMs (Ollama or HuggingFace Transformers), and `stable-baselines3`
> PPO. All data lives in `.sitrep/` under your working directory.

---

## Why it runs out of the box

Heavy dependencies are **lazy-imported** behind optional extras. The core
package installs and imports on Python 3.12 with **zero model downloads**. When
you add an extra (or a model), the corresponding features light up:

| Capability           | Extra                | Fallback when absent                          |
|----------------------|----------------------|-----------------------------------------------|
| Dense embeddings     | `[rag]`              | deterministic hash embedding                  |
| Vector store         | `[rag]`              | SQLite FTS5 + in-memory index                 |
| Knowledge graph      | `[graph]`            | SQLite adjacency / logical fallback           |
| LLM generation       | `[llm]` (ollama/HF)  | DEMO mode (regex extraction, heuristic reply) |
| PPO compression      | `[rl]`               | heuristic compression policy                  |
| KV cache             | `[llm]` + torch      | disabled (no caching)                         |
| Gradio UI            | `[web]`              | CLI scripts still work                        |

LLM backend resolution (`auto`): **Ollama** → **Transformers** → **DEMO**.

---

## Install

```bash
cd sitrep-engine
uv sync                       # core only
uv sync --extra rag --extra web --extra graph        # add capabilities
uv sync --extra all           # everything (downloads large wheels)
```

## Quickstart

```bash
# 1) Launch the web UI (Tabs: Query, Ingest, Stats, Train, Lineage, Versioning)
uv run python scripts/run_web.py

# 2) Ingest documents from a folder
uv run python scripts/ingest_batch.py ./my_docs

# 3) Ask a question on the CLI
uv run python scripts/query_cli.py "What facts do you know about X?"

# 4) Train the PPO compression agent on accumulated feedback
uv run python scripts/train_compression_agent.py

# 5) Precompute KV caches for all passages
uv run python scripts/update_kv_cache.py

# 6) Snapshot / restore the whole .sitrep/
uv run python scripts/backup.py snapshot
```

---

## Architecture (Clean Architecture)

```
domain         → entities (Schema, Fact, TemporalFact, Passage, Episode,
                 Agent, Decision, Skill), value objects, port interfaces
application    → use cases (Ingest, Query, Feedback, Train, Version, Lineage)
adapters       → services (extraction, embedding, compression, ...),
                 repositories (sqlite, kv_cache)
infrastructure → db clients (sqlite/fts5, kuzu, chroma, duckdb),
                 llm gateways, retrieval (hybrid/rerank/temporal),
                 rl (PPO env/agent/reward), kv_cache, chunking,
                 versioning, lineage, monitoring, event_bus
presentation   → Gradio web app
```

Data is partitioned by bounded context under `.sitrep/`:

```
.sitrep/
├── metadata/   SQLite (schemas, facts, passages, episodes, lineage, kv_cache)
├── graph/      KuzuDB knowledge graph (entities + temporal relationships)
├── vectors/    ChromaDB embeddings (passages, facts, schemas)
├── documents/  raw / chunks / archives (Parquet/JSONL)
├── agents/     RL policies (PPO), agent configs
├── lineage/    KuzuDB lineage graph (decision traces)
├── logs/       operation logs
└── config/     project configuration (YAML)
```

---

## Claude Code integration

A `plugin.py` + `claude_plugin.json` are included so SITREP can be loaded as a
Claude Code plugin (query/ingest/train skills exposed as commands).

## Status

Complete and verified across all layers in demo mode (zero model downloads).
Run the test suite with `uv sync --extra dev && uv run pytest`.
