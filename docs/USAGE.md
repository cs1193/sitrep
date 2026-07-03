# SITREP — Usage Guide

**Self-Improving Token-Reduced Embeddable Pipeline.** A fully local, privacy-first
context-engineering system: multi-agent knowledge-graph construction, RL-optimized
adaptive compression, intelligent KV caching, temporal memory, hybrid retrieval
with learnable weights, multimodal + causal reasoning, and a Gradio web UI.

> All data lives in `.sitrep/` under your working directory. No external services.
> Runs out-of-the-box in **demo mode** with zero model downloads; real models
> activate when you install optional extras.

---

## Table of contents
1. [Install](#1-install)
2. [Quickstart (5 minutes)](#2-quickstart-5-minutes)
3. [The `.sitrep/` data directory](#3-the-sitrep-data-directory)
4. [Configuration](#4-configuration)
5. [Backends: LLM & embeddings](#5-backends-llm--embeddings)
6. [Core workflows](#6-core-workflows)
7. [Feature guide (Phases A–G)](#7-feature-guide-phases-ag)
8. [Programmatic API](#8-programmatic-api)
9. [Scripts reference](#9-scripts-reference)
10. [Claude Code integration](#10-claude-code-integration)
11. [Evaluation harness](#11-evaluation-harness)
12. [Testing](#12-testing)
13. [Troubleshooting & caveats](#13-troubleshooting--caveats)

---

## 1. Install

SITREP uses [`uv`](https://docs.astral.sh/uv/) and Python **>=3.10, <3.13**.

```bash
cd sitrep-engine
uv sync                                  # core only (runs in demo mode)
uv sync --extra rag --extra graph --extra web    # add capabilities
uv sync --extra all                      # everything (large downloads)
uv sync --extra dev                      # + pytest for development
```

### Optional extras

| Extra | What it enables | Without it (fallback) |
|-------|-----------------|----------------------|
| `rag` | real dense embeddings + Chroma vector store | hash embedding + SQLite FTS5 scan |
| `graph` | KuzuDB knowledge/lineage graphs | SQLite adjacency / logical fallback |
| `rl` | PPO compression agent (stable-baselines3 + torch) | heuristic compression policy |
| `llm` | real LLM generation (Transformers / Ollama) | DEMO mode (deterministic) |
| `web` | Gradio UI | CLI scripts still work |
| `duckdb` | Parquet document archives / DuckDB analytics | JSONL archives |
| `dev` | pytest, pytest-cov | — |
| `all` | all of the above | — |

> **Python 3.12 recommended.** On 3.10 the full `[rag]` extra can hit an
> `onnxruntime` wheel conflict; installing just `sentence-transformers` (see
> [§5](#5-backends-llm--embeddings)) avoids it.

---

## 2. Quickstart (5 minutes)

```bash
# 1) Launch the web UI (Tabs: Query, Ingest, Stats, Train, Lineage, Versioning)
uv run python scripts/run_web.py                 # → http://127.0.0.1:7860

# 2) Ingest documents from a folder (.txt/.md/.json/.jsonl)
uv run python scripts/ingest_batch.py ./my_docs

# 3) Ask a question on the CLI
uv run python scripts/query_cli.py "What do you know about Acme?"

# 4) Train the compression agent on accumulated feedback
uv run python scripts/train_compression_agent.py

# 5) Snapshot the whole .sitrep/ before a risky change
uv run python scripts/backup.py snapshot --label baseline
```

That's it — in demo mode everything above works with **no model downloads**.

---

## 3. The `.sitrep/` data directory

Created lazily under your current working directory (override with `SITREP_BASE_DIR`):

```
.sitrep/
├── metadata/   SQLite (schemas, facts, passages, episodes, feedback, kv_cache,
│               lineage_events, retrieval_stats, fusion_weights, ccr_store, images,
│               cross_modal_links, media_assets)  + FTS5 indexes
├── graph/      KuzuDB knowledge graph (entities + temporal relationships)
├── vectors/    ChromaDB embeddings (passages, facts, schemas)
├── documents/  raw / chunks / archives (Parquet/JSONL)
├── agents/     RL policies (PPO), agent configs
├── lineage/    KuzuDB lineage graph (decision traces)
├── logs/       operation logs + events.wal (append-only event log)
└── config/     sitrep.yaml (persisted config)
```

Snapshots are written to a sibling `.sitrep_snapshots/` (outside the snapshotted tree).

---

## 4. Configuration

All knobs are **`SITREP_`-prefixed env vars** (or a `.env` file). See `.env.example`
for the full list. Highlights:

```ini
SITREP_BASE_DIR=.sitrep
SITREP_LLM_PROVIDER=auto            # auto | ollama | transformers | demo
SITREP_OLLAMA_URL=http://localhost:11434
SITREP_OLLAMA_MODEL=llama3.1:8b
SITREP_HF_LLM_MODEL=HuggingFaceTB/SmolLM-135M-Instruct
SITREP_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
SITREP_EMBEDDING_DIM=384
SITREP_TOP_K=5
SITREP_CONFIDENCE_THRESHOLD=0.55

# Phase A retrieval (dormant by default — enable with [rag])
SITREP_PPR_WEIGHT=0.0               # → ~0.15 with real embeddings for +Recall
SITREP_DENSITY_WEIGHT=0.0
SITREP_TEMPORAL_WEIGHT=0.0
SITREP_PPR_ALPHA=0.85  SITREP_PPR_GAMMA=0.8

# Phase B memory hygiene
SITREP_TRACK_ACCESS=false           # bump access_count on retrieval
SITREP_FORGETTING_MAX_AGE_DAYS=365
SITREP_FORGETTING_MIN_IMPORTANCE=0.2

# Phase C / E
SITREP_EVENT_LOG_ENABLED=true
SITREP_RESULT_CACHE_TTL=3600
SITREP_AUTO_JUDGE=false

# RL compression
SITREP_PPO_TOTAL_TIMESTEPS=10000
SITREP_COMPRESSION_MIN=0.2  SITREP_COMPRESSION_MAX=0.8
```

**Safe defaults:** Phase A boost weights = 0, `track_access=false`, hard-delete
never automatic. Nothing destructive happens without an explicit flag.

---

## 5. Backends: LLM & embeddings

**LLM cascade** (`SITREP_LLM_PROVIDER=auto`, the default): **Ollama → Transformers → Demo**.

- **Ollama** (recommended for real generation): install [Ollama](https://ollama.com),
  `ollama pull llama3.1:8b`, then run. SITREP talks to its HTTP API (no Python SDK needed).
- **Transformers** (in-process): `uv sync --extra llm`; first run downloads the model.
- **Demo** (default, zero downloads): deterministic, context-aware templated answers.
  The full pipeline (retrieve → compress → answer → confidence) is functional.

Switch explicitly: `SITREP_LLM_PROVIDER=ollama|transformers|demo`.

**Embeddings** (auto): `sentence-transformers` if importable, else a deterministic
hash embedding. To get real embeddings without the full `[rag]` extra:

```bash
uv run --with sentence-transformers --with "numpy<2" python scripts/query_cli.py "..."
```

> Pin `numpy<2` when using torch-based models — torch wheels are built against
> numpy 1.x, so numpy 2.x causes "Numpy is not available" at `.numpy()` time.

**Multimodal** (Phase G): `CLIPEmbedder` (images) and `AVEmbedder` (audio/video)
lazy-load real models; without them they fall back to **caption/transcript hashing**
so cross-modal retrieval still works.

---

## 6. Core workflows

### 6.1 Ingestion
Pipeline: **classify → chunk → embed → extract facts → resolve conflicts → store →
record provenance + lineage**.

```bash
uv run python scripts/ingest_batch.py ./docs --recursive --exts .txt,.md
```
Or paste text in the web UI **Ingest** tab, or the Python API:
```python
app.ingest_uc.execute(text="Acme Corp is a software company based in Berlin.")
```
Facts are extracted (regex in demo, LLM when available), conflicting facts are
detected & resolved (loser → `INVALIDATED`, with a `SUPERSEDES` provenance edge),
and an `Episode` groups the facts.

### 6.2 Querying
Pipeline: **retrieve (hybrid) → RL compression ratio → content-aware compress →
cache-align → LLM answer → confidence + quality → active learning**.

```bash
uv run python scripts/query_cli.py "Where is Acme located?" --top-k 5
```
The web **Query** tab shows the answer, confidence, token-reduction %, ranked
sources, and 👍/👎 feedback buttons. Use the **`QueryOrchestrator`** (Phase D) for
intent routing (simple / comparison / multi-hop / temporal / causal / multimodal /
aggregation).

### 6.3 Feedback (the self-improvement loop)
- **Explicit:** 👍/👎 in the UI updates fusion weights (online SGD) and feeds the
  RL reward model. Each query's per-channel scores are stored in `retrieval_stats`
  so feedback can attribute credit.
- **Automatic (Phase E):** `JudgeUseCase.judge_and_feedback(dto)` scores the answer
  with an LLM-judge (or heuristic) and nudges fusion weights from that score.

```python
dto = app.query_uc.execute("...")
result = app.judge_uc.judge_and_feedback(dto)   # {score, rationale, feedback_applied}
```

### 6.4 Training the compression agent
```bash
uv run python scripts/train_compression_agent.py --timesteps 10000
```
Trains PPO (`[rl]` extra) to pick the compression ratio per query; falls back to a
heuristic policy otherwise. Policy saved to `.sitrep/agents/policies/`.

### 6.5 KV cache precompute (`[llm]` + torch)
```bash
uv run python scripts/update_kv_cache.py            # precompute past_key_values for all passages
```
Stored as pickled BLOBs in SQLite; `KVStitcher` concatenates them along the
sequence dim for generation (handles `DynamicCache` + legacy tuple formats).

### 6.6 Versioning / backup
```bash
uv run python scripts/backup.py snapshot --label pre-experiment
uv run python scripts/backup.py list
uv run python scripts/backup.py restore --name <snapshot>
```
Snapshots are gzipped tarballs of `.sitrep/` (SQLite WAL/SHM sidecars excluded);
restore backs up the current state first.

### 6.7 Lineage inspection
Every decision (ingest, query, compress, retrieve, feedback, conflict, version,
forget, train) is recorded in SQLite **and** the Kuzu lineage graph.
```bash
uv run python scripts/analyze_lineage.py --recent 20
uv run python scripts/analyze_lineage.py --decision-id <id>
uv run python scripts/analyze_lineage.py --episode <episode_id>
```

---

## 7. Feature guide (Phases A–G)

| Phase | Feature | How to use |
|-------|---------|-----------|
| **A** | PPR + temporal-decay + entity-density retrieval | Dormant by default. `SITREP_PPR_WEIGHT=0.15 SITREP_DENSITY_WEIGHT=0.05` with `[rag]` for a Recall@10 boost. |
| **A** | Headroom content-aware compression | Auto: JSON→SmartCrusher, code→AST, logs→LogCompressor, text→Kompress. Reversible via CCR (`plugin.retrieve(key)`). |
| **B** | Importance / consolidation / forgetting | `uv run python scripts/run_forgetting.py` (dry-run by default); `--apply` to mutate (SOFT_DELETE/ARCHIVAL/FADING — never hard-deletes). |
| **B** | Entity graph for PPR | Built from shared fact entities; PPR prefers it over passage-similarity when present. |
| **C** | Atomic writes + provenance + WAL | Automatic: `UnitOfWork` rolls back SQLite on Kuzu/Chroma failure; SUPERSEDES edges on conflict; `events.wal` captures domain events. |
| **D** | Query intelligence | `app.query_orchestrator.execute(q)` classifies intent and routes (multi-hop BFS over the entity graph, comparison decomposition, …). |
| **E** | Explanation + LLM-judge + result cache | Orchestrator attaches `dto.explanation`; caches results (invalidated on ingest); `judge_uc` closes the loop. |
| **F** | Anomaly detection | `app.anomaly_uc.execute()` → z-score outliers over importance/access/novelty. |
| **F** | Allen interval algebra | `app.temporal_reasoning_uc.relate(fact_a_id, fact_b_id)` → one of 13 Allen relations. |
| **G** | Multimodal images (CLIP) | `app.multimodal_uc.ingest_image(caption, linked_passage_ids=[...])` + `retrieve_cross_modal(query)`. |
| **G** | Audio/video | `app.multimodal_uc.ingest_audio(transcript, segments=[...])` / `ingest_video(...)`. |
| **G** | Causal (do-calculus) | `app.causal_uc.add_edge(...)`; `app.causal_uc.effect(x,y)` / `counterfactual(...)`. |

### Memory hygiene detail (Phase B)
```bash
uv run python scripts/run_forgetting.py            # dry-run: reports candidates
uv run python scripts/run_forgetting.py --apply    # apply SOFT_DELETE/FADING/ARCHIVAL
uv run python scripts/run_forgetting.py --decay-only   # daily importance *= 0.95
```
Consolidation merges near-duplicates (cosine ≥ 0.85): the higher-importance
passage wins, the loser is `SOFT_DELETED`. Forgetting criteria (max_age 365d,
inactive 180d, min_importance 0.2, …) are all env-tunable.

### Multimodal detail (Phase G)
```python
app.multimodal_uc.ingest_image("a cat on a sofa", linked_passage_ids=["passage_1"])
hit = app.multimodal_uc.retrieve_cross_modal("cat sofa")
# hit["images"][0] → {caption, score, linked_passages}
```

### Causal detail (Phase G)
```python
app.causal_uc.add_edge("exercise", "fitness", weight=0.6, confidence=0.9)
app.causal_uc.add_edge("age", "exercise", weight=0.3)
app.causal_uc.add_edge("age", "fitness", weight=0.2)
app.causal_uc.effect("exercise", "fitness")
#   → {effect: 0.6, confounders: ['age'], paths: [['exercise','fitness']], ...}
app.causal_uc.counterfactual("exercise","fitness", 2.0, 5.0, 50.0)
#   → {estimated_outcome: 51.8, delta: 1.8, explanation: "..."}
```

---

## 8. Programmatic API

Everything is wired through one composition root:

```python
from src.application import build_application
app = build_application()          # uses SITREP_* env / .sitrep under cwd

# Use cases
app.ingest_uc.execute(text=...)            # IngestDocumentUseCase
app.query_uc.execute("...")                # QueryKnowledgeUseCase (single-shot)
app.query_orchestrator.execute("...")      # intent-routed (Phase D) + cache/explain (E)
app.feedback_uc.submit(query_id, "positive", 1.0)
app.train_uc.execute(total_timesteps=10000)
app.version_uc.snapshot("label"); app.version_uc.list_snapshots()
app.lineage_uc.trace(decision_id); app.lineage_uc.recent(20)
app.conflict_uc.execute()                  # corpus-wide conflict pass
app.consolidation_uc.execute(limit=200)    # near-dup merge
app.forgetting_uc.execute(dry_run=True)    # classify; --apply via dry_run=False
app.multimodal_uc.ingest_image(...); app.multimodal_uc.retrieve_cross_modal(...)
app.causal_uc.effect("x","y"); app.causal_uc.counterfactual(...)
app.anomaly_uc.execute()
app.temporal_reasoning_uc.relate(fact_a_id, fact_b_id)
app.judge_uc.judge_and_feedback(dto)

# Components
app.stats(); app.close()
app._retriever; app._llm; app._embedder; app._agent; app._env
app.kv_repo; app.ccr_repo; app.mm_repo; app.clip_embedder; app.av_embedder
app.result_cache; app.frequency_tracker; app.explanation_service; app.llm_judge
app.event_log; app.unit_of_work()          # UnitOfWork(client, graph_store, vector_store)
app.importance_scorer; app.forgetting_criteria
```

`app.unit_of_work()` returns a context manager for atomic cross-store writes:
```python
with app.unit_of_work() as uow:
    app.client.execute("INSERT INTO skills (...) VALUES (...)")
    uow.register(do=graph_op, undo=compensate)   # fails → SQLite rolls back
```

---

## 9. Scripts reference

| Script | Purpose |
|--------|---------|
| `scripts/run_web.py` | Launch the Gradio UI (Query/Ingest/Stats/Train/Lineage/Versioning) |
| `scripts/ingest_batch.py` | Batch-ingest a folder of documents |
| `scripts/query_cli.py` | Ask one question on the CLI |
| `scripts/train_compression_agent.py` | Train the PPO compression agent |
| `scripts/update_kv_cache.py` | Precompute transformer KV caches (`[llm]`) |
| `scripts/backup.py` | snapshot / list / restore / delete `.sitrep/` |
| `scripts/analyze_lineage.py` | Inspect decision traces / recent / by-episode |
| `scripts/run_forgetting.py` | Memory-hygiene forgetting pass (Phase B) |
| `scripts/eval.py` | Run the eval harness over a labeled JSONL set |

All scripts run from the project root: `uv run python scripts/<name>.py [args]`.

---

## 10. Claude Code integration

`plugin.py` exposes SITREP as an importable API (and `claude_plugin.json` describes
the plugin manifest):

```python
import plugin
plugin.ingest("Acme Corp is located in Berlin.")
plugin.query("Where is Acme?")          # → presenter-formatted dict
plugin.stats()                          # aggregate counts + fusion weights
plugin.train(); plugin.snapshot("v1")
plugin.retrieve(ccr_key)                # Headroom reversible-compression retrieve tool
```

CLI: `python plugin.py stats` (actions: `query`, `ingest`, `train`, `stats`).

---

## 11. Evaluation harness

Real **BEIR/SciFact** sample (5,183-doc corpus, 300 test queries) lives in
`sitrep-engine/eval/`. The harness computes Precision/Recall/MRR/NDCG @K:

```bash
uv run python scripts/eval.py --label myrun --top-k 10
uv run python scripts/eval.py --no-rerank --limit 100     # isolate fusion / speed
```

Results are saved to `eval/results/<label>_<timestamp>.json`; see
`eval/results/SUMMARY.md` for the recorded baseline and Phase-A (hash & real
embedding) comparisons. **Baseline (hash, dormant): nDCG@10 = 0.6414.**

To measure Phase A with real embeddings:
```bash
SITREP_PPR_WEIGHT=0.15 uv run --with sentence-transformers --with "numpy<2" \
  python scripts/eval.py --corpus eval/scifact_corpus_small.jsonl --no-rerank --limit 100
```

---

## 12. Testing

```bash
uv sync --extra dev
uv run pytest                 # full suite (demo mode, no models needed)
uv run pytest tests/test_application/test_phase_d.py   # one phase
```

Tests cover every phase (domain entities, retrieval/fusion, RL env, KV cache,
use-case flows, conflict/provenance, query intelligence, multimodal, causal,
anomaly, Allen algebra) plus the original foundation. All run on core deps only.

---

## 13. Troubleshooting & caveats

- **"Numpy is not available"** when using sentence-transformers → pin `numpy<2`
  (`uv run --with "numpy<2"`).
- **`onnxruntime` install fails on Python 3.10** → the full `[rag]` extra pulls
  chromadb→onnxruntime which lacks cp310 wheels. Either use Python 3.12, or install
  only `sentence-transformers` (`uv run --with sentence-transformers`).
- **Full real-embedding eval is slow** → the per-query vector scan + real
  cross-encoder reranker dominate. Use `--limit`, `--no-rerank`, and/or the small
  corpus, or install Chroma for ANN.
- **Phase A shows no gain with hash embeddings** → expected; PPR needs real
  embeddings + a fact graph to help. Enable `[rag]` and `ppr_weight≈0.15`.
- **No data appears after ingest** → check `SITREP_BASE_DIR` (defaults to
  `./.sitrep` under cwd); each working directory has its own store.
- **Forgetting won't delete my data** → by design. It uses SOFT_DELETE/ARCHIVAL;
  `PERMANENTLY_DELETED` is never chosen by the default strategy mapping.
- **Extraction misses numeric facts** → fixed (prices/quantities/years now
  extract); if a sentence has no recognized verb (is/are/was/were/has/contains/
  located in/…), no fact is extracted — rephrase or extend the patterns.

---

*SITREP is local-first and embeddable: SQLite + FTS5, KuzuDB, ChromaDB, local
LLMs, and stable-baselines3 PPO — no Redis, Postgres, or cloud calls.*
