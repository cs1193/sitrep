# SITREP Architecture Diagrams (ASCII)

---

## Diagram 1: Overall System Architecture (5-Layer Clean Architecture)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          PRESENTATION LAYER (179 LOC)                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────────────┐          ┌────────────────────────────┐          │
│  │   Gradio Web UI      │          │   Claude Code Plugin       │          │
│  │                      │          │   (plugin.py)              │          │
│  │  • Query tab         │◄────────►│  • query()                 │          │
│  │  • Ingest tab        │          │  • ingest()                │          │
│  │  • Train tab         │          │  • train()                 │          │
│  │  • Stats tab         │          │  • stats()                 │          │
│  │  • Lineage tab       │          │  • feedback()              │          │
│  │  • Versioning tab    │          │  • analyze_lineage()       │          │
│  └──────────────────────┘          └────────────────────────────┘          │
│                                                                             │
│        ┌──────────────────────────────────────────────┐                    │
│        │       CLI Entry Points (scripts/)            │                    │
│        │  • run_web.py        • query_cli.py          │                    │
│        │  • ingest_batch.py   • train_compression...  │                    │
│        │  • eval.py           • analyze_lineage.py    │                    │
│        └──────────────────────────────────────────────┘                    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ▲
                                    │
┌─────────────────────────────────────────────────────────────────────────────┐
│                       APPLICATION LAYER (2,881 LOC)                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │   Query      │  │   Ingest     │  │   Train      │  │   Feedback   │  │
│  │   Use Case   │  │   Use Case   │  │   Use Case   │  │   Use Case   │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘  │
│                                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                     │
│  │  Versioning  │  │   Lineage    │  │    Event     │                     │
│  │   Use Case   │  │   Use Case   │  │   System     │                     │
│  └──────────────┘  └──────────────┘  └──────────────┘                     │
│                                                                             │
│               Composition Root: build_application()                        │
│                      (457 LOC, wires all layers)                           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ▲
                                    │
┌─────────────────────────────────────────────────────────────────────────────┐
│                         ADAPTERS LAYER (2,246 LOC)                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  REPOSITORIES (600 LOC)         │    SERVICES (800 LOC)                    │
│  ┌──────────────┐               │    ┌──────────────┐                     │
│  │ FactRepo     │               │    │ Extraction   │                     │
│  ├──────────────┤               │    │ Service      │                     │
│  │ PassageRepo  │               │    ├──────────────┤                     │
│  ├──────────────┤               │    │ Compression  │                     │
│  │ EpisodeRepo  │               │    │ Service      │                     │
│  ├──────────────┤               │    ├──────────────┤                     │
│  │ DecisionRepo │               │    │ Classification                     │
│  ├──────────────┤               │    │ Service      │                     │
│  │ FeedbackRepo │               │    ├──────────────┤                     │
│  ├──────────────┤               │    │ Judgment     │                     │
│  │ EntityRepo   │               │    │ Service      │                     │
│  ├──────────────┤               │    ├──────────────┤                     │
│  │ GraphRepo    │               │    │ Reranker     │                     │
│  │              │               │    │ Service      │                     │
│  └──────────────┘               │    ├──────────────┤                     │
│                                 │    │ Versioning   │                     │
│  (All inherit BaseRepository)    │    │ Service      │                     │
│                                 │    ├──────────────┤                     │
│                                 │    │ Fusion Wt    │                     │
│                                 │    │ Updater      │                     │
│                                 │    └──────────────┘                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ▲
                                    │
┌─────────────────────────────────────────────────────────────────────────────┐
│                        DOMAIN LAYER (1,275 LOC)                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ENTITIES                          │    VALUE OBJECTS                      │
│  ┌──────────────┐                  │    ┌──────────────┐                   │
│  │ Fact         │                  │    │ TimeRange    │                   │
│  │ ├ text       │                  │    │ Entity       │                   │
│  │ ├ importance │                  │    │ Relation     │                   │
│  │ ├ timestamp  │                  │    │ CausalRel   │                   │
│  │ └ metadata   │                  │    └──────────────┘                   │
│  ├──────────────┤                  │                                       │
│  │ Passage      │                  │    ENUMS                              │
│  │ ├ content    │                  │    ┌──────────────┐                   │
│  │ ├ embedding  │                  │    │ DecisionType │                   │
│  │ ├ compressed │                  │    │ QueryIntent  │                   │
│  │ └ ccr_key    │                  │    │ AllenRelation│                   │
│  ├──────────────┤                  │    │ (13 temporal)│                   │
│  │ Episode      │                  │    └──────────────┘                   │
│  │ Decision     │                  │                                       │
│  │ Agent        │                  │                                       │
│  │ Schema       │                  │                                       │
│  └──────────────┘                  │                                       │
│                                                                             │
│             Pure Data Models (Pydantic BaseModel)                          │
│        No Business Logic (Business Logic in Services)                      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ▲
                                    │
┌─────────────────────────────────────────────────────────────────────────────┐
│                     INFRASTRUCTURE LAYER (4,452 LOC)                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  DATABASE CLIENTS (694)      RETRIEVAL (814)       RL (508)                │
│  ┌────────────────┐         ┌─────────────────┐  ┌──────────────┐        │
│  │ SQLiteClient   │         │ HybridRetriever │  │ PPOAgent     │        │
│  ├────────────────┤         ├─────────────────┤  ├──────────────┤        │
│  │ KuzuDBClient   │         │ DenseSearch     │  │ ComprEnv     │        │
│  ├────────────────┤         ├─────────────────┤  ├──────────────┤        │
│  │ ChromaDBClient │         │ SparseSearch    │  │ RewardModel  │        │
│  ├────────────────┤         ├─────────────────┤  └──────────────┘        │
│  │ DuckDBClient   │         │ FusionWeights   │                           │
│  ├────────────────┤         ├─────────────────┤  COMPRESSION (542)        │
│  │ KVCacheClient  │         │ Reranker        │  ┌──────────────┐        │
│  └────────────────┘         ├─────────────────┤  │ SmartCrusher │        │
│                             │ EntityGraphRank │  ├──────────────┤        │
│  LLM GATEWAYS (266)         ├─────────────────┤  │ JSONComp     │        │
│  ┌────────────────┐         │ TemporalFilter  │  ├──────────────┤        │
│  │ OllamaClient   │         └─────────────────┘  │ CodeComp     │        │
│  ├────────────────┤                             ├──────────────┤        │
│  │ TransformersLLM│                             │ LogComp      │        │
│  ├────────────────┤                             ├──────────────┤        │
│  │ DemoLLM        │                             │ TextComp     │        │
│  ├────────────────┤                             │ (Kompress)   │        │
│  │ EmbeddingService                             └──────────────┘        │
│  └────────────────┘                                                      │
│                                                                             │
│  UTILITIES (694 LOC)                                                       │
│  ┌──────────────────────────────────────────────────────────────┐         │
│  │ Config Management  │  Logging  │  Decorators  │  Constants  │         │
│  └──────────────────────────────────────────────────────────────┘         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Diagram 2: Data Model Entity Relationship Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      SITREP DATA MODEL (SQL TABLES)                         │
└─────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────┐
│         PASSAGES TABLE           │
├──────────────────────────────────┤
│ id (PK)                          │
│ content                          │
│ source                           │
│ timestamp                        │
│ token_estimate                   │
│ compressed_form                  │
│ ccr_key (FK → ccr_store)         │───────┐
│ archived (BOOLEAN)               │       │
└──────────────────────────────────┘       │
         │                                 │
         │ 1                              │
         │                                │
         │ * (contains)                   │
         │                                │
         ▼                                │
┌──────────────────────────────────┐      │
│         FACTS TABLE              │      │
├──────────────────────────────────┤      │
│ id (PK)                          │      │
│ text                             │      │
│ source_passage_id (FK)           │◄─────┘
│ importance (0.0-1.0)             │
│ timestamp                        │
│ causal_parent_ids (JSON)         │───────────┐
│ metadata (JSON)                  │           │
│ version                          │           │
│ archived (BOOLEAN)               │           │
└──────────────────────────────────┘           │
         │                                     │
         │ 1                                   │
         │                                     │
         │ * (causes)                          │
         │                                     │
         │◄────────────────────────────────────┘
         │
         │ 1
         │
         │ * (contains)
         │
         ▼
┌──────────────────────────────────┐
│      EPISODES TABLE              │
├──────────────────────────────────┤
│ id (PK)                          │
│ query                            │
│ facts (JSON list of IDs)         │
│ passages (JSON list of IDs)      │
│ timestamp                        │
│ outcome                          │
└──────────────────────────────────┘
         │
         │ 1
         │
         │ * (records)
         │
         ▼
┌──────────────────────────────────┐
│      DECISIONS TABLE             │
├──────────────────────────────────┤
│ id (PK)                          │
│ type (ENUM: INGEST, UPDATE, ..) │
│ input_ids (JSON)                 │
│ output_ids (JSON)                │
│ causal_parents (JSON)            │◄─────────┐
│ timestamp                        │          │
│ reversible (BOOLEAN)             │          │
│ audit_log                        │          │
└──────────────────────────────────┘          │
         │                                    │
         │ (lineage: KuzuDB DAG)              │
         │                                    │
         └────────────────────────────────────┘
              (causal links)

┌──────────────────────────────────┐
│      FEEDBACK TABLE              │
├──────────────────────────────────┤
│ id (PK)                          │
│ query                            │
│ fact_id (FK)                     │
│ relevance (0.0-1.0)              │
│ quality (0.0-1.0)                │
│ timestamp                        │
└──────────────────────────────────┘

┌──────────────────────────────────┐
│   FUSION_WEIGHTS TABLE           │
├──────────────────────────────────┤
│ id (PK)                          │
│ w_dense (REAL)                   │
│ w_sparse (REAL)                  │
│ w_entity_rank (REAL)             │
│ updated_at (TIMESTAMP)           │
└──────────────────────────────────┘

┌──────────────────────────────────┐
│     KV_CACHE TABLE               │
├──────────────────────────────────┤
│ id (PK)                          │
│ fact_id (FK)                     │
│ model_id                         │
│ cache (BLOB)                     │
│ expires_at (TIMESTAMP)           │
└──────────────────────────────────┘

┌──────────────────────────────────┐
│    SCHEMAS TABLE                 │
├──────────────────────────────────┤
│ id (PK)                          │
│ content                          │
│ type                             │
│ version                          │
│ timestamp                        │
└──────────────────────────────────┘
```

---

## Diagram 3: Hybrid Retrieval Architecture

```
USER QUERY: "What is SITREP's compression mechanism?"
│
├────────────────────────────────────────────────────────────┐
│                                                            │
▼                                                            │
┌──────────────────────────────────────────────────────────┐ │
│         QUERY INTENT CLASSIFIER                         │ │
├──────────────────────────────────────────────────────────┤ │
│ Input: "What is SITREP's compression mechanism?"       │ │
│                                                          │ │
│ Output: Intent = "SIMPLE" (direct lookup question)     │ │
│         Entities: ["SITREP", "compression"]            │ │
└──────────────────────────────────────────────────────────┘ │
         │                                                   │
         ▼                                                   │
┌──────────────────────────────────────────────────────────┐ │
│              HYBRID SEARCH ENGINE                       │ │
├──────────────────────────────────────────────────────────┤ │
│                                                          │ │
│  ┌─────────────────────────────────────────────────┐   │ │
│  │ 1. DENSE SEARCH (ChromaDB + Sentence-Transformers)  │ │
│  ├─────────────────────────────────────────────────┤   │ │
│  │ Query embedding: [0.34, -0.12, 0.87, ...]     │   │ │
│  │ (384-dimensional vector)                       │   │ │
│  │                                                 │   │ │
│  │ FAISS k-NN search → top-10 by cosine sim:      │   │ │
│  │  • Passage A: 0.92 (compression strategies)    │   │ │
│  │  • Passage B: 0.87 (RL agent details)          │   │ │
│  │  • Passage C: 0.81 (token reduction)           │   │ │
│  │  • Passage D: 0.76 (SmartCrusher)              │   │ │
│  │  • ...                                          │   │ │
│  └─────────────────────────────────────────────────┘   │ │
│                                                          │ │
│  ┌─────────────────────────────────────────────────┐   │ │
│  │ 2. SPARSE SEARCH (BM25 + SQLite FTS5)          │   │ │
│  ├─────────────────────────────────────────────────┤   │ │
│  │ Query terms: ["compression", "mechanism"]      │   │ │
│  │                                                 │   │ │
│  │ BM25 ranking → top-10 by text relevance:      │   │ │
│  │  • Passage A: 0.78 (contains both terms)       │   │ │
│  │  • Passage D: 0.85 (higher term frequency)     │   │ │
│  │  • Passage E: 0.72 (one term)                  │   │ │
│  │  • ...                                          │   │ │
│  └─────────────────────────────────────────────────┘   │ │
│                                                          │ │
│  ┌─────────────────────────────────────────────────┐   │ │
│  │ 3. FUSION (Learned Weights)                    │   │ │
│  ├─────────────────────────────────────────────────┤   │ │
│  │ w_dense = 0.62  (learned from feedback)        │   │ │
│  │ w_sparse = 0.38  (learned from feedback)       │   │ │
│  │                                                 │   │ │
│  │ Fused score = 0.62 * dense_score               │   │ │
│  │            + 0.38 * sparse_score               │   │ │
│  │                                                 │   │ │
│  │ Passage A: 0.62*0.92 + 0.38*0.78 = 0.86       │   │ │
│  │ Passage B: 0.62*0.87 + 0.38*0.45 = 0.71       │   │ │
│  │ Passage D: 0.62*0.76 + 0.38*0.85 = 0.80       │   │ │
│  │ Passage C: 0.62*0.81 + 0.38*0.42 = 0.66       │   │ │
│  │ ...                                             │   │ │
│  │                                                 │   │ │
│  │ → Ranked: [Passage A, Passage D, Passage B]   │   │ │
│  └─────────────────────────────────────────────────┘   │ │
│                                                          │ │
│  ┌─────────────────────────────────────────────────┐   │ │
│  │ 4. ENTITY GRAPH RANKING (Optional, PPR)        │   │ │
│  ├─────────────────────────────────────────────────┤   │ │
│  │ If entity graph enabled:                       │   │ │
│  │  • Node "SITREP" → neighbors = {"compression",│   │ │
│  │                    "retrieval", "RL", ...}    │   │ │
│  │  • Personalized PageRank seeded at "SITREP"  │   │ │
│  │  • Boost passages mentioning neighbor nodes   │   │ │
│  │                                                 │   │ │
│  │ (Dormant by default, activate via [graph])    │   │ │
│  └─────────────────────────────────────────────────┘   │ │
│                                                          │ │
└──────────────────────────────────────────────────────────┘ │
         │                                                   │
         ▼                                                   │
┌──────────────────────────────────────────────────────────┐ │
│           RERANKER (Importance + Recency)              │ │
├──────────────────────────────────────────────────────────┤ │
│                                                          │ │
│ For each passage:                                       │ │
│  score' = fusion_score                                  │ │
│         + importance_weight * fact.importance          │ │
│         + recency_weight * time_decay_factor           │ │
│         + quality_weight * fact.quality_score          │ │
│                                                          │ │
│ Reranked top-5:                                         │ │
│  1. Passage A (compression strategies)                  │ │
│  2. Passage D (SmartCrusher details)                    │ │
│  3. Passage B (RL agent training loop)                  │ │
│  4. Passage E (PPO compression agent)                   │ │
│  5. Passage C (token reduction examples)                │ │
│                                                          │ │
└──────────────────────────────────────────────────────────┘ │
         │                                                   │
         ▼                                                   │
┌──────────────────────────────────────────────────────────┐ │
│        COMPRESSION (Optional, RL Policy)               │ │
├──────────────────────────────────────────────────────────┤ │
│                                                          │ │
│ For each passage:                                       │ │
│  state = (text, embedding, importance, type)          │ │
│  compression_ratio ~ π(state)  [PPO policy sample]    │ │
│  compressed_text = compress(text, ratio)              │ │
│                                                          │ │
│ Passage A: 245 tokens → 68 tokens (72% reduction)     │ │
│ Passage D: 189 tokens → 52 tokens (73% reduction)     │ │
│ Passage B: 312 tokens → 95 tokens (70% reduction)     │ │
│ ...                                                     │ │
│                                                          │ │
└──────────────────────────────────────────────────────────┘ │
         │                                                   │
         ▼                                                   │
┌──────────────────────────────────────────────────────────┐ │
│    TEMPORAL FILTERING (Optional, Allen Algebra)        │ │
├──────────────────────────────────────────────────────────┤ │
│                                                          │ │
│ If query has temporal intent:                          │ │
│  • Query time range: [2025-07-01, 2025-07-31]         │ │
│  • Filter facts: fact_time_range OVERLAPS query_time  │ │
│  • Allen relations: BEFORE, AFTER, DURING, ...        │ │
│                                                          │ │
│ (Dormant by default, activate in query UI)            │ │
│                                                          │ │
└──────────────────────────────────────────────────────────┘ │
         │                                                   │
         ▼                                                   │
┌──────────────────────────────────────────────────────────┐ │
│    LLM EXPLANATION (Optional, Ollama/Transformers)    │ │
├──────────────────────────────────────────────────────────┤ │
│                                                          │ │
│ Input: Query + top-5 compressed facts                  │ │
│ Prompt: "Explain SITREP's compression mechanism based  │ │
│          on these facts: [facts]"                       │ │
│                                                          │ │
│ Output: "SITREP uses RL-optimized compression with...  │ │
│          The PPO agent learns optimal compression...   │ │
│          SmartCrusher handles content-aware strategies"│ │
│                                                          │ │
└──────────────────────────────────────────────────────────┘ │
         │                                                   │
         ▼                                                   │
└─────────────────────────────────────────────────────────────┘
         RETURN QueryResult(facts, explanation)
         │
         └──→ ResultCache.store(query, result)
              (for later feedback + weight update)
```

---

## Diagram 4: RL Training Loop (PPO Compression Agent)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│              PPO COMPRESSION AGENT TRAINING LOOP                             │
└─────────────────────────────────────────────────────────────────────────────┘

INITIALIZATION
│
├─ Policy Network π (Actor)
│  └─ MLP: state → action distribution (compression_ratio ∈ [0.1, 0.9])
│
├─ Value Network V (Critic)
│  └─ MLP: state → scalar value estimate V(s)
│
├─ Experience Buffer (rollout memory)
│  └─ Stores: (state, action, reward, next_state, done)
│
└─ Reward Model (downstream task accuracy)
   └─ Measures: "Does compressed fact still preserve meaning for QA/NER/Classification?"

┌─────────────────────────────────────────────────────────────────────────────┐
│                      FOR EPISODE t = 1 .. N_EPISODES:                       │
└─────────────────────────────────────────────────────────────────────────────┘

    ┌──────────────────────────────────────────────────────────────┐
    │ STEP 1: ROLLOUT (Sample trajectory from current policy)     │
    ├──────────────────────────────────────────────────────────────┤
    │                                                              │
    │  Sample batch of M passages from SQLite:                    │
    │    passages = [p1, p2, p3, ..., p_M]                       │
    │                                                              │
    │  FOR each passage p in passages:                            │
    │    state_t = encode(p)                                      │
    │             = {p.text, p.embedding, p.importance, p.type}  │
    │                                                              │
    │    action_t ~ π(· | state_t)   [sample compression ratio]  │
    │    (ε-greedy: argmax with prob (1-ε), random with prob ε)  │
    │                                                              │
    │    compressed_p = compress(p, action_t)                    │
    │    [Apply SmartCrusher or RL-selected strategy]            │
    │                                                              │
    │    reward_t = -len(compressed_p.tokens)                    │
    │              + α * downstream_accuracy_delta(p, compressed_p)
    │              + β * preservation_penalty                    │
    │    [Negative token delta: reward for reducing tokens]      │
    │    [Accuracy delta: penalty if compression hurts accuracy] │
    │                                                              │
    │    Append (state_t, action_t, reward_t) → experience buffer│
    │                                                              │
    │  ┌──────────────────────────────────────────────────────┐  │
    │  │ Example: Passage about "RL compression agent"        │  │
    │  │ ─────────────────────────────────────────────────────│  │
    │  │ Original tokens: 312                                 │  │
    │  │ Compressed tokens: 95  (ratio = 0.30)               │  │
    │  │                                                      │  │
    │  │ reward = -95            (token penalty)             │  │
    │  │        + 0.8 * 0.05     (QA acc: 98% → 95%, Δ=-3%) │  │
    │  │        + 0 * 0.95       (semantic preservation OK)  │  │
    │  │ ─────────────────────────────────────────────────────│  │
    │  │ total_reward ≈ -95.06                               │  │
    │  │                                                      │  │
    │  │ (Negative, but RL will learn if reward improves     │  │
    │  │  over iterations)                                   │  │
    │  └──────────────────────────────────────────────────────┘  │
    │                                                              │
    └──────────────────────────────────────────────────────────────┘

    ┌──────────────────────────────────────────────────────────────┐
    │ STEP 2: COMPUTE ADVANTAGES (Generalized Advantage Estimate) │
    ├──────────────────────────────────────────────────────────────┤
    │                                                              │
    │  FOR each trajectory in buffer:                             │
    │    V_t = V(state_t)     [Critic: expected cumulative reward]
    │    V_{t+1} = V(state_{t+1})                                 │
    │    TD_error = reward_t + γ * V_{t+1} - V_t                  │
    │               (Temporal Difference error, γ=0.99)           │
    │                                                              │
    │    advantage_t = TD_error + λ * advantage_{t+1}             │
    │                 (GAE: Generalized Advantage Est., λ=0.95)   │
    │                                                              │
    │    return_t = advantage_t + V_t                             │
    │              (Bootstrap value estimate)                     │
    │                                                              │
    └──────────────────────────────────────────────────────────────┘

    ┌──────────────────────────────────────────────────────────────┐
    │ STEP 3: POLICY GRADIENT UPDATE (PPO Loss)                   │
    ├──────────────────────────────────────────────────────────────┤
    │                                                              │
    │  Actor loss (Policy gradient):                              │
    │    L_actor = - mean( log π(action | state) * advantage )    │
    │            [Maximize log-prob of good actions]              │
    │                                                              │
    │  Critic loss (Value prediction):                            │
    │    L_critic = MSE(V(state), return)                         │
    │            [Minimize value prediction error]                │
    │                                                              │
    │  PPO Clipping (prevents large updates):                     │
    │    r_t = π_new(a|s) / π_old(a|s)  [probability ratio]      │
    │    L_clip = - mean( min(r_t * A_t,                         │
    │                         clip(r_t, 1-ε, 1+ε) * A_t) )       │
    │           [ε=0.2 typically]                                 │
    │                                                              │
    │  Total loss:                                                │
    │    L_total = L_actor + 0.5 * L_critic                       │
    │                                                              │
    │  Backprop & gradient descent (Adam optimizer):              │
    │    θ ← θ - learning_rate * ∇ L_total                        │
    │                                                              │
    └──────────────────────────────────────────────────────────────┘

    ┌──────────────────────────────────────────────────────────────┐
    │ STEP 4: CHECKPOINT & LOGGING                                │
    ├──────────────────────────────────────────────────────────────┤
    │                                                              │
    │  Every K episodes:                                          │
    │    Save policy weights: .sitrep/agents/ppo_<timestamp>.pt   │
    │    Save latest symlink: .sitrep/agents/ppo_latest.pt        │
    │                                                              │
    │  Log metrics:                                               │
    │    episode, mean_reward, mean_loss, entropy, KL_divergence  │
    │                                                              │
    │  Plot training curves:                                      │
    │    reward_history, loss_history, value_error                │
    │                                                              │
    └──────────────────────────────────────────────────────────────┘

    [Clear experience buffer, continue to next episode]

┌─────────────────────────────────────────────────────────────────────────────┐
│                    DEPLOYMENT (Inference Time)                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Load checkpoint: policy_weights ← load(.sitrep/agents/ppo_latest.pt)     │
│                                                                             │
│  For each query result (top-K passages):                                   │
│    state = encode(passage)                                                 │
│    compression_ratio = argmax_a π(a | state)  [deterministic at inference] │
│    compressed = compress(passage, compression_ratio)                       │
│    return compressed passage                                               │
│                                                                             │
│  Metrics:                                                                  │
│    avg_compression_ratio = 70% (70% tokens removed)                        │
│    downstream_task_accuracy: maintained or improved                        │
│    inference_latency: ~5-10ms per passage                                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Diagram 5: Ingest Pipeline (End-to-End)

```
SOURCE DOCUMENT (PDF, Markdown, CSV, Web)
│
├──────────────────────────────────────────────────────────────┐
│                                                              │
▼                                                              │
┌──────────────────────────────────────────────────────────────┐ │
│              PARSER (Document Type Detection)              │ │
├──────────────────────────────────────────────────────────────┤ │
│                                                              │ │
│  Auto-detect file type OR explicit type parameter:          │ │
│    • PDF → pdfplumber extract text + metadata               │ │
│    • Markdown → parse frontmatter + content                 │ │
│    • CSV → read rows, convert to passages                   │ │
│    • HTML/Web → BeautifulSoup scrape                        │ │
│    • TXT → plain text split by paragraphs                   │ │
│                                                              │ │
│  Output: [RawPassage{text, source, metadata}, ...]         │ │
│                                                              │ │
└──────────────────────────────────────────────────────────────┘ │
         │                                                       │
         ▼                                                       │
┌──────────────────────────────────────────────────────────────┐ │
│             CHUNKER (Split Large Documents)                │ │
├──────────────────────────────────────────────────────────────┤ │
│                                                              │ │
│  For each passage:                                           │ │
│    if len(passage) > max_tokens (e.g., 512):               │ │
│      split by sentences, paragraphs, or sliding window      │ │
│    else:                                                     │ │
│      keep as-is                                             │ │
│                                                              │ │
│  Output: [Chunk{id, content, source, token_count}, ...]    │ │
│                                                              │ │
└──────────────────────────────────────────────────────────────┘ │
         │                                                       │
         ▼                                                       │
┌──────────────────────────────────────────────────────────────┐ │
│          EMBEDDING SERVICE (Dense Vectors)                 │ │
├──────────────────────────────────────────────────────────────┤ │
│                                                              │ │
│  Batch encode chunks:                                        │ │
│    embeddings = model.encode([c.content for c in chunks])   │ │
│    (sentence-transformers/all-MiniLM-L6-v2)                 │ │
│                                                              │ │
│  Output: {chunk_id → embedding_vector (384-dim)}            │ │
│                                                              │ │
│  (If [rag] extra not installed, fallback to hash-based)    │ │
│                                                              │ │
└──────────────────────────────────────────────────────────────┘ │
         │                                                       │
         ├─────────────────────────┬──────────────────┬─────────┘
         │                         │                  │
         ▼                         ▼                  ▼
    ┌────────────┐           ┌──────────┐       ┌────────────┐
    │ EXTRACTION │           │ENTITY    │       │CLASSIFICATION
    │SERVICE     │           │EXTRACTION│       │SERVICE
    │(LLM-based) │           │          │       │
    ├────────────┤           ├──────────┤       ├────────────┤
    │Extract     │           │Extract   │       │Detect:
    │atomic      │           │entities, │       │• code
    │claims from │           │relations,│       │• docs
    │each chunk  │           │temporal  │       │• log
    │            │           │relations │       │• narrative
    │Output:     │           │          │       │• other
    │Facts[]     │           │Output:   │       │
    │(with       │           │Edges[]   │       │Output:
    │importance) │           │(for KG)  │       │type enum
    │            │           │          │       │
    └────────────┘           └──────────┘       └────────────┘
         │                         │                  │
         └─────────────────────────┴──────────────────┘
                                   │
                                   ▼
                    ┌───────────────────────────────┐
                    │   ATOMIC WRITE TRANSACTION    │
                    ├───────────────────────────────┤
                    │ async with txn.begin():       │
                    │                               │
                    │ 1. SQLite (Transactional)    │
                    │    ├─ INSERT passages        │
                    │    │  (FTS5 indexed)         │
                    │    └─ INSERT facts           │
                    │                               │
                    │ 2. ChromaDB (Logged)         │
                    │    └─ ADD embeddings + index │
                    │                               │
                    │ 3. KuzuDB (Logged)           │
                    │    ├─ ADD entity nodes       │
                    │    └─ ADD relationship edges │
                    │                               │
                    │ 4. DuckDB (Logged)           │
                    │    └─ APPEND to Parquet      │
                    │       (if [duckdb] enabled)  │
                    │                               │
                    │ 5. Decision Log (SQLite)     │
                    │    └─ CREATE Decision record │
                    │       (type=INGEST, inputs, │
                    │        outputs)              │
                    │                               │
                    │ If error: txn.rollback()     │
                    │           (SQLite atomic)    │
                    │           delete from KuzuDB │
                    │           delete from Chroma │
                    │                               │
                    └───────────────────────────────┘
                                   │
                                   ▼
                      ┌─────────────────────────┐
                      │  .sitrep/ Directory    │
                      ├─────────────────────────┤
                      │ metadata/sitrep.db      │
                      │  ├─ passages table       │
                      │  ├─ facts table          │
                      │  ├─ decisions table      │
                      │  └─ FTS5 indexes         │
                      │                          │
                      │ graph/kuzu.db/           │
                      │  ├─ EntityNodes          │
                      │  └─ Relations            │
                      │                          │
                      │ vectors/                 │
                      │  └─ embeddings.db        │
                      │                          │
                      │ documents/               │
                      │  └─ archives/            │
                      │     (Parquet files)      │
                      │                          │
                      └─────────────────────────┘
                                   │
                                   ▼
                    ┌───────────────────────────────┐
                    │  RETURN IngestResult         │
                    ├───────────────────────────────┤
                    │ facts_added: 47               │
                    │ passages_added: 12            │
                    │ entities_added: 156           │
                    │ relations_added: 283          │
                    │ total_tokens_stored: 18432    │
                    │ timestamp: 2025-07-09T...     │
                    └───────────────────────────────┘
```

---

## Diagram 6: Dependency Graph (Simplified)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         DEPENDENCY HIERARCHY                                │
│                      (Arrow → "depends on")                                 │
└─────────────────────────────────────────────────────────────────────────────┘

                        ┌─────────────────┐
                        │  Application    │
                        │   (Use Cases)   │
                        └────────┬────────┘
                                 │
                    ┌────────────┼────────────┬─────────────┐
                    │            │            │             │
                    ▼            ▼            ▼             ▼
            ┌───────────────┐ ┌────────┐ ┌───────┐  ┌─────────────┐
            │QueryUseCase   │ │Ingest  │ │Train  │  │Feedback &   │
            │               │ │UseCase │ │Case   │  │Versioning   │
            └───┬───────────┘ └───┬────┘ └───┬───┘  └──────┬──────┘
                │                 │          │             │
    ┌───────────┼─────────────────┼──────────┼─────────────┼─────────┐
    │           │                 │          │             │         │
    ▼           ▼                 ▼          ▼             ▼         ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         ADAPTER LAYER                                   │
│  Repositories (FactRepo, PassageRepo, EpisodeRepo, DecisionRepo, ...)  │
│  Services (ExtractionService, CompressionService, RerankerService, ... │
│  Managers (FusionWeightUpdater, LineageTracker, VersioningService, ...) │
└─────────┬──────────────────────────────────┬──────────────────────────┘
          │                                  │
    ┌─────┴──────┬──────────────┬────────────┴────┬──────────┐
    │            │              │                 │          │
    ▼            ▼              ▼                 ▼          ▼
┌──────────┐ ┌──────────┐ ┌────────────┐ ┌─────────────┐ ┌────────┐
│Retrieval │ │Compression│ │LLM Gateways│ │RL Agent     │ │Embedding
│Engine    │ │Service    │ │(Ollama,   │ │(PPO)        │ │Service
│(Hybrid   │ │(SmartCrush│ │Transform..│ │             │ │
│Search)   │ │, heuristic)│ │)          │ │             │ │
└──────┬───┘ └─────┬─────┘ └─────┬──────┘ └──────┬──────┘ └───┬────┘
       │           │             │              │            │
       └───────────┼─────────────┴──────────────┴────────────┘
                   │
        ┌──────────┴────────────┐
        │                       │
        ▼                       ▼
    ┌──────────────────────────────────────────────────┐
    │         INFRASTRUCTURE LAYER                      │
    │                                                  │
    │  Database Clients:                               │
    │  ├─ SQLiteClient (Transactional metadata)       │
    │  ├─ KuzuDBClient (Knowledge graph)              │
    │  ├─ ChromaDBClient (Vector embeddings)          │
    │  ├─ DuckDBClient (Analytics, archives)          │
    │  └─ KVCacheClient (Transformer KV cache)        │
    │                                                  │
    │  External Dependencies (Lazy-imported):          │
    │  ├─ sentence-transformers ([rag] extra)         │
    │  ├─ chromadb ([rag] extra)                      │
    │  ├─ torch, transformers ([llm] extra)           │
    │  ├─ ollama-python (Ollama client, [llm] extra)  │
    │  ├─ kuzudb ([graph] extra)                      │
    │  ├─ duckdb ([duckdb] extra)                     │
    │  └─ gradio ([web] extra)                        │
    │                                                  │
    │  System Dependencies:                            │
    │  ├─ sqlite3 (built-in)                          │
    │  ├─ numpy, rank-bm25 (core)                     │
    │  ├─ pydantic (core validation)                  │
    │  └─ pyyaml (core config)                        │
    │                                                  │
    └──────────────────────────────────────────────────┘

OPTIONAL FEATURE ACTIVATION
(Lazy imports behind [extra] flags)

    Core (always available)
    │
    ├─ [rag]   → Dense embedding + ChromaDB + Reranking
    ├─ [graph] → KuzuDB + Entity graph + Personalized PageRank
    ├─ [llm]   → LLM generation (Ollama, Transformers, KV cache)
    ├─ [rl]    → PPO compression agent training + deployment
    ├─ [web]   → Gradio web UI
    ├─ [duckdb]→ Parquet analytics + archival
    └─ [all]   → All of the above

    Example: uv sync --extra rag --extra graph --extra web
             → Enables dense retrieval, graph ranking, and UI
```

---

## Diagram 7: State Transitions (Fact Lifecycle)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    FACT LIFECYCLE & STATE TRANSITIONS                       │
└─────────────────────────────────────────────────────────────────────────────┘

START: Fact created during ingestion
│
▼
┌─────────────────────────────────────────────────────────────────┐
│ STATE: ACTIVE                                                   │
├─────────────────────────────────────────────────────────────────┤
│ • Stored in SQLite facts table                                  │
│ • Indexed in ChromaDB embeddings                                │
│ • Included in KuzuDB graph                                      │
│ • Importance score: [0.0, 1.0] (initialized at 0.5)            │
│ • Access frequency: 0 (incremented on retrieval)                │
│ • Last accessed: NULL                                           │
└─────────────────────────────────────────────────────────────────┘
│
├─────────────────────────────────────────┬────────────────┬──────────────┐
│                                         │                │              │
▼ (user feedback)                         ▼ (time passes)  ▼ (merge)      ▼ (deletion)
┌──────────────────┐            ┌─────────────────┐   ┌──────────┐  ┌────────────┐
│UPDATE: IMPORTANCE │            │SOFT DELETE:     │   │MERGED    │  │SOFT DELETE:│
│                   │            │FADE             │   │(archived)│  │(archived)  │
├──────────────────┤            ├─────────────────┤   ├──────────┤  ├────────────┤
│Relevance feedback │            │If importance    │   │Merged    │  │User request│
│increases/decreases│            │decays below     │   │into      │  │or system   │
│importance weight  │            │threshold:       │   │parent    │  │cleanup:    │
│                   │            │mark archived=1  │   │fact (1:1)│  │mark        │
│• Query-level      │            │                 │   │          │  │archived=1  │
│• Fact-level       │            │Memory hygiene   │   │Decision: │  │            │
│• Fusion weight    │            │consolidation    │   │MERGE     │  │Decision:   │
│  update           │            │schedule runs    │   │          │  │DELETE      │
│                   │            │periodically     │   │v1 → v2   │  │            │
│Decision: UPDATE   │            │                 │   │          │  │Lineage:    │
│                   │            │Decision: FADE   │   │Lineage:  │  │Records undo│
│Lineage: Records   │            │                 │   │Records   │  │target      │
│fact version change│            │Lineage: Records │   │merge     │  │            │
│                   │            │forgetting event │   │event     │  │v1 + v2 →   │
│                   │            │                 │   │          │  │[archived]  │
└──────────────────┘            └─────────────────┘   └──────────┘  └────────────┘
│                                    │                    │              │
│                                    ▼                    ▼              ▼
│                        ┌──────────────────────────────────────────┐   │
│                        │ STATE: ARCHIVED                          │   │
│                        ├──────────────────────────────────────────┤   │
│                        │ • Still stored (never hard-deleted)      │   │
│                        │ • Excluded from retrieval (archived=1)   │   │
│                        │ • Queries skip these facts               │   │
│                        │ • Can be recovered (reversible)          │   │
│                        │ • Lineage preserved (decision log)       │   │
│                        └──────────────────────────────────────────┘   │
│                                    │                                  │
│                                    │ (if user requests recovery)      │
│                                    │ OR (if important, restart from) │
│                                    │     lineage & causal parents    │
│                                    │                                 │
│                                    ▼                                 │
│        ┌──────────────────────────────────────────────────┐          │
│        │ RECOVERY: Recreate archived fact (reversible)   │          │
│        │ • Reapply decision operations in reverse         │          │
│        │ • Restore embeddings                            │          │
│        │ • Reset to previous version                     │          │
│        │ Decision: RESTORE                               │          │
│        │ Lineage: Records recovery event                 │          │
│        └──────────────────────────────────────────────────┘          │
│                                    │                                  │
│                                    └──────────┬───────────────────────┘
│                                               │
└───────────────────────────────────────────────┘
                    │
                    ▼
        ┌──────────────────────────┐
        │ STATE: ACTIVE (RESTORED) │
        │ (back to start, updated) │
        └──────────────────────────┘

KEY PRINCIPLES:
 ✓ No hard deletes (reversibility)
 ✓ Soft-delete with archive flag
 ✓ Lineage preserved (decision DAG)
 ✓ Can reconstruct history
 ✓ Temporal importance decay
 ✓ Memory hygiene (consolidation, forgetting)
```

---

## Diagram 8: Module Interaction Flow (One Complete Query)

```
USER ENTERS QUERY: "How does SITREP compress text?"
│
├──────────────────────────────────────────────────────────────────┐
│                                                                  │
▼                                                                  │
┌──────────────────────────────────────────────────────────────────┐ │
│ web/cli/plugin interface → application.query(query, top_k=5)   │ │
├──────────────────────────────────────────────────────────────────┤ │
│ Calls: QueryUseCase.execute()                                   │ │
└──────────────────────────────────────────────────────────────────┘ │
         │                                                         │
         ▼                                                         │
┌──────────────────────────────────────────────────────────────────┐ │
│ Step 1: CLASSIFY INTENT                                         │ │
├──────────────────────────────────────────────────────────────────┤ │
│ intent_classifier.classify(query)                               │ │
│  → "simple" (direct topic query)                                │ │
│  → query terms: ["compress", "text", "SITREP"]                 │ │
└──────────────────────────────────────────────────────────────────┘ │
         │                                                         │
         ▼                                                         │
┌──────────────────────────────────────────────────────────────────┐ │
│ Step 2: HYBRID SEARCH                                           │ │
├──────────────────────────────────────────────────────────────────┤ │
│ hybrid_retriever.search(query, top_k=10)                        │ │
│   → embedding_service.embed(query)                              │ │
│   → chromadb.search(embedding, k=10)  [dense scores]           │ │
│   → sqlite_fts5.search(["compress", "text"])  [sparse scores]  │ │
│   → fusion_weights.get()  [w_dense=0.62, w_sparse=0.38]        │ │
│   → combined_score = 0.62*dense + 0.38*sparse                  │ │
│   → top-10 passages: [Passage_A, Passage_B, ...]               │ │
└──────────────────────────────────────────────────────────────────┘ │
         │                                                         │
         ▼                                                         │
┌──────────────────────────────────────────────────────────────────┐ │
│ Step 3: RERANKING                                               │ │
├──────────────────────────────────────────────────────────────────┤ │
│ reranker.rerank(passages, query)                                │ │
│   → incorporate: importance, recency, quality score             │ │
│   → reranked: [Passage_A, Passage_D, Passage_B, ...]          │ │
│   → top-5 selected: [p1, p2, p3, p4, p5]                       │ │
└──────────────────────────────────────────────────────────────────┘ │
         │                                                         │
         ▼                                                         │
┌──────────────────────────────────────────────────────────────────┐ │
│ Step 4: COMPRESSION (RL Policy)                                 │ │
├──────────────────────────────────────────────────────────────────┤ │
│ FOR each passage in top-5:                                       │ │
│   state = encode(passage)  [text, embedding, importance, type] │ │
│   compression_ratio ~ π(state)  [PPO policy inference]         │ │
│   compressed = compress(passage, compression_ratio)            │ │
│   store original→compressed mapping in ccr_store               │ │
│                                                                  │ │
│ Example:                                                         │ │
│   p1: 245 tokens → 68 tokens (SmartCrusher, ratio=0.28)        │ │
│   p2: 312 tokens → 97 tokens (TextComp, ratio=0.31)            │ │
│   p3: 189 tokens → 52 tokens (CodeComp, ratio=0.27)            │ │
│   p4: 267 tokens → 71 tokens (LogComp, ratio=0.27)             │ │
│   p5: 201 tokens → 54 tokens (SmartCrusher, ratio=0.27)        │ │
└──────────────────────────────────────────────────────────────────┘ │
         │                                                         │
         ▼                                                         │
┌──────────────────────────────────────────────────────────────────┐ │
│ Step 5: GENERATE EXPLANATION (LLM)                              │ │
├──────────────────────────────────────────────────────────────────┤ │
│ llm_client.generate(                                            │ │
│   prompt = f"""                                                  │ │
│   Answer this query: {query}                                    │ │
│   Based on these facts:                                         │ │
│   {compressed_facts}                                            │ │
│   """                                                            │ │
│ )                                                                │ │
│                                                                  │ │
│ Output:                                                          │ │
│   "SITREP compresses text using multiple strategies:           │ │
│    1. SmartCrusher: Content-aware key selection                │ │
│    2. Code compression: AST-based reduction                    │ │
│    3. RL optimization: PPO learns ratio per passage            │ │
│    ..."                                                          │ │
└──────────────────────────────────────────────────────────────────┘ │
         │                                                         │
         ▼                                                         │
┌──────────────────────────────────────────────────────────────────┐ │
│ Step 6: CACHE RESULT                                            │ │
├──────────────────────────────────────────────────────────────────┤ │
│ result_cache.store(                                             │ │
│   query=query,                                                   │ │
│   result=QueryResult(                                           │ │
│     facts=[compressed facts],                                   │ │
│     explanation=explanation,                                   │ │
│     timestamp=now()                                             │ │
│   )                                                              │ │
│ )                                                                │ │
│                                                                  │ │
│ (Cache enables later feedback → fusion weight updates)          │ │
└──────────────────────────────────────────────────────────────────┘ │
         │                                                         │
         ▼                                                         │
┌──────────────────────────────────────────────────────────────────┐ │
│ RETURN QueryResult to user                                       │ │
├──────────────────────────────────────────────────────────────────┤ │
│ {                                                                │ │
│   "facts": [                                                    │ │
│     {                                                            │ │
│       "id": "fact_123",                                         │ │
│       "text": "[compressed] SmartCrusher: Identify high-...",   │ │
│       "importance": 0.87,                                       │ │
│       "ccr_key": "ccr_2025_07_09_001",  # Decompression key   │ │
│       "original_tokens": 245,                                   │ │
│       "compressed_tokens": 68                                   │ │
│     },                                                           │ │
│     ...                                                          │ │
│   ],                                                             │ │
│   "explanation": "SITREP compresses text using...",            │ │
│   "query_latency_ms": 342,                                      │ │
│   "total_tokens_in_result": 342                                 │ │
│ }                                                                │ │
└──────────────────────────────────────────────────────────────────┘ │
         │                                                         │
         └─────────────────────────────────────────────────────────┘
                        Render in UI / CLI
         
         ┌──────────────────────────────────────────────────────────┐
         │ USER PROVIDES FEEDBACK (Optional)                       │
         ├──────────────────────────────────────────────────────────┤
         │ User clicks: "Fact_123: Very Relevant" (rating: 0.95)  │
         │                                                          │
         │ Triggers: feedback_usecase.provide_feedback(            │
         │   query=query,                                          │
         │   fact_id="fact_123",                                   │
         │   relevance=0.95                                        │
         │ )                                                        │
         │   → Updates: fact.importance += (0.95 - old_importance) │
         │   → Updates: fusion_weights (w_dense ↑, w_sparse ↓)    │
         │   → Stores: FeedbackRecord in SQLite                    │
         │                                                          │
         │ Next query uses updated weights → better results        │
         └──────────────────────────────────────────────────────────┘
```

---

## Diagram 9: Error Recovery & Transaction Rollback

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     ATOMIC WRITE WITH ROLLBACK                              │
│                (Transaction spanning SQLite + KuzuDB + ChromaDB)            │
└─────────────────────────────────────────────────────────────────────────────┘

SCENARIO: User ingests 50 passages; ChromaDB fails mid-indexing
│
├──────────────────────────────────────────────────────────────────┐
│                                                                  │
▼                                                                  │
┌──────────────────────────────────────────────────────────────────┐ │
│ BEGIN TRANSACTION (txn = transaction_manager.begin())           │ │
├──────────────────────────────────────────────────────────────────┤ │
│ • SQLite: BEGIN TRANSACTION (implicit, WAL mode)                │ │
│ • KuzuDB: Begin logical transaction (logged)                    │ │
│ • ChromaDB: Batch index (no explicit txn, logged)               │ │
│ • DuckDB: Append to Parquet (transactional append)              │ │
│                                                                  │ │
│ Transaction ID: txn_id_20250709_001                             │ │
│ Operation: INGEST                                               │ │
└──────────────────────────────────────────────────────────────────┘ │
         │                                                         │
         ├─────────────────────┬──────────────┬──────────┐        │
         │                     │              │          │        │
         ▼                     ▼              ▼          ▼        │
    (1) SQLite         (2) ChromaDB      (3) KuzuDB  (4) DuckDB │
        INSERT            INDEX          ADD NODES   APPEND     │
        ✓ passages        passages        entities    parquet    │
        ✓ facts           embedding       relations   ✓          │
        ✓ fts5 index      batch...                               │
        ✓                 index 1-45:  OK               │        │
        ✓                 index 46...  ERROR!           │        │
        ✓                                               │        │
        │                 ChromaDB fails at index 46   │        │
        │                                               │        │
        └───────────────────┬──────────────────────────┘        │
                            │                                    │
                            ▼                                    │
              ┌──────────────────────────────┐                  │
              │ EXCEPTION CAUGHT             │                  │
              │ "ChromaDB batch_add failed"  │                  │
              │ Reason: Network timeout      │                  │
              └──────────────────────────────┘                  │
                            │                                    │
                            ▼                                    │
              ┌──────────────────────────────────────┐           │
              │ AUTOMATIC ROLLBACK INITIATED         │           │
              │ async with transaction_manager:      │           │
              │   ...                                │           │
              │ except Exception as e:               │           │
              │   await txn.rollback()               │           │
              └──────────────────────────────────────┘           │
                            │                                    │
        ┌───────────────────┼───────────────┬──────────┐        │
        │                   │               │          │        │
        ▼                   ▼               ▼          ▼        │
    SQLite            KuzuDB            ChromaDB   DuckDB     │
    ────────────────────────────────────────────────────────── │
    ROLLBACK:         Delete nodes:     No delete   Rollback  │
    • Undo INSERT     • Remove 156      needed       parquet   │
      passages         entity nodes     (index 1-45 append     │
    • Undo INSERT     • Remove 283      already        log     │
      facts            relations         existed)              │
    • WAL journal                                              │
      cleared          (Transaction log  (Eventual             │
                       cleaned)          consistency)          │
                                                               │
    Database reverts to pre-ingest state                       │
                            │                                 │
                            ▼                                 │
                ┌──────────────────────────┐                 │
                │ STATE: Consistent        │                 │
                │ ✓ SQLite: No passages    │                 │
                │ ✓ KuzuDB: No entities    │                 │
                │ ✓ ChromaDB: Partial     │                 │
                │   (embeddings 1-45 may  │                 │
                │    remain; need cleanup) │                 │
                │ ✓ DuckDB: No append     │                 │
                │                          │                 │
                │ Decision log: Records    │                 │
                │   failed INGEST attempt │                 │
                └──────────────────────────┘                 │
                            │                                 │
                            ▼                                 │
                ┌──────────────────────────────────┐         │
                │ RECOVERY OPTIONS                 │         │
                ├──────────────────────────────────┤         │
                │ 1. Retry:                        │         │
                │    Fix ChromaDB connection       │         │
                │    Re-ingest same 50 passages   │         │
                │                                  │         │
                │ 2. Resume:                       │         │
                │    Checkpoint: passages 1-45     │         │
                │    Ingest failed passage 46 only │         │
                │    Ingest 47-50                  │         │
                │                                  │         │
                │ 3. Manual Fix:                   │         │
                │    ChromaDB cleanup script       │         │
                │    Delete orphaned embeddings    │         │
                │    Then retry                    │         │
                │                                  │         │
                │ 4. Abort:                        │         │
                │    Skip ingest, raise error      │         │
                └──────────────────────────────────┘         │
                            │                                 │
                            └─────────────────────────────────┘
```

---

## Diagram 10: Performance Profile (Expected Latencies)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    SITREP LATENCY PROFILE (Estimates)                       │
└─────────────────────────────────────────────────────────────────────────────┘

QUERY LATENCY BREAKDOWN (1 query, top-5 results)
═════════════════════════════════════════════════════════════════════════════

┌─────────────────────────────────────────────────────────────────┐
│ Dense Embedding (query)                                         │ 15-25 ms
├─────────────────────────────────────────────────────────────────┤
│  query_text: "How does SITREP compress text?"                  │
│  model: sentence-transformers/all-MiniLM-L6-v2                 │
│  output: 384-dim embedding                                      │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ Dense Search (ChromaDB k-NN)                                    │ 10-20 ms
├─────────────────────────────────────────────────────────────────┤
│  k = 10 (retrieve top-10)                                       │
│  index: FAISS (Flat or IVF)                                     │
│  corpus: 5,000 passages (100K+ scale: 50-100 ms)               │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ Sparse Search (BM25 on FTS5)                                    │ 5-15 ms
├─────────────────────────────────────────────────────────────────┤
│  query terms: ["compress", "text"]                              │
│  index: SQLite FTS5 (inverted index)                             │
│  corpus: 5,000 passages (100K+ scale: 20-50 ms)               │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ Fusion (Linear combination)                                     │ 1-2 ms
├─────────────────────────────────────────────────────────────────┤
│  w_dense = 0.62, w_sparse = 0.38                                │
│  combine top-10 from both → rank                                │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ Reranking (Importance + Recency + Quality)                      │ 2-5 ms
├─────────────────────────────────────────────────────────────────┤
│  reorder top-10 by composite score                              │
│  select top-5                                                    │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ Compression (RL Policy Inference)                               │ 20-50 ms
├─────────────────────────────────────────────────────────────────┤
│  5 passages × 5-10 ms each (forward pass)                       │
│  model: 2-layer MLP (small, efficient)                          │
│  gpu: Optional (10-15 ms if GPU available, 20-50 if CPU)       │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ LLM Generation (Explanation, optional)                          │ 500-2000 ms
├─────────────────────────────────────────────────────────────────┤
│  model: Mistral-7B (Ollama) or Transformers                     │
│  input: ~300 tokens (query + 5 compressed facts)                │
│  output: ~150 tokens (typical explanation)                      │
│  latency: GPU-dependent (5-10 tokens/sec on GPU)                │
│  latency: CPU-dependent (100-500ms fallback)                    │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ Result Cache Store                                              │ 2-5 ms
├─────────────────────────────────────────────────────────────────┤
│  write QueryResult to SQLite                                    │
└─────────────────────────────────────────────────────────────────┘

═════════════════════════════════════════════════════════════════════════════

                          TOTAL (NO LLM):  73-127 ms
                          TOTAL (WITH LLM): 573-2127 ms

                          P50: ~100 ms (no LLM)
                          P95: ~200 ms (no LLM)
                          P99: ~500 ms (no LLM)

═════════════════════════════════════════════════════════════════════════════

INGEST LATENCY BREAKDOWN (50 passages, 3K tokens total)
═════════════════════════════════════════════════════════════════════════════

┌─────────────────────────────────────────────────────────────────┐
│ Parsing (PDF, MD, etc.)                                         │ 50-200 ms
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ Chunking (split into passages)                                  │ 5-10 ms
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ Batch Embedding (50 passages → 384-dim vectors)                 │ 100-150 ms
├─────────────────────────────────────────────────────────────────┤
│  model: sentence-transformers (optimized batching)              │
│  throughput: ~300-500 passages/sec                              │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ Extraction (LLM or heuristic, fact mining)                      │ 500-2000 ms
├─────────────────────────────────────────────────────────────────┤
│  if heuristic (sentence-level): 50-100 ms                       │
│  if LLM (Mistral): 500-2000 ms (50 passages × 10-40 ms each)   │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ Entity Extraction (for KuzuDB graph)                            │ 100-300 ms
├─────────────────────────────────────────────────────────────────┤
│  NER or dependency parsing                                      │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ Transaction Write (SQLite + KuzuDB + ChromaDB)                  │ 200-500 ms
├─────────────────────────────────────────────────────────────────┤
│  SQLite: INSERT 50 passages + 150 facts (batched)               │
│  KuzuDB: ADD 300 entity nodes + 500 relations                   │
│  ChromaDB: INDEX 50 embeddings                                  │
│  DuckDB: APPEND to Parquet (if enabled)                         │
└─────────────────────────────────────────────────────────────────┘

═════════════════════════════════════════════════════════════════════════════

                    TOTAL (HEURISTIC): 455-1260 ms
                    TOTAL (LLM): 955-3260 ms

                    P50: ~800 ms (heuristic)
                    P95: ~2000 ms (heuristic)
                    P99: ~3000 ms (LLM-based)

═════════════════════════════════════════════════════════════════════════════

SCALING NOTES:
 • 10K passages: Dense search ~80-100 ms (FAISS with IVF), sparse ~50 ms
 • 100K passages: Dense search ~200-300 ms, sparse ~100-200 ms (need indexing)
 • GPU available: RL compression 5-10 ms/passage, extraction 50-100 ms/passage
 • CPU only: Latencies 2-5x higher for LLM/RL inference
```

---

End of Architecture Diagrams

**All diagrams show:**
- ✅ System boundaries (what's inside vs. outside SITREP)
- ✅ Data flow (queries → results, ingestion → storage)
- ✅ Module dependencies (what calls what)
- ✅ Error recovery (rollback, consistency)
- ✅ Performance characteristics (latency breakdown)
- ✅ State machines (fact lifecycle)
- ✅ Scaling considerations (10K → 100K corpus)
