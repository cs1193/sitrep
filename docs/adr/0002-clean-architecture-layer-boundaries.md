# ADR-0002: Clean Architecture Layer Boundaries

**Status:** Accepted  
**Date:** 2026-07-09  
**Deciders:** Architecture Team  
**Co-Authors:** Claude Haiku 4.5

---

## Context

SITREP handles multiple responsibilities: data persistence (SQLite, KuzuDB, ChromaDB), business logic (retrieval, compression, RL training), domain modeling (Fact, Passage, Episode), and external integrations (LLMs, embeddings, RL frameworks).

Without clear layer boundaries, these responsibilities tend to bleed into each other, leading to:
- **Hard-to-test code:** Business logic tightly coupled to database access
- **Difficult refactoring:** Changing persistence strategy requires touching business logic
- **Inconsistent patterns:** Different modules use different approaches to similar problems
- **Tight coupling:** Adding a new database or LLM requires changes throughout the codebase

The team needed a **clear, unidirectional dependency structure** that would:
1. Isolate domain logic from infrastructure details
2. Enable testing without external services
3. Allow swapping implementations (databases, LLMs, compression strategies)
4. Provide a consistent mental model for developers

---

## Decision

**Adopt Clean Architecture with 5 explicit layers, strict unidirectional dependencies (inward only), and segregation by responsibility.**

### Layer Structure

**Layer 1: Domain** (1,275 LOC) — Outermost boundary
- Pure data models: `Fact`, `Passage`, `Episode`, `Decision`, `Agent`, `Schema`
- Value objects: `TimeRange`, `Entity`, `Relation`, `CausalRelation`
- Enums: `DecisionType` (INGEST, UPDATE, MERGE, DELETE), `AllenRelation` (13 temporal predicates)
- **No business logic, no I/O, no external dependencies**
- Changes to domain models only affect inner layers

**Layer 2: Adapters** (2,246 LOC) — Service layer
- **Repositories:** Abstract data persistence (CRUD + domain-specific queries)
  - Persist domain models without domain knowing about storage backend
  - Example: `FactRepository.find_by_importance()` works same regardless of SQLite vs PostgreSQL
- **Services:** Business logic encapsulated in focused classes
  - Extraction, Compression, Classification, Judgment, Reranking, Versioning, Conflict Resolution
  - Each service has single responsibility
  - No knowledge of how data is retrieved (that's repositories' job)
- **Dependency:** Domain ← Adapters only (adapters depend on domain, not vice versa)

**Layer 3: Application** (2,881 LOC) — Use cases
- **Use Cases:** Orchestrate repositories + services to implement workflows
  - `QueryUseCase`: Retrieves facts, reranks, compresses, generates explanation
  - `IngestUseCase`: Parses document, extracts facts, builds graph, stores atomically
  - `TrainUseCase`: Samples passages, runs RL rollout, updates policy
  - `FeedbackUseCase`: Updates importance, fusion weights, stores signals
  - `VersioningUseCase`: Creates fact versions, maintains lineage
  - `LineageUseCase`: Traces decision DAG, enables audit + rollback
- **Composition Root:** `build_application()` wires all dependencies
  - Single entry point for dependency injection
  - Configuration flows in, fully-wired Application flows out
- **Dependency:** Adapters ← Application only (application orchestrates adapters)

**Layer 4: Infrastructure** (4,452 LOC) — External service adapters
- **Database Clients:** Concrete implementations for each database
  - `SQLiteClient`: Transactional metadata store
  - `KuzuDBClient`: Graph with temporal relationships
  - `ChromaDBClient`: Vector embeddings
  - `DuckDBClient`: OLAP + Parquet archival
  - `KVCacheClient`: Transformer KV precomputation
  - All expose clean interfaces that repositories use
- **Retrieval Engine:** Hybrid search implementation
  - Dense search (embeddings), sparse search (BM25), fusion (learnable weights)
  - Reranking, entity graph ranking, temporal filtering
- **Compression Strategies:** Content-aware compression implementations
  - SmartCrusher, JSON compressor, Code AST compressor, Log compressor, Text compressor
  - Reversible compression via CCR store
- **RL Agent:** PPO compression agent + environment + reward model
- **LLM Gateways:** Ollama, Transformers, Demo mode adapters
- **Embedding Service:** sentence-transformers, fallback to hash-based
- **Dependency:** Application ← Infrastructure only (infrastructure provides capabilities)

**Layer 5: Presentation** (179 LOC) — Entry points
- **Gradio Web UI:** Interactive tabs (Query, Ingest, Train, Stats, Lineage, Versioning)
- **CLI Scripts:** Entry points that call Application methods
  - `query_cli.py`, `ingest_batch.py`, `train_compression_agent.py`, etc.
- **Plugin Interface:** `plugin.py` exposes Application methods to Claude Code
- **Dependency:** Application ← Presentation only (presentation calls application)

### Dependency Rule

```
Domain (no dependencies)
  ↑
Adapters (depends on Domain only)
  ↑
Application (depends on Adapters only)
  ↑
Infrastructure (depends on Application & Domain for interfaces)
  ↑
Presentation (depends on Application only)
```

**No downward dependencies:** Inner layers never import from outer layers.

### File Structure Reflects Layers

```
src/
├── domain/              ← Layer 1: Pure models
│   ├── schemas.py       (Fact, Passage, Episode, Decision, Agent, Schema)
│   ├── value_objects.py (TimeRange, Entity, Relation)
│   └── enums.py         (DecisionType, AllenRelation)
│
├── adapters/            ← Layer 2: Repositories + Services
│   ├── repositories.py   (FactRepo, PassageRepo, etc.)
│   ├── services.py       (ExtractionService, CompressionService, etc.)
│   └── managers.py       (LineageTracker, VersioningService)
│
├── application/         ← Layer 3: Use cases + Composition
│   ├── use_cases.py      (QueryUseCase, IngestUseCase, etc.)
│   ├── __init__.py       (build_application() composition root)
│   └── event_system.py   (Optional: event bus for decoupling)
│
├── infrastructure/      ← Layer 4: External service adapters
│   ├── database.py       (SQLite, KuzuDB, ChromaDB, DuckDB, KVCache clients)
│   ├── retrieval.py      (HybridRetriever, Reranker, EntityGraphRank)
│   ├── compression.py    (SmartCrusher, strategy implementations)
│   ├── rl.py             (PPOAgent, CompressionEnv, RewardModel)
│   ├── llm.py            (OllamaClient, TransformersLLM, DemoLLM)
│   ├── embeddings.py     (EmbeddingService with fallback)
│   └── utils.py          (Shared infrastructure utilities)
│
└── utils/               ← Cross-layer utilities
    ├── config.py        (Configuration management)
    ├── logging.py       (Logging setup)
    ├── decorators.py    (Caching, retry, timing decorators)
    └── constants.py     (Magic numbers, defaults)
```

---

## Rationale

### Why Clean Architecture?

**Testability:** Unit tests don't need databases or LLMs
```python
# Test extraction service without DB
service = ExtractionService(mock_llm)
facts = service.extract(passage)
assert len(facts) == 3
assert facts[0].importance > 0.5
```

**Swappability:** Change database without touching business logic
```python
# Old: SQLite
app = build_application(config_sqlite)

# New: PostgreSQL (just swap repository)
app = build_application(config_postgres)
# Application code unchanged!
```

**Discoverability:** Clear where each concern lives
- Need to understand retrieval? → `src/infrastructure/retrieval.py`
- Need to understand versioning? → `src/adapters/services.py` + `src/application/use_cases.py`
- Need to understand compression? → `src/infrastructure/compression.py`

**Flexibility:** Add new retrieval strategy without touching query logic
```python
# Old: HybridRetriever
retriever = HybridRetriever(...)

# New: EntityGraphOnlyRetriever
retriever = EntityGraphOnlyRetriever(...)
# QueryUseCase doesn't change
```

### Why These 5 Layers?

- **Domain** — What we're modeling (facts, decisions, temporal relations)
- **Adapters** — How we implement it (repositories abstract storage, services encapsulate logic)
- **Application** — What we do with it (query, ingest, train, feedback)
- **Infrastructure** — What tools we use (databases, LLMs, ML models)
- **Presentation** — How users interact (UI, CLI, API)

Each layer has single responsibility; dependencies flow inward only.

---

## Consequences

### Positive

✅ **Testable:** Mock repositories, test use cases in isolation  
✅ **Swappable:** Add PostgreSQL, Pinecone, Claude API without touching core logic  
✅ **Clear ownership:** Each layer has well-defined responsibilities  
✅ **Refactoring safety:** Changes in one layer don't cascade outward  
✅ **Onboarding:** New developers understand structure immediately  
✅ **Extensibility:** Add new compression strategy, new LLM, new database without refactoring  

### Negative

⚠️ **Indirection:** Extra interfaces (Repository, Service) may feel over-engineered for simple operations  
⚠️ **Initial setup:** More files, more boilerplate (but template reduces friction)  
⚠️ **Performance:** Layer crossing has minimal cost, but adds function call overhead  
⚠️ **Learning curve:** Team needs to understand unidirectional dependency rule  

### Mitigation

1. **Templating:** Provide blueprints for new services (copy/paste pattern)
2. **Documentation:** Link architecture docs in PRs that add new components
3. **Code review:** Check for layer violations (imports from outer layers)
4. **Automated checks:** Lint rule to catch invalid imports (optional: `py-architecture-lint`)

---

## Implementation

### Composition Root Pattern

All dependencies wired in one place (`src/application/__init__.py`):

```python
def build_application(config: SitrepConfig) -> Application:
    # 1. Database clients (Infrastructure)
    sqlite = SQLiteClient(config.db_path)
    kuzu = KuzuDB(config.graph_path)
    chroma = ChromaDB(config.vectors_path)
    
    # 2. Repositories (Adapters) depend on DB clients
    fact_repo = FactRepository(sqlite)
    passage_repo = PassageRepository(sqlite, chroma)
    graph_repo = GraphRepository(kuzu)
    
    # 3. Services (Adapters) depend on repositories
    extraction = ExtractionService(llm_client, embedding_service)
    compression = CompressionService(strategies, rl_policy)
    
    # 4. Use cases (Application) depend on services
    query_uc = QueryUseCase(retriever, reranker, compression, llm)
    ingest_uc = IngestUseCase(parser, extraction, repositories, graph)
    
    # 5. Return fully-wired Application
    return Application(
        query=query_uc,
        ingest=ingest_uc,
        # ... more use cases
    )
```

**Benefit:** Single point of change when wiring changes (e.g., swap SQLite → PostgreSQL)

### Repository Abstraction

Repositories hide storage details:

```python
# Domain doesn't know how facts are stored
class FactRepository(ABC):
    @abstractmethod
    async def find_by_id(fact_id: str) -> Fact: ...
    
    @abstractmethod
    async def find_by_importance(min_imp: float, limit: int) -> List[Fact]: ...

# Concrete implementation (SQLite)
class SQLiteFactRepository(FactRepository):
    def __init__(self, client: SQLiteClient):
        self.client = client
    
    async def find_by_importance(self, min_imp: float, limit: int) -> List[Fact]:
        rows = self.client.execute(
            "SELECT * FROM facts WHERE importance >= ? ORDER BY importance DESC LIMIT ?",
            (min_imp, limit)
        )
        return [Fact.from_row(row) for row in rows]
```

**Benefit:** Swap to PostgreSQL by implementing same interface; no code in services/use cases changes.

### Dependency Inversion

Services don't directly use infrastructure; they depend on abstractions:

```python
# Service depends on abstract Repository
class RerankerService:
    def __init__(self, fact_repo: FactRepository):
        self.fact_repo = fact_repo
    
    async def rerank(self, facts: List[Fact]) -> List[Fact]:
        # Use repository abstraction
        importances = await self.fact_repo.get_importance_stats()
        return sorted(facts, key=lambda f: f.importance, reverse=True)

# At composition time, inject concrete repository
reranker = RerankerService(fact_repo=SQLiteFactRepository(...))
```

**Benefit:** Service tested with mock repository; production uses real repository.

---

## Violations & Prevention

### Common Violations to Avoid

❌ **Domain importing from Infrastructure**
```python
# BAD: Domain model depends on ChromaDB
class Fact(BaseModel):
    embeddings: chromadb.Collection  # ← Domain shouldn't know about ChromaDB
```

✅ **Correct:** Infrastructure depends on Domain
```python
# GOOD: Domain defines interface; Infrastructure implements
class Fact(BaseModel):
    text: str
    embedding: Optional[np.ndarray]  # ← Platform-agnostic

# Infrastructure handles actual storage
class PassageRepository:
    def __init__(self, chroma: ChromaDB):
        self.chroma = chroma  # ← Infrastructure owns ChromaDB
```

❌ **Application importing from Infrastructure directly**
```python
# BAD: Use case depends on concrete SQLite
class QueryUseCase:
    def __init__(self, sqlite: SQLiteClient):  # ← Hard to test, hard to swap
        self.sqlite = sqlite
```

✅ **Correct:** Application depends on abstract Repositories
```python
# GOOD: Use case depends on repository interface
class QueryUseCase:
    def __init__(self, fact_repo: FactRepository):  # ← Testable, swappable
        self.fact_repo = fact_repo
```

### Prevention

1. **Code review checklist:** Check imports don't cross layer boundary
2. **Package structure:** Make violations obvious (layer violation would be `from infrastructure import ...` inside `adapters/`)
3. **Type hints:** Abstract types (`FactRepository`) instead of concrete (`SQLiteClient`) make intent clear

---

## Evolution & Extension

### Adding a New Feature

**Example: Add PostgreSQL support**

1. Create `PostgreSQLFactRepository(FactRepository)` in `infrastructure/database.py`
2. Update `build_application()` to accept `db_type` config parameter
3. Instantiate correct repository based on config
4. All use cases, services, adapters unchanged ✓

**Files changed:** ~20 lines in 2 files (database.py, __init__.py)  
**Existing tests pass:** Yes ✓ (because application depends on abstraction)

### Adding a New Use Case

**Example: Add "Consolidate memory" use case**

1. Create `ConsolidateMemoryUseCase` in `application/use_cases.py`
2. Depends on existing repositories + services (no new infrastructure needed)
3. Add to `Application` dataclass
4. Wire in `build_application()`

**Files changed:** ~100 lines in 2 files (use_cases.py, __init__.py)  
**Breaking changes:** None ✓

### Adding a New Compression Strategy

**Example: Add "Abstractive summarization" compressor**

1. Create `AbstractiveSummarizerCompressor` in `infrastructure/compression.py`
2. Inherit from `CompressionStrategy` interface
3. Add to `CompressionService` registry
4. Services, use cases, adapters unchanged ✓

**Files changed:** ~80 lines in 1 file (compression.py)

---

## Related ADRs

- **ADR-0003:** Lazy imports for optional infrastructure dependencies
- **ADR-0004:** Multi-database approach (SQLite + KuzuDB + ChromaDB)
- **ADR-0005:** Hybrid retrieval fusion weight learning (Adapters layer)

---

## References

- **Code:** `src/` directory structure
- **Composition Root:** `src/application/__init__.py` (457 LOC)
- **Example Repositories:** `src/adapters/repositories.py` (600 LOC)
- **Example Services:** `src/adapters/services.py` (800 LOC)
- **Example Use Cases:** `src/application/use_cases.py` (1,200 LOC)
- **Architecture Analysis:** `docs/SITREP_CODEBASE_ANALYSIS.md` (Section 2)
- **Architecture Diagrams:** `docs/ARCHITECTURE_DIAGRAMS.md` (Diagrams 1 & 6)

---

**Status:** ✅ Accepted and implemented  
**Last Updated:** 2026-07-09  
**Next Review:** When adding new database or major infrastructure change
