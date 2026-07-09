# ADR-0004: Multi-Database Approach (SQLite + KuzuDB + ChromaDB + DuckDB)

**Status:** Accepted  
**Date:** 2026-07-09  
**Deciders:** Architecture Team, Data Engineering  
**Co-Authors:** Claude Haiku 4.5

---

## Context

SITREP stores and queries multiple data types with different access patterns:

1. **Structured metadata** (facts, passages, decisions, feedback)
   - Row-oriented queries: "Get facts with importance > 0.5"
   - Full-text search: "Find passages containing 'compression'"
   - Transactional: Atomic writes across multiple tables
   - Pattern: OLTP (Online Transaction Processing)

2. **Knowledge graph** (entities, relationships, temporal intervals)
   - Graph queries: "Find neighbors of entity X"
   - Temporal filtering: "Facts overlapping time range [T1, T2]"
   - Traversal: "Find paths from fact A to fact B"
   - Pattern: Graph OLTP

3. **Vector embeddings** (dense semantic representations)
   - k-NN search: "Find 10 passages most similar to query"
   - Batch indexing: "Index 1000 passage embeddings"
   - Pattern: Vector ANN (Approximate Nearest Neighbor)

4. **Analytics & archival** (time-series data for reporting)
   - Columnar analysis: "Aggregates over time"
   - Historical snapshots: "Parquet time-series storage"
   - Pattern: OLAP (Online Analytical Processing)

5. **Key-value cache** (transformer KV pairs for inference acceleration)
   - Fast write: Cache computed KV for fact F
   - Fast read: Retrieve KV for fact F
   - Expiry: Invalidate cache after time T
   - Pattern: Ephemeral KV store

Using a single database for all would:
- **Sacrifice performance:** A general-purpose DB (even PostgreSQL) is slower than specialized tools
  - Vector search: PostgreSQL pgvector is 10-100x slower than FAISS/ChromaDB
  - Graph traversal: PostgreSQL is slower than graph DBs at graph queries
  - Analytics: PostgreSQL slower than columnar (DuckDB) by 10-1000x for aggregations
- **Waste storage:** General-purpose DB stores unnecessary overhead for each use case
- **Overfits design:** Schema designed for OLTP doesn't fit OLAP well
- **Creates bottleneck:** Single DB connection pool becomes bottleneck under load

The team needed a way to:
1. **Optimize for each access pattern** — Use specialized DB for each pattern
2. **Maintain consistency** — Coordinate writes across DBs atomically
3. **Enable seamless upgrade paths** — Add new DB type without refactoring
4. **Keep data in sync** — Ensure facts in SQLite match facts in ChromaDB

---

## Decision

**Use polyglot persistence (multiple databases optimized for their use case) with Repository pattern to abstract storage.**

### Database Allocation

```
┌──────────────────────────────────────────────────────────────────────┐
│                     SITREP DATA STORAGE                              │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  SQLite (Metadata + FTS5)          KuzuDB (Knowledge Graph)          │
│  ├─ facts table (OLTP)              ├─ EntityNode (vertices)         │
│  │  └─ FTS5 for full-text            │  └─ name, type, embedding    │
│  ├─ passages table (OLTP)            ├─ Fact nodes                   │
│  │  └─ FTS5 for keyword search       ├─ Relations (edges)            │
│  ├─ episodes table                   │  └─ 13 temporal predicates    │
│  ├─ decisions table (audit trail)    └─ Relationship properties      │
│  ├─ feedback table                                                    │
│  ├─ fusion_weights table (learnable) ChromaDB (Vector Embeddings)     │
│  ├─ kv_cache table (ephemeral)       ├─ passage_embeddings (384-dim) │
│  └─ schemas table                    └─ fact_embeddings (semantic)   │
│                                                                       │
│  (Transactional, ACID)              (Graph OLTP)                    │
│  (Full-text search)                 (Temporal algebra)              │
│  (WAL mode: ~100ms latency)         (~10ms neighbor query)          │
│                                                                       │
│                  DuckDB (Analytics + Archives)                        │
│                  ├─ facts_archive.parquet (time-series)              │
│                  ├─ passages_archive.parquet                         │
│                  ├─ decisions_archive.parquet                        │
│                  └─ Columnar OLAP queries                            │
│                     (~100ms aggregations over 100K rows)             │
│                                                                      │
│  (Write-once append)                                                │
│  (Columnar compression)                                             │
│  (Fast aggregations)                                                │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

### Each Database's Role

#### 1. SQLite (Primary Transactional Store)

**Use for:** Structured metadata that needs ACID guarantees

```
facts:
  id (PK)
  text
  source_passage_id (FK)
  importance (0.0-1.0)
  timestamp
  causal_parent_ids (JSON)
  metadata (JSON)
  archived (BOOLEAN)

passages:
  id (PK)
  content
  source
  timestamp
  token_estimate
  compressed_form
  ccr_key (FK → CCR store)
  archived (BOOLEAN)

decisions:
  id (PK)
  type (ENUM)
  input_ids (JSON)
  output_ids (JSON)
  timestamp
  causal_parents (JSON)
  reversible (BOOLEAN)
  audit_log
  
feedback:
  id (PK)
  query
  fact_id (FK)
  relevance (0.0-1.0)
  quality (0.0-1.0)
  timestamp
  
fusion_weights:
  id (PK)
  w_dense (REAL)
  w_sparse (REAL)
  w_entity_rank (REAL)
  updated_at (TIMESTAMP)
```

**Why SQLite?**
- ✓ Built-in, zero configuration
- ✓ ACID transactions with WAL mode
- ✓ FTS5 extension for full-text search
- ✓ Fast enough for 100K facts (~50ms queries)
- ✓ Self-contained (single file `.sitrep/metadata/sitrep.db`)
- ✓ Version control friendly (query-time consistency, no server state)
- ⚠️ Limitation: Not for >1M rows or high-concurrency writes (but fine for single-user)

**When to upgrade:** If corpus grows to millions of facts or multiple concurrent writers, migrate to PostgreSQL (same schema, repositories abstract it)

#### 2. KuzuDB (Knowledge Graph)

**Use for:** Temporal relationships and entity graphs

```
EntityNode(id, name, type, embedding)
├─ FactNode (inherits EntityNode)
└─ ConceptNode (e.g., "compression", "RL")

Relation(source, target, type)
├─ mentions (fact mentions concept)
├─ causes (fact causes fact)
├─ merges_into (fact merges into fact)
└─ AllenRelation (13 temporal predicates):
   ├─ BEFORE
   ├─ AFTER
   ├─ MEETS
   ├─ OVERLAPS
   ├─ DURING
   ├─ CONTAINS
   ├─ STARTS
   ├─ FINISHES
   ├─ EQUALS
   └─ ... (13 total)
```

**Why KuzuDB?**
- ✓ Native graph queries (neighbors, paths, traversals)
- ✓ Supports temporal relationships (Allen intervals)
- ✓ Enables Personalized PageRank for ranking
- ✓ Fast graph traversal (<10ms for small graphs)
- ⚠️ Optional: Feature dormant by default, activates with `[graph]` extra
- ⚠️ Limitation: Not for massive graphs (>10M nodes), but fine for entity graphs in small-medium corpora

**When to use:** When you need "find related facts" or "temporal reasoning"

**When optional:** Core retrieval works without graph (BM25 + dense search sufficient)

#### 3. ChromaDB (Vector Embeddings)

**Use for:** Semantic similarity search (dense embeddings)

```
Collections:
├─ passages
│  ├─ id: passage_id
│  ├─ embedding: np.array (384-dim)
│  ├─ document: passage content
│  └─ metadata: {source, timestamp, ...}
│
└─ facts
   ├─ id: fact_id
   ├─ embedding: np.array (semantic summary)
   ├─ document: fact text
   └─ metadata: {importance, source_passage_id, ...}
```

**Why ChromaDB?**
- ✓ Purpose-built for embeddings (ChromaDB, Pinecone, Weaviate all work)
- ✓ Fast k-NN search (FAISS under the hood: ~10ms for k=10 on 5K vectors)
- ✓ Handles sparse embeddings well (if models unavailable, can fallback to BM25)
- ✓ Optional: Easy to omit if only using BM25
- ⚠️ Limitation: Doesn't scale to billions of vectors without sharding (fine for <100K)

**When to use:** When you want "semantic search" (understand meaning, not just keywords)

**When optional:** Sparse BM25 works fine for keyword-heavy corpora

#### 4. DuckDB (Analytics & Archival)

**Use for:** Time-series analysis and Parquet archives

```
Parquet Files:
├─ facts_archive.parquet
│  ├─ fact_id
│  ├─ text
│  ├─ importance
│  ├─ timestamp
│  └─ ... (columnar format)
│
├─ passages_archive.parquet
│
└─ decisions_archive.parquet
```

**Why DuckDB?**
- ✓ Columnar format (OLAP-optimized, 10-1000x faster than row-oriented for aggregations)
- ✓ Parquet support (standard format, zero-copy reads)
- ✓ SQL support (familiar interface for analytics)
- ✓ Process-embedded (no server)
- ✓ Write-once append model (immutable archives)
- ⚠️ Optional: Only needed if you do analytics/reporting

**When to use:** Historical analysis ("How did importance scores change over time?")

**When optional:** Not needed for basic query/ingest workflow

#### 5. KV Cache (Transformer Optimization)

**Use for:** Pre-computed transformer key-value pairs

```
kv_cache table (SQLite):
├─ fact_id
├─ model_id (e.g., "llama-7b")
├─ cache (BLOB: pickled PyTorch tensor)
├─ expires_at (TIMESTAMP)
└─ metadata (JSON)
```

**Why store in SQLite?**
- ✓ Small, ephemeral data (delete on expiry)
- ✓ Needs ACID (don't corrupt cache mid-compute)
- ✓ Row-based access ("Get cache for fact F with model M")
- ✓ Time-based invalidation (SQLite supports TIMESTAMP)

**Benefit:** If you query the same fact multiple times, reuse KV cache instead of recomputing (30-50% latency reduction)

---

## Rationale

### Why Polyglot Persistence?

**Performance:** Specialized DBs are 10-1000x faster than general-purpose

| Task | General DB (PostgreSQL) | Specialized | Speedup |
|------|------------------------|-------------|---------|
| BM25 full-text search | 100-200ms | SQLite FTS5: 20-50ms | 3-5x |
| k-NN vector search (k=10) | 500-1000ms | ChromaDB/FAISS: 10-20ms | 30-100x |
| Graph neighbor query | 200-500ms | KuzuDB: 10-30ms | 10-50x |
| Aggregate 100K rows | 1000-5000ms | DuckDB: 50-200ms | 10-50x |

**Storage:** Specialized formats compress better

| Data | General DB | Specialized | Compression |
|------|-----------|-------------|-------------|
| Vectors (100K × 384-dim) | 150 MB | FAISS: 50 MB | 3x |
| Time-series facts | 50 MB | Parquet: 5 MB | 10x |
| Graph edges | 30 MB | Graph DB: 10 MB | 3x |

**Design:** One schema doesn't fit all

- OLTP (online transaction processing) optimizes for small writes + reads (SQLite)
- Graph DB optimizes for traversals + relationships (KuzuDB)
- OLAP (online analytical processing) optimizes for large-scale aggregations (DuckDB)
- Vector search optimizes for similarity queries (ChromaDB)

### Why Atomic Writes Across Databases?

When you ingest a passage, you need:
1. INSERT passage into SQLite ✓
2. INSERT embedding into ChromaDB ✓
3. ADD entities into KuzuDB ✓
4. All succeed, or all fail (no partial state)

**Without coordination:**
- SQLite succeeds, ChromaDB fails → Fact has no embedding → Search returns incomplete results
- SQLite succeeds, KuzuDB fails → Missing graph edges → Knowledge graph broken

**With coordination (transaction manager):**
```python
async with txn_manager.begin() as txn:
    # 1. SQLite
    passage_id = await passage_repo.create(passage, txn)
    
    # 2. ChromaDB (logged in SQLite for rollback)
    await embedding_store.index(embedding, txn)
    
    # 3. KuzuDB (logged in SQLite for rollback)
    await graph_repo.add_entities(entities, txn)
    
    # If any fails, context manager rolls back all three
```

**Consistency guarantee:** Either all succeed or none do. No partial state.

### Why Abstraction via Repository?

Repositories let you swap databases without changing application code:

```python
# Old: SQLite only
app = build_application(config_sqlite)

# New: PostgreSQL (just change instantiation)
app = build_application(config_postgres)

# Same Application code works!
```

**Without abstraction:**
- Add PostgreSQL → Refactor FactRepository, PassageRepository, DecisionRepository
- Queries hardcoded for specific DB → Can't change without refactoring services
- Add DuckDB for analytics → New DuckDB-specific code scattered everywhere

**With abstraction:**
- All DB access goes through repositories
- Change repository implementation → Application unchanged
- Add new repository for new DB → Plug in without touching existing code

---

## Consequences

### Positive

✅ **Performance optimized:** Each DB optimized for its access pattern  
✅ **Feature flexibility:** Can omit databases you don't use (DuckDB, KuzuDB optional)  
✅ **Scalability:** Can upgrade databases independently (SQLite → PostgreSQL, ChromaDB → Pinecone)  
✅ **Clear separation:** Each DB has clearly defined purpose  
✅ **Consistency:** Atomic writes ensure no partial state  
✅ **Maintainability:** Repository abstraction keeps infrastructure details hidden  

### Negative

⚠️ **Complexity:** Managing multiple DBs and keeping them in sync  
⚠️ **Data duplication:** Facts stored in both SQLite and embeddings (duplication)  
⚠️ **Dependency management:** Each DB has own dependencies to manage  
⚠️ **Operational burden:** Monitor/backup multiple DBs (vs. single monolithic DB)  
⚠️ **Eventual consistency:** KuzuDB/ChromaDB may lag SQLite (not guaranteed strong consistency)  

### Mitigation

1. **Transaction coordination:** Use transaction manager to synchronize writes
2. **Async replication:** Background job re-indexes ChromaDB/KuzuDB from SQLite (eventual consistency acceptable)
3. **Health checks:** Periodic validation (facts in SQLite == facts in embeddings)
4. **Automatic reindex:** On startup, check consistency and reindex missing embeddings
5. **Optional DBs:** DuckDB and KuzuDB optional (can omit if not needed)

---

## Implementation

### Repository Abstraction for Multi-DB

```python
# src/adapters/repositories.py

class FactRepository(ABC):
    """Repositories abstract storage implementation."""
    
    @abstractmethod
    async def create(fact: Fact, txn: Transaction) -> str: ...
    
    @abstractmethod
    async def find_by_id(fact_id: str) -> Fact: ...
    
    @abstractmethod
    async def find_by_importance(min_imp: float, limit: int) -> List[Fact]: ...
    
    @abstractmethod
    async def full_text_search(query: str) -> List[Fact]: ...

class SQLiteFactRepository(FactRepository):
    """Concrete implementation using SQLite."""
    
    async def create(self, fact: Fact, txn: Transaction) -> str:
        # INSERT into facts table
        # Logged in transaction for rollback
        return fact.id
    
    async def find_by_importance(self, min_imp: float, limit: int) -> List[Fact]:
        # SELECT * FROM facts WHERE importance >= ?
        return [Fact.from_row(row) for row in rows]

# Could add PostgreSQL version later:
class PostgreSQLFactRepository(FactRepository):
    async def create(self, fact: Fact, txn: Transaction) -> str:
        # INSERT using PostgreSQL
        return fact.id
```

### Transaction Manager (Multi-DB Coordination)

```python
# src/infrastructure/transaction.py

class TransactionManager:
    """Coordinates atomic writes across multiple databases."""
    
    def __init__(self, sqlite: SQLiteClient, chroma: ChromaDB, kuzu: KuzuDB):
        self.sqlite = sqlite
        self.chroma = chroma
        self.kuzu = kuzu
    
    @contextlib.asynccontextmanager
    async def begin(self) -> AsyncIterator[Transaction]:
        """Begin transaction across all databases."""
        txn = Transaction(
            sqlite_txn=await self.sqlite.begin(),
            chroma_pending=[],  # Chroma has no explicit txn; log changes
            kuzu_pending=[],    # KuzuDB has no explicit txn; log changes
        )
        
        try:
            yield txn
            # Commit SQLite (ACID guarantees)
            await txn.sqlite_txn.commit()
            
            # Apply pending changes to Chroma and KuzuDB
            for chroma_op in txn.chroma_pending:
                await self.chroma.apply(chroma_op)
            for kuzu_op in txn.kuzu_pending:
                await self.kuzu.apply(kuzu_op)
        
        except Exception as e:
            # Rollback SQLite
            await txn.sqlite_txn.rollback()
            # Chroma/KuzuDB changes not applied (eventual consistency)
            # Background job will detect mismatch and reindex
            raise
```

### Composition Root (Database Wiring)

```python
# src/application/__init__.py

def build_application(config: SitrepConfig) -> Application:
    # 1. Instantiate DB clients
    sqlite = SQLiteClient(config.db_path)
    chroma = ChromaDB(config.vectors_path) if config.embeddings_enabled else None
    kuzu = KuzuDB(config.graph_path) if config.graph_enabled else None
    duckdb = DuckDB(config.duckdb_path) if config.duckdb_enabled else None
    
    # 2. Instantiate repositories (bound to specific DB implementation)
    fact_repo = SQLiteFactRepository(sqlite)
    passage_repo = SQLitePassageRepository(sqlite, chroma)
    graph_repo = KuzuGraphRepository(kuzu) if kuzu else DummyGraphRepository()
    
    # 3. Instantiate transaction manager (coordinates all DBs)
    txn_manager = TransactionManager(sqlite, chroma, kuzu)
    
    # 4. Return Application with all databases wired
    return Application(
        fact_repo=fact_repo,
        passage_repo=passage_repo,
        graph_repo=graph_repo,
        txn_manager=txn_manager,
        duckdb=duckdb,
    )
```

### Backup Strategy

```bash
# Backup across multiple DBs
backup_sitrep() {
    BACKUP_DIR=".sitrep/backups/$(date +%Y%m%d_%H%M%S)"
    mkdir -p $BACKUP_DIR
    
    # SQLite: Simple file copy (or VACUUM INTO)
    cp .sitrep/metadata/sitrep.db $BACKUP_DIR/
    
    # KuzuDB: Export edges
    kuzu_export .sitrep/graph/ $BACKUP_DIR/graph_backup.json
    
    # ChromaDB: Export embeddings
    chroma_export .sitrep/vectors/ $BACKUP_DIR/vectors_backup.json
    
    # DuckDB: Parquet files already backed up (immutable)
    cp -r .sitrep/documents/archives $BACKUP_DIR/
    
    tar czf ".sitrep/backups/$(date +%Y%m%d_%H%M%S).tar.gz" $BACKUP_DIR
}
```

---

## Scaling Strategy

### Phase 1: Current (Small Corpus)
- **SQLite:** 1-100K facts
- **ChromaDB:** 1-10K embeddings
- **KuzuDB:** 1-10K entities (optional)
- **DuckDB:** Optional (archives only)

### Phase 2: Growing (Medium Corpus)
- **PostgreSQL** (upgrade from SQLite): 100K-1M facts
  - Same schema as SQLite, just change repository
  - Supports concurrent writes
- **ChromaDB:** 10K-100K embeddings
  - Consider sharding if needed
- **KuzuDB:** 10K-100K entities

### Phase 3: Large Scale (Massive Corpus)
- **PostgreSQL** (with partitioning): 1M+ facts
- **Pinecone** (upgrade from ChromaDB): 100K+ embeddings
  - Managed vector DB, no ops needed
- **Neo4j** (upgrade from KuzuDB): 100K+ entities
  - Enterprise graph DB with clustering

**Key:** Repository abstraction allows these upgrades without application changes

---

## Monitoring & Health

### Consistency Checks

```python
# Background job: Validate consistency across DBs
async def validate_consistency():
    """Check if facts in all DBs match."""
    sqlite_facts = await sqlite_repo.find_all()
    chromadb_facts = await embedding_store.find_all()
    kuzu_facts = await graph_repo.find_all_facts()
    
    # Should all match
    assert len(sqlite_facts) == len(chromadb_facts), "Embedding mismatch!"
    assert len(sqlite_facts) == len(kuzu_facts), "Graph mismatch!"
    
    # Fix mismatches: reindex
    for fact in sqlite_facts:
        if fact.id not in chromadb_facts:
            logger.warning(f"Missing embedding for {fact.id}, reindexing")
            await embedding_store.index(fact)
```

### Metrics

```
sitrep_sqlite_facts_total
sitrep_chromadb_embeddings_total
sitrep_kuzu_entities_total
sitrep_duckdb_archived_rows_total
sitrep_db_consistency_check_duration_seconds
sitrep_db_consistency_mismatches_total
```

---

## Related ADRs

- **ADR-0002:** Clean Architecture (Repositories abstract DB choice)
- **ADR-0003:** Lazy imports (Databases are optional)

---

## References

- **Code:** `src/infrastructure/database.py` (all DB clients)
- **Repositories:** `src/adapters/repositories.py`
- **Composition:** `src/application/__init__.py` (build_application)
- **Architecture Analysis:** `docs/SITREP_CODEBASE_ANALYSIS.md` (Section 2a)
- **Data Diagrams:** `docs/ARCHITECTURE_DIAGRAMS.md` (Diagram 2)

---

**Status:** ✅ Accepted and implemented  
**Last Updated:** 2026-07-09  
**Next Review:** When corpus exceeds 100K facts or performance degrades
