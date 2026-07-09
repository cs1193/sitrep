# ADR-0007: Memory Consolidation and Forgetting Policy

**Status:** Accepted  
**Date:** 2026-07-09  
**Deciders:** Memory Management Team  
**Co-Authors:** Claude Haiku 4.5

---

## Context

As SITREP ingests documents and users query facts, the database grows:
- Day 1: 100 facts
- Week 1: 1,000 facts
- Month 1: 10,000 facts
- Year 1: 100,000+ facts

Storage and query performance both degrade over time:
- Disk usage grows (vectors, embeddings, full text indices)
- Query latency increases (more data to search)
- Maintenance overhead increases (backup, reindex, analyze)

Without cleanup, the system becomes:
- **Bloated:** Storing facts nobody accesses anymore
- **Slow:** Old irrelevant facts pollute search results
- **Expensive:** More disk, more memory, slower backups

But facts shouldn't be permanently deleted:
- Might become relevant again (e.g., "compression" irrelevant today, critical next month)
- Need audit trail (compliance, debugging)
- Should be recoverable (reversible)

**Problem:** How to distinguish important facts worth keeping from forgettable noise?

**Key challenge:** No explicit "importance" signal in most corpora
- User doesn't label facts
- No explicit feedback
- Only implicit signals: access frequency, recency, semantic similarity to new queries

**Strategies seen in literature:**

1. **LRU (Least Recently Used)** — Delete facts not accessed in 30 days
   - ✓ Simple
   - ✗ Ignores inherent importance (might delete important-but-unused facts)

2. **Access frequency** — Keep top-K most-accessed facts
   - ✓ Objective
   - ✗ Biased toward early facts (accumulate more accesses)

3. **Importance scoring** — Define importance = f(access_frequency, recency, novelty, semantic_similarity)
   - ✓ Holistic
   - ✗ Complex, requires tuning

4. **Manual curation** — Users explicitly mark facts as keep/delete
   - ✓ Accurate
   - ✗ Labor-intensive

---

## Decision

**Use multi-signal importance scoring with soft-delete archival. Periodically consolidate low-importance facts without hard-deleting.**

### Importance Scoring Formula

```
importance(fact) = α * access_frequency_score(fact)
                 + β * recency_score(fact)
                 + γ * semantic_novelty_score(fact)
                 + δ * user_signal_score(fact)
```

Where:
- `access_frequency_score` ∈ [0, 1]: Normalized access count
- `recency_score` ∈ [0, 1]: Time decay (recent = high)
- `semantic_novelty_score` ∈ [0, 1]: Uniqueness in corpus
- `user_signal_score` ∈ [0, 1]: Explicit feedback (if available)
- Weights: α=0.25, β=0.25, γ=0.25, δ=0.25 (equal weight, tunable)

### Signal 1: Access Frequency

```python
def access_frequency_score(fact: Fact, corpus_stats: CorpusStats) -> float:
    """
    How often is this fact accessed?
    
    Normalized to max access count in corpus (prevents saturation).
    """
    max_accesses = corpus_stats.max_access_count or 100
    
    return min(1.0, fact.access_count / max_accesses)

# Example:
# fact.access_count = 50, max in corpus = 200
# score = 50 / 200 = 0.25 (moderately accessed)
```

### Signal 2: Recency (Time Decay)

```python
def recency_score(fact: Fact, now: datetime, half_life: timedelta) -> float:
    """
    How recently was this fact accessed?
    
    Exponential decay: importance_decay(t) = 2^(-t / half_life)
    
    half_life: Time for importance to drop to 50%
              Tunable (default: 30 days)
    """
    age = now - fact.last_accessed
    
    if fact.last_accessed is None:
        # Never accessed: conservative, assume important
        creation_age = now - fact.created_at
        return 2.0 ** (-creation_age / half_life)
    
    decay = 2.0 ** (-(age / half_life))
    return min(1.0, decay)

# Example (half_life=30 days):
# Accessed 1 day ago: 2^(-1/30) = 0.977 (almost important as today)
# Accessed 30 days ago: 2^(-30/30) = 0.5 (50% importance loss)
# Accessed 90 days ago: 2^(-90/30) = 0.125 (87.5% importance loss)
```

### Signal 3: Semantic Novelty

```python
def semantic_novelty_score(fact: Fact, corpus: List[Fact]) -> float:
    """
    How unique is this fact in the corpus?
    
    Measures Jaccard similarity to all other facts.
    High novelty = low similarity to others = worth keeping.
    """
    # Compute Jaccard similarity to 10 nearest neighbors
    embedding = fact.embedding  # 384-dim from sentence-transformers
    
    similarities = []
    for other in corpus:
        if other.id == fact.id:
            continue
        sim = cosine_similarity(embedding, other.embedding)
        similarities.append(sim)
    
    # Average similarity to neighbors
    if not similarities:
        return 1.0  # Only fact in corpus = perfectly novel
    
    avg_similarity = np.mean(similarities[:10])  # Top 10 neighbors
    
    # Invert: high similarity → low novelty → low score
    novelty = 1.0 - avg_similarity
    
    return novelty

# Example:
# fact.embedding very similar to 10 others: avg_sim=0.9 → novelty=0.1 (low)
# fact.embedding unique: avg_sim=0.4 → novelty=0.6 (high)
```

### Signal 4: User Feedback

```python
def user_signal_score(fact: Fact, feedback_repo) -> float:
    """
    Has user explicitly marked this as relevant?
    
    If user marked relevant multiple times: score increases
    If user never interacted: score = 0.5 (neutral)
    """
    feedbacks = await feedback_repo.find_by_fact_id(fact.id)
    
    if not feedbacks:
        return 0.5  # Neutral (unknown)
    
    avg_relevance = np.mean([f.relevance for f in feedbacks])
    
    return avg_relevance  # 0-1 from user ratings

# Example:
# User marked relevant 5x with avg relevance 0.9: score=0.9 (high)
# User never marked: score=0.5 (neutral)
```

### Combined Score & Action

```python
async def compute_importance(
    fact: Fact,
    corpus_stats: CorpusStats,
    corpus: List[Fact],
    feedback_repo,
) -> float:
    """Compute multi-signal importance score [0, 1]."""
    
    now = datetime.now()
    
    scores = {
        "access": access_frequency_score(fact, corpus_stats),
        "recency": recency_score(fact, now, half_life=timedelta(days=30)),
        "novelty": semantic_novelty_score(fact, corpus),
        "user": await user_signal_score(fact, feedback_repo),
    }
    
    importance = (
        0.25 * scores["access"]
        + 0.25 * scores["recency"]
        + 0.25 * scores["novelty"]
        + 0.25 * scores["user"]
    )
    
    return importance

# Action logic:
importance = await compute_importance(fact, ...)

if importance < 0.2:
    # Archive: Mark archived=True, exclude from retrieval
    await fact_repo.archive(fact.id)
    logger.info(f"Archived {fact.id} (importance={importance:.2f})")

elif 0.2 <= importance < 0.5:
    # Gray zone: Keep but don't index in dense search (BM25 only)
    await chroma.delete(fact.id)  # Remove embedding
    logger.info(f"De-indexed {fact.id} (importance={importance:.2f})")

else:
    # Keep fully indexed
    await chroma.ensure_indexed(fact.id)
```

### Consolidation Policies

**Three levels of archival (reversible):**

```python
class ArchivalLevel(Enum):
    ACTIVE = "active"           # Full index (SQLite + ChromaDB + KuzuDB)
    ARCHIVED_SOFT = "archived_soft"    # In SQLite, not indexed
    ARCHIVED_HARD = "archived_hard"    # In Parquet, not in active DB

async def consolidate_memory():
    """Periodically clean up low-importance facts."""
    
    # 1. Compute importance for all facts
    facts = await fact_repo.find_all(archived=False)
    importances = []
    
    for fact in facts:
        imp = await compute_importance(fact, ...)
        importances.append((fact.id, imp))
    
    # 2. Archive low-importance facts (soft-delete)
    archived_count = 0
    for fact_id, importance in importances:
        if importance < 0.2:
            await fact_repo.update(fact_id, archived=True)
            archived_count += 1
    
    # 3. Move very old archived facts to cold storage (Parquet)
    # Facts archived >6 months ago
    cutoff_date = datetime.now() - timedelta(days=180)
    old_archived = await fact_repo.find_archived_before(cutoff_date)
    
    for fact in old_archived:
        await duckdb_client.append_to_archive(fact)
        await fact_repo.delete_permanently(fact.id)  # OK: in archive
    
    logger.info(f"Consolidated: archived {archived_count}, cold-stored {len(old_archived)}")

# Schedule: Run daily at 3 AM (low-traffic time)
# Async: Doesn't block queries
```

### Recovery (Reversible)

```python
async def recover_fact(fact_id: str):
    """Unarchive a fact (bring back to active)."""
    
    # 1. Check if in cold storage (Parquet)
    archived_fact = await duckdb_client.find_in_archive(fact_id)
    
    if archived_fact:
        # Restore from Parquet
        fact = Fact.from_dict(archived_fact)
        await fact_repo.create(fact)
        logger.info(f"Recovered {fact_id} from cold storage")
    else:
        # Restore from soft-delete (already in SQLite)
        await fact_repo.update(fact_id, archived=False)
        logger.info(f"Unarchived {fact_id}")
    
    # 2. Re-index in embeddings
    embedding = await embedding_service.embed(fact.text)
    await chroma.index(fact_id, embedding)
    
    # 3. Re-add to graph
    entities = await entity_extractor.extract(fact.text)
    await kuzu.add_nodes(entities)
    
    # 4. Reset access count (treat as new)
    await fact_repo.update(fact_id, access_count=0, last_accessed=datetime.now())
```

---

## Rationale

### Why Multi-Signal Scoring?

Single signal fails:
- **Access frequency alone:** Biased toward facts accessed early (accumulate count over time)
- **Recency alone:** Bias toward recently-accessed facts (old important facts discarded)
- **Novelty alone:** Keep duplicate facts, discard common ones (wrong)
- **User signal alone:** No signal if user never interacts (most facts)

Multi-signal balances:
- ✓ Access frequency: Popular facts usually important
- ✓ Recency: Recent access suggests ongoing relevance
- ✓ Novelty: Unique facts less redundant
- ✓ User feedback: Direct signal when available

### Why Soft-Delete (Archival)?

Hard delete problems:
- ✗ Irreversible (can't recover if mistaken)
- ✗ Violates audit trail (compliance issues)
- ✗ Breaks lineage (decision references deleted fact)

Soft delete benefits:
- ✓ Reversible (can unarchive if needed)
- ✓ Maintains audit trail (archived_at timestamp)
- ✓ Keeps lineage intact
- ✓ Transparent (can query archived facts if needed)

### Why Exponential Time Decay?

Linear decay problems:
```
Linear: importance_decay(t) = 1 - (t / 90 days)
  Day 0: 1.0
  Day 45: 0.5
  Day 90: 0.0 (immediate obsolescence)
```

Exponential benefits:
```
Exponential: importance_decay(t) = 2^(-t / 30 days)
  Day 0: 1.0
  Day 30: 0.5
  Day 60: 0.25
  Day 90: 0.125 (asymptotic, never quite 0)
```

Exponential makes sense:
- Recent access more important than old
- But very old facts don't instantly become useless
- Allows facts to persist if occasionally accessed

### Why Async Consolidation?

Consolidation in background (not blocking queries):
- ✓ Queries unaffected
- ✓ Can run during low-traffic hours
- ✓ Doesn't spike latency

---

## Consequences

### Positive

✅ **Keeps database lean** — Automatically archives low-value facts  
✅ **Improves performance** — Fewer facts → faster searches  
✅ **Reduces storage** — Archived facts in cold storage (Parquet)  
✅ **Reversible** — Can unarchive facts if needed  
✅ **No manual work** — Automatic, no user involvement needed  
✅ **Audit-friendly** — Maintains lineage, timestamps, decision records  

### Negative

⚠️ **Complex scoring** — Multiple signals, weights to tune  
⚠️ **Tuning challenge** — Different corpora need different weights  
⚠️ **Cold start problem** — New facts (no access history) treated as low importance  
⚠️ **Potential data loss** — Might archive important facts (mitigated by reversibility)  
⚠️ **Storage overhead** — Parquet archives duplicate data (trade-off with cleanup)  

### Mitigation

1. **Conservative thresholds:** Archive only if importance < 0.2 (high bar)
2. **Regular audits:** Review archived facts monthly, adjust weights if needed
3. **User override:** Let users mark facts as "never archive"
4. **Backup before archival:** Always backup before cold storage migration
5. **Monitoring:** Alert if archival rate exceeds threshold (e.g., >10% per month)

---

## Implementation

### Storage Layout

```
Active facts: SQLite + ChromaDB + KuzuDB
├── ACTIVE tier: importance ≥ 0.5
│   └─ Fully indexed (text, embeddings, graph)
├── GRAY tier: 0.2 ≤ importance < 0.5
│   └─ Text indexed (BM25 only, no embeddings)
└─ ARCHIVED tier: importance < 0.2
   └─ In SQLite but archived=True, not searchable

Cold storage: DuckDB Parquet
└─ ARCHIVED_COLD: older than 180 days
   └─ Parquet immutable archive, not in active DB
```

### Consolidation Schedule

```python
# In application startup or cron job

async def schedule_consolidation(app):
    """Schedule daily consolidation at 3 AM."""
    scheduler = BackgroundScheduler()
    
    scheduler.add_job(
        func=app.consolidate_memory,
        trigger="cron",
        hour=3,
        minute=0,
        max_instances=1,  # Don't run twice
        misfire_grace_time=3600,
    )
    
    scheduler.start()
```

### Telemetry

```python
# Track archival behavior

class ArchivalMetrics:
    """Monitor consolidation health."""
    
    async def get_status(self) -> Dict:
        """Return current archival status."""
        total_facts = await fact_repo.count(archived=False)
        archived_facts = await fact_repo.count(archived=True)
        cold_facts = await duckdb.count_archived()
        
        importance_distribution = await self._compute_distribution()
        
        return {
            "active_facts": total_facts,
            "archived_facts": archived_facts,
            "cold_storage_facts": cold_facts,
            "archival_rate_%": 100 * archived_facts / total_facts,
            "importance_median": importance_distribution["median"],
            "importance_p95": importance_distribution["p95"],
            "last_consolidation": await self._get_last_run(),
        }
```

### Testing

```python
# tests/unit/adapters/test_importance_scoring.py

@pytest.mark.asyncio
async def test_importance_decreases_with_time():
    """Older facts should have lower importance."""
    fact_old = Fact(
        id="f1",
        last_accessed=datetime.now() - timedelta(days=90),
    )
    fact_new = Fact(
        id="f2",
        last_accessed=datetime.now(),
    )
    
    imp_old = recency_score(fact_old, datetime.now(), timedelta(days=30))
    imp_new = recency_score(fact_new, datetime.now(), timedelta(days=30))
    
    assert imp_old < imp_new

@pytest.mark.asyncio
async def test_consolidation_archives_low_importance():
    """Facts with importance < 0.2 should be archived."""
    facts = [mock_fact(importance=0.1), mock_fact(importance=0.5)]
    
    await consolidate_memory(facts)
    
    assert await fact_repo.is_archived(facts[0].id)
    assert not await fact_repo.is_archived(facts[1].id)

@pytest.mark.asyncio
async def test_recovery_unarchives_fact():
    """Unarchive should restore soft-deleted facts."""
    fact = mock_fact()
    await fact_repo.archive(fact.id)
    
    await recover_fact(fact.id)
    
    assert not await fact_repo.is_archived(fact.id)
```

---

## Future Enhancements

### Short Term
- [ ] User-defined "never archive" list (keep certain facts permanently)
- [ ] Importance visualization (show why fact was archived)
- [ ] Manual recovery UI (unarchive facts with one click)

### Medium Term
- [ ] Learned archival policy (train model on manual keep/delete decisions)
- [ ] Query-specific importance (recent queries boost related facts)
- [ ] Temporal patterns (some facts become important seasonally)

### Long Term
- [ ] Automated weight tuning (optimize weights on real archival/query data)
- [ ] Hierarchical consolidation (multiple archival tiers with different costs)
- [ ] Fact summarization (summarize archived facts to preserve knowledge)

---

## Related ADRs

- **ADR-0004:** Multi-database approach enables tiered storage
- **ADR-0002:** Repositories abstract archival logic

---

## References

- **Code:** `src/adapters/services.py` (ImportanceScorer, ConsolidationService)
- **Storage:** `src/infrastructure/database.py` (ArchiveRepository)
- **Scheduler:** `src/application/__init__.py` (build_application includes scheduler)
- **Metrics:** `src/utils/metrics.py` (ArchivalMetrics)

---

**Status:** ✅ Accepted and implemented  
**Last Updated:** 2026-07-09  
**Next Review:** After 90 days, assess archival rate + user feedback on recovered facts
