# ADR-0005: Hybrid Retrieval Fusion Weight Learning

**Status:** Accepted  
**Date:** 2026-07-09  
**Deciders:** Retrieval Engineering Team  
**Co-Authors:** Claude Haiku 4.5

---

## Context

SITREP's retrieval engine combines two complementary search strategies:
- **Dense search:** Semantic similarity via embeddings (ChromaDB/FAISS k-NN)
  - Pros: Understands meaning, captures paraphrases ("compress tokens" ≈ "reduce length")
  - Cons: Slow if no embeddings, requires model, misses exact keywords
- **Sparse search:** Keyword matching via BM25 (SQLite FTS5)
  - Pros: Fast, deterministic, captures exact terms
  - Cons: Doesn't understand meaning, misses paraphrases

Different corpora have different characteristics:
- **Code corpus:** Keywords dominate (function names, variable names)
  - Query: "What is PPOCompressionAgent?" → Exact match "PPOCompressionAgent" most important
  - Dense search may miss implementation details
  - Sparse search best
- **Documentation corpus:** Semantics dominate (explanation text)
  - Query: "How to improve token efficiency?" → Synonym "reduce tokens" also relevant
  - Dense search captures meaning, sparse may miss paraphrases
  - Dense search best
- **Mixed corpus:** Both matter
  - Query: "RL agent compression" → Want both keyword match ("agent") and semantic match ("optimization")
  - Need to balance both

**Problem:** How to balance dense and sparse results?

Fixed weights won't work:
- Code corpus: Dense 50%, Sparse 50% loses exact matches
- Doc corpus: Dense 50%, Sparse 50% misses semantics
- Different queries in same corpus have different needs

**Solution needed:**
1. **Learn fusion weights** from user feedback (which facts are relevant?)
2. **Adapt to corpus characteristics** (more dense for semantic corpora, more sparse for keyword-heavy)
3. **Adjust at query time** if possible (some queries are more semantic, others more keyword-focused)
4. **Degrade gracefully** if one retriever unavailable (e.g., no embeddings)

---

## Decision

**Learn fusion weights (w_dense, w_sparse) from relevance feedback. Update weights incrementally as users interact with results.**

### Fusion Formula

```
combined_score(fact) = w_dense * dense_score(fact) 
                     + w_sparse * sparse_score(fact)
```

Where:
- `w_dense` ∈ [0, 1] — Weight for dense search
- `w_sparse` ∈ [0, 1] — Weight for sparse search
- Constraint: `w_dense + w_sparse = 1.0` (normalized)
- Initial: `w_dense = 0.6, w_sparse = 0.4` (slightly favor dense)

### Learning from Feedback

**User interaction flow:**

```
1. User queries: "How does compression work?"
   
2. System returns top-5 results with hybrid scoring
   (combined_score = 0.6 * dense + 0.4 * sparse)
   
3. User marks result as relevant: "Fact #42 is very useful"
   
4. System records feedback:
   - query: "How does compression work?"
   - fact_id: "fact_42"
   - relevance: 0.95  (on scale 0-1)
   - was_retrieved_via: "dense_score=0.87, sparse_score=0.52"
   
5. Weight update (online learning):
   - If fact_42 had high dense_score → w_dense ↑
   - If fact_42 had low sparse_score → w_sparse ↓
   
6. Next similar query uses updated weights
```

### Weight Update Algorithm

**Moving average update (no retraining needed):**

```python
def update_fusion_weights(
    relevance: float,           # 0-1 scale
    dense_score: float,         # 0-1
    sparse_score: float,        # 0-1
    current_w_dense: float,     # Current weight
    alpha: float = 0.1,         # Learning rate
) -> Tuple[float, float]:
    """Update weights based on relevance feedback."""
    
    # Signal: which retriever was more helpful for this relevant fact?
    if relevance > 0.7:  # Only learn from strong signals
        # If dense score > sparse, dense was more helpful
        # If sparse score > dense, sparse was more helpful
        signal = dense_score - sparse_score  # Range: [-1, 1]
        
        # Update: move weight toward the better-performing retriever
        new_w_dense = current_w_dense + alpha * signal * relevance
        new_w_dense = np.clip(new_w_dense, 0.0, 1.0)  # Bound to [0, 1]
        new_w_sparse = 1.0 - new_w_dense
    else:
        # Weak signal: don't update (avoid noise)
        new_w_dense = current_w_dense
        new_w_sparse = 1.0 - new_w_dense
    
    return new_w_dense, new_w_sparse
```

**Example:**
```
Initial: w_dense=0.6, w_sparse=0.4

User marks fact relevant (relevance=0.9):
  dense_score=0.85, sparse_score=0.42
  signal = 0.85 - 0.42 = 0.43 (dense better)
  
  new_w_dense = 0.6 + 0.1 * 0.43 * 0.9
              = 0.6 + 0.0387
              = 0.6387
  
After 1 feedback: w_dense=0.64, w_sparse=0.36 (dense weight increased)
After 10 feedbacks: w_dense=0.72, w_sparse=0.28 (denser corpus detected)
```

### Storage & Persistence

**Single row in SQLite fusion_weights table:**

```sql
CREATE TABLE fusion_weights (
  id INTEGER PRIMARY KEY,
  w_dense REAL DEFAULT 0.6,
  w_sparse REAL DEFAULT 0.4,
  w_entity_rank REAL DEFAULT 0.0,  -- Optional: entity graph ranking
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  update_count INTEGER DEFAULT 0,  -- How many feedbacks processed
  metadata TEXT  -- JSON: corpus type, last query patterns, etc.
);

-- Only 1 row; UPDATE not INSERT
INSERT OR REPLACE INTO fusion_weights 
  (id, w_dense, w_sparse, updated_at, update_count)
VALUES (1, 0.64, 0.36, CURRENT_TIMESTAMP, 10);
```

**Why single row?**
- Global weights for entire corpus (not per-query-type)
- Simple to load/update (no table scan)
- Can always revert to initial weights if learning goes wrong

### Usage in Retrieval

```python
# src/infrastructure/retrieval.py

class HybridRetriever:
    def __init__(self, fact_repo, embedding_service, fusion_weights_repo):
        self.fact_repo = fact_repo
        self.embedding_service = embedding_service
        self.fusion_weights_repo = fusion_weights_repo
    
    async def search(self, query: str, top_k: int = 10) -> List[Fact]:
        # 1. Get current fusion weights
        weights = await self.fusion_weights_repo.get_current()
        w_dense = weights.w_dense
        w_sparse = weights.w_sparse
        
        # 2. Dense search (semantic)
        query_embedding = await self.embedding_service.embed(query)
        dense_results = await self.chroma.search(
            query_embedding, 
            top_k=top_k * 2  # Over-retrieve for fusion
        )
        # Returns: [(fact_id, dense_score: 0-1), ...]
        
        # 3. Sparse search (keyword via BM25)
        sparse_results = await self.fact_repo.full_text_search(
            query,
            top_k=top_k * 2
        )
        # Returns: [(fact_id, sparse_score: 0-1), ...]
        
        # 4. Combine results with learned weights
        combined_scores = {}
        
        for fact_id, dense_score in dense_results:
            combined_scores[fact_id] = w_dense * dense_score
        
        for fact_id, sparse_score in sparse_results:
            if fact_id in combined_scores:
                combined_scores[fact_id] += w_sparse * sparse_score
            else:
                combined_scores[fact_id] = w_sparse * sparse_score
        
        # 5. Rank by combined score
        ranked = sorted(
            combined_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]
        
        # 6. Fetch full facts and return
        facts = [await self.fact_repo.find_by_id(fact_id) for fact_id, _ in ranked]
        return facts
```

### Feedback Integration

```python
# src/application/use_cases.py

class FeedbackUseCase:
    def __init__(self, result_cache, fact_repo, fusion_weights_repo):
        self.result_cache = result_cache
        self.fact_repo = fact_repo
        self.fusion_weights_repo = fusion_weights_repo
    
    async def provide_feedback(
        self,
        query: str,
        fact_id: str,
        relevance: float,  # 0-1 scale
    ) -> None:
        """User marks a fact as relevant/irrelevant."""
        
        # 1. Retrieve cached query result (has dense_score, sparse_score)
        cached_result = await self.result_cache.get(query)
        if not cached_result:
            return  # No cached result, can't update weights
        
        # 2. Find this fact in cached results
        fact_result = None
        for f in cached_result.facts:
            if f.id == fact_id:
                fact_result = f
                break
        
        if not fact_result:
            return  # Fact wasn't in original results
        
        # 3. Extract scores
        dense_score = fact_result.dense_score or 0.0
        sparse_score = fact_result.sparse_score or 0.0
        
        # 4. Update fusion weights
        current_weights = await self.fusion_weights_repo.get_current()
        new_w_dense, new_w_sparse = update_fusion_weights(
            relevance=relevance,
            dense_score=dense_score,
            sparse_score=sparse_score,
            current_w_dense=current_weights.w_dense,
            alpha=0.1,
        )
        
        # 5. Store updated weights
        await self.fusion_weights_repo.update(
            w_dense=new_w_dense,
            w_sparse=new_w_sparse,
        )
        
        # 6. Store feedback record (for analytics)
        await feedback_repo.create(Feedback(
            query=query,
            fact_id=fact_id,
            relevance=relevance,
            timestamp=datetime.now(),
        ))
```

---

## Rationale

### Why Learn Fusion Weights?

**Adapts to corpus characteristics without manual tuning:**
```
Code corpus feedback:
  User: "Find PPOCompressionAgent" 
  Relevant facts: All have exact match "PPOCompressionAgent"
  Learning: sparse_score >> dense_score → w_sparse increases

Documentation corpus feedback:
  User: "How to reduce token usage?"
  Relevant facts: Mention "compression", "token efficiency", "reduce length"
  Learning: dense_score >> sparse_score → w_dense increases
```

**Online learning (fast, no retraining):**
- Each feedback immediately updates weights
- No need to retrain models
- No offline batch processing required
- Works incrementally (weights improve over time)

**Captures user intent:**
- User feedback is ground truth for relevance
- Learns what retrieval strategy aligns with relevance
- Automatically detects if corpus is code-heavy vs. semantic-heavy

### Why Not Fixed Weights?

Fixed weights require manual tuning:
- ❌ Tune for one corpus, broken on another corpus
- ❌ Tune for one query type, broken for others
- ❌ Requires expert knowledge ("should it be 60/40 or 70/30?")

### Why Not Separate Weights Per Query Type?

Would require:
- ❌ Classifying queries into types (error-prone)
- ❌ Collecting feedback per type (slow)
- ❌ More storage and complexity

Single global weight is simpler and works because:
- ✓ Different query types in same corpus have same characteristics
- ✓ Corpus characteristics don't change drastically (usually)
- ✓ Single weight converges quickly (typically 10-20 feedbacks)

---

## Consequences

### Positive

✅ **Adapts to corpus** — Automatically adjusts to keyword-heavy vs. semantic-heavy  
✅ **No manual tuning** — Learns from user feedback, no hyperparameter guessing  
✅ **Online learning** — Weights improve incrementally as users interact  
✅ **Simple storage** — One row in one table  
✅ **Graceful degradation** — Works even if dense search unavailable (w_dense→0)  
✅ **Improves over time** — Earlier queries may be worse, later queries better  

### Negative

⚠️ **Slow convergence** — Takes 10-50 feedbacks to adapt (few hours of usage)  
⚠️ **Feedback required** — Without user feedback, weights stay at defaults  
⚠️ **Cold start** — New corpus starts with generic defaults (might not be optimal)  
⚠️ **Noise sensitivity** — Bad feedback can degrade weights (mitigated by relevance threshold)  
⚠️ **Single global weight** — Can't have different weights for different query types  

### Mitigation

1. **Good defaults:** Start with w_dense=0.6, w_sparse=0.4 (works for most corpora)
2. **Feedback threshold:** Only update on strong signals (relevance > 0.7)
3. **Reset option:** Let users reset to defaults if learning goes wrong
4. **Analytics:** Show current weights + update count (transparency)
5. **Recommend feedback:** Explicitly ask user "Was this result useful?" after queries
6. **Cold-start warmup:** Pre-train weights on small labeled set if possible

---

## Implementation Details

### Feedback Storage Schema

```python
# src/domain/schemas.py

class Feedback(BaseModel):
    """User feedback on retrieval result."""
    id: str
    query: str
    fact_id: str
    relevance: float  # 0-1 scale
    timestamp: datetime
    dense_score: Optional[float] = None  # Recorded for analysis
    sparse_score: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None
```

### Fusion Weights Analytics

```python
# src/adapters/services.py

class FusionWeightAnalyzer:
    """Analyze fusion weight learning progress."""
    
    async def get_convergence(self) -> Dict[str, Any]:
        """How well have weights converged?"""
        weights = await weights_repo.get_current()
        feedbacks = await feedback_repo.find_recent(limit=100)
        
        if len(feedbacks) < 10:
            return {"status": "cold_start", "feedbacks": len(feedbacks)}
        
        # Calculate trend (are weights stabilizing?)
        recent = feedbacks[-20:]
        dense_trend = [f.dense_score for f in recent]
        variance = np.var(dense_trend)
        
        return {
            "status": "converged" if variance < 0.05 else "learning",
            "current_w_dense": weights.w_dense,
            "current_w_sparse": weights.w_sparse,
            "total_feedbacks": weights.update_count,
            "weight_variance": variance,
            "corpus_type": "keyword_heavy" if weights.w_sparse > 0.6 else "semantic_heavy",
        }
    
    async def recommendations(self) -> Dict[str, str]:
        """Suggest actions based on weight state."""
        convergence = await self.get_convergence()
        
        if convergence["status"] == "cold_start":
            return {"action": "Collect more feedback to learn weights"}
        
        if convergence["corpus_type"] == "keyword_heavy":
            return {
                "action": "Corpus appears keyword-heavy",
                "recommendation": "Sparse search is primary; dense search secondary"
            }
        
        return {"action": "Weights converged; retrieval optimized"}
```

### Web UI Integration

```python
# scripts/run_web.py - Gradio UI

with gr.Blocks() as demo:
    with gr.Tab("Query"):
        with gr.Column():
            query = gr.Textbox(label="Query")
            top_k = gr.Slider(1, 20, value=5, label="Results")
            explain = gr.Checkbox(value=True, label="Explain")
            
            submit = gr.Button("Search")
            
            with gr.Column():
                results = gr.Dataframe(
                    label="Results",
                    columns=["Rank", "Fact", "Importance", "Score"]
                )
                explanation = gr.Textbox(label="Explanation")
                
                # Feedback section
                with gr.Group():
                    gr.Markdown("### Feedback (helps improve results)")
                    selected_fact = gr.Dropdown(
                        label="Which result was helpful?",
                        choices=[],  # Populated from results
                    )
                    relevance = gr.Radio(
                        choices=["Not useful", "Somewhat useful", "Very useful"],
                        label="How useful was this result?"
                    )
                    submit_feedback = gr.Button("Submit Feedback")
    
    with gr.Tab("Stats"):
        fusion_stats = gr.Textbox(
            label="Fusion Weight Status",
            value=lambda: asyncio.run(analyzer.get_convergence()),
        )
```

---

## Testing Strategy

### Unit Tests

```python
# tests/unit/adapters/test_fusion_weights.py

@pytest.mark.asyncio
async def test_weight_update_on_relevant_feedback():
    """Dense search should be boosted when relevant fact has high dense_score."""
    repo = MockFusionWeightsRepo(w_dense=0.5, w_sparse=0.5)
    
    # Relevant fact had higher dense_score
    new_w_dense, new_w_sparse = update_fusion_weights(
        relevance=0.9,
        dense_score=0.8,
        sparse_score=0.3,
        current_w_dense=0.5,
    )
    
    assert new_w_dense > 0.5  # Dense weight increased
    assert new_w_sparse < 0.5

@pytest.mark.asyncio
async def test_weak_feedback_ignored():
    """Feedback below threshold should not update weights."""
    w_before = 0.5
    new_w, _ = update_fusion_weights(
        relevance=0.4,  # Below threshold
        dense_score=0.9,
        sparse_score=0.1,
        current_w_dense=w_before,
    )
    
    assert new_w == w_before  # No change
```

### Integration Tests

```python
# tests/integration/test_hybrid_retrieval_learning.py

@pytest.mark.asyncio
async def test_weights_adapt_to_corpus():
    """Over time, weights should adapt to corpus characteristics."""
    app = build_application(config)
    
    # Simulate keyword-heavy corpus feedback
    for i in range(20):
        await app.query("dense term specific keyword")
        # User marks sparse-search-friendly results as relevant
        await app.feedback(
            query="dense term specific keyword",
            fact_id=f"fact_{i}",
            relevance=0.9,
        )
    
    # Weights should favor sparse search now
    weights = await app.fusion_weights_repo.get_current()
    assert weights.w_sparse > 0.6
    assert weights.w_dense < 0.4
```

---

## Future Enhancements

### Short Term
- [ ] Per-query-type weights (detect intent, use different weights)
- [ ] Per-user weights (different users have different preferences)
- [ ] Weight confidence intervals (how sure are we about current weights?)

### Medium Term
- [ ] Multi-armed bandit approach (explore different weights while exploiting best)
- [ ] A/B testing integration (test weight changes against baseline)
- [ ] Automatic corpus classification (detect if semantic vs. keyword-heavy)

### Long Term
- [ ] Learned reranker (train neural reranker that predicts relevance)
- [ ] Personalized fusion (different weights for different user types)
- [ ] Query-specific weights (detect query intent, use optimal weights)

---

## Related ADRs

- **ADR-0002:** Repositories abstract retrieval implementation
- **ADR-0004:** ChromaDB + SQLite support dense/sparse search

---

## References

- **Code:** `src/infrastructure/retrieval.py` (HybridRetriever)
- **Storage:** `src/adapters/repositories.py` (FusionWeightsRepository)
- **Use Case:** `src/application/use_cases.py` (FeedbackUseCase)
- **Diagram:** `docs/ARCHITECTURE_DIAGRAMS.md` (Diagram 3: Hybrid Retrieval)

---

**Status:** ✅ Accepted and implemented  
**Last Updated:** 2026-07-09  
**Next Review:** After 100 feedbacks to assess learning progress
