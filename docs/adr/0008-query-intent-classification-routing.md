# ADR-0008: Query Intent Classification and Routing

**Status:** Accepted  
**Date:** 2026-07-09  
**Deciders:** Query Engineering Team  
**Co-Authors:** Claude Haiku 4.5

---

## Context

Users query SITREP with many different intentions:

```
"What is SITREP?" 
  → SIMPLE: Direct lookup, return matching facts

"How does SITREP differ from traditional RAG?"
  → COMPARISON: Compare two things, highlight differences

"How does compression reduce tokens without losing meaning?"
  → CAUSATION: Explain cause-effect relationship

"When did we implement PPO compression?"
  → TEMPORAL: Find facts within time range

"Did implementing RL compression improve retrieval accuracy?"
  → CAUSAL_COUNTERFACTUAL: Estimate effect of intervention

"Summarize compression strategies and their trade-offs"
  → SYNTHESIS: Combine multiple facts into coherent answer

"Find facts about compression that are NOT about RL"
  → NEGATION: Filter facts with exclusion constraints

"Which compression strategy is best for JSON?"
  → CONTEXTUAL: Answer depends on domain (JSON vs. code vs. logs)
```

Different intent types need different retrieval strategies:

| Intent | Optimal Retrieval | Why |
|--------|------------------|-----|
| SIMPLE | Hybrid (dense + sparse) | Direct keyword + semantic match |
| COMPARISON | Entity-graph + dense | Find related concepts, compare |
| CAUSATION | Causal graph + temporal | Link cause → effect, timeline |
| TEMPORAL | Temporal filtering + dense | Date range + semantic match |
| COUNTERFACTUAL | Causal reasoning + RL | Do-calculus, simulate effect |
| SYNTHESIS | Dense + importance + novelty | Find key facts, avoid redundant |
| NEGATION | Sparse inverse + dense | Keyword NOT + semantic |
| CONTEXTUAL | Domain-aware compression | Compress based on query domain |

**Problem:** Without routing, all queries use same retrieval strategy (hybrid search)
- Simple queries: Works fine
- Temporal queries: Misses time constraints
- Causal queries: Misses causal links
- Comparison queries: Returns semantically similar but doesn't compare
- Counterfactual: Can't estimate effects

**Solution needed:**
1. Classify query intent automatically
2. Route to specialized retriever
3. Rerank results based on intent
4. Generate intent-specific explanations

---

## Decision

**Implement multi-intent query classifier that detects intent, routes to specialized retrievers, and adapts ranking/explanation.**

### Intent Classification

**Classification approach:**

```python
class QueryIntentClassifier:
    """Classify query intent using pattern matching + semantic similarity."""
    
    # Hand-coded patterns for high-confidence detection
    PATTERNS = {
        "TEMPORAL": {
            "keywords": ["when", "date", "time", "before", "after", "during"],
            "templates": [
                r"when (did|was|were) .+",
                r"(before|after|during) .+ (did|happened)",
            ]
        },
        "COMPARISON": {
            "keywords": ["compare", "difference", "vs", "versus", "better", "worse"],
            "templates": [
                r"(compare|difference) .+ (and|with|vs)",
                r"(how|why) .+ (different|differ)",
            ]
        },
        "CAUSATION": {
            "keywords": ["cause", "effect", "why", "result", "led to", "because"],
            "templates": [
                r"why (did|does) .+",
                r"(cause|effect) of .+",
            ]
        },
        "NEGATION": {
            "keywords": ["not", "without", "exclude", "except", "minus"],
            "templates": [
                r".+ (not|without) .+",
            ]
        },
        "SYNTHESIS": {
            "keywords": ["summarize", "overview", "overview", "summary", "list"],
            "templates": [
                r"summarize .+",
                r"list (all|the) .+",
            ]
        },
    }
    
    async def classify(self, query: str) -> Tuple[Intent, float]:
        """
        Classify query intent.
        
        Returns: (Intent, confidence: 0-1)
        """
        query_lower = query.lower()
        
        # 1. Pattern matching (high confidence)
        for intent_name, patterns in self.PATTERNS.items():
            for keyword in patterns["keywords"]:
                if keyword in query_lower:
                    confidence = 0.9  # High confidence
                    return Intent[intent_name], confidence
            
            for template in patterns["templates"]:
                if re.search(template, query_lower):
                    confidence = 0.95
                    return Intent[intent_name], confidence
        
        # 2. Semantic similarity (lower confidence)
        intent_examples = {
            "SIMPLE": [
                "What is X?",
                "Tell me about X",
                "Define X",
            ],
            "COMPARISON": [
                "Compare X and Y",
                "How are X and Y different?",
            ],
            # ... more examples
        }
        
        query_embedding = await embedding_service.embed(query)
        best_intent = None
        best_score = 0.0
        
        for intent, examples in intent_examples.items():
            example_embeddings = await embedding_service.embed_batch(examples)
            similarities = [
                cosine_similarity(query_embedding, ex_emb)
                for ex_emb in example_embeddings
            ]
            avg_sim = np.mean(similarities)
            
            if avg_sim > best_score:
                best_score = avg_sim
                best_intent = intent
        
        confidence = best_score * 0.7  # Lower confidence than patterns
        
        return Intent[best_intent], confidence
```

**Intent enum:**

```python
class Intent(Enum):
    SIMPLE = "simple"                       # Direct lookup
    COMPARISON = "comparison"               # Compare entities
    CAUSATION = "causation"                 # Cause-effect
    TEMPORAL = "temporal"                   # Time-based
    COUNTERFACTUAL = "counterfactual"       # Effect estimation
    SYNTHESIS = "synthesis"                 # Multi-fact summary
    NEGATION = "negation"                   # Exclusion constraints
    CONTEXTUAL = "contextual"               # Domain-dependent
    UNKNOWN = "unknown"                     # Fallback
```

### Routing Strategy

```python
class IntentRouter:
    """Route queries to specialized retrievers based on intent."""
    
    async def retrieve(
        self,
        query: str,
        intent: Intent,
        top_k: int = 5,
    ) -> List[Fact]:
        """Route query to appropriate retriever."""
        
        if intent == Intent.SIMPLE:
            return await self._retrieve_simple(query, top_k)
        
        elif intent == Intent.TEMPORAL:
            return await self._retrieve_temporal(query, top_k)
        
        elif intent == Intent.COMPARISON:
            return await self._retrieve_comparison(query, top_k)
        
        elif intent == Intent.CAUSATION:
            return await self._retrieve_causation(query, top_k)
        
        elif intent == Intent.COUNTERFACTUAL:
            return await self._retrieve_counterfactual(query, top_k)
        
        elif intent == Intent.SYNTHESIS:
            return await self._retrieve_synthesis(query, top_k)
        
        elif intent == Intent.NEGATION:
            return await self._retrieve_negation(query, top_k)
        
        elif intent == Intent.CONTEXTUAL:
            return await self._retrieve_contextual(query, top_k)
        
        else:  # UNKNOWN
            return await self._retrieve_simple(query, top_k)
    
    async def _retrieve_simple(self, query: str, top_k: int) -> List[Fact]:
        """Direct hybrid search (dense + sparse)."""
        return await self.hybrid_retriever.search(query, top_k)
    
    async def _retrieve_temporal(self, query: str, top_k: int) -> List[Fact]:
        """Extract time constraints, filter by temporal relationships."""
        
        # Extract dates from query: "between 2025-06 and 2025-07"
        time_range = extract_time_range(query)
        
        # Hybrid search
        facts = await self.hybrid_retriever.search(query, top_k * 2)
        
        # Filter by temporal constraints
        if time_range:
            facts = [f for f in facts if temporal_overlap(f.time_range, time_range)]
        
        return facts[:top_k]
    
    async def _retrieve_comparison(self, query: str, top_k: int) -> List[Fact]:
        """Extract entities being compared, find facts about each."""
        
        # Extract "X and Y" from "Compare X and Y"
        entities = extract_entities_for_comparison(query)
        
        # Retrieve facts about each entity
        all_facts = []
        for entity in entities:
            facts = await self.hybrid_retriever.search(entity, top_k)
            all_facts.extend(facts)
        
        # Deduplicate and rank by relevance to original query
        unique_facts = deduplicate_by_id(all_facts)
        ranked = await self.reranker.rerank(unique_facts, query)
        
        return ranked[:top_k]
    
    async def _retrieve_causation(self, query: str, top_k: int) -> List[Fact]:
        """Use causal graph to find cause-effect chains."""
        
        # Extract cause/effect from "Why did X happen?" or "What caused X?"
        effect = extract_effect_entity(query)
        
        # Find causal parents in decision graph
        causal_facts = await self.graph_retriever.find_causal_chain(effect, depth=3)
        
        return causal_facts[:top_k]
    
    async def _retrieve_counterfactual(self, query: str, top_k: int) -> List[Fact]:
        """Estimate effect of intervention using do-calculus."""
        
        # Extract intervention and outcome: "If we didn't use RL, would accuracy..."
        intervention = extract_intervention(query)
        outcome = extract_outcome(query)
        
        # Use RL agent + causal graph to estimate effect
        effect_estimate = await self.causal_reasoner.estimate_effect(intervention, outcome)
        
        # Return facts supporting the estimate
        supporting_facts = await self.hybrid_retriever.search(
            f"{intervention} {outcome}",
            top_k,
        )
        
        return supporting_facts
    
    async def _retrieve_synthesis(self, query: str, top_k: int) -> List[Fact]:
        """Find diverse, complementary facts for synthesis."""
        
        # Retrieve more facts than usual (synthesis needs depth)
        facts = await self.hybrid_retriever.search(query, top_k * 3)
        
        # Rank by: importance + novelty + recency (avoid redundant summaries)
        scored_facts = []
        for fact in facts:
            score = (
                0.4 * fact.importance
                + 0.3 * compute_novelty(fact, scored_facts)
                + 0.3 * compute_recency_score(fact)
            )
            scored_facts.append((fact, score))
        
        # Sort by score, return top-k
        sorted_facts = sorted(scored_facts, key=lambda x: x[1], reverse=True)
        return [f for f, _ in sorted_facts[:top_k]]
    
    async def _retrieve_negation(self, query: str, top_k: int) -> List[Fact]:
        """Find facts matching positive part, exclude negative part."""
        
        # Parse "X but NOT Y" → positive="X", negative="Y"
        positive, negative = extract_negation_parts(query)
        
        # Retrieve facts matching positive
        facts = await self.hybrid_retriever.search(positive, top_k * 2)
        
        # Filter out facts matching negative
        filtered = [f for f in facts if not matches_query(f.text, negative)]
        
        return filtered[:top_k]
    
    async def _retrieve_contextual(self, query: str, top_k: int) -> List[Fact]:
        """Detect domain (code/log/doc), use domain-specific compression."""
        
        # Detect domain: "compression strategy for JSON"
        domain = extract_domain_context(query)
        
        # Retrieve facts
        facts = await self.hybrid_retriever.search(query, top_k)
        
        # Compress based on domain
        for fact in facts:
            if domain == "json":
                fact.compressed_form = json_specific_compress(fact.text)
            elif domain == "code":
                fact.compressed_form = code_specific_compress(fact.text)
            # ... more domains
        
        return facts
```

### Explanation Generation

```python
class IntentAwareExplainer:
    """Generate explanations tailored to query intent."""
    
    async def explain(
        self,
        query: str,
        intent: Intent,
        facts: List[Fact],
        llm: LLMClient,
    ) -> str:
        """Generate intent-specific explanation."""
        
        if intent == Intent.TEMPORAL:
            prompt = f"""
            User asked about timeline: "{query}"
            
            Facts (with timestamps):
            {self._format_with_timestamps(facts)}
            
            Explain in chronological order, highlighting key dates.
            """
        
        elif intent == Intent.COMPARISON:
            prompt = f"""
            User wants to compare: "{query}"
            
            Facts about entity 1 and entity 2:
            {self._format_for_comparison(facts)}
            
            Compare and contrast the entities, highlighting key differences.
            """
        
        elif intent == Intent.CAUSATION:
            prompt = f"""
            User asks why something happened: "{query}"
            
            Causal chain (cause → effect → outcome):
            {self._format_causal_chain(facts)}
            
            Explain the cause-effect relationships step by step.
            """
        
        elif intent == Intent.SYNTHESIS:
            prompt = f"""
            User wants synthesis: "{query}"
            
            Multiple facts on this topic:
            {self._format_for_synthesis(facts)}
            
            Summarize the key points and how they relate.
            """
        
        else:
            # Default explanation
            prompt = f"""
            User asked: "{query}"
            
            Relevant facts:
            {self._format_basic(facts)}
            
            Answer the user's question based on these facts.
            """
        
        return await llm.generate(prompt)
```

---

## Rationale

### Why Classify Intent?

Different intents have different optimal retrieval strategies:
- Temporal: Need date filtering (hybrid alone won't catch "2025-06")
- Comparison: Need to find facts about both entities (not just semantically similar)
- Causation: Need causal graph (hybrid search won't find "cause → effect")
- Counterfactual: Need do-calculus reasoning (hybrid can't estimate effects)

One-size-fits-all retrieval leaves performance on the table.

### Why Pattern Matching First?

Fast, high-confidence:
- "When did X happen?" → Temporal (95% confidence, instant)
- "Compare X and Y" → Comparison (95% confidence, instant)
- Semantic similarity: Slower, lower confidence (70%)

Hybrid approach: Try patterns first, fallback to semantic if no match.

### Why Route to Specialized Retrievers?

Each retriever optimized for its intent:
- Simple: Hybrid search (general-purpose)
- Temporal: Hybrid + temporal filter (focused)
- Comparison: Multi-entity search + diff (targeted)
- Causation: Graph traversal (specialized)

Specialized retrievers outperform one-size-fits-all.

### Why Adapt Explanations?

Same facts, different context:
- Temporal query: Explain in chronological order
- Comparison query: Highlight differences
- Causation query: Explain cause-effect chain
- Synthesis query: Summarize key points

Intent-aware explanations more helpful to users.

---

## Consequences

### Positive

✅ **Better retrieval accuracy** — Specialized strategies for each intent  
✅ **Faster answers** — Route to most efficient retriever  
✅ **Clearer explanations** — Tailored to what user is asking  
✅ **Handles complex queries** — Temporal, causal, counterfactual, synthesis  
✅ **Extensible** — Easy to add new intents and retrievers  

### Negative

⚠️ **Classification errors** — Might classify intent wrong (mitigated by fallback)  
⚠️ **Increased complexity** — More code paths, more to test  
⚠️ **Pattern brittleness** — Hand-coded patterns may not cover all cases  
⚠️ **Semantic classifier needs embedding** — Extra latency (mitigated by fast patterns)  
⚠️ **UNKNOWN fallback** — Low-confidence queries fall back to simple retrieval  

### Mitigation

1. **Conservative classification** — Only route if confidence > threshold
2. **Fallback chain** — If specialized retriever fails, try hybrid
3. **Feedback loop** — Track misclassifications, improve patterns
4. **Hybrid approach** — Combine pattern + semantic for higher accuracy
5. **Logging** — Log intent + confidence for debugging

---

## Implementation

### Integration with QueryUseCase

```python
# src/application/use_cases.py

class QueryUseCase:
    def __init__(
        self,
        intent_classifier: QueryIntentClassifier,
        intent_router: IntentRouter,
        explainer: IntentAwareExplainer,
        llm: LLMClient,
    ):
        self.classifier = intent_classifier
        self.router = intent_router
        self.explainer = explainer
        self.llm = llm
    
    async def execute(self, query: str, top_k: int = 5) -> QueryResult:
        """Execute query with intent-aware routing."""
        
        # 1. Classify intent
        intent, confidence = await self.classifier.classify(query)
        
        if confidence < 0.5:
            intent = Intent.UNKNOWN  # Fallback
        
        # 2. Route to specialized retriever
        facts = await self.router.retrieve(query, intent, top_k)
        
        # 3. Generate intent-aware explanation
        explanation = await self.explainer.explain(
            query,
            intent,
            facts,
            self.llm,
        )
        
        return QueryResult(
            facts=facts,
            explanation=explanation,
            intent=intent,
            intent_confidence=confidence,
        )
```

### Testing

```python
# tests/unit/application/test_query_intent.py

@pytest.mark.asyncio
async def test_temporal_intent_detection():
    """Query with dates should be classified as TEMPORAL."""
    classifier = QueryIntentClassifier()
    
    intent, conf = await classifier.classify("When did RL compression launch?")
    
    assert intent == Intent.TEMPORAL
    assert conf > 0.9

@pytest.mark.asyncio
async def test_comparison_routing():
    """Comparison intent should retrieve facts about both entities."""
    router = IntentRouter(retriever, graph_retriever, reranker)
    
    facts = await router.retrieve(
        "Compare SMartCrusher and RL compression",
        Intent.COMPARISON,
        top_k=5,
    )
    
    # Should have facts about both strategies
    assert any("SmartCrusher" in f.text for f in facts)
    assert any("RL" in f.text for f in facts)

@pytest.mark.asyncio
async def test_fallback_on_unknown_intent():
    """Unknown intent should use simple hybrid retrieval."""
    router = IntentRouter(retriever, ...)
    
    facts = await router.retrieve(
        "Xyzzy flibbertigibbet",  # Nonsense query
        Intent.UNKNOWN,
        top_k=5,
    )
    
    # Should still return results (using simple retrieval)
    assert len(facts) > 0
```

---

## Future Enhancements

### Short Term
- [ ] Add more patterns (question detection, entity extraction)
- [ ] Feedback-based pattern refinement (log misclassifications)
- [ ] Confidence threshold tuning (when to fallback)

### Medium Term
- [ ] Learn patterns from user clicks (which retrievers work best per intent)
- [ ] Multi-intent detection (query can have multiple intents)
- [ ] Intent-specific ranking (boost facts relevant to intent)

### Long Term
- [ ] Question answering pipeline (QA-specific retrieval)
- [ ] Multi-turn conversation (remember previous intents)
- [ ] Intent prediction (suggest intent to user before querying)

---

## Related ADRs

- **ADR-0005:** Fusion weights adapt based on query intent
- **ADR-0007:** Memory consolidation importance depends on query intent
- **ADR-0009:** Conflict resolution strategy varies by intent

---

## References

- **Code:** `src/application/use_cases.py` (QueryUseCase)
- **Classifier:** `src/infrastructure/query_intent.py` (QueryIntentClassifier)
- **Router:** `src/infrastructure/intent_router.py` (IntentRouter)
- **Diagram:** `docs/ARCHITECTURE_DIAGRAMS.md` (Diagram 8: Query Flow)

---

**Status:** ✅ Accepted and implemented  
**Last Updated:** 2026-07-09  
**Next Review:** After 100 queries, analyze intent distribution and misclassifications
