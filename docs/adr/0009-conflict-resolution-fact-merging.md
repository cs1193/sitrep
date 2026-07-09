# ADR-0009: Conflict Resolution and Fact Merging

**Status:** Accepted  
**Date:** 2026-07-09  
**Deciders:** Data Integrity Team  
**Co-Authors:** Claude Haiku 4.5

---

## Context

As SITREP ingests documents, it will encounter contradictory facts:

```
Fact A: "RL compression achieves 70% token reduction"
  Source: sitrep-engine/docs/SITREP_CODEBASE_ANALYSIS.md (2026-07-09)
  Importance: 0.8

Fact B: "RL compression reduces tokens by up to 90%"
  Source: sitrep-engine/eval/compression_eval.py (2026-07-08)
  Importance: 0.7
```

Are these contradictory or complementary?
- 70% reduction (different benchmark/config)
- "up to 90%" (best case vs. average case)

Or genuine conflicts:

```
Fact C: "PPO compression agent converges in 50 episodes"
  Source: paper1.md
  
Fact D: "PPO compression agent requires 1000 episodes to converge"
  Source: paper2.md
```

Clearly contradictory.

**Challenge:** How to handle conflicts?

**Options:**

1. **Keep both** (flag as conflict)
   - ✓ Preserves information
   - ✗ Confuses users (which is right?)
   - ✗ Clutters retrieval results

2. **Delete older** (assume newer is correct)
   - ✓ Clean dataset
   - ✗ Might lose valuable context
   - ✗ Assumes recency = correctness (false)

3. **Merge into resolution** (synthesize)
   - ✓ Resolves contradiction
   - ✓ Preserves both views
   - ✗ Complex, requires judgment

4. **Expert resolution** (manual intervention)
   - ✓ Accurate
   - ✗ Doesn't scale (100K facts = 100K decisions?)

**Requirement:**
- Automatic resolution where possible
- Flag conflicts that need human review
- Preserve context (why the conflict? source difference?)
- Document reasoning (for audit trail)

---

## Decision

**Implement heuristic conflict detection + automatic merging when high-confidence, manual review queue for ambiguous cases.**

### Conflict Detection

```python
class ConflictDetector:
    """Detect contradictions between facts."""
    
    async def find_conflicts(
        self,
        fact_a: Fact,
        fact_b: Fact,
    ) -> Optional[Conflict]:
        """
        Check if fact_a and fact_b contradict.
        
        Returns: Conflict object if detected, None if compatible.
        """
        
        # 1. Same source?
        if fact_a.source_passage_id == fact_b.source_passage_id:
            return None  # Not a conflict (same passage)
        
        # 2. Semantic similarity (are they about same topic?)
        sim = cosine_similarity(fact_a.embedding, fact_b.embedding)
        if sim < 0.7:
            return None  # Different topics, not a conflict
        
        # 3. Negation detection (does one negate the other?)
        contradiction_score = await self._compute_contradiction_score(fact_a, fact_b)
        if contradiction_score < 0.5:
            return None  # No contradiction detected
        
        # 4. Confirm contradiction with LLM
        llm_verdict = await self._llm_contradiction_check(fact_a, fact_b)
        if not llm_verdict.is_contradiction:
            return None  # False alarm
        
        return Conflict(
            fact_a_id=fact_a.id,
            fact_b_id=fact_b.id,
            contradiction_score=contradiction_score,
            llm_verdict=llm_verdict,
            detected_at=datetime.now(),
        )
    
    async def _compute_contradiction_score(
        self,
        fact_a: Fact,
        fact_b: Fact,
    ) -> float:
        """
        Heuristic: Do the facts contradict?
        
        Returns: [0, 1] where 1 = clear contradiction
        """
        
        # Extract numerical claims
        numbers_a = extract_numbers(fact_a.text)  # [70, 90] for "70-90%"
        numbers_b = extract_numbers(fact_b.text)
        
        if numbers_a and numbers_b:
            # Check if ranges overlap
            range_a = (min(numbers_a), max(numbers_a))
            range_b = (min(numbers_b), max(numbers_b))
            
            overlap = max(0, min(range_a[1], range_b[1]) - max(range_a[0], range_b[0]))
            if overlap == 0:
                # No overlap: likely contradiction
                return 0.9
            elif overlap > 0:
                # Overlap: might be compatible (e.g., "70%" and "up to 90%")
                return 0.3
        
        # Extract negation words
        negation_words_a = {"not", "no", "never", "without"}
        negation_words_b = {"not", "no", "never", "without"}
        
        has_negation_a = any(w in fact_a.text.lower() for w in negation_words_a)
        has_negation_b = any(w in fact_b.text.lower() for w in negation_words_b)
        
        # One says X, other says NOT X (likely contradiction)
        if has_negation_a != has_negation_b:
            return 0.7
        
        # Semantic contradiction detection (via embedding distance + semantics)
        # Higher contradiction if opposite sentiment
        sentiment_a = compute_sentiment(fact_a.text)  # -1 (negative) to +1 (positive)
        sentiment_b = compute_sentiment(fact_b.text)
        
        if abs(sentiment_a - sentiment_b) > 1.5:
            return 0.6
        
        return 0.2  # Default: not contradictory
    
    async def _llm_contradiction_check(
        self,
        fact_a: Fact,
        fact_b: Fact,
    ) -> LLMVerdict:
        """Use LLM as final arbiter."""
        
        prompt = f"""
        Fact A: "{fact_a.text}"
        Source A: {fact_a.source}
        
        Fact B: "{fact_b.text}"
        Source B: {fact_b.source}
        
        Do these facts contradict each other?
        - Yes: They make opposing claims (e.g., "X is true" vs. "X is false")
        - No: They are compatible (e.g., "X achieved 70%" and "X achieved up to 90%")
        - Unclear: Cannot determine from the text alone
        
        Answer: Yes / No / Unclear
        Reasoning: [1-2 sentences]
        """
        
        response = await llm.generate(prompt)
        
        return LLMVerdict(
            is_contradiction=("Yes" in response),
            reasoning=extract_reasoning(response),
        )
```

### Conflict Resolution Strategies

```python
class ConflictResolver:
    """Resolve detected conflicts."""
    
    async def resolve(
        self,
        conflict: Conflict,
        fact_a: Fact,
        fact_b: Fact,
    ) -> ConflictResolution:
        """
        Resolve conflict automatically if high-confidence, else flag for review.
        """
        
        # 1. Try automatic resolution
        auto_resolution = await self._try_auto_resolve(fact_a, fact_b)
        if auto_resolution:
            return auto_resolution
        
        # 2. Otherwise: Flag for manual review
        return ConflictResolution(
            status="PENDING_REVIEW",
            fact_a_id=fact_a.id,
            fact_b_id=fact_b.id,
            reason="Could not auto-resolve; flagged for human review",
        )
    
    async def _try_auto_resolve(
        self,
        fact_a: Fact,
        fact_b: Fact,
    ) -> Optional[ConflictResolution]:
        """
        Try automatic resolution strategies.
        
        Return None if no high-confidence resolution found.
        """
        
        # Strategy 1: Source credibility
        # (Example: official docs > blogs > random sources)
        credibility_a = score_source_credibility(fact_a.source)
        credibility_b = score_source_credibility(fact_b.source)
        
        if credibility_a > credibility_b + 0.3:
            return ConflictResolution(
                status="KEEP_A",
                fact_a_id=fact_a.id,
                fact_b_id=fact_b.id,
                reason=f"Source credibility: A ({credibility_a:.2f}) > B ({credibility_b:.2f})",
                decision_confidence=min(credibility_a, 1.0),
            )
        
        if credibility_b > credibility_a + 0.3:
            return ConflictResolution(
                status="KEEP_B",
                fact_a_id=fact_a.id,
                fact_b_id=fact_b.id,
                reason=f"Source credibility: B ({credibility_b:.2f}) > A ({credibility_a:.2f})",
                decision_confidence=min(credibility_b, 1.0),
            )
        
        # Strategy 2: Recency (assume newer is more accurate)
        age_diff = abs((fact_a.created_at - fact_b.created_at).days)
        
        if age_diff > 30:  # Significant time difference
            if fact_a.created_at > fact_b.created_at:
                return ConflictResolution(
                    status="KEEP_A",
                    reason=f"Newer source (A: {fact_a.created_at.date()}, B: {fact_b.created_at.date()})",
                    decision_confidence=0.7,
                )
            else:
                return ConflictResolution(
                    status="KEEP_B",
                    reason=f"Newer source (B: {fact_b.created_at.date()}, A: {fact_a.created_at.date()})",
                    decision_confidence=0.7,
                )
        
        # Strategy 3: Merge if compatible
        merge_result = await self._try_merge(fact_a, fact_b)
        if merge_result:
            return merge_result
        
        # Strategy 4: User importance (user feedback signal)
        importance_diff = fact_a.importance - fact_b.importance
        
        if abs(importance_diff) > 0.3:
            if importance_diff > 0:
                return ConflictResolution(
                    status="KEEP_A",
                    reason=f"Higher user importance (A: {fact_a.importance:.2f}, B: {fact_b.importance:.2f})",
                    decision_confidence=0.6,
                )
            else:
                return ConflictResolution(
                    status="KEEP_B",
                    reason=f"Higher user importance (B: {fact_b.importance:.2f}, A: {fact_a.importance:.2f})",
                    decision_confidence=0.6,
                )
        
        # No high-confidence resolution found
        return None
    
    async def _try_merge(
        self,
        fact_a: Fact,
        fact_b: Fact,
    ) -> Optional[ConflictResolution]:
        """
        Attempt to merge facts into single comprehensive fact.
        
        Example: "70% reduction" + "up to 90%" → "70-90% reduction"
        """
        
        # Try numerical merging
        numbers_a = extract_numbers(fact_a.text)
        numbers_b = extract_numbers(fact_b.text)
        
        if numbers_a and numbers_b:
            merged_text = synthesize_numerical_facts(fact_a.text, fact_b.text)
            
            merged_fact = Fact(
                id=generate_id(),
                text=merged_text,
                source_passage_id=fact_a.source_passage_id,  # Pick A's source
                importance=(fact_a.importance + fact_b.importance) / 2,
                created_at=datetime.now(),
                metadata={
                    "merged_from": [fact_a.id, fact_b.id],
                    "merge_strategy": "numerical",
                },
            )
            
            return ConflictResolution(
                status="MERGE",
                fact_a_id=fact_a.id,
                fact_b_id=fact_b.id,
                merged_fact=merged_fact,
                reason="Merged numerical facts into range",
                decision_confidence=0.8,
            )
        
        # Try semantic merging
        prompt = f"""
        Fact A: {fact_a.text}
        Fact B: {fact_b.text}
        
        Create a single comprehensive statement that incorporates both facts.
        If they conflict, indicate the conflict in the merged statement.
        
        Merged: ...
        """
        
        merged_text = await llm.generate(prompt)
        
        merged_fact = Fact(
            id=generate_id(),
            text=merged_text,
            source_passage_id=fact_a.source_passage_id,
            importance=(fact_a.importance + fact_b.importance) / 2,
            metadata={
                "merged_from": [fact_a.id, fact_b.id],
                "merge_strategy": "semantic",
            },
        )
        
        return ConflictResolution(
            status="MERGE",
            fact_a_id=fact_a.id,
            fact_b_id=fact_b.id,
            merged_fact=merged_fact,
            reason="Merged facts semantically",
            decision_confidence=0.6,
        )
```

### Manual Review Queue

```python
class ConflictReviewQueue:
    """Manage conflicts pending human review."""
    
    async def add_to_queue(self, conflict: Conflict, resolution: ConflictResolution):
        """Add conflict to manual review queue."""
        await conflict_repo.create(conflict)
        
        logger.warning(
            f"Conflict flagged for review: {conflict.fact_a_id} vs {conflict.fact_b_id}\n"
            f"Reason: {resolution.reason}"
        )
    
    async def get_pending_reviews(self) -> List[Conflict]:
        """Get all conflicts awaiting manual resolution."""
        return await conflict_repo.find_pending()
    
    async def resolve_manually(
        self,
        conflict: Conflict,
        resolution: ConflictResolution,
    ):
        """User manually resolves conflict."""
        
        if resolution.status == "KEEP_A":
            # Archive fact B
            await fact_repo.archive(conflict.fact_b_id)
        elif resolution.status == "KEEP_B":
            # Archive fact A
            await fact_repo.archive(conflict.fact_a_id)
        elif resolution.status == "MERGE":
            # Create merged fact, archive both A and B
            await fact_repo.create(resolution.merged_fact)
            await fact_repo.archive(conflict.fact_a_id)
            await fact_repo.archive(conflict.fact_b_id)
        
        # Mark resolution as complete
        await conflict_repo.update(conflict.id, resolved=True, resolution=resolution)
```

---

## Rationale

### Why Automatic Detection First?

Prevents conflicts from silently corrupting data:
- Without detection: User sees contradictory facts, doesn't know which to trust
- With detection: Flag for review, prevent bad merges

### Why Automatic Resolution When Possible?

Doesn't scale to have humans review every conflict:
- Code corpus: 10K facts → 100s of conflicts?
- Need automatic resolution for 80% of cases
- Manual review for edge cases

### Why Merge When Compatible?

Loses less information:
- Delete: "70%" gone, "up to 90%" kept (artificial)
- Keep both: Confuses user (which is right?)
- Merge: "70-90%" combines both signals

### Why Confidence Scores?

Transparency and future learning:
- High-confidence decision: Trust the auto-resolution
- Low-confidence: Flag for review
- Learn from manual resolutions: Improve heuristics

---

## Consequences

### Positive

✅ **Automatic conflict handling** — Most conflicts resolved without manual intervention  
✅ **Preserves information** — Merging keeps both signals  
✅ **Transparent reasoning** — Decisions logged with rationale  
✅ **Escalation path** — Difficult cases flagged for review  
✅ **Learns from feedback** — Manual resolutions improve heuristics  

### Negative

⚠️ **Merge quality** — Merged facts may not read naturally  
⚠️ **False resolution** — Might resolve as "keep A" when "merge" better  
⚠️ **Edge cases** — Complex conflicts need human judgment  
⚠️ **LLM calls** — Conflict detection uses LLM (latency, cost)  
⚠️ **Source credibility** — Hard-coded scores may be wrong  

### Mitigation

1. **Conservative merging** — Only merge if high-confidence
2. **Manual review queue** — Escalate low-confidence decisions
3. **Feedback loop** — Log manual resolutions, improve scoring
4. **Configurable thresholds** — Tune sensitivity per corpus
5. **Audit trail** — Full reasoning stored for compliance

---

## Implementation

### Integration with IngestUseCase

```python
# src/application/use_cases.py

class IngestUseCase:
    def __init__(
        self,
        fact_repo,
        conflict_detector,
        conflict_resolver,
        conflict_queue,
    ):
        self.fact_repo = fact_repo
        self.detector = conflict_detector
        self.resolver = conflict_resolver
        self.queue = conflict_queue
    
    async def execute(self, passages: List[Passage]) -> IngestResult:
        """Ingest passages, detect and resolve conflicts."""
        
        # ... extract facts from passages ...
        
        # For each extracted fact, check for conflicts with existing facts
        for new_fact in extracted_facts:
            existing_facts = await self.fact_repo.find_similar(
                new_fact.embedding,
                top_k=5,
            )
            
            for existing_fact in existing_facts:
                conflict = await self.detector.find_conflicts(new_fact, existing_fact)
                
                if conflict:
                    # Try auto-resolve
                    resolution = await self.resolver.resolve(
                        conflict,
                        new_fact,
                        existing_fact,
                    )
                    
                    if resolution.status == "PENDING_REVIEW":
                        # Add to manual review queue
                        await self.queue.add_to_queue(conflict, resolution)
                    else:
                        # Auto-resolved: apply resolution
                        if resolution.status == "KEEP_A":
                            await self.fact_repo.archive(existing_fact.id)
                        elif resolution.status == "KEEP_B":
                            # Don't ingest new_fact
                            continue
                        elif resolution.status == "MERGE":
                            await self.fact_repo.create(resolution.merged_fact)
                            await self.fact_repo.archive(existing_fact.id)
                            continue
        
        # Ingest remaining facts
        await self.fact_repo.create_batch(final_facts)
        
        return IngestResult(...)
```

### Testing

```python
# tests/unit/adapters/test_conflict_resolution.py

@pytest.mark.asyncio
async def test_conflict_detection_numerical():
    """Detect numerical contradictions."""
    detector = ConflictDetector(embedding_service)
    
    fact_a = Fact(text="Compression achieves 70% reduction")
    fact_b = Fact(text="Compression achieves 10% reduction")
    
    conflict = await detector.find_conflicts(fact_a, fact_b)
    
    assert conflict is not None
    assert conflict.contradiction_score > 0.7

@pytest.mark.asyncio
async def test_merge_compatible_facts():
    """Merge numerical ranges intelligently."""
    resolver = ConflictResolver(embedding_service, llm)
    
    fact_a = Fact(text="Achieves 70% reduction")
    fact_b = Fact(text="Can achieve up to 90% reduction")
    
    resolution = await resolver.resolve(Conflict(...), fact_a, fact_b)
    
    assert resolution.status == "MERGE"
    assert "70" in resolution.merged_fact.text
    assert "90" in resolution.merged_fact.text
```

---

## Future Enhancements

### Short Term
- [ ] Improve source credibility scoring (learn from user feedback)
- [ ] Add more merge strategies (domain-specific)
- [ ] Batch conflict detection (on schedule, not just on ingest)

### Medium Term
- [ ] Learn from manual resolutions (improve heuristics)
- [ ] Temporal conflict resolution (same fact from different time periods)
- [ ] Multi-way conflicts (A vs B vs C)

### Long Term
- [ ] Zero-shot conflict resolution (train on labeled conflicts)
- [ ] Fine-grained merge (keep both facts, link them as "alternative claims")
- [ ] Source tracking (credit claims to sources, enable source-based filtering)

---

## Related ADRs

- **ADR-0008:** Query intent affects conflict resolution strategy
- **ADR-0010:** Lineage tracks which facts were merged/archived

---

## References

- **Code:** `src/adapters/services.py` (ConflictResolver)
- **Detection:** `src/infrastructure/conflict_detection.py` (ConflictDetector)
- **Queue:** `src/adapters/conflict_queue.py` (ConflictReviewQueue)

---

**Status:** ✅ Accepted and implemented  
**Last Updated:** 2026-07-09  
**Next Review:** After 1000 facts, analyze auto-resolution accuracy
