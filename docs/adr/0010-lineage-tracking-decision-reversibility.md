# ADR-0010: Lineage Tracking and Decision Reversibility

**Status:** Accepted  
**Date:** 2026-07-09  
**Deciders:** Governance & Audit Team  
**Co-Authors:** Claude Haiku 4.5

---

## Context

SITREP transforms facts throughout their lifecycle:
1. **Ingest:** Document → Passages → Extracted Facts
2. **Merge:** Fact A + Fact B → Merged Fact (archives A & B)
3. **Compress:** Fact → Compressed Fact (stores original in CCR)
4. **Archive:** Fact → Archived Fact (soft-deleted)
5. **Recover:** Archived Fact → Active Fact (unarchived)
6. **Update:** Fact v1 → Fact v2 (new version)
7. **Delete:** Fact → (removed from all indices)

Without tracking, users can't answer:
- "Why is fact F in the results?" (what decisions led here?)
- "Can I undo the merge of A and B?" (is it reversible?)
- "Which facts are derived from document D?" (traceability)
- "What changed in fact F over time?" (audit trail)
- "Why was fact F archived?" (decision reasoning)

**Compliance requirement:**
- Some jurisdictions require data lineage (GDPR, audit trails)
- Financial/medical: Need to prove why data included/excluded
- ML systems: Explainability requires tracing data provenance

**User benefit:**
- Transparency: See why system decided something
- Reversibility: Undo mistakes (archives, bad merges)
- Debugging: Trace bad information to source
- Learning: Understand what decisions worked

**Challenge:** Tracking all decisions without bloating the system
- Decision DAG can be large (100K decisions for 10K facts)
- Need efficient storage and query
- Must be queryable (which facts came from document X?)

---

## Decision

**Implement decision graph using KuzuDB (graph database). Track all transformations as decision nodes with causal links. Support both forward (what did this fact come from?) and backward (what facts did this decision create?) traversal.**

### Decision Types

```python
class DecisionType(Enum):
    """All transformations tracked as decisions."""
    INGEST = "ingest"              # Document → Passages → Facts
    EXTRACT = "extract"             # Passage → Facts (explicit extraction)
    MERGE = "merge"                 # Fact A + B → Merged Fact
    COMPRESS = "compress"           # Fact → Compressed Fact
    ARCHIVE = "archive"             # Fact → Archived (soft-delete)
    UNARCHIVE = "unarchive"         # Archived → Active (recover)
    UPDATE = "update"               # Fact v1 → Fact v2
    CLASSIFY = "classify"           # Fact → Classified (type detection)
    RANK = "rank"                   # Facts reranked by importance
    DELETE = "delete"               # Fact → (hard delete)
    FEEDBACK = "feedback"           # User marks fact relevant
    CONFLICT_RESOLVE = "conflict"   # Conflicting facts → resolution
```

### Decision Graph Schema (KuzuDB)

```
Decision Node:
  id: UUID
  type: DecisionType (ENUM)
  timestamp: DATETIME
  input_ids: LIST<Fact IDs>        # What went in
  output_ids: LIST<Fact IDs>       # What came out
  metadata: JSON                    # Decision-specific data
  reversible: BOOLEAN               # Can this be undone?
  audit_log: TEXT                   # Free-form reasoning

Example INGEST decision:
  id: "dec_20250709_001"
  type: INGEST
  timestamp: 2026-07-09 17:30:00
  input_ids: ["doc_sitrep_codebase"]
  output_ids: ["fact_42", "fact_43", ..., "fact_89"]  (48 facts)
  metadata: {
    source_path: "/path/to/SITREP_CODEBASE_ANALYSIS.md",
    extraction_method: "llm",
    passages_created: 12,
  }
  reversible: true
  audit_log: "Ingested SITREP analysis document, extracted 48 facts"

Causal edges (Decision → Fact):
  dec_20250709_001 --creates--> fact_42
  dec_20250709_001 --creates--> fact_43
  ...
  dec_20250709_002 --merges--> fact_42 (merges fact A into fact B)
  dec_20250709_002 --archives--> fact_43 (archives merged fact)
```

### Forward Traversal (What caused this fact?)

```python
async def trace_fact_origins(fact_id: str) -> FactLineage:
    """
    Trace fact back to source: Fact ← Decision ← Input Facts ← Decisions ← ...
    """
    
    lineage = FactLineage(fact_id=fact_id)
    
    # 1. Find decision that created this fact
    creating_decision = await kuzu.query(f"""
        MATCH (d:Decision)-[:creates]->(f:Fact)
        WHERE f.id = '{fact_id}'
        RETURN d
    """)
    
    if not creating_decision:
        return lineage  # Original fact (no decision)
    
    lineage.creating_decision = creating_decision[0]
    
    # 2. Recursively trace input facts
    input_fact_ids = creating_decision.input_ids
    for input_id in input_fact_ids:
        input_lineage = await trace_fact_origins(input_id)
        lineage.inputs.append(input_lineage)
    
    return lineage

# Example output:
# Fact(compression reduces tokens) ← Decision(MERGE)
#   ← Fact(compression achieves 70% reduction)
#   ← Decision(INGEST from sitrep_codebase.md)
#     ← Document(SITREP_CODEBASE_ANALYSIS.md)

# Fact(compression achieves 70% reduction) [ORIGINAL, not created by decision]
```

### Backward Traversal (What facts did this decision create?)

```python
async def find_decision_consequences(
    decision_id: str,
) -> List[Fact]:
    """
    Find all facts derived from this decision.
    
    Decision ← Current Decision ← Descendant Decisions ← Final Facts
    """
    
    # 1. Direct outputs
    direct_outputs = await kuzu.query(f"""
        MATCH (d:Decision)-[:creates]->(f:Fact)
        WHERE d.id = '{decision_id}'
        RETURN f.id
    """)
    
    affected_facts = set(direct_outputs)
    
    # 2. Indirect: facts created by decisions that used these outputs
    dependent_decisions = await kuzu.query(f"""
        MATCH (d1:Decision)-[:creates]->(f:Fact),
              (d2:Decision)-[:uses]->(f:Fact),
              (d2:Decision)-[:creates]->(f2:Fact)
        WHERE d1.id = '{decision_id}'
        RETURN d2.id
    """)
    
    # Recursively find consequences
    for dep_decision_id in dependent_decisions:
        consequences = await find_decision_consequences(dep_decision_id)
        affected_facts.update([f.id for f in consequences])
    
    return [Fact(id=fid) for fid in affected_facts]

# Example: If we ARCHIVE a fact, find all merged facts that depended on it
```

### Reversibility Support

```python
async def can_reverse_decision(decision_id: str) -> Tuple[bool, str]:
    """Check if a decision can be reversed."""
    
    decision = await decision_repo.find_by_id(decision_id)
    
    if not decision.reversible:
        return False, "Decision marked as non-reversible"
    
    # Check if any output facts have been merged with others
    for output_id in decision.output_ids:
        dependent = await find_decision_consequences(decision_id)
        if len(dependent) > 1:
            return False, f"Fact {output_id} used in {len(dependent)} dependent decisions"
    
    return True, "Can be reversed"

async def reverse_decision(decision_id: str) -> ReverseResult:
    """Undo a decision (if possible)."""
    
    can_reverse, reason = await can_reverse_decision(decision_id)
    if not can_reverse:
        raise ValueError(f"Cannot reverse: {reason}")
    
    decision = await decision_repo.find_by_id(decision_id)
    
    if decision.type == DecisionType.INGEST:
        # Delete all facts created by this ingest
        for fact_id in decision.output_ids:
            await fact_repo.delete_permanently(fact_id)
        
        logger.info(f"Reversed INGEST decision: deleted {len(decision.output_ids)} facts")
    
    elif decision.type == DecisionType.MERGE:
        # Restore archived facts
        for fact_id in decision.output_ids:
            if fact_id in decision.input_ids:
                await fact_repo.unarchive(fact_id)
        
        logger.info(f"Reversed MERGE decision: restored {len(decision.output_ids)} facts")
    
    elif decision.type == DecisionType.ARCHIVE:
        # Unarchive the fact
        for fact_id in decision.output_ids:
            await fact_repo.unarchive(fact_id)
        
        logger.info(f"Reversed ARCHIVE decision: unarchived {len(decision.output_ids)} facts")
    
    # Record the reversal as a decision itself
    await decision_repo.create(Decision(
        type=DecisionType.REVERSE,
        input_ids=decision.output_ids,
        metadata={"reversed_decision_id": decision_id},
        reversible=False,  # Reversal itself cannot be reversed
        audit_log=f"Reversed decision {decision_id}",
    ))
    
    return ReverseResult(success=True, message=reason)
```

### Query Interface for Lineage

```python
class LineageUseCase:
    """Allow users to query lineage."""
    
    async def explain_fact_provenance(fact_id: str) -> str:
        """
        User-facing: Explain where a fact came from.
        
        Returns: Human-readable explanation with source chain.
        """
        
        lineage = await trace_fact_origins(fact_id)
        
        # Generate explanation
        explanation = f"""
        Fact: {lineage.fact.text}
        
        Origin: Created by {lineage.creating_decision.type.value} decision
        Time: {lineage.creating_decision.timestamp}
        
        Sources:
        """
        
        for input_lineage in lineage.inputs:
            if input_lineage.is_original:
                explanation += f"\n  - Original fact: {input_lineage.fact.text}"
            else:
                explanation += f"\n  - Derived from {input_lineage.creating_decision.type.value}"
        
        return explanation
    
    async def find_facts_from_document(doc_path: str) -> List[Fact]:
        """
        User query: "Show me all facts from this document"
        
        Returns: All facts directly or indirectly created from ingesting doc.
        """
        
        # Find INGEST decision for this document
        ingest_decisions = await decision_repo.find_by_type_and_metadata(
            type=DecisionType.INGEST,
            metadata_filter={"source_path": doc_path},
        )
        
        all_facts = []
        for decision in ingest_decisions:
            facts = await find_decision_consequences(decision.id)
            all_facts.extend(facts)
        
        return all_facts
```

### Audit Trail

```python
class AuditLog:
    """Immutable audit trail of all decisions."""
    
    async def log_decision(decision: Decision):
        """Record decision in append-only log."""
        
        log_entry = {
            "timestamp": datetime.now(),
            "decision_id": decision.id,
            "decision_type": decision.type,
            "user_id": current_user.id,  # Who made this decision
            "input_ids": decision.input_ids,
            "output_ids": decision.output_ids,
            "reason": decision.audit_log,
            "metadata": decision.metadata,
        }
        
        # Append to immutable log (JSONL file)
        await append_to_event_log(".sitrep/logs/decisions.jsonl", log_entry)
        
        # Also store in KuzuDB for queryability
        await kuzu.create_node("Decision", decision)
    
    async def get_audit_trail(fact_id: str) -> List[AuditEntry]:
        """Get full audit trail for a fact."""
        
        entries = []
        
        # Read append-only log
        with open(".sitrep/logs/decisions.jsonl", "r") as f:
            for line in f:
                entry = json.loads(line)
                if fact_id in entry["output_ids"]:
                    entries.append(AuditEntry.from_dict(entry))
        
        return entries  # Chronologically ordered
```

---

## Rationale

### Why Graph Database (KuzuDB)?

Decision lineage is inherently graph-like:
- Fact A ← Decision D1 ← Fact B ← Decision D2
- Multiple sources → Single merged fact (diamond pattern)
- Reversals create backward links

Graph queries natural:
```sql
-- "What facts did decision D create?"
MATCH (d:Decision)-[:creates]->(f:Fact)
WHERE d.id = 'D'
RETURN f

-- "Trace fact F back to sources"
MATCH path = (f:Fact)-[*]->(source:Fact)
WHERE f.id = 'F' AND source IS ORIGINAL
RETURN path
```

Relational DB would require joins and recursive CTEs (slow).

### Why Decision as First-Class Object?

Could track in fact metadata:
```json
{
  "fact": {
    "id": "f1",
    "text": "...",
    "derived_from": ["f0"]
  }
}
```

Problem: Doesn't capture decision metadata, non-reversible.

Decision object captures:
- WHAT was decided
- WHEN it was decided
- WHY (reasoning/audit log)
- Whether it's reversible
- All inputs/outputs atomically

### Why Reversibility Flag?

Some decisions shouldn't be reversed:
- INGEST from trusted source: Remove and re-ingest (reversible)
- FEEDBACK from user: Can't undo (non-reversible, affects weights)
- REVERSE itself: Can't undo undoing (non-reversible)

Explicit flag makes intent clear.

### Why Append-Only Event Log?

Immutable audit trail (can't alter history):
- Compliance: Prove what decisions were made when
- Security: Detect tampering (JSONL signature verification possible)
- Recovery: Replay decisions to rebuild state

---

## Consequences

### Positive

✅ **Full traceability** — Every fact traceable to source  
✅ **Explainability** — Users see why facts included  
✅ **Audit compliance** — Immutable decision log  
✅ **Reversibility** — Undo mistakes (if possible)  
✅ **Debugging** — Trace bad info to source  
✅ **Learned from data** — Understand what works  

### Negative

⚠️ **Storage overhead** — Decision graph can grow large (1-2x fact count)  
⚠️ **Query latency** — Graph traversal slower than flat queries (but still <100ms)  
⚠️ **Reverse constraints** — Some decisions can't be reversed (complex logic)  
⚠️ **Update complexity** — Every fact change creates decision node  
⚠️ **Cleanup challenges** — Hard to delete old lineage (breaks audit trail)  

### Mitigation

1. **Lazy decision creation** — Only create decision nodes for "important" operations
2. **Lineage pruning** — Archive old lineage (but keep hash of audit log)
3. **Query optimization** — Index common paths (document → facts)
4. **Compression** — Store decision IDs in facts, not full lineage inline
5. **Sampling** — For very large graphs, sample ancestors instead of full trace

---

## Implementation

### Integration

```python
# src/application/__init__.py

async def build_application(config: SitrepConfig) -> Application:
    # ... DB clients ...
    
    # Decision tracking
    decision_repo = DecisionRepository(sqlite)
    lineage_tracker = LineageTracker(kuzu, decision_repo)
    
    # Wrap repositories to track decisions
    fact_repo = FactRepository(sqlite)
    fact_repo = TrackedRepository(fact_repo, decision_repo)  # Decorator
    
    return Application(
        fact_repo=fact_repo,
        lineage=lineage_tracker,
    )
```

### Hooks in Operations

```python
# Every operation creates a decision

class IngestUseCase:
    async def execute(self, passages):
        # ... ingest logic ...
        
        # Create decision node
        decision = Decision(
            type=DecisionType.INGEST,
            input_ids=[doc_id],
            output_ids=[f.id for f in facts],
            reversible=True,
            audit_log="Ingested document",
        )
        
        await decision_repo.create(decision)
        await kuzu.add_edge("INGEST", decision.id, [f.id for f in facts])
```

### Testing

```python
@pytest.mark.asyncio
async def test_trace_fact_origins():
    """Trace fact back to source document."""
    fact = await fact_repo.find_by_id("fact_1")
    lineage = await lineage_tracker.trace_origins(fact.id)
    
    # Should trace back to original ingest
    assert lineage.creating_decision.type == DecisionType.INGEST

@pytest.mark.asyncio
async def test_reverse_ingest():
    """Reverse an ingest decision."""
    # Ingest facts
    decision = await ingest_usecase.execute([passage])
    fact_ids = decision.output_ids
    
    # Reverse
    await lineage_tracker.reverse_decision(decision.id)
    
    # Facts should be deleted
    for fid in fact_ids:
        assert await fact_repo.find_by_id(fid) is None

@pytest.mark.asyncio
async def test_cannot_reverse_if_dependent():
    """Cannot reverse if fact used in downstream decision."""
    # Ingest fact A
    decision_1 = await ingest_usecase.execute([passage_a])
    fact_a_id = decision_1.output_ids[0]
    
    # Merge fact A into fact B
    decision_2 = await merge_usecase.execute(fact_a_id, fact_b_id)
    
    # Cannot reverse ingest (fact A used in merge)
    can_reverse, _ = await lineage_tracker.can_reverse_decision(decision_1.id)
    assert not can_reverse
```

---

## Future Enhancements

### Short Term
- [ ] Lineage visualization (DAG view in UI)
- [ ] "What changed?" queries (diff lineage between versions)
- [ ] Batch lineage (query all facts from document X)

### Medium Term
- [ ] Causal counterfactuals (what if we removed source Y?)
- [ ] Lineage-based importance (facts from trusted sources valued higher)
- [ ] Auto-cleanup (archive old lineage while preserving audit hash)

### Long Term
- [ ] Linked data format (export lineage as RDF/PROV)
- [ ] Blockchain audit trail (immutable lineage verification)
- [ ] AI-driven lineage discovery (auto-infer undocumented decisions)

---

## Related ADRs

- **ADR-0004:** Multi-database approach (KuzuDB stores lineage)
- **ADR-0009:** Conflict resolution decisions tracked in lineage
- **ADR-0010:** (This ADR)

---

## References

- **Code:** `src/infrastructure/lineage.py` (LineageTracker)
- **Repository:** `src/adapters/decision_repository.py` (DecisionRepository)
- **Use Case:** `src/application/use_cases.py` (LineageUseCase)
- **Audit:** `src/infrastructure/audit_log.py` (AuditLog)
- **Diagram:** `docs/ARCHITECTURE_DIAGRAMS.md` (Diagram 7: Fact Lifecycle, Diagram 9: Error Recovery)

---

**Status:** ✅ Accepted and implemented  
**Last Updated:** 2026-07-09  
**Next Review:** After 10K decisions, assess query performance
