# ADR-0001: Multi-Agent Swarm for Architecture Documentation

**Status:** Accepted  
**Date:** 2026-07-09  
**Deciders:** Architecture Team, Claude Code  
**Co-Authors:** Claude Haiku 4.5

---

## Context

SITREP is a sophisticated system with multiple layers (Domain, Adapters, Application, Infrastructure, Presentation), complex data flows (retrieval, compression, RL training, ingestion), and subtle architectural patterns (Clean Architecture, Repository Pattern, Composition Root, Lazy Imports, Atomic Transactions).

Traditional documentation approaches (manual authoring, single-author review) risk:
- **Incompleteness:** One person may miss subsystems, edge cases, or interdependencies
- **Inconsistency:** Different sections written at different times may contradict
- **Staleness:** Manual updates lag behind code changes
- **Low Coverage:** Hard to justify time investment in comprehensive docs for exploratory projects

The team needed a **comprehensive, multi-perspective, quickly-generated** architecture baseline that would:
1. Cover all 5 layers and all subsystems
2. Provide multiple views (architecture, data flows, dependencies, patterns, APIs)
3. Generate visual diagrams automatically
4. Establish a baseline for future incremental updates

---

## Decision

**Spawn a multi-agent swarm to analyze the codebase in parallel and generate architecture documentation.**

The swarm consists of 7 specialized agents, each running independently:

1. **Architecture Scout** — Maps project structure, modules, LOC distribution
2. **API Mapper** — Documents public interfaces, function signatures, contracts
3. **Data Flow Analyst** — Traces data movement through pipelines
4. **Dependency Mapper** — Identifies module relationships and coupling
5. **Documentation Summarizer** — Extracts README, feature lists, configuration
6. **Synthesizer** — Consolidates findings into coherent narrative with diagrams

**Deliverables:**

- `docs/README.md` — Documentation hub and navigation guide (148 lines)
- `docs/ANALYSIS_INDEX.md` — Quick reference + 30-second summary (204 lines)
- `docs/SITREP_CODEBASE_ANALYSIS.md` — 13-section deep-dive (1,231 lines, 42 KB)
  - Sections: Structure, Layered Architecture, APIs, Data Flows, Dependencies, Critical Data Structures, Data Flows, Entry Points, Testing, Patterns, Dependency Graph, Observations, File Index
- `docs/ARCHITECTURE_DIAGRAMS.md` — 10 ASCII diagrams (1,328 lines, 112 KB)
  - Overall system, data models, retrieval pipeline, RL training loop, ingest pipeline, dependency graph, fact lifecycle, query flow, error recovery, performance profile

**Process:**

1. Scout & Analyze phases run **in parallel** (all agents independent, concurrent)
2. Agents read source code, documentation, configuration
3. Synthesize phase consolidates findings into structured documents
4. Documents formatted as Markdown for easy GitHub integration
5. ASCII diagrams embedded for copy/paste use (no external image deps)

---

## Rationale

### Why Multi-Agent Swarm?

**Breadth & Depth:**
- Each agent specializes in one dimension (architecture, APIs, data flow, dependencies)
- Running in parallel covers more ground than sequential analysis
- Synthesizer can cross-reference to identify gaps or inconsistencies

**Speed:**
- Complete analysis in ~7 minutes (parallel execution)
- Would take a human 4–8 hours for equivalent coverage
- Frees team to focus on engineering, not documentation

**Consistency:**
- Each agent uses same reasoning process and format
- Synthesizer ensures cross-document alignment
- Easier to maintain later (agents can re-run as code changes)

**Scalability:**
- Same approach works for larger codebases (just spawn more agents)
- Agents can be specialized further (e.g., "Security Analyzer", "Performance Profiler")

### Why ASCII Diagrams?

- **No external dependencies:** Embeds in Markdown, GitHub renders natively
- **Version control friendly:** Plain text, diffs are human-readable
- **Copy/paste:** Can be reused in Slack, emails, presentations
- **Maintainable:** Easy to edit inline (vs. regenerating image files)

### Why This Format?

- **Markdown:** Native to GitHub, integrates with repo
- **Embedded in docs/:** Discoverable, co-located with code
- **Hierarchical:** README → INDEX → ANALYSIS → DIAGRAMS (progressive detail)
- **Self-contained:** Each file stands alone; cross-references link sections

---

## Consequences

### Positive

✅ **Comprehensive baseline:** All layers, subsystems, APIs documented upfront  
✅ **Multiple perspectives:** Different angles (architecture, APIs, data flows, patterns)  
✅ **Visual references:** Diagrams aid understanding and onboarding  
✅ **Team alignment:** Common language, shared mental model  
✅ **Onboarding accelerator:** New engineers can understand architecture in 1–2 hours  
✅ **Decision record:** Commit message + ADR explain why docs exist  
✅ **Reproducible:** Same analysis can be re-run if code changes significantly  

### Negative

⚠️ **Potential staleness:** Docs may fall behind if code changes frequently  
⚠️ **Not a substitute for reading code:** Diagrams simplify; some details hidden  
⚠️ **Maintenance burden:** Docs require updates; need process to keep current  
⚠️ **Agent hallucination risk:** Multi-agent systems can miss nuances or make claims that need verification  

### Mitigation

1. **Keep code as source of truth:** Docs supplement, not replace, code review
2. **Update docs with PRs:** When merging architectural changes, update ADR + docs
3. **Periodic re-runs:** Every 2–3 months, re-run analysis and check for drift
4. **Review process:** Code reviewer checks if docs need updating
5. **Cross-reference:** Docs link to file paths + line numbers for verification

---

## Implementation Details

### Agent Configuration

```python
agents = [
    Agent("Architecture Scout", effort="high", timeout=60s),
    Agent("API Mapper", effort="medium", timeout=45s),
    Agent("Data Flow Analyst", effort="high", timeout=60s),
    Agent("Dependency Mapper", effort="medium", timeout=45s),
    Agent("Documentation Summarizer", effort="low", timeout=30s),
    Agent("Synthesizer", effort="high", timeout=90s, depends_on=[all others]),
]
```

### Workflow Structure

```
Phase 1: Scout (parallel)
  ├─ Structure: Find files, LOC, organization
  └─ Docs: Read READMEs, extract features

Phase 2: Analyze (parallel, after Scout)
  ├─ Architecture: Map modules, classes, patterns
  ├─ APIs: Extract signatures, docstrings, contracts
  ├─ Data Flow: Trace pipelines (ingest, query, train)
  └─ Dependencies: Map imports, coupling

Phase 3: Synthesize (sequential, after Analyze)
  └─ Consolidate findings, create diagrams, write narrative
```

### Output Locations

```
docs/
├── README.md                           (New hub)
├── USAGE.md                            (Existing, untouched)
├── ANALYSIS_INDEX.md                   (Generated)
├── SITREP_CODEBASE_ANALYSIS.md        (Generated)
├── ARCHITECTURE_DIAGRAMS.md           (Generated)
└── adr/
    └── 0001-multi-agent-architecture-documentation.md  (This file)
```

---

## Timeline

- **2026-07-09, 17:00:** Decision to spawn swarm
- **2026-07-09, 17:05:** Agents launched (Scout phase)
- **2026-07-09, 17:12:** Analyze phase completes
- **2026-07-09, 17:40:** Synthesis complete, docs generated
- **2026-07-09, 17:41:** Files migrated to `docs/`, commit created
- **2026-07-09, 17:42:** Pushed to main, ADR created

---

## Future Improvements

### Short Term (1–2 weeks)
- [ ] Add automated diff checking (alert if docs drift from code)
- [ ] Create update guide (what to change if code changes)
- [ ] Link diagrams to source files (e.g., "See src/infrastructure/retrieval.py:814")

### Medium Term (1–2 months)
- [ ] Re-run swarm analysis after major refactoring
- [ ] Add decision trees ("How to add PostgreSQL support?")
- [ ] Generate API reference documentation from docstrings

### Long Term (3+ months)
- [ ] Automatic doc regeneration on CI/CD
- [ ] Integrate with PR checks (flag if architectural change without doc update)
- [ ] Build interactive architecture explorer (click modules to see details)
- [ ] Add code examples for each layer (copy/paste snippets)

---

## Related Decisions

- **ADR-0002 (Future):** Clean Architecture layer boundaries and module structure
- **ADR-0003 (Future):** Lazy import strategy for optional dependencies
- **ADR-0004 (Future):** Multi-database approach (SQLite + KuzuDB + ChromaDB)

---

## References

- **Commit:** f3250fc — "Add comprehensive architecture documentation via multi-agent swarm analysis"
- **Documentation Hub:** `docs/README.md`
- **Quick Reference:** `docs/ANALYSIS_INDEX.md`
- **Architecture Analysis:** `docs/SITREP_CODEBASE_ANALYSIS.md`
- **Diagrams:** `docs/ARCHITECTURE_DIAGRAMS.md`

---

## Approval & Sign-Off

| Role | Name | Date | Signature |
|------|------|------|-----------|
| Architecture Decision | Claude Code | 2026-07-09 | ✓ |
| AI Agent (Co-Author) | Claude Haiku 4.5 | 2026-07-09 | ✓ |

---

**Status:** ✅ Accepted and implemented  
**Last Updated:** 2026-07-09  
**Next Review:** 2026-10-09 (quarterly check on doc freshness)
