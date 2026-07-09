# Architecture Decision Records (ADRs)

This directory contains Architecture Decision Records — a way to document architectural decisions, their context, and consequences.

## Format

Each ADR follows the format:
- **Status:** Proposed, Accepted, Deprecated, Superseded
- **Date:** When the decision was made
- **Context:** The issue or problem being addressed
- **Decision:** What was decided
- **Rationale:** Why this decision was made
- **Consequences:** Positive and negative outcomes
- **Implementation:** How the decision was implemented
- **Related Decisions:** Links to other relevant ADRs

## Current ADRs

| # | Title | Status | Date | Summary |
|---|-------|--------|------|---------|
| [0001](./0001-multi-agent-architecture-documentation.md) | Multi-Agent Swarm for Architecture Documentation | Accepted | 2026-07-09 | Generate comprehensive architecture docs via 7-agent parallel swarm |

## Future ADRs (Planned)

- **ADR-0002:** Clean Architecture layer boundaries and module structure
- **ADR-0003:** Lazy import strategy for optional dependencies
- **ADR-0004:** Multi-database approach (SQLite + KuzuDB + ChromaDB)
- **ADR-0005:** Hybrid retrieval fusion weight learning strategy
- **ADR-0006:** RL compression agent training environment design

## How to Add a New ADR

1. Create a new file: `000N-short-title-slug.md` (increment the number)
2. Copy the template below
3. Fill in all sections (especially Context, Decision, Consequences)
4. Link from this README in the table above
5. Commit with message: `docs: add ADR-000N: Short title`

## Template

```markdown
# ADR-000N: Short Title

**Status:** Proposed | Accepted | Deprecated | Superseded  
**Date:** YYYY-MM-DD  
**Deciders:** Name1, Name2  

---

## Context

The issue or problem we're addressing...

## Decision

What we decided...

## Rationale

Why we chose this over alternatives...

## Consequences

### Positive
- ...

### Negative
- ...

## Implementation

How we actually did it...

## Related Decisions

- ADR-XXXX: Related decision

---

**Status:** ✅ Accepted  
**Last Updated:** YYYY-MM-DD  
```

## Decision Tracking Workflow

1. **Identify** a significant architectural decision needed
2. **Create** ADR-000N with full context and rationale
3. **Discuss** with team (PR review)
4. **Accept** once consensus reached (status = Accepted)
5. **Implement** the decision
6. **Reference** ADR in related code changes (commit messages, comments)
7. **Update** ADR if consequences discovered differ from expectations
8. **Supersede** if a better decision is found later (mark as Superseded, link to new ADR)

## Guidelines

### When to Write an ADR

✅ **Write an ADR for:**
- Major architectural changes (new layer, new database, new integration pattern)
- Decisions that affect multiple teams or subsystems
- Long-term technical debt decisions (what to defer, what to prioritize)
- Trade-offs between approaches (why X over Y over Z)
- Decisions that are non-obvious or have consequences that need documenting

❌ **Don't write an ADR for:**
- Bug fixes or refactoring (use commit messages, PR descriptions)
- Minor implementation details (use code comments)
- Decisions that are reversible with no downstream impact

### Writing Tips

1. **Use present tense:** "We decide to..." not "We decided to..."
2. **Be concrete:** Link to code, files, line numbers
3. **Document trade-offs:** What did you consider and reject?
4. **Think forward:** What will someone need to know in 6 months?
5. **Keep it short:** 1–2 pages is typical (use docs/ for deep dives)

## Reviewing ADRs

When reviewing an ADR, check:
- [ ] Context is clear (what problem are we solving?)
- [ ] Decision is unambiguous (someone else could implement it)
- [ ] Rationale is sound (why this, not the alternatives?)
- [ ] Consequences are realistic (positive and negative)
- [ ] Implementation is feasible (do we have tools/skills?)
- [ ] Related decisions are linked (what does this depend on?)

---

**Total ADRs:** 1 (Accepted)  
**Deprecated:** 0  
**Next ADR Number:** 0002  
**Last Updated:** 2026-07-09
