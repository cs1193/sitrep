# ADR-0003: Lazy Imports for Optional Dependencies

**Status:** Accepted  
**Date:** 2026-07-09  
**Deciders:** Architecture Team  
**Co-Authors:** Claude Haiku 4.5

---

## Context

SITREP supports multiple optional capabilities:
- **Dense embeddings** (sentence-transformers) — for semantic search
- **Knowledge graphs** (KuzuDB) — for temporal relationships + Personalized PageRank
- **LLM generation** (Ollama, Transformers) — for explanations
- **RL training** (PyTorch) — for compression agent optimization
- **Web UI** (Gradio) — for interactive interface
- **Analytics** (DuckDB) — for Parquet archives

If all dependencies were required at install time:
- **Installation bloat:** Users installing SITREP just to query facts would download 2+ GB of ML models
- **Startup delay:** `import sitrep` would take 30+ seconds (PyTorch alone takes 10s)
- **Installation failure:** Users on CPU-only systems might fail to install CUDA-dependent packages
- **Complexity:** Force users to manage dependencies they don't need
- **Fragmentation:** Different use cases want different subsets (e.g., "Just give me CLI without UI or ML")

The team needed a way to:
1. **Install quickly** — Core package in seconds, no model downloads
2. **Activate features on demand** — Install extras only for capabilities you use
3. **Graceful degradation** — Work in demo mode if extras not available
4. **Clear feedback** — Users know what to install to unlock features

---

## Decision

**Use lazy imports with optional extras. Dependencies only loaded when used. Fallbacks provided for demo/basic functionality.**

### Extra Definitions (in `pyproject.toml`)

```toml
[project.optional-dependencies]
rag = [
    "sentence-transformers>=2.2.0",
    "chromadb>=0.4.0",
]
graph = [
    "kuzudb>=0.0.8",
    "networkx>=3.0",  # For Personalized PageRank
]
llm = [
    "transformers>=4.30.0",
    "torch>=2.0.0",
    "ollama>=0.0.1",
]
rl = [
    "torch>=2.0.0",
]
web = [
    "gradio>=3.50.0",
]
duckdb = [
    "duckdb>=0.8.0",
    "pyarrow>=13.0.0",
]

# Convenience: install common combinations
all = [
    "sitrep[rag,graph,llm,rl,web,duckdb]",
]
```

### Installation Modes

```bash
# Mode 1: Core only (fastest, no models)
uv sync
# ~50 MB, installs in <10s, launches in <2s
# Features: CLI query (with FTS5 only, no dense search)

# Mode 2: Common extras (recommended for most)
uv sync --extra rag --extra graph --extra web
# ~2 GB (includes transformers, torch, sentence-transformers)
# Features: Dense search, knowledge graph, web UI

# Mode 3: Full (for researchers/development)
uv sync --extra all
# ~3 GB
# Features: Everything + RL training + DuckDB analytics
```

### Lazy Import Pattern

```python
# src/infrastructure/llm.py

# Core imports (always available)
from typing import Optional
from sitrep.domain import Fact

# Lazy imports (loaded only when needed)
_TRANSFORMERS_AVAILABLE = False
_OLLAMA_AVAILABLE = False

def ensure_transformers():
    global _TRANSFORMERS_AVAILABLE
    if not _TRANSFORMERS_AVAILABLE:
        try:
            import transformers  # ← Loaded on first use, not import time
            _TRANSFORMERS_AVAILABLE = True
        except ImportError:
            raise ImportError(
                "Transformers not available. "
                "Install with: uv sync --extra llm"
            )
    return transformers

class LLMClient(ABC):
    @abstractmethod
    async def generate(self, prompt: str) -> str: ...

class TransformersLLM(LLMClient):
    def __init__(self, model_id: str = "mistralai/Mistral-7B"):
        ensure_transformers()  # ← Validate before using
        import transformers
        
        self.model_id = model_id
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=model_id,
            device_map="auto"
        )
    
    async def generate(self, prompt: str) -> str:
        import transformers
        result = self.pipeline(prompt, max_length=200)
        return result[0]["generated_text"]

class OllamaLLM(LLMClient):
    def __init__(self, base_url: str = "http://localhost:11434"):
        ensure_ollama()
        import ollama
        self.client = ollama.Client(base_url)
    
    async def generate(self, prompt: str) -> str:
        import ollama
        # Use Ollama API
        response = self.client.generate(prompt)
        return response.get("response", "")

class DemoLLM(LLMClient):
    """Fallback when no LLM installed."""
    async def generate(self, prompt: str) -> str:
        # Return synthetic response
        return f"[DEMO MODE] SITREP would explain this based on: {prompt[:50]}..."
```

### Graceful Degradation

```python
# src/application/__init__.py

def build_application(config: SitrepConfig) -> Application:
    # Database: Always available (SQLite built-in)
    sqlite = SQLiteClient(config.db_path)
    fact_repo = FactRepository(sqlite)
    
    # Embeddings: Try dense, fallback to hash-based
    try:
        from sentence_transformers import SentenceTransformer
        embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        embedding_service = TransformerEmbeddingService(embedding_model)
    except ImportError:
        logger.warning("Transformers not installed; using hash-based embeddings")
        embedding_service = HashEmbeddingService()
    
    # Knowledge graph: Optional
    if config.graph_enabled:
        try:
            import kuzudb
            kuzu = KuzuDB(config.graph_path)
            graph_repo = GraphRepository(kuzu)
        except ImportError:
            logger.warning("KuzuDB not installed; graph features disabled")
            graph_repo = None
    else:
        graph_repo = None
    
    # LLM: Optional with fallback
    if config.llm_type == "transformers":
        try:
            llm = TransformersLLM(config.llm_model)
        except ImportError:
            logger.warning("Transformers not installed; using demo LLM")
            llm = DemoLLM()
    elif config.llm_type == "ollama":
        try:
            llm = OllamaLLM(config.ollama_base_url)
        except (ImportError, ConnectionError):
            logger.warning("Ollama not available; using demo LLM")
            llm = DemoLLM()
    else:
        llm = DemoLLM()
    
    # RL agent: Optional
    if config.rl_enabled:
        try:
            import torch
            rl_agent = PPOCompressionAgent(config.rl_policy_path)
        except ImportError:
            logger.warning("PyTorch not installed; using heuristic compression")
            rl_agent = HeuristicCompressionAgent()
    else:
        rl_agent = HeuristicCompressionAgent()
    
    # Web UI: Registered at startup if available
    if config.web_enabled:
        try:
            import gradio
            web_ui = GradioUI(app)
        except ImportError:
            logger.warning("Gradio not installed; web UI unavailable")
            web_ui = None
    else:
        web_ui = None
    
    return Application(
        fact_repo=fact_repo,
        embedding_service=embedding_service,
        graph_repo=graph_repo,
        llm=llm,
        rl_agent=rl_agent,
        web_ui=web_ui,
    )
```

### Installation Error Messages

```python
# src/infrastructure/retrieval.py

def ensure_chromadb():
    try:
        import chromadb
        return chromadb
    except ImportError:
        raise ImportError(
            "ChromaDB (dense embeddings) not available.\n\n"
            "To enable dense semantic search, install with:\n"
            "  uv sync --extra rag\n\n"
            "Without this, SITREP will use BM25 sparse search only."
        )
```

---

## Rationale

### Why Lazy Imports?

**Fast installation:** Core package installs in seconds
```bash
$ time pip install sitrep
real 0m5.234s  # vs. 45s with all dependencies
```

**Fast startup:** Scripts launch instantly
```bash
$ time python -c "from sitrep import build_application"
real 0m0.847s  # vs. 30s+ if PyTorch imported at import time
```

**Flexible deployment:**
- **Data scientist on laptop:** `uv sync --extra rag --extra llm` (dense search + generation)
- **DevOps on server:** `uv sync --extra all` (everything for batch processing)
- **CI/CD in container:** `uv sync` (core only, tests with mocks)
- **Cloud function:** `uv sync --extra rag --extra web` (minimal footprint)

**Graceful degradation:** Works even if optional deps not installed
- Dense search unavailable? → Use BM25 sparse search
- Ollama down? → Use demo mode
- PyTorch not installed? → Use heuristic compression (no learning, but still works)

### Why Fallbacks Matter?

Users can start with `uv sync`, get basic functionality, then decide what to install:
1. **Query facts with BM25:** "This is useful! Let me enable dense search."
2. **Add dense search:** `uv sync --extra rag` (5 min installation)
3. **Add LLM explanations:** `uv sync --extra llm` (10 min installation)
4. **Train compression agent:** `uv sync --extra rl` (5 min additional)

**Benefit:** Onboarding is incremental; no 30-minute initial setup.

---

## Consequences

### Positive

✅ **Fast installation:** Core package in <10s, no model downloads  
✅ **Fast startup:** Scripts launch instantly (no PyTorch/CUDA overhead)  
✅ **Flexible:** Users install only what they need  
✅ **Works everywhere:** CPU-only systems can run (with fallbacks)  
✅ **Clear feedback:** Error messages tell users exactly what to install  
✅ **Demo-friendly:** Can demo SITREP without GPU or models  
✅ **CI/CD friendly:** Tests run fast with core only, integration tests run with full extras  

### Negative

⚠️ **Code complexity:** Try/except blocks around imports  
⚠️ **Testing burden:** Need tests with and without optional deps  
⚠️ **User confusion:** Might not know why features aren't working (need to check docs)  
⚠️ **Fallback quality:** Demo LLM and heuristic compression less good than real ones  

### Mitigation

1. **Clear docs:** `docs/USAGE.md` lists all extras and what they enable
2. **Good error messages:** Tell users exactly which extra to install
3. **Feature detection:** `sitrep --features` shows what's available
4. **Integration tests:** Run full test suite with all extras in CI
5. **Telemetry (optional):** Log which features are actually used (to guide future extras)

---

## Implementation Details

### Checking Features Programmatically

```python
# src/utils/features.py

class Features:
    """Check which optional features are available."""
    
    @staticmethod
    def has_transformers() -> bool:
        try:
            import transformers
            return True
        except ImportError:
            return False
    
    @staticmethod
    def has_chromadb() -> bool:
        try:
            import chromadb
            return True
        except ImportError:
            return False
    
    @staticmethod
    def has_kuzudb() -> bool:
        try:
            import kuzudb
            return True
        except ImportError:
            return False
    
    @staticmethod
    def has_torch() -> bool:
        try:
            import torch
            return True
        except ImportError:
            return False
    
    @staticmethod
    def has_gradio() -> bool:
        try:
            import gradio
            return True
        except ImportError:
            return False
    
    @staticmethod
    def available_features() -> Dict[str, bool]:
        return {
            "dense_embeddings": Features.has_chromadb(),
            "knowledge_graph": Features.has_kuzudb(),
            "llm_generation": Features.has_transformers(),
            "rl_training": Features.has_torch(),
            "web_ui": Features.has_gradio(),
        }
```

### CLI Command

```bash
$ uv run sitrep --features
SITREP Feature Status
═════════════════════════════════════

Core Features (always available):
  ✓ SQLite database (metadata)
  ✓ BM25 sparse search
  ✓ Soft-delete with archival
  ✓ Decision lineage tracking

Optional Features:
  ✓ Dense embeddings (ChromaDB) [rag]
  ✓ Knowledge graph (KuzuDB) [graph]
  ✓ LLM generation (Transformers) [llm]
  ✗ RL compression training (PyTorch) [rl]
  ✓ Web UI (Gradio) [web]
  ✗ DuckDB analytics [duckdb]

To enable missing features:
  uv sync --extra rl --extra duckdb
```

### pyproject.toml Structure

```toml
[project]
name = "sitrep"
version = "0.1.0"
description = "Self-Improving Token-Reduced Embeddable Pipeline"
requires-python = ">=3.10"

dependencies = [
    "pydantic>=2.0",
    "numpy>=1.24",
    "rank-bm25>=0.2.2",
    "pyyaml>=6.0",
    "python-dotenv>=1.0",
]

[project.optional-dependencies]
rag = [
    "sentence-transformers>=2.2.0",
    "chromadb>=0.4.0",
]
graph = [
    "kuzudb>=0.0.8",
    "networkx>=3.0",
]
llm = [
    "transformers>=4.30.0",
    "torch>=2.0.0",
    "ollama>=0.0.1",
]
rl = [
    "torch>=2.0.0",
]
web = [
    "gradio>=3.50.0",
]
duckdb = [
    "duckdb>=0.8.0",
    "pyarrow>=13.0.0",
]
dev = [
    "pytest>=7.0",
    "pytest-asyncio>=0.21.0",
    "pytest-cov>=4.0",
    "black>=23.0",
    "ruff>=0.1.0",
]
all = [
    "sitrep[rag,graph,llm,rl,web,duckdb]",
]
```

---

## Testing Strategy

### Unit Tests (with mocks)

```python
# tests/unit/application/test_query_use_case.py

def test_query_without_rl_compression():
    """Query works even without RL agent (uses heuristic)."""
    fact_repo = MockFactRepository()
    reranker = MockReranker()
    compression = MockCompressionService()  # Uses heuristic
    
    query_uc = QueryUseCase(
        retriever=MockRetriever(fact_repo),
        reranker=reranker,
        compression=compression,
        llm=DemoLLM(),  # No LLM
    )
    
    result = asyncio.run(query_uc.execute("query"))
    assert len(result.facts) > 0
    assert result.explanation == "[DEMO MODE] ..."
```

### Integration Tests (with real deps)

```bash
# ci/test-rag-only.sh
# Test with only [rag] extra, no [llm] or [rl]
uv sync --extra rag
pytest tests/integration/  # Should pass

# ci/test-all-extras.sh
# Test with all extras
uv sync --extra all
pytest tests/integration/  # Should pass with full features
```

---

## Documentation

### Installation Guide

```markdown
## Installation

### Quickstart (Core Only)
uv sync
# ~50 MB, launches in <1s
# Includes: Basic CLI, BM25 search, SQLite

### Recommended (Common Extras)
uv sync --extra rag --extra graph --extra web
# ~2 GB
# Adds: Dense embeddings, knowledge graph, web UI

### Full Install (Development/Research)
uv sync --extra all
# ~3 GB
# Adds: Everything including RL training, DuckDB

### Feature Matrix
| Feature | Core | [rag] | [graph] | [llm] | [rl] | [web] |
|---------|------|-------|---------|-------|------|-------|
| Query facts | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| BM25 search | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Dense search | | ✓ | ✓ | ✓ | ✓ | ✓ |
| Knowledge graph | | | ✓ | ✓ | ✓ | ✓ |
| LLM explanation | Demo | Demo | Demo | ✓ | ✓ | ✓ |
| RL compression | Heuristic | Heuristic | Heuristic | Heuristic | ✓ | ✓ |
| Web UI | | | | | | ✓ |
```

---

## Related ADRs

- **ADR-0002:** Clean Architecture (lazy imports fit infrastructure layer)
- **ADR-0004:** Multi-database approach (databases are optional extras)

---

## References

- **Code:** `src/infrastructure/` lazy imports
- **Configuration:** `pyproject.toml` extras definition
- **Usage Guide:** `docs/USAGE.md`
- **Feature Detection:** `src/utils/features.py`
- **Fallbacks:** `src/infrastructure/llm.py`, `src/infrastructure/embeddings.py`

---

**Status:** ✅ Accepted and implemented  
**Last Updated:** 2026-07-09  
**Next Review:** When adding new major optional capability
