"""Project-wide constants: model defaults, table/collection names, roles, thresholds.

Centralizing identifiers here keeps SQL, Cypher, and Chroma references consistent
and refactor-safe across layers.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Defaults (mirror .env.example)
# ---------------------------------------------------------------------------
EMBEDDING_DIM = 384
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_OLLAMA_MODEL = "llama3.1:8b"
DEFAULT_HF_LLM_MODEL = "HuggingFaceTB/SmolLM-135M-Instruct"

# ---------------------------------------------------------------------------
# .sitrep/ subdirectories (bounded contexts)
# ---------------------------------------------------------------------------
DIR_METADATA = "metadata"
DIR_GRAPH = "graph"
DIR_VECTORS = "vectors"
DIR_DOCUMENTS = "documents"
DIR_AGENTS = "agents"
DIR_LINEAGE = "lineage"
DIR_LOGS = "logs"
DIR_CONFIG = "config"
ALL_SUBDIRS = (
    DIR_METADATA,
    DIR_GRAPH,
    DIR_VECTORS,
    DIR_DOCUMENTS,
    DIR_AGENTS,
    DIR_LINEAGE,
    DIR_LOGS,
    DIR_CONFIG,
)

# ---------------------------------------------------------------------------
# SQLite metadata
# ---------------------------------------------------------------------------
SQLITE_DB_FILENAME = "sitrep.db"

# Logical table names
TBL_SCHEMAS = "schemas"
TBL_SCHEMA_FIELDS = "schema_fields"
TBL_FACTS = "facts"
TBL_PASSAGES = "passages"
TBL_EPISODES = "episodes"
TBL_AGENTS = "agents"
TBL_DECISIONS = "decisions"
TBL_SKILLS = "skills"
TBL_FEEDBACK = "feedback"
TBL_KV_CACHE = "kv_cache"
TBL_LINEAGE_EVENTS = "lineage_events"
TBL_FUSION_WEIGHTS = "fusion_weights"
TBL_RETRIEVAL_STATS = "retrieval_stats"

# FTS5 virtual tables
FTS_PASSAGES = "passages_fts"
FTS_FACTS = "facts_fts"

# ---------------------------------------------------------------------------
# ChromaDB collections
# ---------------------------------------------------------------------------
COLL_PASSAGES = "passages"
COLL_FACTS = "facts"
COLL_SCHEMAS = "schemas"

# ---------------------------------------------------------------------------
# KuzuDB
# ---------------------------------------------------------------------------
KUZU_GRAPH_DIR = DIR_GRAPH
KUZU_LINEAGE_DIR = DIR_LINEAGE

# ---------------------------------------------------------------------------
# Agent roles
# ---------------------------------------------------------------------------
ROLE_EXTRACTION = "extraction"
ROLE_CONFLICT_DETECTION = "conflict_detection"
ROLE_CONFLICT_RESOLUTION = "conflict_resolution"
ROLE_TEMPORAL = "temporal"

# ---------------------------------------------------------------------------
# Lineage decision types
# ---------------------------------------------------------------------------
DEC_INGEST = "ingest"
DEC_QUERY = "query"
DEC_COMPRESS = "compress"
DEC_RETRIEVE = "retrieve"
DEC_FEEDBACK = "feedback"
DEC_CONFLICT = "conflict"
DEC_VERSION = "version"

# ---------------------------------------------------------------------------
# Quality / confidence
# ---------------------------------------------------------------------------
DEFAULT_CONFIDENCE_THRESHOLD = 0.55
DEFAULT_TOP_K = 5
DEFAULT_FUSION_WEIGHTS = (0.34, 0.33, 0.33)  # (bm25, vector, graph)
SCHEMA_PROMOTION_THRESHOLD = 5

# ---------------------------------------------------------------------------
# KV cache
# ---------------------------------------------------------------------------
KV_CACHE_TABLE = TBL_KV_CACHE
