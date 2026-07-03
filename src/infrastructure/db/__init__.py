"""Database clients (SQLite/FTS5, KuzuDB, ChromaDB, DuckDB)."""
from src.infrastructure.db.sqlite_client import SQLiteClient

__all__ = ["SQLiteClient"]
# KuzuClient / ChromaClient / DuckDBClient are exported lazily on import to
# avoid forcing optional dependencies; import them directly when needed:
#   from src.infrastructure.db.kuzu_client import KuzuClient
#   from src.infrastructure.db.chroma_client import ChromaClient
#   from src.infrastructure.db.duckdb_client import DuckDBClient
