"""Central configuration via Pydantic Settings.

All runtime knobs are defined here and overridable through ``SITREP_``-prefixed
environment variables or a ``.env`` file. ``get_config()`` returns a process-wide
singleton; ``ensure_directories()`` materializes the ``.sitrep/`` layout lazily.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import yaml
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.utils.constants import ALL_SUBDIRS

_logger = logging.getLogger("sitrep.config")


class SitrepConfig(BaseSettings):
    """Typed, environment-driven configuration for SITREP."""

    model_config = SettingsConfigDict(
        env_prefix="SITREP_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- data location ---
    base_dir: Path = Field(default=Path(".sitrep"), description="Root data directory under CWD.")

    # --- LLM backend ---
    llm_provider: Literal["auto", "ollama", "transformers", "demo"] = "auto"
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1:8b"
    ollama_timeout: float = 120.0
    hf_llm_model: str = "HuggingFaceTB/SmolLM-135M-Instruct"
    llm_max_new_tokens: int = 256
    llm_temperature: float = 0.2

    # --- embeddings ---
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384
    embedding_batch_size: int = 32

    # --- reranker ---
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # --- retrieval / fusion ---
    top_k: int = 5
    fusion_weights: List[float] = Field(default_factory=lambda: [0.34, 0.33, 0.33])
    fusion_lr: float = 0.05
    confidence_threshold: float = 0.55

    # --- chunking ---
    chunk_size: int = 512
    chunk_overlap: int = 64
    min_chunk_size: int = 32

    # --- PPO compression agent ---
    ppo_total_timesteps: int = 10000
    ppo_learning_rate: float = 3e-4
    ppo_n_steps: int = 2048
    ppo_batch_size: int = 64
    ppo_gamma: float = 0.99
    compression_min: float = 0.2
    compression_max: float = 0.8

    # --- Phase A retrieval upgrade (PPR + temporal decay + entity density) ---
    # The engines are constructed when ``ppr_enabled`` is true, but the BOOST
    # weights default to 0 (dormant): on hash-embedding + passage-only graphs
    # (e.g. the SciFact eval) PPR/density do not improve over the strong BM25
    # baseline and can regress. Enable these once real embeddings ([rag]) and a
    # fact/entity graph (Phase B/C) are in place. All are env-overridable.
    ppr_enabled: bool = True
    ppr_alpha: float = 0.85
    ppr_gamma: float = 0.8
    ppr_max_iter: int = 100
    ppr_tol: float = 1e-6
    ppr_weight: float = 0.0
    density_weight: float = 0.0
    bridge_theta: float = 0.3
    bridge_degree: int = 5
    temporal_strategy: str = "exponential"
    temporal_half_life_days: float = 180.0
    temporal_weight: float = 0.0

    # --- Phase B memory hygiene (importance / consolidation / forgetting) ---
    track_access: bool = False            # bump access_count on retrieval (off in eval)
    entity_graph_enabled: bool = True      # build passage adjacency from shared fact entities
    consolidation_theta: float = 0.85
    consolidation_max_pairs: int = 200
    importance_recency_w: float = 0.3
    importance_frequency_w: float = 0.3
    importance_redundancy_w: float = 0.2
    importance_source_w: float = 0.2
    forgetting_max_age_days: float = 365
    forgetting_inactive_period_days: float = 180
    forgetting_min_importance: float = 0.2
    forgetting_decay_rate: float = 0.95
    forgetting_min_access_frequency: float = 0.1
    forgetting_redundancy_threshold: float = 0.85
    forgetting_min_redundancy_count: int = 3
    forgetting_target_memory_size: int = 0

    # --- Phase C write safety ---
    event_log_enabled: bool = True

    # --- Phase E quality loop ---
    result_cache_ttl: int = 3600
    auto_judge: bool = False

    # --- schema promotion ---
    schema_promotion_threshold: int = 5

    # --- logging ---
    log_level: str = "INFO"

    # ----------------------------------------------------------------- validators
    @field_validator("base_dir", mode="after")
    @classmethod
    def _resolve_base_dir(cls, v: Path) -> Path:
        """Resolve relative paths against the current working directory."""
        v = Path(v).expanduser()
        return v.resolve() if v.is_absolute() else (Path.cwd() / v).resolve()

    @field_validator("fusion_weights", mode="after")
    @classmethod
    def _check_fusion_weights(cls, v: List[float]) -> List[float]:
        if len(v) != 3:
            raise ValueError("fusion_weights must have exactly 3 entries (bm25, vector, graph)")
        total = sum(v)
        return [w / total for w in v] if total > 0 else [1 / 3, 1 / 3, 1 / 3]

    @field_validator("log_level", mode="after")
    @classmethod
    def _upper_log(cls, v: str) -> str:
        return v.upper()

    # ----------------------------------------------------------------- derived paths
    @property
    def metadata_dir(self) -> Path:
        return self.base_dir / "metadata"

    @property
    def graph_dir(self) -> Path:
        return self.base_dir / "graph"

    @property
    def vectors_dir(self) -> Path:
        return self.base_dir / "vectors"

    @property
    def documents_dir(self) -> Path:
        return self.base_dir / "documents"

    @property
    def agents_dir(self) -> Path:
        return self.base_dir / "agents"

    @property
    def lineage_dir(self) -> Path:
        return self.base_dir / "lineage"

    @property
    def logs_dir(self) -> Path:
        return self.base_dir / "logs"

    @property
    def config_dir(self) -> Path:
        return self.base_dir / "config"

    @property
    def policies_dir(self) -> Path:
        return self.agents_dir / "policies"

    @property
    def sqlite_db_path(self) -> Path:
        return self.metadata_dir / "sitrep.db"

    @property
    def config_yaml_path(self) -> Path:
        return self.config_dir / "sitrep.yaml"

    def all_dirs(self) -> List[Path]:
        """Return every managed subdirectory under ``base_dir``."""
        return [self.base_dir / name for name in ALL_SUBDIRS]

    # ----------------------------------------------------------------- side effects
    def ensure_directories(self) -> None:
        """Create the full ``.sitrep/`` layout if it does not exist."""
        self.base_dir.mkdir(parents=True, exist_ok=True)
        for d in self.all_dirs():
            d.mkdir(parents=True, exist_ok=True)
        self.policies_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------------- persistence
    def to_dict(self) -> Dict[str, Any]:
        """Serialize config to a JSON/YAML-safe dict (paths as strings)."""
        data = self.model_dump(mode="json")
        return data

    def save_yaml(self, path: Optional[Path] = None) -> Path:
        """Persist the configuration to ``config/sitrep.yaml`` (or *path*)."""
        target = path or self.config_yaml_path
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(self.to_dict(), fh, sort_keys=True, allow_unicode=True)
        _logger.info("config saved → %s", target)
        return target

    @classmethod
    def load_yaml(cls, path: Path) -> "SitrepConfig":
        """Load configuration from a YAML file (env vars do not override)."""
        with Path(path).open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        return cls(**{k: v for k, v in data.items() if k}, _env_file=None)

    @classmethod
    def bootstrap(cls, persist: bool = True) -> "SitrepConfig":
        """Construct config, create directories, and optionally persist YAML."""
        cfg = cls()
        cfg.ensure_directories()
        if persist:
            try:
                cfg.save_yaml()
            except Exception as exc:  # pragma: no cover
                _logger.warning("could not persist config yaml: %s", exc)
        return cfg


# --------------------------------------------------------------------------- singleton
_CONFIG: Optional[SitrepConfig] = None


def get_config(reload: bool = False, bootstrap: bool = True) -> SitrepConfig:
    """Return the process-wide config singleton, bootstrapping directories once."""
    global _CONFIG
    if _CONFIG is None or reload:
        _CONFIG = SitrepConfig.bootstrap() if bootstrap else SitrepConfig()
    return _CONFIG


def set_config(cfg: SitrepConfig) -> None:
    """Inject a config (useful for tests / dependency injection)."""
    global _CONFIG
    _CONFIG = cfg


def setup_logging(cfg: Optional[SitrepConfig] = None) -> logging.Logger:
    """Configure root logging and return the ``sitrep`` logger."""
    cfg = cfg or get_config(bootstrap=False)
    level = getattr(logging, cfg.log_level, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger("sitrep")
