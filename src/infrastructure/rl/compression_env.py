"""Gymnasium environment for learning the compression ratio.

Observation: query embedding (D dims) + 3 normalized stats
(retrieval confidence, result-count ratio, context-length ratio).
Action: continuous compression ratio in [compression_min, compression_max].
Reward: supplied by an :class:`LLMRewardModel` comparing the compressed-context
answer to the full-context answer.

``gymnasium`` is optional: when absent the class still works as a plain Python
object (used by the heuristic policy path); it simply has no registered spaces.
"""
from __future__ import annotations

import logging
import random
from typing import Any, Callable, List, Optional, Sequence, Tuple

from src.adapters.services.compression import CompressionService
from src.domain.interfaces import EmbeddingGateway, LLMGateway, Retriever, RewardModel
from src.utils.common import count_tokens_heuristic

_logger = logging.getLogger("sitrep.rl.env")

try:  # optional dependency
    import gymnasium as gym  # type: ignore
    from gymnasium import spaces  # type: ignore

    HAVE_GYM = True
    _EnvBase = gym.Env
except ImportError:  # pragma: no cover - optional dep
    gym = None  # type: ignore
    spaces = None  # type: ignore
    HAVE_GYM = False
    _EnvBase = object


class CompressionEnv(_EnvBase):  # type: ignore[misc, valid-type]
    """Episodic env: one query → one compression decision → reward."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        embedder: EmbeddingGateway,
        retriever: Retriever,
        compression: CompressionService,
        llm: LLMGateway,
        reward_model: RewardModel,
        queries: Optional[Sequence[str]] = None,
        query_provider: Optional[Callable[[], str]] = None,
        top_k: int = 5,
        compression_min: float = 0.2,
        compression_max: float = 0.8,
        max_ctx_tokens: int = 2048,
        seed: Optional[int] = None,
    ) -> None:
        """Wire all dependencies and (when gymnasium is present) define spaces."""
        self.embedder = embedder
        self.retriever = retriever
        self.compression = compression
        self.llm = llm
        self.reward_model = reward_model
        self.queries = list(queries) if queries else []
        self.query_provider = query_provider
        self.top_k = top_k
        self.compression_min = compression_min
        self.compression_max = compression_max
        self.max_ctx_tokens = max_ctx_tokens
        self._rng = random.Random(seed)
        self._current_query: str = ""
        self._last_conf: float = 0.0
        self._last_n_results: int = 0
        self._last_ctx_tokens: int = 0

        self.embedding_dim = getattr(embedder, "dim", 384)
        if HAVE_GYM and spaces is not None:
            self.observation_space = spaces.Box(
                low=-1.0, high=1.0, shape=(self.embedding_dim + 3,), dtype="float32"
            )
            self.action_space = spaces.Box(
                low=compression_min, high=compression_max, shape=(1,), dtype="float32"
            )

    # ----------------------------------------------------------------- helpers
    def _next_query(self) -> str:
        """Pick the next query from the configured source."""
        if self.query_provider is not None:
            return self.query_provider()
        if self.queries:
            return self._rng.choice(self.queries)
        return "summarize the available context"

    def _build_answer(self, query: str, context: str) -> str:
        """Generate an answer from *context* for *query*."""
        if not context:
            return ""
        prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer concisely:"
        try:
            return self.llm.generate(prompt)
        except Exception as exc:  # pragma: no cover
            _logger.warning("answer generation failed: %s", exc)
            return ""

    def _observe(self) -> List[float]:
        """Build the observation vector (embedding + 3 normalized stats)."""
        emb = list(self.embedder.embed(self._current_query))
        conf = float(max(0.0, min(1.0, self._last_conf)))
        n_results = float(min(1.0, self._last_n_results / max(1, self.top_k)))
        ctx_ratio = float(min(1.0, self._last_ctx_tokens / max(1, self.max_ctx_tokens)))
        return emb + [conf, n_results, ctx_ratio]

    # ----------------------------------------------------------------- gym API
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        """Begin a new episode; return ``(observation, info)``."""
        if seed is not None:
            self._rng = random.Random(seed)
        self._current_query = self._next_query()
        self._last_conf = 0.0
        self._last_n_results = 0
        self._last_ctx_tokens = 0
        return self._observe(), {"query": self._current_query}

    def step(self, action):
        """Apply *action* (compression ratio), return gym step tuple."""
        ratio = float(action[0]) if hasattr(action, "__len__") else float(action)
        ratio = max(self.compression_min, min(self.compression_max, ratio))

        results = self.retriever.retrieve(self._current_query, top_k=self.top_k)
        self._last_n_results = len(results)
        self._last_conf = max((r.score for r in results), default=0.0)
        context = "\n\n".join(r.text for r in results)
        self._last_ctx_tokens = count_tokens_heuristic(context)

        compressed, full_tokens, comp_tokens = self.compression.compress(
            context, ratio=ratio, query=self._current_query
        )
        full_answer = self._build_answer(self._current_query, context)
        compressed_answer = self._build_answer(self._current_query, compressed)
        reward = self.reward_model.score(
            self._current_query, compressed_answer, full_answer, context
        )
        info = {
            "query": self._current_query,
            "ratio": ratio,
            "full_tokens": full_tokens,
            "compressed_tokens": comp_tokens,
            "full_answer": full_answer,
            "compressed_answer": compressed_answer,
        }
        return self._observe(), float(reward), True, False, info

    def render(self):  # pragma: no cover
        """No-op render."""
        return None
