"""PPO compression agent (``stable-baselines3``) with a heuristic fallback.

Implements the :class:`~src.domain.interfaces.CompressionPolicy` port. When
``stable_baselines3``/``torch`` are unavailable, it degrades to a confidence-
aware heuristic policy that still selects sensible compression ratios.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional, Sequence

from src.domain.interfaces import CompressionPolicy
from src.utils.config import SitrepConfig, get_config

_logger = logging.getLogger("sitrep.rl.agent")


def _have_sb3() -> bool:
    """Return True if ``stable_baselines3`` and ``torch`` are importable."""
    try:
        import stable_baselines3  # type: ignore  # noqa: F401
        import torch  # type: ignore  # noqa: F401

        return True
    except ImportError:
        return False


class HeuristicCompressionPolicy(CompressionPolicy):
    """Confidence-aware compression policy (no learning required)."""

    def __init__(self, low: float = 0.2, high: float = 0.8) -> None:
        """Set the ratio range [low=most compression, high=least compression]."""
        self.low = low
        self.high = high

    def select_ratio(self, observation: Sequence[float]) -> float:
        """Return a ratio in [low, high]: higher confidence → more compression.

        Observation's last 3 entries are assumed to be
        ``[confidence, n_results_ratio, ctx_length_ratio]``.
        """
        stats = list(observation[-3:]) if len(observation) >= 3 else [0.5, 0.5, 0.5]
        conf = max(0.0, min(1.0, float(stats[0])))
        # High confidence → we can afford to retain less (lower ratio).
        ratio = self.high - conf * (self.high - self.low)
        return float(max(self.low, min(self.high, ratio)))

    def save(self, path: str) -> None:
        """Persist heuristic parameters as JSON."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"low": self.low, "high": self.high, "kind": "heuristic"}, fh)

    def load(self, path: str) -> None:
        """Load heuristic parameters from JSON (best-effort)."""
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            self.low = float(data.get("low", self.low))
            self.high = float(data.get("high", self.high))
        except FileNotFoundError:
            pass


class PPOCompressionAgent(CompressionPolicy):
    """PPO agent wrapping a Gymnasium compression env, with heuristic fallback."""

    def __init__(
        self,
        env: Any,
        config: Optional[SitrepConfig] = None,
        policy_path: Optional[str] = None,
    ) -> None:
        """Construct the agent, creating/loading a PPO model when sb3 is present."""
        self.env = env
        self.cfg = config or get_config(bootstrap=False)
        self._fallback = HeuristicCompressionPolicy(
            self.cfg.compression_min, self.cfg.compression_max
        )
        self._model: Any = None
        if _have_sb3():
            try:
                from stable_baselines3 import PPO  # type: ignore

                self._model = PPO(
                    "MlpPolicy",
                    env,
                    learning_rate=self.cfg.ppo_learning_rate,
                    n_steps=self.cfg.ppo_n_steps,
                    batch_size=self.cfg.ppo_batch_size,
                    gamma=self.cfg.ppo_gamma,
                    verbose=0,
                )
                if policy_path and Path(policy_path).exists():
                    self.load(policy_path)
                _logger.info("PPO compression agent ready (stable-baselines3)")
            except Exception as exc:  # pragma: no cover
                _logger.warning("PPO init failed, using heuristic: %s", exc)
                self._model = None
        else:
            _logger.info("stable-baselines3 unavailable; using heuristic compression policy")

    @property
    def is_trained_backend(self) -> bool:
        """Return True when a real PPO model is active."""
        return self._model is not None

    def select_ratio(self, observation: Sequence[float]) -> float:
        """Select a compression ratio from the observation (PPO or heuristic)."""
        if self._model is not None:
            try:
                import numpy as np  # type: ignore

                action, _ = self._model.predict(
                    np.asarray(observation, dtype="float32"), deterministic=True
                )
                value = float(action[0]) if hasattr(action, "__len__") else float(action)
                return max(self.cfg.compression_min, min(self.cfg.compression_max, value))
            except Exception as exc:  # pragma: no cover
                _logger.warning("PPO predict failed, using heuristic: %s", exc)
        return self._fallback.select_ratio(observation)

    def train(self, total_timesteps: Optional[int] = None, callbacks: Any = None) -> int:
        """Train the PPO model; no-op (returns 0) when only the heuristic is active."""
        if self._model is None:
            _logger.warning("training skipped (heuristic policy active)")
            return 0
        steps = int(total_timesteps or self.cfg.ppo_total_timesteps)
        _logger.info("training PPO for %d timesteps...", steps)
        self._model.learn(total_timesteps=steps, callback=callbacks or [])
        return steps

    def save(self, path: str) -> None:
        """Persist the policy (PPO zip or heuristic JSON)."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        if self._model is not None:
            self._model.save(path)
        else:
            self._fallback.save(path if path.endswith(".json") else path + ".json")

    def load(self, path: str) -> None:
        """Load the policy from *path*."""
        if self._model is not None and Path(path).exists():
            try:
                from stable_baselines3 import PPO  # type: ignore

                self._model = PPO.load(path, env=self.env)
                return
            except Exception as exc:  # pragma: no cover
                _logger.warning("PPO load failed: %s", exc)
        self._fallback.load(path if path.endswith(".json") else path + ".json")
