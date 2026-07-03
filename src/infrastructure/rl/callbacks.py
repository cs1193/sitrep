"""Training callbacks for the PPO compression agent (``stable_baselines3``).

Lazy-imported; provide no-op stubs when sb3 is absent so the module always
imports. ``MetricsCallback`` mirrors episode rewards into the SITREP metrics
collector; ``PeriodicCheckpointCallback`` saves the policy periodically.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

_logger = logging.getLogger("sitrep.rl.callbacks")

try:  # optional dependency
    from stable_baselines3.common.callbacks import BaseCallback  # type: ignore

    HAVE_SB3 = True
except ImportError:  # pragma: no cover - optional dep

    class BaseCallback:  # type: ignore[no-redef]
        """Minimal stand-in so subclasses are defined without sb3."""

        def __init__(self, verbose: int = 0) -> None:
            self.verbose = verbose

        def _on_training_start(self) -> None: ...

        def _on_step(self) -> bool:
            return True

        def on_training_start(self, *a, **k) -> None:
            self._on_training_start()

        def on_step(self) -> bool:
            return self._on_step()

    HAVE_SB3 = False


class MetricsCallback(BaseCallback):
    """Forward episode rewards to the SITREP metrics collector."""

    def __init__(self, verbose: int = 0) -> None:
        """Initialize the callback."""
        super().__init__(verbose)

    def _on_step(self) -> bool:
        """Inspect local episode info buffers and record rewards."""
        try:
            infos = self.locals.get("infos", []) if hasattr(self, "locals") else []
            for info in infos:
                reward = info.get("reward") if isinstance(info, dict) else None
                if reward is not None:
                    from src.infrastructure.monitoring.metrics import get_metrics

                    get_metrics().observe("rl.episode_reward", float(reward))
        except Exception:  # pragma: no cover
            pass
        return True


class PeriodicCheckpointCallback(BaseCallback):
    """Save the policy every *save_freq* steps to *save_path``."""

    def __init__(self, save_freq: int, save_path: str, verbose: int = 0) -> None:
        """Configure checkpoint frequency and destination."""
        super().__init__(verbose)
        self.save_freq = max(1, int(save_freq))
        self.save_path = Path(save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)

    def _on_step(self) -> bool:
        """Checkpoint the model when the step counter hits a multiple of save_freq."""
        if self.n_calls % self.save_freq == 0 and hasattr(self, "model") and self.model is not None:
            try:
                target = self.save_path / f"ppo_{self.n_calls}.zip"
                self.model.save(target)
                _logger.debug("checkpoint saved: %s", target)
            except Exception:  # pragma: no cover
                pass
        return True


def default_callbacks(save_path: Optional[str] = None, save_freq: int = 1000) -> list:
    """Return the standard callback list (metrics + optional checkpoint)."""
    cbs: list = [MetricsCallback()]
    if save_path:
        cbs.append(PeriodicCheckpointCallback(save_freq, save_path))
    return cbs


__all__ = ["MetricsCallback", "PeriodicCheckpointCallback", "default_callbacks", "HAVE_SB3"]
