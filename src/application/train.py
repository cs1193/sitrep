"""Train-agent use case: runs PPO and evaluates the compression policy."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional, Tuple

from src.application.dto import TrainResultDTO
from src.application.events import agent_trained
from src.domain.interfaces import RewardModel
from src.domain.schemas import Decision
from src.infrastructure.lineage import LineageTracker
from src.infrastructure.rl.compression_agent import PPOCompressionAgent
from src.utils.decorators import log_execution

_logger = logging.getLogger("sitrep.usecase.train")


class TrainAgentUseCase:
    """Trains the RL compression agent and persists the policy."""

    def __init__(
        self,
        env: Any,
        agent: PPOCompressionAgent,
        reward_model: RewardModel,
        lineage_tracker: LineageTracker,
        config: Optional[Any] = None,
        eval_episodes: int = 5,
    ) -> None:
        """Wire the env, agent, reward model, lineage, and config."""
        self.env = env
        self.agent = agent
        self.reward_model = reward_model
        self.lineage_tracker = lineage_tracker
        self.config = config
        self.eval_episodes = eval_episodes

    @log_execution
    def execute(self, total_timesteps: Optional[int] = None) -> TrainResultDTO:
        """Train (or warm-start) the agent, evaluate, and save the policy."""
        steps = self.agent.train(total_timesteps)
        mean_reward, n_eval = self._evaluate(self.eval_episodes)

        policy_path = self._policy_path()
        try:
            self.agent.save(str(policy_path))
        except Exception as exc:  # pragma: no cover
            _logger.warning("policy save failed: %s", exc)
            policy_path = None

        backend = "ppo" if self.agent.is_trained_backend else "heuristic"
        self.lineage_tracker.record(
            Decision(
                agent_id="rl",
                decision_type="train",
                action="train_compression_agent",
                inputs={"total_timesteps": steps},
                outputs={"mean_reward": mean_reward, "episodes_evaluated": n_eval},
                rationale=f"backend={backend}; policy={'saved' if policy_path else 'unsaved'}",
            )
        )
        agent_trained(backend, steps, mean_reward).publish()
        return TrainResultDTO(
            timesteps=steps,
            backend=backend,
            policy_path=str(policy_path) if policy_path else None,
            mean_reward=mean_reward,
            episodes_evaluated=n_eval,
        )

    # ----------------------------------------------------------------- helpers
    def _policy_path(self) -> Path:
        """Return the configured policy destination path."""
        if self.config is None:
            return Path(".sitrep/agents/policies/ppo_policy")
        return Path(self.config.policies_dir) / "ppo_policy"

    def _evaluate(self, episodes: int) -> Tuple[float, int]:
        """Roll out the current policy and return (mean_reward, n_episodes)."""
        if episodes <= 0:
            return 0.0, 0
        rewards = []
        for _ in range(episodes):
            try:
                obs, _info = self.env.reset()
                action = self.agent.select_ratio(obs)
                _obs, reward, _term, _trunc, _info = self.env.step([action])
                rewards.append(float(reward))
            except Exception as exc:  # pragma: no cover
                _logger.warning("eval rollout failed: %s", exc)
        mean = sum(rewards) / len(rewards) if rewards else 0.0
        return mean, len(rewards)
