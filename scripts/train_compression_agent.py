#!/usr/bin/env python3
"""Train the PPO compression agent on accumulated feedback (extra ``[rl]``)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    """Parse args and run training."""
    parser = argparse.ArgumentParser(description="Train the SITREP compression RL agent.")
    parser.add_argument(
        "--timesteps", type=int, default=None, help="Total PPO timesteps (heuristic mode ignores this)"
    )
    args = parser.parse_args()

    from src.application import build_application

    app = build_application()
    result = app.train_uc.execute(total_timesteps=args.timesteps)
    app.close()
    print(
        f"\nTraining complete ({result.backend} backend):\n"
        f"  timesteps         : {result.timesteps}\n"
        f"  mean_reward       : {result.mean_reward:.4f}\n"
        f"  episodes_evaluated: {result.episodes_evaluated}\n"
        f"  policy_path       : {result.policy_path}"
    )
    if result.backend == "heuristic":
        print(
            "\nℹ️  Heuristic policy active. Install RL extras to train a real PPO model:\n"
            "    uv sync --extra rl"
        )


if __name__ == "__main__":
    main()
