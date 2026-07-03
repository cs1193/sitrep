"""Causal query use case (Phase G3): effect estimation + counterfactuals."""
from __future__ import annotations

import logging
from typing import Dict, Any

from src.infrastructure.reasoning.causal import CausalEngine

_logger = logging.getLogger("sitrep.usecase.causal")


class CausalQueryUseCase:
    """Answers causal questions over a :class:`CausalEngine`."""

    def __init__(self, engine: CausalEngine) -> None:
        """Store the causal engine."""
        self.engine = engine

    def add_edge(self, cause: str, effect: str, weight: float = 1.0, confidence: float = 1.0) -> None:
        """Add a causal edge to the underlying graph (build the model incrementally)."""
        self.engine.graph.add_edge(cause, effect, weight=weight, confidence=confidence)

    def effect(self, treatment: str, outcome: str) -> Dict[str, Any]:
        """Estimate the causal effect of *treatment* on *outcome* with explanation."""
        return self.engine.estimate_effect(treatment, outcome)

    def counterfactual(
        self,
        treatment: str,
        outcome: str,
        factual_value: float,
        intervention_value: float,
        factual_outcome: float,
    ) -> Dict[str, Any]:
        """Estimate the counterfactual outcome had *treatment* been different."""
        return self.engine.counterfactual(
            treatment, outcome, factual_value, intervention_value, factual_outcome
        )
