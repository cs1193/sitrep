"""Tests for domain entities and value objects."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.domain.schemas import Fact, Passage, Schema, TemporalFact
from src.domain.value_objects import (
    CompressionRatio,
    Confidence,
    FactStatus,
    TimeRange,
    WeightTriple,
)


def test_schema_promotion_threshold():
    """A schema promotes only after its usage crosses the threshold."""
    schema = Schema(name="Person")
    assert schema.maybe_promote(5) is False
    schema.increment_usage(5)
    assert schema.maybe_promote(5) is True
    assert schema.is_promoted is True


def test_schema_rejects_negative_increment():
    """Negative increments are invalid."""
    with pytest.raises(ValueError):
        Schema(name="Person").increment_usage(-1)


def test_fact_invalidation_flips_status():
    """Invalidation marks a fact invalidated and no longer valid."""
    fact = Fact(subject="a", predicate="is", object_value="b", confidence=0.5)
    assert fact.is_valid_at(datetime.now(timezone.utc)) is True
    fact.invalidate(reason="superseded")
    assert fact.status == FactStatus.INVALIDATED
    assert fact.is_valid_at(datetime.now(timezone.utc)) is False
    assert fact.valid_to is not None


def test_fact_confidence_must_be_in_range():
    """Out-of-range confidence raises."""
    with pytest.raises(ValueError):
        Fact(subject="a", predicate="is", object_value="b", confidence=1.5)


def test_temporal_fact_point_in_time():
    """TemporalFact.at_time respects the recorded timestamp."""
    now = datetime.now(timezone.utc)
    tf = TemporalFact(subject="a", predicate="is", object_value="b")
    assert tf.at_time(now + timedelta(days=1)) is not None
    assert tf.at_time(now - timedelta(days=2)) is None  # before it was recorded


def test_passage_counts_tokens():
    """Passages auto-count tokens when not provided."""
    passage = Passage(text="a short passage with several words", document_id="d1")
    assert passage.tokens > 0


def test_value_objects_enforce_invariants():
    """Value objects reject out-of-range / invalid inputs."""
    assert float(Confidence(0.5)) == 0.5
    with pytest.raises(ValueError):
        Confidence(2.0)
    with pytest.raises(ValueError):
        CompressionRatio(-0.1)
    weights = WeightTriple(1, 1, 1)
    assert abs(sum(weights.as_tuple()) - 1.0) < 1e-9
    with pytest.raises(ValueError):
        WeightTriple(-1, 0, 0)


def test_time_range_contains_and_overlaps():
    """TimeRange membership and overlap behave correctly."""
    start = datetime.now(timezone.utc)
    end = start + timedelta(hours=1)
    tr = TimeRange(start, end)
    assert tr.contains(start + timedelta(minutes=10)) is True
    other = TimeRange(start + timedelta(minutes=30), end + timedelta(hours=1))
    assert tr.overlaps(other) is True
