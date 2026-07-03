"""Service adapters: extraction, embedding, conflict, compression, classification, quality, confidence."""
from src.adapters.services.classification import ClassificationService
from src.adapters.services.compression import CompressionService
from src.adapters.services.confidence import ConfidenceEstimator
from src.adapters.services.conflict import (
    Conflict,
    ConflictDetectionService,
    ConflictResolutionService,
)
from src.adapters.services.embedding import EmbeddingService
from src.adapters.services.extraction import ExtractionResult, ExtractionService
from src.adapters.services.quality import QualityEstimator

__all__ = [
    "EmbeddingService",
    "ExtractionService",
    "ExtractionResult",
    "ConflictDetectionService",
    "ConflictResolutionService",
    "Conflict",
    "CompressionService",
    "ClassificationService",
    "QualityEstimator",
    "ConfidenceEstimator",
]
