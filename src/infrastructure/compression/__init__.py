"""Headroom content-aware compression.

Routes input by content type (JSON / code / logs / natural language) to a
specialised compressor, with the existing extractive compressor as the
natural-language fallback. All compressors implement
``compress(text: str, ratio: float) -> str``.
"""
from src.infrastructure.compression.code_compressor import CodeCompressor
from src.infrastructure.compression.kompress import Kompress
from src.infrastructure.compression.log_compressor import LogCompressor
from src.infrastructure.compression.router import ContentRouter, ContentType
from src.infrastructure.compression.smart_crusher import SmartCrusher

__all__ = [
    "ContentRouter",
    "ContentType",
    "SmartCrusher",
    "CodeCompressor",
    "LogCompressor",
    "Kompress",
]
