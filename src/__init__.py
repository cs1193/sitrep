"""SITREP — Self-Improving Token-Reduced Embeddable Pipeline.

This package root (`src`) is the importable package; submodules are organized
following Clean Architecture:

    src.domain          – entities, value objects, port interfaces
    src.application     – use cases, DTOs, events, agents
    src.adapters        – service adapters, repositories
    src.infrastructure  – db clients, llm gateways, retrieval, rl, kv_cache, ...
    src.presentation    – web (Gradio)
    src.utils           – config, constants, decorators, common helpers
"""

__version__ = "0.1.0"
__all__ = ["__version__"]
