"""Infrastructure layer: concrete adapters for storage, LLM, retrieval, RL, KV cache.

Everything here implements the ports declared in ``src.domain.interfaces``.
Heavy third-party dependencies are imported lazily so the package imports even
when optional extras are not installed.
"""
