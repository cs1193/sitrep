"""Adapter layer: concrete implementations of the domain ports.

This layer turns the abstract ``src.domain.interfaces`` protocols into working
SQLite/Chroma/Kuzu/LLM-backed components. The application layer depends on the
ports, never on these concrete classes.
"""
