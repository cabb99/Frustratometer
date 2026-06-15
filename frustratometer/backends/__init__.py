"""Compute-engine registry (numpy reference; numba/cuda accelerators registered as added)."""
from .base import Backend
from .numpy_engine import NumpyEngine

__all__ = ['Backend', 'NumpyEngine', 'get_engine', 'register_engine', 'BACKEND_REGISTRY']

BACKEND_REGISTRY = {'numpy': NumpyEngine()}


def register_engine(name, engine):
    BACKEND_REGISTRY[name] = engine


def get_engine(name='numpy'):
    name = name or 'numpy'
    try:
        return BACKEND_REGISTRY[name]
    except KeyError:
        raise ValueError(f"Unknown compute engine {name!r}; available: {sorted(BACKEND_REGISTRY)}")
