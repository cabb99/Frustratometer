"""Registry of fast frustration backends and a generic explicit-Potts dispatcher.

Backends (numba, cuda) are interchangeable executors of the Potts IR. ``get_backend``
resolves a backend by name; ``frustration_potts`` runs single-residue / mutational
frustration for an explicit sparse Potts model ``{h, J, contact_i, contact_j, L}`` on a
chosen backend — the path shared by DCA and the static-context reduced model (any
front-end that already holds an explicit per-contact ``J``).
"""
import numpy as np

from . import numba as _numba_backend

__all__ = ['BACKEND_REGISTRY', 'get_backend', 'frustration_potts']

BACKEND_REGISTRY = {'numba': _numba_backend}
try:  # cuda import is GPU-free (only kernel launches need a device)
    from . import cuda as _cuda_backend
    BACKEND_REGISTRY['cuda'] = _cuda_backend
except Exception:  # pragma: no cover - cuda toolchain absent
    pass


def get_backend(name):
    """Return the backend module registered under ``name`` ('numba' or 'cuda')."""
    name = name or 'numba'
    try:
        return BACKEND_REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"Unknown fast backend {name!r}; available: {sorted(BACKEND_REGISTRY)}")


def frustration_potts(potts, seq_index, kind='singleresidue', aa_freq=None,
                      contact_freq=None, correction=0.0, backend='numba', dense=True):
    """Frustration of an explicit sparse Potts model via a fast backend.

    ``potts`` is ``{h (L,21), J (Nc,21,21), contact_i, contact_j, L}``; ``seq_index`` is
    the native sequence in the model's alphabet. Returns ``(L,)`` for 'singleresidue', and
    for 'mutational' either a dense ``(L, L)`` matrix (``dense=True``) or the per-contact
    ``(Nc,)`` values.
    """
    mod = get_backend(backend)
    fn_name = {'singleresidue': 'singleresidue_frustration_potts',
               'mutational': 'mutational_frustration_potts'}.get(kind)
    if fn_name is None:
        raise ValueError(f"kind must be 'singleresidue' or 'mutational', got {kind!r}")
    if not hasattr(mod, fn_name):
        raise NotImplementedError(
            f"Backend {backend!r} does not implement the explicit-Potts path ({fn_name}).")

    # Match the numpy defaults: a missing frequency table means uniform weighting.
    if aa_freq is None:
        aa_freq = np.ones(21)
    if contact_freq is None:
        contact_freq = np.ones((21, 21))

    if kind == 'singleresidue':
        return mod.singleresidue_frustration_potts(seq_index, potts, aa_freq, correction)

    values = mod.mutational_frustration_potts(seq_index, potts, contact_freq, correction)
    if not dense:
        return values
    out = np.zeros((potts['L'], potts['L']))
    out[np.asarray(potts['contact_i']), np.asarray(potts['contact_j'])] = values
    return out
