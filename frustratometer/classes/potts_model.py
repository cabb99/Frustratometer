"""The shared Potts representation for AWSEM and DCA.

``PottsModel`` is the common currency every compute engine (numpy/numba/cuda) consumes:
fields ``h (L, Q)`` and couplings ``J``. Sparse is the canonical coupling form — per-contact
``J (Nc, Q, Q)`` with ``contact_i``/``contact_j`` — and a dense ``(L, L, Q, Q)`` view is
materialized on demand via :meth:`to_dense` for the few consumers that need a full matrix (DCA
``scores()``, ``optimization``). Electrostatics rides along as a rank-1 :class:`ElectrostaticsSidecar`
(it is not foldable into the factored couplings), and static-context folding contributes a constant
``offset``.

The ``as_sparse_dict`` / ``as_dense_dict`` accessors return the legacy dicts that the existing
``frustration.compute_*`` functions and the numba/cuda kernels already accept, so the model can be
introduced without rewriting the compute layer.

During the migration ``PottsModel`` can also hold a pre-materialized dense ``J`` (``J_dense``) so a
"dense" model round-trips unchanged; once the codebase is sparse-canonical, dense is only ever the
on-demand view.
"""
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

__all__ = ['PottsModel', 'ElectrostaticsSidecar']


@dataclass
class ElectrostaticsSidecar:
    """Rank-1 electrostatic term carried alongside the Potts couplings.

    Holds what ``frustration.build_elec_data`` / ``build_elec_data_sparse`` produce today; the
    backends apply it as an additive screened-Coulomb correction (it lives on a wider contact set
    than the structural couplings and is rank-1 in identity, so it is not a factored channel)."""
    data: dict  # the existing _elec_data payload (transitional; structured fields land in Step C)

    @classmethod
    def from_elec_data(cls, elec_data):
        return None if elec_data is None else cls(data=elec_data)


@dataclass
class PottsModel:
    """Fields + couplings over ``L`` sites in a ``Q``-letter alphabet (gap at 0)."""
    h: np.ndarray                              # (L, Q)
    L: int
    J_sparse: Optional[np.ndarray] = None      # (Nc, Q, Q)
    contact_i: Optional[np.ndarray] = None     # (Nc,)
    contact_j: Optional[np.ndarray] = None     # (Nc,)
    J_dense: Optional[np.ndarray] = None       # (L, L, Q, Q) — pre-materialized or cached view
    offset: float = 0.0
    elec: Optional[ElectrostaticsSidecar] = None
    structural: Optional[object] = None        # StructuralModel companion (AWSEM only)

    @property
    def Q(self) -> int:
        return self.h.shape[1]

    @property
    def is_sparse(self) -> bool:
        return self.J_sparse is not None

    @property
    def Nc(self) -> int:
        return 0 if self.contact_i is None else len(self.contact_i)

    def to_dense(self) -> np.ndarray:
        """Materialize (and cache) the dense ``(L, L, Q, Q)`` coupling tensor."""
        if self.J_dense is None:
            J = np.zeros((self.L, self.L, self.Q, self.Q), dtype=np.float64)
            if self.is_sparse and self.Nc:
                np.add.at(J, (self.contact_i, self.contact_j), self.J_sparse)
            self.J_dense = J
        return self.J_dense

    def as_sparse_dict(self) -> dict:
        """Legacy ``{h, J, contact_i, contact_j, L}`` dict (the ``sparse_potts_model`` shape)."""
        if not self.is_sparse:
            raise ValueError("PottsModel has no sparse couplings; build it sparse-canonically.")
        return {'h': self.h, 'J': self.J_sparse,
                'contact_i': self.contact_i, 'contact_j': self.contact_j, 'L': self.L}

    def as_dense_dict(self) -> dict:
        """Legacy ``{h, J}`` dense dict (the ``potts_model`` shape)."""
        return {'h': self.h, 'J': self.to_dense()}

    @classmethod
    def from_sparse_dict(cls, d: dict, **kw) -> 'PottsModel':
        return cls(h=d['h'], L=int(d.get('L', d['h'].shape[0])),
                   J_sparse=d['J'], contact_i=d['contact_i'], contact_j=d['contact_j'], **kw)

    @classmethod
    def from_dense_dict(cls, d: dict, **kw) -> 'PottsModel':
        return cls(h=d['h'], L=int(d['h'].shape[0]), J_dense=d['J'], **kw)
