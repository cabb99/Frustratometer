"""Potts intermediate representation (IR) shared by AWSEM and DCA.

The IR is the narrow waist of the frustration stack: physics front-ends
(AWSEM terms, DCA inference) *lower* to it, and the compute backends
(numpy/numba/cuda) *execute* it. It knows about fields and couplings, not
about membrane/electrostatics/DCA/DNA — those are folded in at lowering time.

A :class:`PottsIR` is a field matrix ``h (L, Q)`` plus a list of
:class:`CouplingBlock` and a scalar ``offset``. Each block stores its couplings
in one of two modes:

- **explicit** — ``J (Nb, Q, Q)`` arbitrary per-contact matrices (DCA).
- **factored** — ``coeff (Nb, T)`` and ``G (T, Q, Q)`` with pair energy
  ``Σ_t coeff[b, t] · G[t, a, b]`` (AWSEM contact channels; memory-lean and the
  form the configurational engine needs).

The energy of a sequence ``s`` is
``-Σ_i h[i, s_i] - 0.5 Σ_blocks Σ_b J_b(s_i, s_j) + offset`` (couplings are
stored symmetrically, hence the 0.5).
"""

from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np

__all__ = ['CouplingBlock', 'PottsIR']


@dataclass
class CouplingBlock:
    """A set of pair couplings over contacts ``(contact_i, contact_j)``.

    Exactly one storage mode is populated: ``J`` (explicit) or ``coeff``+``G``
    (factored). ``scale`` is a global multiplier applied to the pair energy
    (e.g. ``k_contact`` for the AWSEM structural block).
    """
    contact_i: np.ndarray
    contact_j: np.ndarray
    J: Optional[np.ndarray] = None          # explicit (Nb, Q, Q)
    coeff: Optional[np.ndarray] = None      # factored (Nb, T)
    G: Optional[np.ndarray] = None          # factored (T, Q, Q)
    scale: float = 1.0
    name: str = ''

    def __post_init__(self):
        explicit = self.J is not None
        factored = self.coeff is not None and self.G is not None
        if explicit == factored:
            raise ValueError("CouplingBlock needs exactly one of J or (coeff, G)")

    @property
    def mode(self) -> str:
        return 'explicit' if self.J is not None else 'factored'

    @property
    def Nb(self) -> int:
        return len(self.contact_i)

    @property
    def Q(self) -> int:
        return self.J.shape[1] if self.J is not None else self.G.shape[1]

    def explicit_J(self) -> np.ndarray:
        """Materialize this block's couplings as ``(Nb, Q, Q)`` (scale folded in)."""
        if self.J is not None:
            return self.scale * self.J
        # Σ_t coeff[b, t] · G[t, a, b]  →  (Nb, Q, Q)
        return self.scale * np.einsum('bt,tij->bij', self.coeff, self.G)

    def accumulate_dense(self, J_dense: np.ndarray) -> None:
        """Add this block into a dense ``(L, L, Q, Q)`` coupling tensor in place."""
        Jb = self.explicit_J()
        np.add.at(J_dense, (self.contact_i, self.contact_j), Jb)


@dataclass
class PottsIR:
    """Fields + coupling blocks + constant offset over ``L`` sites, alphabet ``Q``."""
    L: int
    q: int
    h: np.ndarray                                   # (L, Q)
    blocks: List[CouplingBlock] = field(default_factory=list)
    offset: float = 0.0

    def to_dense(self) -> dict:
        """Materialize a dense Potts model ``{'h': (L, Q), 'J': (L, L, Q, Q)}``."""
        J = np.zeros((self.L, self.L, self.q, self.q), dtype=np.float64)
        for block in self.blocks:
            block.accumulate_dense(J)
        return {'h': self.h, 'J': J}

    @classmethod
    def from_sparse_potts_dict(cls, d: dict, offset: float = 0.0) -> 'PottsIR':
        """Build from the legacy ``{h, J (Nc,Q,Q), contact_i, contact_j, L}`` dict."""
        h = d['h']
        block = CouplingBlock(contact_i=d['contact_i'], contact_j=d['contact_j'],
                              J=d['J'], name='potts')
        return cls(L=d.get('L', h.shape[0]), q=h.shape[1], h=h, blocks=[block],
                   offset=offset)

    def to_sparse_potts_dict(self) -> dict:
        """Collapse to the legacy single-block ``{h, J, contact_i, contact_j, L}`` dict.

        Blocks sharing the same contact list are summed; differing contact lists
        are concatenated (duplicate pairs are allowed by the downstream sparse
        energy functions, which sum over rows).
        """
        if len(self.blocks) == 1:
            b = self.blocks[0]
            return {'h': self.h, 'J': b.explicit_J(),
                    'contact_i': b.contact_i, 'contact_j': b.contact_j, 'L': self.L}
        ci = np.concatenate([b.contact_i for b in self.blocks])
        cj = np.concatenate([b.contact_j for b in self.blocks])
        J = np.concatenate([b.explicit_J() for b in self.blocks], axis=0)
        return {'h': self.h, 'J': J, 'contact_i': ci, 'contact_j': cj, 'L': self.L}
