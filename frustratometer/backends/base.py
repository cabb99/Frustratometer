"""Backend (compute engine) abstraction.

A backend is an *engine* — numpy (reference), numba, or cuda — that evaluates energies and
frustration for a model. ``Frustratometer`` holds ``self.engine`` and delegates to it, replacing
the dense/sparse and fast-path ``if`` branches that used to be scattered across its methods.

Transitionally the engine methods take the ``Frustratometer`` instance (``fm``) and read the
model/aux data off it; once the codebase is on ``PottsModel`` they take the model directly.
"""
from abc import ABC, abstractmethod

__all__ = ['Backend']


class Backend(ABC):
    name = 'base'
    supports_configurational = False

    @abstractmethod
    def native_energy(self, fm, sequence, ignore_couplings_of_gaps=False,
                      ignore_fields_of_gaps=False) -> float: ...

    @abstractmethod
    def couplings_energy(self, fm, sequence, ignore_couplings_of_gaps=False) -> float: ...

    @abstractmethod
    def sequences_energies(self, fm, sequences, split_couplings_and_fields=False): ...

    @abstractmethod
    def decoy_fluctuation(self, fm, sequence, kind, mask): ...
