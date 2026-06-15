"""AWSEM physics: one file per term, each the single source for that contribution.

- ``burial`` — per-residue burial field
- ``contact`` — direct/water/protein contact couplings (+ membrane blend) → sparse Potts J
- ``exposure`` — dense indicator maps for the optimization consumers
- ``selection`` — static-context reduction (fold static residues into active fields + offset)
- ``dna_charge`` — external (DNA/ligand) screened-Coulomb charge field
"""
from .burial import build_burial_energy
from .contact import build_sparse_potts
from .exposure import pair_indicators_dense
from .selection import fold_static_context
from .dna_charge import external_charge_field

__all__ = ['build_burial_energy', 'build_sparse_potts', 'pair_indicators_dense',
           'fold_static_context', 'external_charge_field']
