"""DNA / external static charge field.

An external static charge (DNA phosphate, ligand) acts on the active protein residues through the
same Debye-Hückel-screened Coulomb as the protein electrostatics. Because a residue's energy
scales with its identity's charge, the field is rank-1 in identity and folds into ``h``.
"""
import numpy as np
from scipy.spatial.distance import cdist

__all__ = ['external_charge_field']


def external_charge_field(residue_coords, charge_coords, charges, aa_charges,
                          k_electrostatics, screening_length):
    """Field on ``h`` from external static point charges, screened Coulomb.

        ``h_ext[i, a] = -k · aa_charges[a] · Σ_d q_d · exp(-r_id/L)/r_id``   shape (N, Q).

    Parameters
    ----------
    residue_coords : (N, 3)   representative (CB; CA for Gly) coordinates of the protein residues.
    charge_coords  : (M, 3)   coordinates of the external charges.
    charges        : (M,)     external charge values (e.g. -1 per DNA phosphate).
    aa_charges     : (Q,)     per-identity residue charge in the model alphabet.
    k_electrostatics, screening_length : Coulomb prefactor and screening length (Angstrom).
    """
    residue_coords = np.asarray(residue_coords, dtype=np.float64)
    charge_coords = np.asarray(charge_coords, dtype=np.float64)
    charges = np.asarray(charges, dtype=np.float64)
    d = cdist(residue_coords, charge_coords)  # (N, M)
    with np.errstate(divide='ignore', invalid='ignore'):
        ind = np.exp(-d / screening_length) / d
    ind = np.nan_to_num(ind, nan=0.0, posinf=0.0, neginf=0.0)
    phi = ind @ charges  # (N,) screened potential from the external charges at each residue
    return -k_electrostatics * np.outer(phi, np.asarray(aa_charges, dtype=np.float64))
