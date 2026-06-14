"""Static-context reduction of a sparse Potts model.

A "static context" is a set of residues whose identity is held fixed (a frozen
protein region, or — via an added field — DNA/external charges) while the rest of
the protein ("active" residues) is what we vary and measure frustration on. The
static residues still shape the geometry (rho/burial were computed on the full
structure), but because their identity is fixed their couplings to active residues
reduce to *fields* on the active residues, and static-static couplings reduce to a
constant energy offset.

``fold_static_context`` performs that reduction on an assembled sparse Potts model,
returning a smaller model over the active residues plus the scalar offset. The
result is provably equal to evaluating the full model with the non-active positions
pinned to their native identity (see ``optimization.AwsemEnergySelected``):

    E_full(sigma) = -Σ_i h[i,σ_i] - 0.5 Σ_c J[c, σ_ci, σ_cj]

splits, for active set A / static set S with fixed identities ŝ, into

    E = [-Σ_{i∈A} h'[i,σ_i] - 0.5 Σ_{c∈AA} J[c,σ,σ]] + offset

with the folded active field and constant offset

    h'[i,a] = h[i,a] + Σ_{j∈S, (i,j) contact} J_{ij}(a, ŝ_j)
    offset  = -Σ_{i∈S} h[i,ŝ_i] - 0.5 Σ_{c∈SS} J[c, ŝ_ci, ŝ_cj]

Contacts are stored symmetrically (both (i,j) and (j,i)); the 0.5 factors and the
two directed active-static contributions combine to the unordered-pair field above.
"""
import numpy as np
from scipy.spatial.distance import cdist

__all__ = ['fold_static_context', 'external_charge_field']


def external_charge_field(residue_coords, charge_coords, charges, aa_charges,
                          k_electrostatics, screening_length):
    """Field on ``h`` from external static point charges (e.g. DNA), screened Coulomb.

    Each external charge ``q_d`` at ``charge_coords[d]`` interacts with a protein residue
    of identity ``a`` (charge ``aa_charges[a]``) at ``residue_coords[i]`` through the same
    Debye-Hückel-screened Coulomb used for protein electrostatics:
    ``E = k * Σ_i aa_charges[σ_i] · Σ_d q_d · exp(-r_id/L)/r_id``. Since the native energy
    is ``-Σ_i h[i, σ_i]``, this returns the additive field

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


def fold_static_context(potts, seq_index, active_index, extra_field=None):
    """Reduce a sparse Potts model to its active residues.

    Parameters
    ----------
    potts : dict
        ``{'h': (L, q), 'J': (Nc, q, q), 'contact_i', 'contact_j', 'L'}`` with
        symmetric contacts, in the alphabet of ``seq_index``.
    seq_index : np.ndarray (L,)
        Native identity index of every residue; only the static entries are read.
    active_index : array-like
        Positions (subset of ``range(L)``) that remain active.
    extra_field : np.ndarray (L, q), optional
        An additional field on ``h`` over all residues (e.g. an external DNA/ligand
        charge field from :func:`external_charge_field`). The active rows fold into the
        reduced ``h``; the static rows (fixed identity) fold into ``offset``.

    Returns
    -------
    reduced : dict
        Sparse Potts model over the active residues (``L' = len(active_index)``),
        with the static couplings folded into ``h`` and active-active contacts
        reindexed to active-local positions.
    offset : float
        Constant static contribution to the native energy.
    """
    h = np.asarray(potts['h'])
    if extra_field is not None:
        h = h + np.asarray(extra_field, dtype=np.float64)
    J = np.asarray(potts['J'])
    ci = np.asarray(potts['contact_i'])
    cj = np.asarray(potts['contact_j'])
    L = int(potts['L'])
    seq_index = np.asarray(seq_index)

    active_index = np.sort(np.asarray(active_index, dtype=np.intp))
    is_active = np.zeros(L, dtype=bool)
    is_active[active_index] = True
    local = np.full(L, -1, dtype=np.intp)
    local[active_index] = np.arange(active_index.size)

    h_fold = h[active_index].copy()  # (L', q): active own fields (incl. extra_field)

    static_index = np.where(~is_active)[0]
    offset = -float(h[static_index, seq_index[static_index]].sum())

    ai = is_active[ci]
    aj = is_active[cj]

    # active-active: keep, reindex to active-local
    aa = ai & aj
    reduced = {
        'h': h_fold,
        'J': J[aa],
        'contact_i': local[ci[aa]],
        'contact_j': local[cj[aa]],
        'L': int(active_index.size),
    }

    # active i, static j -> fold onto active i (half; mirror contact supplies the rest)
    m = ai & ~aj
    if m.any():
        u = local[ci[m]]
        sv = seq_index[cj[m]]
        np.add.at(h_fold, u, 0.5 * J[m][np.arange(u.size), :, sv])
    # static i, active j -> fold onto active j
    m = ~ai & aj
    if m.any():
        u = local[cj[m]]
        sv = seq_index[ci[m]]
        np.add.at(h_fold, u, 0.5 * J[m][np.arange(u.size), sv, :])

    # static-static -> constant offset
    ss = ~ai & ~aj
    if ss.any():
        offset -= 0.5 * float(J[ss][np.arange(ss.sum()),
                                    seq_index[ci[ss]], seq_index[cj[ss]]].sum())

    return reduced, offset
