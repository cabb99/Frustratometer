"""Selection / static-context physics: reduce a sparse Potts model to its active residues.

A "static context" is a set of residues whose identity is held fixed (a frozen protein region,
or — via an added field — DNA/external charges) while the rest ("active" residues) is varied and
measured. Static residues still shape the geometry (rho/burial were computed on the full
structure), but because their identity is fixed their couplings to active residues reduce to
*fields* on the active residues, and static-static couplings reduce to a constant offset.

The reduction is provably equal to evaluating the full model with the non-active positions pinned
to their native identity (see ``optimization.AwsemEnergySelected``):

    E_full(sigma) = -Σ_i h[i,σ_i] - 0.5 Σ_c J[c, σ_ci, σ_cj]

splits, for active set A / static set S with fixed identities ŝ, into

    E = [-Σ_{i∈A} h'[i,σ_i] - 0.5 Σ_{c∈AA} J[c,σ,σ]] + offset
    h'[i,a] = h[i,a] + Σ_{j∈S, (i,j) contact} J_{ij}(a, ŝ_j)
    offset  = -Σ_{i∈S} h[i,ŝ_i] - 0.5 Σ_{c∈SS} J[c, ŝ_ci, ŝ_cj]

Contacts are stored symmetrically; the 0.5 factors + the two directed active-static
contributions combine to the unordered-pair field above.
"""
import numpy as np

__all__ = ['fold_static_context']


def fold_static_context(potts, seq_index, active_index, extra_field=None):
    """Reduce a sparse Potts model to its active residues.

    Parameters
    ----------
    potts : dict
        ``{'h': (L, q), 'J': (Nc, q, q), 'contact_i', 'contact_j', 'L'}`` with symmetric
        contacts, in the alphabet of ``seq_index``.
    seq_index : np.ndarray (L,)
        Native identity index of every residue; only the static entries are read.
    active_index : array-like
        Positions (subset of ``range(L)``) that remain active.
    extra_field : np.ndarray (L, q), optional
        Additional field on ``h`` over all residues (e.g. a DNA/ligand charge field). Active rows
        fold into the reduced ``h``; static rows fold into ``offset``.

    Returns
    -------
    (reduced, offset) : (dict, float)
        Sparse Potts model over the active residues + the constant static contribution.
    """
    h = np.asarray(potts['h'])
    if extra_field is not None:
        h = h + np.asarray(extra_field, dtype=np.float64)
    J = np.asarray(potts['J'])
    contact_i = np.asarray(potts['contact_i'])
    contact_j = np.asarray(potts['contact_j'])
    n_residues = int(potts['L'])
    seq_index = np.asarray(seq_index)

    active_index = np.sort(np.asarray(active_index, dtype=np.intp))
    is_active = np.zeros(n_residues, dtype=bool)
    is_active[active_index] = True
    to_local = np.full(n_residues, -1, dtype=np.intp)
    to_local[active_index] = np.arange(active_index.size)

    active_h = h[active_index].copy()

    static_index = np.where(~is_active)[0]
    offset = -float(h[static_index, seq_index[static_index]].sum())

    i_active = is_active[contact_i]
    j_active = is_active[contact_j]

    both_active = i_active & j_active
    reduced = {
        'h': active_h,
        'J': J[both_active],
        'contact_i': to_local[contact_i[both_active]],
        'contact_j': to_local[contact_j[both_active]],
        'L': int(active_index.size),
    }

    # An active-static contact reduces to half a field on the active residue, selecting the
    # coupling slice fixed by the static partner's identity; the mirror contact gives the rest.
    def fold_active_static(mask, active_residue, static_residue, static_is_j):
        if not mask.any():
            return
        rows = to_local[active_residue[mask]]
        partner = seq_index[static_residue[mask]]
        blocks = J[mask]
        pick = np.arange(rows.size)
        fields = blocks[pick, :, partner] if static_is_j else blocks[pick, partner, :]
        np.add.at(active_h, rows, 0.5 * fields)

    fold_active_static(i_active & ~j_active, contact_i, contact_j, static_is_j=True)
    fold_active_static(~i_active & j_active, contact_j, contact_i, static_is_j=False)

    both_static = ~i_active & ~j_active
    if both_static.any():
        pick = np.arange(int(both_static.sum()))
        pair_energy = J[both_static][pick, seq_index[contact_i[both_static]],
                                     seq_index[contact_j[both_static]]]
        offset -= 0.5 * float(pair_energy.sum())

    return reduced, offset
