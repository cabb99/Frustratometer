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

__all__ = ['fold_static_context']


def fold_static_context(potts, seq_index, active_index):
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

    h_fold = h[active_index].copy()  # (L', q): active own fields

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
