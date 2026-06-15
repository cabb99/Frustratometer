"""Contact physics: the direct/water/protein contact couplings → sparse Potts J.

This is the single place that defines how contact energy is computed (the channel
contraction ``k·(direct·θ + water·tsw + protein·tsp)``), plus the membrane blend and the
assembly of the sparse Potts model dict. Edit the contact-energy formula here.
"""
import numpy as np

__all__ = ['build_sparse_potts']


def _gamma21(gamma20, ax, ay):
    """Map a (20,20) AWSEM gamma to (21,21) DCA alphabet with the gap row/col zeroed."""
    g = gamma20[ax, ay].copy()
    g[0, :] = 0
    g[:, 0] = 0
    return g


def _contact_J(model, p, theta_c, tsw_c, tsp_c,
               direct_gamma, water_gamma, protein_gamma):
    """k_contact * (direct*theta + water*tsw + protein*tsp) per contact, (Nc, 21, 21)."""
    ax, ay = model.aa_map_awsem_x, model.aa_map_awsem_y
    dg = _gamma21(direct_gamma, ax, ay)
    wg = _gamma21(water_gamma, ax, ay)
    pg = _gamma21(protein_gamma, ax, ay)
    return p.k_contact * (
        dg[np.newaxis, :, :] * theta_c[:, np.newaxis, np.newaxis]
        + wg[np.newaxis, :, :] * tsw_c[:, np.newaxis, np.newaxis]
        + pg[np.newaxis, :, :] * tsp_c[:, np.newaxis, np.newaxis]
    )


def build_sparse_potts(model, p, ci, cj, theta_c, thetaII_c,
                       sigma_water_c, sigma_protein_c, burial_energy):
    """Assemble the sparse Potts model dict from per-contact structural quantities.

    Returns ``{'h': (L,21), 'J': (Nc,21,21), 'contact_i', 'contact_j', 'L'}`` in the
    DCA 21-letter alphabet, including the membrane blend
    ``(1-alpha_i*alpha_j)*water + alpha_i*alpha_j*membrane`` and the Zim field.
    """
    tsw_c = thetaII_c * sigma_water_c
    tsp_c = thetaII_c * sigma_protein_c

    J = _contact_J(model, p, theta_c, tsw_c, tsp_c,
                   model.direct_gamma, model.water_gamma, model.protein_gamma)

    if p.membrane:
        J_membrane = _contact_J(model, p, theta_c, tsw_c, tsp_c,
                                model.membrane_direct_gamma, model.membrane_water_gamma,
                                model.membrane_protein_gamma)
        alpha_ij = (model.alpha[ci] * model.alpha[cj])[:, np.newaxis, np.newaxis]
        J = (1 - alpha_ij) * J + alpha_ij * J_membrane

    h = burial_energy.sum(axis=-1)[:, model.aa_map_awsem_list]
    if model._zim_h is not None:
        h = h + model._zim_h
    h[:, 0] = 0

    return {
        'h': h,
        'J': J,
        'contact_i': ci.astype(np.intp),
        'contact_j': cj.astype(np.intp),
        'L': model.N,
    }
