"""Lowering of AWSEM structural indicators into a Potts model.

Single source of truth for the burial->h and contact-channels->J assembly that
was duplicated across ``AWSEM._build_sparse`` and ``AWSEM._build_sparse_from_contacts``
(and, for the indicator-exposure pair maps, a third time each). The functions take
the constructed ``AWSEM`` model plus the validated ``AWSEMParameters`` and the
per-contact structural quantities, and return the data the builders store.

Couplings are returned in DCA 21-letter alphabet, gap row/col zeroed. The membrane
blend ``(1-alpha_i*alpha_j)*water + alpha_i*alpha_j*membrane`` and the per-residue
Zim field are applied here when ``p.membrane`` is set.
"""
import numpy as np

__all__ = ['build_sparse_potts', 'pair_indicators_dense']


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


def build_burial_energy(model, p, burial_indicator):
    """Per-residue burial energy tensor (N, q, 3): ``0.5*k_contact*bg[a,w]*burial_indicator[i,w]``.

    When membrane is enabled the burial gamma is blended per residue,
    ``(1-alpha_i)*water + alpha_i*membrane`` — matching the configurational burial blend.
    """
    bi = burial_indicator[:, np.newaxis, :]
    water = 0.5 * p.k_contact * model.burial_gamma[np.newaxis, :, :] * bi
    if not p.membrane:
        return water
    mem = 0.5 * p.k_contact * model.membrane_burial_gamma[np.newaxis, :, :] * bi
    a = model.alpha[:, np.newaxis, np.newaxis]
    return (1 - a) * water + a * mem


def build_sparse_potts(model, p, ci, cj, theta_c, thetaII_c,
                       sigma_water_c, sigma_protein_c, burial_energy):
    """Assemble the sparse Potts model dict from per-contact structural quantities.

    Returns ``{'h': (L,21), 'J': (Nc,21,21), 'contact_i', 'contact_j', 'L'}`` in the
    DCA 21-letter alphabet. Mirrors the construction previously inlined in both sparse
    builders, including the membrane blend and the Zim field.
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


def pair_indicators_dense(model, p, ci, cj, theta_c, thetaII_c,
                          sigma_water_c, sigma_protein_c):
    """Dense (L, L) pairwise indicator maps for ``expose_indicator_functions``.

    Returns the list of indicator arrays in the exact order the optimization code
    expects: water (direct, protein, water) and, under membrane, the membrane triple,
    each weighted by the per-contact blend.
    """
    L = model.N
    dense_theta = np.zeros((L, L))
    dense_theta[ci, cj] = theta_c
    dense_protein = np.zeros((L, L))
    dense_protein[ci, cj] = thetaII_c * sigma_protein_c
    dense_water = np.zeros((L, L))
    dense_water[ci, cj] = thetaII_c * sigma_water_c

    if p.membrane:
        alpha_ij_dense = np.zeros((L, L))
        alpha_ij_dense[ci, cj] = model.alpha[ci] * model.alpha[cj]
        return [
            (1 - alpha_ij_dense) * dense_theta,
            (1 - alpha_ij_dense) * dense_protein,
            (1 - alpha_ij_dense) * dense_water,
            alpha_ij_dense * dense_theta,
            alpha_ij_dense * dense_protein,
            alpha_ij_dense * dense_water,
        ]
    return [dense_theta, dense_protein, dense_water]
