"""Indicator exposure for the optimization consumers (AwsemEnergyAverage/Variance/Std).

The list order and shapes here are a load-bearing contract with ``optimization.py`` — do not
reorder. Returns the dense (L, L) pairwise indicator maps in the expected order.
"""
import numpy as np

__all__ = ['pair_indicators_dense']


def pair_indicators_dense(model, p, ci, cj, theta_c, thetaII_c,
                          sigma_water_c, sigma_protein_c):
    """Dense (L, L) pairwise indicator maps for ``expose_indicator_functions``.

    Order: water (direct, protein, water) and, under membrane, the membrane triple, each
    weighted by the per-contact blend.
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
