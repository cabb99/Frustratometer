"""Burial physics: the per-residue burial field contribution."""
import numpy as np

__all__ = ['build_burial_energy']


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
