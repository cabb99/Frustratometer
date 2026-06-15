"""Parity tests for membrane support in the fast (numba/cuda) backends.

Membrane lowers the contact coupling to six channels (water + membrane, blended by
the per-contact alpha product) and folds the Zim field into h_struct, so the frozen
kernels (native energy, single-residue, mutational) reproduce the slow membrane model
with no kernel changes. Configurational on a membrane model routes through the numpy
path (the fast configurational kernels are water-only for now), so we only check it runs.
"""
import numpy as np
import pytest

import frustratometer
from frustratometer.frustration.data import FrustrationData
from frustratometer.frustration import numba as fnumba

try:
    from numba import cuda as _nbcuda
    from frustratometer.frustration import cuda as fcuda
    CUDA_AVAILABLE = _nbcuda.is_available()
except Exception:
    CUDA_AVAILABLE = False


@pytest.fixture(scope="module")
def membrane_models():
    """A sparse 2xov membrane model (slow) plus its fast FrustrationData."""
    import os
    data = os.path.join(os.path.dirname(__file__), 'data')
    structure = frustratometer.Structure(os.path.join(data, '2xov.pdb'), 'A',
                                         repair_pdb=False, sparse=True)
    zim = np.where(np.loadtxt(os.path.join(data, 'PredictedZim')) == 2, -1.0, 1.0)
    slow = frustratometer.AWSEM(
        structure, membrane=True, zim=zim,
        min_sequence_separation_contact=10, min_sequence_separation_rho=2,
        distance_cutoff_contact=9.5, k_electrostatics=0.0)
    data_fd = FrustrationData.from_awsem(slow)
    return slow, data_fd


def test_membrane_lowered_to_six_channels(membrane_models):
    _, d = membrane_models
    assert d.coeff.shape[1] == 6
    assert d.gammas.shape == (6, 21, 21)


def test_build_burial_energy_blends_membrane():
    """Burial energy blends (1-alpha)*water + alpha*membrane per residue under membrane,
    and is water-only otherwise. (No-op for the shipped gammas, which have equal burials.)"""
    import types
    from frustratometer.awsem import physics as assembly
    rng = np.random.default_rng(0)
    N, q = 5, 20
    model = types.SimpleNamespace(
        burial_gamma=rng.normal(size=(q, 3)),
        membrane_burial_gamma=rng.normal(size=(q, 3)),
        alpha=rng.random(N))
    bi = rng.normal(size=(N, 3))
    p_mem = types.SimpleNamespace(k_contact=4.184, membrane=True)
    p_water = types.SimpleNamespace(k_contact=4.184, membrane=False)

    water = 0.5 * p_mem.k_contact * model.burial_gamma[None, :, :] * bi[:, None, :]
    mem = 0.5 * p_mem.k_contact * model.membrane_burial_gamma[None, :, :] * bi[:, None, :]
    a = model.alpha[:, None, None]
    np.testing.assert_allclose(assembly.build_burial_energy(model, p_mem, bi),
                               (1 - a) * water + a * mem)
    np.testing.assert_allclose(assembly.build_burial_energy(model, p_water, bi), water)


def test_numba_membrane_native_matches_slow(membrane_models):
    slow, d = membrane_models
    np.testing.assert_allclose(fnumba.native_energy(d), slow.native_energy(), rtol=1e-6)


def test_numba_membrane_singleresidue_matches_slow(membrane_models):
    slow, d = membrane_models
    ref = slow.frustration(kind='singleresidue')
    got = fnumba.singleresidue_frustration(d)
    np.testing.assert_allclose(got, ref, atol=1e-5, rtol=1e-4)


def test_numba_membrane_mutational_matches_slow(membrane_models):
    slow, d = membrane_models
    ref = slow.frustration(kind='mutational')
    got = fnumba.mutational_frustration_dense(d)
    mask = ref != 0
    np.testing.assert_allclose(got[mask], ref[mask], atol=1e-5, rtol=1e-4)


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="no CUDA GPU available")
def test_cuda_membrane_matches_slow(membrane_models):
    slow, d = membrane_models
    np.testing.assert_allclose(fcuda.native_energy(d), slow.native_energy(), rtol=1e-6)
    np.testing.assert_allclose(fcuda.singleresidue_frustration(d),
                               slow.frustration(kind='singleresidue'), atol=1e-5, rtol=1e-4)
    ref = slow.frustration(kind='mutational')
    got = fcuda.mutational_frustration_dense(d)
    mask = ref != 0
    np.testing.assert_allclose(got[mask], ref[mask], atol=1e-5, rtol=1e-4)


def test_fast_membrane_model_native_matches_nonfast():
    """A numba-backend membrane model's native energy matches the equivalent numpy model."""
    import os
    data = os.path.join(os.path.dirname(__file__), 'data')
    structure = frustratometer.Structure(os.path.join(data, '2xov.pdb'), 'A',
                                         repair_pdb=False, sparse=True)
    zim = np.where(np.loadtxt(os.path.join(data, 'PredictedZim')) == 2, -1.0, 1.0)
    kw = dict(membrane=True, zim=zim, min_sequence_separation_contact=10,
              min_sequence_separation_rho=2, distance_cutoff_contact=9.5, k_electrostatics=0.0)
    slow = frustratometer.AWSEM(structure, **kw)
    fast = frustratometer.AWSEM(structure, backend='numba', **kw)
    np.testing.assert_allclose(fast.native_energy(), slow.native_energy(), rtol=1e-6)
    np.testing.assert_allclose(fast.frustration(kind='singleresidue'),
                               slow.frustration(kind='singleresidue'), atol=1e-5, rtol=1e-4)
    # configurational on a membrane fast model must run (routes through numpy) and be finite
    conf = fast.configurational_frustration(n_decoys=2000, seed=1)
    assert np.isfinite(conf[np.isfinite(conf)]).all()
