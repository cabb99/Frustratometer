"""Tests for the explicit-J fast path (frustratometer.frustration.backends).

The explicit-J kernels run frustration for any sparse Potts model {h, J, contacts}
without AWSEM channels or electrostatics — the path shared by DCA and the static-context
reduced model. They must match the established numpy sparse frustration, and a DCA model
computed with backend='numba' must match its numpy result.
"""
import numpy as np
import pytest

import frustratometer
from frustratometer import frustration
from frustratometer.frustration.frustration import _AA, compute_seq_index
from frustratometer.frustration import numba as fnumba
from frustratometer.frustration import backends

try:
    from numba import cuda as _nbcuda
    CUDA_AVAILABLE = _nbcuda.is_available()
except Exception:
    CUDA_AVAILABLE = False


@pytest.fixture(scope="module")
def awsem_sparse():
    s = frustratometer.Structure('tests/data/6u5e.pdb', 'A', sparse=True)
    return frustratometer.AWSEM(s, distance_cutoff_contact=9.5,
                                min_sequence_separation_contact=2, k_electrostatics=0)


def test_explicit_native_matches_numpy(awsem_sparse):
    m = awsem_sparse
    si = compute_seq_index(m.sequence)
    got = fnumba.native_energy_potts(si, m.sparse_potts_model)
    np.testing.assert_allclose(got, m.native_energy(), rtol=1e-7, atol=1e-7)


def test_explicit_singleresidue_matches_numpy(awsem_sparse):
    m = awsem_sparse
    si = compute_seq_index(m.sequence)
    got = fnumba.singleresidue_frustration_potts(si, m.sparse_potts_model, m.aa_freq)
    np.testing.assert_allclose(got, m.frustration(kind='singleresidue'), atol=1e-6, rtol=1e-5)


def test_explicit_mutational_matches_numpy(awsem_sparse):
    m = awsem_sparse
    si = compute_seq_index(m.sequence)
    got = fnumba.mutational_frustration_potts(si, m.sparse_potts_model, m.contact_freq)
    ref = m.frustration(kind='mutational')
    ci, cj = m.sparse_potts_model['contact_i'], m.sparse_potts_model['contact_j']
    np.testing.assert_allclose(got, ref[ci, cj], atol=1e-6, rtol=1e-5)


def test_backends_registry():
    assert 'numba' in backends.BACKEND_REGISTRY
    assert backends.get_backend('numba') is fnumba
    with pytest.raises(ValueError):
        backends.get_backend('nonsense')


def test_frustration_potts_dense_and_sparse(awsem_sparse):
    m = awsem_sparse
    si = compute_seq_index(m.sequence)
    potts = m.sparse_potts_model
    dense = backends.frustration_potts(potts, si, kind='mutational',
                                       contact_freq=m.contact_freq, backend='numba', dense=True)
    sparse = backends.frustration_potts(potts, si, kind='mutational',
                                        contact_freq=m.contact_freq, backend='numba', dense=False)
    assert dense.shape == (m.N, m.N)
    np.testing.assert_allclose(dense[potts['contact_i'], potts['contact_j']], sparse)


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="no CUDA GPU available")
def test_cuda_explicit_matches_numba(awsem_sparse):
    from frustratometer.frustration import cuda as fcuda
    if not hasattr(fcuda, 'singleresidue_frustration_potts'):
        pytest.skip("cuda explicit-Potts path not implemented yet")
    m = awsem_sparse
    si = compute_seq_index(m.sequence)
    np.testing.assert_allclose(
        fcuda.singleresidue_frustration_potts(si, m.sparse_potts_model, m.aa_freq),
        fnumba.singleresidue_frustration_potts(si, m.sparse_potts_model, m.aa_freq),
        atol=1e-6, rtol=1e-5)


# ── DCA on the fast engine ──────────────────────────────────────────────────

@pytest.fixture(scope="module")
def dca_model():
    s = frustratometer.Structure('examples/data/1cyo.pdb', 'A',
                                 distance_matrix_method='minimum', repair_pdb=False)
    return frustratometer.DCA.from_potts_model_file(
        s, 'examples/data/PottsModel1cyoA.mat', distance_cutoff=4, sequence_cutoff=0)


def test_dca_singleresidue_numba_matches_numpy(dca_model):
    ref = dca_model.frustration(kind='singleresidue')
    got = dca_model.frustration(kind='singleresidue', backend='numba')
    np.testing.assert_allclose(got, ref, atol=1e-6, rtol=1e-5)


def test_dca_mutational_numba_matches_numpy(dca_model):
    ref = dca_model.frustration(kind='mutational')  # dense (L,L) over the mask
    got = dca_model.frustration(kind='mutational', backend='numba')
    mask = dca_model.mask.astype(bool)
    np.testing.assert_allclose(got[mask], ref[mask], atol=1e-6, rtol=1e-5)
