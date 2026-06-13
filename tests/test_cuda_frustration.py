"""
Tests for frustratometer.frustration.cuda — on-the-fly AWSEM frustration.

Compares the cuda implementation against the existing AWSEM class results.
"""
import pytest
import numpy as np
from pathlib import Path
import frustratometer
from frustratometer.frustration.data import FrustrationData
from frustratometer.frustration import cuda as fcuda
from frustratometer.frustration import numba as fnumba

test_data_path = Path('tests/data')


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_data_from_awsem(model):
    """Build a FrustrationData instance from an existing AWSEM object."""
    return FrustrationData.from_awsem(model)


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def awsem_model():
    """6u5e sparse AWSEM model without electrostatics, seq-sep contact=2."""
    structure = frustratometer.Structure(
        test_data_path / '6u5e.pdb', "A", sparse=True,
    )
    return frustratometer.AWSEM(
        structure,
        distance_cutoff_contact=9.5,
        min_sequence_separation_contact=2,
        k_electrostatics=0,
    )


@pytest.fixture(scope="module")
def awsem_model_elec():
    """6u5e sparse AWSEM with electrostatics enabled."""
    structure = frustratometer.Structure(
        test_data_path / '6u5e.pdb', "A", sparse=True,
    )
    return frustratometer.AWSEM(
        structure,
        distance_cutoff_contact=9.5,
        min_sequence_separation_contact=2,
        k_electrostatics=4.184,
        min_sequence_separation_electrostatics=1,
    )


@pytest.fixture(scope="module")
def cuda_data(awsem_model):
    return _make_data_from_awsem(awsem_model)


@pytest.fixture(scope="module")
def cuda_data_elec(awsem_model_elec):
    return _make_data_from_awsem(awsem_model_elec)


# ── tests ─────────────────────────────────────────────────────────────────────

class TestNativeEnergy:
    def test_native_energy_no_elec(self, awsem_model, cuda_data):
        ref = awsem_model.native_energy()
        got = fcuda.native_energy(cuda_data)
        np.testing.assert_allclose(got, ref, rtol=1e-6)

    def test_native_energy_with_elec(self, awsem_model_elec, cuda_data_elec):
        ref = awsem_model_elec.native_energy()
        got = fcuda.native_energy(cuda_data_elec)
        np.testing.assert_allclose(got, ref, rtol=1e-4)


class TestSingleresideFrustration:
    def test_singleresidue_no_elec(self, awsem_model, cuda_data):
        ref = awsem_model.frustration(kind='singleresidue')
        got = fcuda.singleresidue_frustration(cuda_data)
        # Allow some tolerance due to on-the-fly vs materialized computation
        np.testing.assert_allclose(got, ref, atol=1e-5, rtol=1e-4)

    def test_singleresidue_with_elec(self, awsem_model_elec, cuda_data_elec):
        ref = awsem_model_elec.frustration(kind='singleresidue')
        got = fcuda.singleresidue_frustration(cuda_data_elec)
        np.testing.assert_allclose(got, ref, atol=1e-3, rtol=1e-3)


class TestMutationalFrustration:
    def test_mutational_no_elec(self, awsem_model, cuda_data):
        ref_dense = awsem_model.frustration(kind='mutational')
        got_dense = fcuda.mutational_frustration_dense(cuda_data)
        # Compare only at contact positions
        mask = got_dense != 0
        np.testing.assert_allclose(got_dense[mask], ref_dense[mask], atol=1e-5, rtol=1e-4)

    def test_mutational_with_elec(self, awsem_model_elec, cuda_data_elec):
        ref_dense = awsem_model_elec.frustration(kind='mutational')
        got_dense = fcuda.mutational_frustration_dense(cuda_data_elec)
        mask = got_dense != 0
        np.testing.assert_allclose(got_dense[mask], ref_dense[mask], atol=1e-3, rtol=1e-3)

    def test_mutational_sparse_matches_dense(self, cuda_data):
        sparse = fcuda.mutational_frustration(cuda_data)
        dense = fcuda.mutational_frustration_dense(cuda_data)
        assert sparse.shape == (cuda_data.Nc,)
        # Two separate kernel runs may differ by ~fp64 epsilon (atomic ordering in V build).
        np.testing.assert_allclose(dense[cuda_data.contact_i, cuda_data.contact_j], sparse, atol=1e-12)


@pytest.mark.stochastic
class TestConfigurationalFrustration:
    def test_configurational_correlation(self, awsem_model, cuda_data):
        """Check that configurational frustration is well correlated with AWSEM reference."""
        n_decoys = 100000
        ref = awsem_model.configurational_frustration(n_decoys=n_decoys)
        got = fcuda.configurational_frustration(cuda_data, n_decoys=n_decoys, seed=123)
        # Both are (L, L) with NaN at non-contact positions
        mask = np.isfinite(ref) & np.isfinite(got)
        corr = np.corrcoef(ref[mask], got[mask])[0, 1]
        assert corr > 0.90, f"Correlation {corr:.4f} too low"


# ── fast=True API tests ──────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def fast_model():
    """6u5e sparse AWSEM model with fast=True, backend='cuda', no electrostatics."""
    structure = frustratometer.Structure(
        test_data_path / '6u5e.pdb', "A", sparse=True,
    )
    return frustratometer.AWSEM(
        structure,
        distance_cutoff_contact=9.5,
        min_sequence_separation_contact=2,
        k_electrostatics=0,
        fast=True, backend='cuda',
    )


@pytest.fixture(scope="module")
def fast_model_elec():
    """6u5e sparse AWSEM model with fast=True and electrostatics."""
    structure = frustratometer.Structure(
        test_data_path / '6u5e.pdb', "A", sparse=True,
    )
    return frustratometer.AWSEM(
        structure,
        distance_cutoff_contact=9.5,
        min_sequence_separation_contact=2,
        k_electrostatics=4.184,
        min_sequence_separation_electrostatics=1,
        fast=True, backend='cuda',
    )


class TestFastNativeEnergy:
    def test_fast_native_energy_no_elec(self, awsem_model, fast_model):
        ref = awsem_model.native_energy()
        got = fast_model.native_energy()
        np.testing.assert_allclose(got, ref, rtol=1e-6)

    def test_fast_native_energy_with_elec(self, awsem_model_elec, fast_model_elec):
        ref = awsem_model_elec.native_energy()
        got = fast_model_elec.native_energy()
        np.testing.assert_allclose(got, ref, rtol=1e-4)


class TestFastSingleresideFrustration:
    def test_fast_singleresidue_no_elec(self, awsem_model, fast_model):
        ref = awsem_model.frustration(kind='singleresidue')
        got = fast_model.frustration(kind='singleresidue')
        np.testing.assert_allclose(got, ref, atol=1e-5, rtol=1e-4)

    def test_fast_singleresidue_with_elec(self, awsem_model_elec, fast_model_elec):
        ref = awsem_model_elec.frustration(kind='singleresidue')
        got = fast_model_elec.frustration(kind='singleresidue')
        np.testing.assert_allclose(got, ref, atol=1e-3, rtol=1e-3)


class TestFastMutationalFrustration:
    def test_fast_mutational_no_elec(self, awsem_model, fast_model):
        ref = awsem_model.frustration(kind='mutational')
        got = fast_model.frustration(kind='mutational')
        mask = got != 0
        np.testing.assert_allclose(got[mask], ref[mask], atol=1e-5, rtol=1e-4)

    def test_fast_mutational_with_elec(self, awsem_model_elec, fast_model_elec):
        ref = awsem_model_elec.frustration(kind='mutational')
        got = fast_model_elec.frustration(kind='mutational')
        mask = got != 0
        np.testing.assert_allclose(got[mask], ref[mask], atol=1e-3, rtol=1e-3)

    def test_fast_mutational_sparse(self, fast_model):
        sparse = fast_model.frustration(kind='mutational', dense=False)
        dense = fast_model.frustration(kind='mutational')
        assert sparse.shape == fast_model.N
        np.testing.assert_allclose(dense[sparse.row, sparse.col], sparse.data)


class TestFastAttributes:
    def test_has_frustration_data(self, fast_model):
        assert fast_model._frustration_data is not None

    def test_sequence_preserved(self, awsem_model, fast_model):
        assert fast_model.sequence == awsem_model.sequence

    def test_N_preserved(self, awsem_model, fast_model):
        assert fast_model.N == awsem_model.N

    def test_no_potts_model(self, fast_model):
        assert fast_model.sparse_potts_model is None


# ── FrustrationCUDA class-level API ──────────────────────────────────────────

@pytest.fixture(scope="module")
def cuda_backend(cuda_data):
    """Pre-built FrustrationCUDA instance reused across backend-level tests."""
    return fcuda.FrustrationCUDA(cuda_data)


class TestFrustrationCUDAClass:
    def test_constructor_returns_frustration_cuda(self, cuda_data):
        backend = fcuda.FrustrationCUDA(cuda_data)
        assert isinstance(backend, fcuda.FrustrationCUDA)

    def test_init_L_matches_model(self, awsem_model, cuda_data):
        backend = fcuda.FrustrationCUDA(cuda_data)
        assert backend.L == awsem_model.N

    def test_mutational_sparse_shape(self, cuda_backend):
        values = cuda_backend.mutational(dense=False)
        assert values.shape == (cuda_backend.Nc,)

    def test_configurational_sparse_shape(self, cuda_backend):
        values = cuda_backend.configurational(n_decoys=1000, dense=False)
        assert values.shape == (cuda_backend.n_conf,)

    def test_configurational_dense_shape(self, cuda_backend):
        dense = cuda_backend.configurational(n_decoys=1000, dense=True)
        assert dense.shape == (cuda_backend.L, cuda_backend.L)

    def test_configurational_reseeds_per_call(self, cuda_backend):
        """Repeated configurational() with the same seed is reproducible: the RNG states are
        recreated per call instead of reusing states a prior call advanced. The decoy samples
        are identical; only the atomic-reduction of the decoy statistics adds ~1e-14 noise,
        so the results match to well within the ~few-percent spread of independent samplings."""
        a = cuda_backend.configurational(n_decoys=1000, seed=7, dense=False)
        b = cuda_backend.configurational(n_decoys=1000, seed=7, dense=False)
        np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-6)


# ── Backend reuse ─────────────────────────────────────────────────────────────

class TestBackendReuse:
    """Passing a pre-built FrustrationCUDA to legacy functions must not re-upload."""

    def test_native_energy_prebuilt(self, cuda_backend, awsem_model):
        ref = awsem_model.native_energy()
        got = fcuda.native_energy(cuda_backend)
        np.testing.assert_allclose(got, ref, rtol=1e-6)

    def test_mutational_dense_prebuilt(self, cuda_backend, awsem_model):
        ref = awsem_model.frustration(kind='mutational')
        got = fcuda.mutational_frustration_dense(cuda_backend)
        mask = got != 0
        np.testing.assert_allclose(got[mask], ref[mask], atol=1e-5, rtol=1e-4)


# ── Correction parameter ──────────────────────────────────────────────────────

class TestCorrectionParameter:
    def test_singleresidue_correction_changes_output(self, cuda_backend):
        base = cuda_backend.singleresidue(correction=0.0)
        shifted = cuda_backend.singleresidue(correction=1.0)
        assert not np.allclose(base, shifted)

    def test_mutational_correction_changes_output(self, cuda_backend):
        base = cuda_backend.mutational(correction=0.0, dense=True)
        shifted = cuda_backend.mutational(correction=1.0, dense=True)
        mask = base != 0
        assert not np.allclose(base[mask], shifted[mask])


# ── CUDA vs numba parity ──────────────────────────────────────────────────────

class TestCUDAvsNumbaParity:
    """CUDA and numba backends must agree to within float64 tolerance."""

    def test_native_energy_parity(self, cuda_data):
        cuda_val = fcuda.native_energy(cuda_data)
        numba_val = fnumba.native_energy(cuda_data)
        np.testing.assert_allclose(cuda_val, numba_val, rtol=1e-6)

    def test_singleresidue_parity(self, cuda_data):
        cuda_val = fcuda.singleresidue_frustration(cuda_data)
        numba_val = fnumba.singleresidue_frustration(cuda_data)
        np.testing.assert_allclose(cuda_val, numba_val, atol=1e-5, rtol=1e-4)

    def test_mutational_parity(self, cuda_data):
        cuda_dense = fcuda.mutational_frustration_dense(cuda_data)
        numba_dense = fnumba.mutational_frustration_dense(cuda_data)
        mask = cuda_dense != 0
        np.testing.assert_allclose(cuda_dense[mask], numba_dense[mask], atol=1e-5, rtol=1e-4)


# ── Output dtype ───────────────────────────────────────────────

class TestOutputDtype:
    """Host-facing outputs are always float64 (a Python float for the scalar)."""

    def test_native_energy_is_python_float(self, cuda_backend):
        assert isinstance(cuda_backend.native_energy(), float)

    def test_singleresidue_output_is_float64(self, cuda_backend):
        assert cuda_backend.singleresidue().dtype == np.float64

    def test_mutational_dense_output_is_float64(self, cuda_backend):
        assert cuda_backend.mutational(dense=True).dtype == np.float64

    def test_configurational_output_is_float64(self, cuda_backend):
        assert cuda_backend.configurational(n_decoys=500, dense=True).dtype == np.float64
