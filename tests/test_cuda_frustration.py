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
    return fcuda.prepare_cuda(cuda_data)


class TestFrustrationCUDAClass:
    def test_prepare_cuda_returns_frustration_cuda(self, cuda_data):
        backend = fcuda.prepare_cuda(cuda_data)
        assert isinstance(backend, fcuda.FrustrationCUDA)

    def test_device_data_is_frustration_cuda(self, cuda_data):
        dd = fcuda.DeviceData(cuda_data)
        assert isinstance(dd, fcuda.FrustrationCUDA)

    def test_init_L_matches_model(self, awsem_model, cuda_data):
        backend = fcuda.prepare_cuda(cuda_data)
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


# ── Legacy tuple API ──────────────────────────────────────────────────────────

class TestLegacyTupleAPI:
    def test_mutational_frustration_returns_4_tuple(self, cuda_data):
        result = fcuda.mutational_frustration(cuda_data)
        assert len(result) == 4

    def test_mutational_frustration_sparse_matches_dense(self, cuda_data):
        values, ci, cj, _ = fcuda.mutational_frustration(cuda_data)
        dense = fcuda.mutational_frustration_dense(cuda_data)
        # Two separate kernel runs may differ by ~fp64 machine epsilon due to
        # non-deterministic atomic ordering in _build_v_kernel.
        np.testing.assert_allclose(dense[ci, cj], values, atol=1e-12)

    def test_mutational_frustration_L_matches_model(self, awsem_model, cuda_data):
        _, _, _, L = fcuda.mutational_frustration(cuda_data)
        assert L == awsem_model.N


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


# ── Phase 2: mixed-precision workspace ───────────────────────────────────────

@pytest.fixture(scope="module")
def f32_backend(cuda_data):
    return fcuda.prepare_cuda(cuda_data, workspace_dtype=np.float32)


@pytest.fixture(scope="module")
def f64_backend(cuda_data):
    return fcuda.prepare_cuda(cuda_data, workspace_dtype=np.float64)


class TestMixedPrecisionAPI:
    """workspace_dtype parameter is accepted and workspace arrays have the right dtype."""

    def test_f32_backend_is_frustration_cuda(self, cuda_data):
        backend = fcuda.prepare_cuda(cuda_data, workspace_dtype=np.float32)
        assert isinstance(backend, fcuda.FrustrationCUDA)

    def test_default_workspace_dtype_is_float64(self, f64_backend):
        assert f64_backend.d_V.dtype == np.float64

    def test_v_matrix_is_float32(self, f32_backend):
        assert f32_backend.d_V.dtype == np.float32

    def test_config_decoy_is_float32(self, f32_backend):
        assert f32_backend.d_config_decoy.dtype == np.float32

    def test_config_stats_stays_float64(self, f32_backend):
        # Accumulation buffer must stay float64 to preserve score precision.
        assert f32_backend.d_config_stats.dtype == np.float64


class TestMixedPrecisionOutputDtype:
    """Host-facing outputs must always be float64 regardless of workspace_dtype."""

    def test_native_energy_is_python_float(self, f32_backend):
        assert isinstance(f32_backend.native_energy(), float)

    def test_singleresidue_output_is_float64(self, f32_backend):
        assert f32_backend.singleresidue().dtype == np.float64

    def test_mutational_sparse_output_is_float64(self, f32_backend):
        assert f32_backend.mutational(dense=False).dtype == np.float64

    def test_mutational_dense_output_is_float64(self, f32_backend):
        assert f32_backend.mutational(dense=True).dtype == np.float64

    def test_configurational_output_is_float64(self, f32_backend):
        result = f32_backend.configurational(n_decoys=500, dense=True)
        assert result.dtype == np.float64


class TestMixedPrecisionDrift:
    """float32 workspace results must agree with float64 within agreed tolerances.

    Tolerances:
    - native_energy: rtol=1e-3  (energy is a single scalar, tight)
    - singleresidue: atol=5e-2  (z-score; ~2 decimal places)
    - mutational:    atol=5e-2  (z-score; ~2 decimal places)
    """

    def test_native_energy_drift(self, f32_backend, f64_backend):
        ref = f64_backend.native_energy()
        got = f32_backend.native_energy()
        np.testing.assert_allclose(got, ref, rtol=1e-3)

    def test_singleresidue_drift(self, f32_backend, f64_backend):
        ref = f64_backend.singleresidue()
        got = f32_backend.singleresidue()
        np.testing.assert_allclose(got, ref, atol=5e-2, rtol=5e-2)

    def test_mutational_drift(self, f32_backend, f64_backend):
        ref = f64_backend.mutational(dense=True)
        got = f32_backend.mutational(dense=True)
        mask = ref != 0
        np.testing.assert_allclose(got[mask], ref[mask], atol=5e-2, rtol=5e-2)


# ── Phase 3: CPU-precomputed V upload ─────────────────────────────────────────

@pytest.fixture(scope="module")
def cpu_v_backend(cuda_data):
    return fcuda.prepare_cuda(cuda_data, build_v_on_cpu=True)


class TestCPUPrecomputedV:
    """build_v_on_cpu=True must upload a CPU-built V matrix instead of using the GPU kernel."""

    def test_build_v_on_cpu_is_accepted(self, cuda_data):
        backend = fcuda.prepare_cuda(cuda_data, build_v_on_cpu=True)
        assert isinstance(backend, fcuda.FrustrationCUDA)

    def test_v_shape_preserved(self, cpu_v_backend):
        assert cpu_v_backend.d_V.shape == (cpu_v_backend.L, 21)

    def test_v_dtype_preserved(self, cpu_v_backend):
        assert cpu_v_backend.d_V.dtype == np.float64

    def test_native_energy_matches_gpu_build(self, cuda_data, cpu_v_backend):
        gpu_val = fcuda.native_energy(cuda_data)
        cpu_val = cpu_v_backend.native_energy()
        np.testing.assert_allclose(cpu_val, gpu_val, rtol=1e-10)

    def test_singleresidue_matches_gpu_build(self, cuda_data, cpu_v_backend):
        gpu_val = fcuda.singleresidue_frustration(cuda_data)
        cpu_val = cpu_v_backend.singleresidue()
        np.testing.assert_allclose(cpu_val, gpu_val, atol=1e-10, rtol=1e-10)

    def test_mutational_matches_gpu_build(self, cuda_data, cpu_v_backend):
        gpu_dense = fcuda.mutational_frustration_dense(cuda_data)
        cpu_dense = cpu_v_backend.mutational(dense=True)
        mask = gpu_dense != 0
        np.testing.assert_allclose(cpu_dense[mask], gpu_dense[mask], atol=1e-10, rtol=1e-10)

    def test_v_matches_numba_compute_v(self, cuda_data, cpu_v_backend):
        expected = fnumba.compute_V(cuda_data)
        got = cpu_v_backend.d_V.copy_to_host()
        np.testing.assert_allclose(got, expected, rtol=1e-12)

    def test_build_v_on_cpu_with_float32_workspace(self, cuda_data):
        backend = fcuda.prepare_cuda(cuda_data, build_v_on_cpu=True, workspace_dtype=np.float32)
        assert backend.d_V.dtype == np.float32
