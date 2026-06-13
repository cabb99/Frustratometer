"""
Tests for frustratometer.frustration.numba — on-the-fly AWSEM frustration.

Compares the numba implementation against the existing AWSEM class results.
"""
import pytest
import numpy as np
from pathlib import Path
import frustratometer
from frustratometer.frustration.data import FrustrationData
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
def numba_data(awsem_model):
    return _make_data_from_awsem(awsem_model)


@pytest.fixture(scope="module")
def numba_data_elec(awsem_model_elec):
    return _make_data_from_awsem(awsem_model_elec)


# ── tests ─────────────────────────────────────────────────────────────────────

class TestNativeEnergy:
    def test_native_energy_no_elec(self, awsem_model, numba_data):
        ref = awsem_model.native_energy()
        got = fnumba.native_energy(numba_data)
        np.testing.assert_allclose(got, ref, rtol=1e-6)

    def test_native_energy_with_elec(self, awsem_model_elec, numba_data_elec):
        ref = awsem_model_elec.native_energy()
        got = fnumba.native_energy(numba_data_elec)
        np.testing.assert_allclose(got, ref, rtol=1e-4)


class TestSingleresideFrustration:
    def test_singleresidue_no_elec(self, awsem_model, numba_data):
        ref = awsem_model.frustration(kind='singleresidue')
        got = fnumba.singleresidue_frustration(numba_data)
        # Allow some tolerance due to on-the-fly vs materialized computation
        np.testing.assert_allclose(got, ref, atol=1e-5, rtol=1e-4)

    def test_singleresidue_with_elec(self, awsem_model_elec, numba_data_elec):
        ref = awsem_model_elec.frustration(kind='singleresidue')
        got = fnumba.singleresidue_frustration(numba_data_elec)
        np.testing.assert_allclose(got, ref, atol=1e-3, rtol=1e-3)


class TestMutationalFrustration:
    def test_mutational_no_elec(self, awsem_model, numba_data):
        ref_dense = awsem_model.frustration(kind='mutational')
        got_dense = fnumba.mutational_frustration_dense(numba_data)
        # Compare only at contact positions
        mask = got_dense != 0
        np.testing.assert_allclose(got_dense[mask], ref_dense[mask], atol=1e-5, rtol=1e-4)

    def test_mutational_with_elec(self, awsem_model_elec, numba_data_elec):
        ref_dense = awsem_model_elec.frustration(kind='mutational')
        got_dense = fnumba.mutational_frustration_dense(numba_data_elec)
        mask = got_dense != 0
        np.testing.assert_allclose(got_dense[mask], ref_dense[mask], atol=1e-3, rtol=1e-3)

    def test_mutational_sparse_matches_dense(self, numba_data):
        sparse = fnumba.mutational_frustration(numba_data)
        dense = fnumba.mutational_frustration_dense(numba_data)
        assert sparse.shape == (numba_data.Nc,)
        np.testing.assert_allclose(dense[numba_data.contact_i, numba_data.contact_j], sparse)


@pytest.mark.stochastic
class TestConfigurationalFrustration:
    def test_configurational_correlation(self, awsem_model, numba_data):
        """Check that configurational frustration is well correlated with AWSEM reference."""
        n_decoys = 100000
        ref = awsem_model.configurational_frustration(n_decoys=n_decoys)
        got = fnumba.configurational_frustration(numba_data, n_decoys=n_decoys, seed=123)
        # Both are (L, L) with NaN at non-contact positions
        mask = np.isfinite(ref) & np.isfinite(got)
        corr = np.corrcoef(ref[mask], got[mask])[0, 1]
        assert corr > 0.90, f"Correlation {corr:.4f} too low"


# ── fast=True API tests ──────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def fast_model():
    """6u5e sparse AWSEM model with fast=True, no electrostatics."""
    structure = frustratometer.Structure(
        test_data_path / '6u5e.pdb', "A", sparse=True,
    )
    return frustratometer.AWSEM(
        structure,
        distance_cutoff_contact=9.5,
        min_sequence_separation_contact=2,
        k_electrostatics=0,
        fast=True,
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
        fast=True,
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


class TestFastFallback:
    """Fast models lazily materialize a sparse Potts model for inherited operations.

    Each test builds a *fresh* fast model so materialization does not leak into the
    module-scoped ``fast_model`` fixture (which other tests assert is un-materialized).
    """

    @staticmethod
    def _fresh_fast_model(k_electrostatics=0):
        structure = frustratometer.Structure(test_data_path / '6u5e.pdb', "A", sparse=True)
        return frustratometer.AWSEM(
            structure, distance_cutoff_contact=9.5, min_sequence_separation_contact=2,
            k_electrostatics=k_electrostatics, min_sequence_separation_electrostatics=1,
            fast=True)

    def test_materialized_potts_matches_nonfast(self, awsem_model):
        m = self._fresh_fast_model()
        assert m.sparse_potts_model is None
        m._ensure_potts_model()
        got, ref = m.sparse_potts_model, awsem_model.sparse_potts_model
        np.testing.assert_array_equal(got['contact_i'], ref['contact_i'])
        np.testing.assert_array_equal(got['contact_j'], ref['contact_j'])
        np.testing.assert_allclose(got['J'], ref['J'], atol=1e-6)
        np.testing.assert_allclose(got['h'], ref['h'], atol=1e-6)

    def test_mutant_native_energy_matches_nonfast(self, awsem_model):
        m = self._fresh_fast_model()
        mutant = ('A' if m.sequence[0] != 'A' else 'C') + m.sequence[1:]
        np.testing.assert_allclose(
            m.native_energy(sequence=mutant), awsem_model.native_energy(sequence=mutant),
            rtol=1e-5, atol=1e-5)

    def test_mutant_native_energy_elec_matches_nonfast(self, awsem_model_elec):
        m = self._fresh_fast_model(k_electrostatics=4.184)
        mutant = ('A' if m.sequence[0] != 'A' else 'C') + m.sequence[1:]
        np.testing.assert_allclose(
            m.native_energy(sequence=mutant), awsem_model_elec.native_energy(sequence=mutant),
            rtol=1e-4, atol=1e-4)

    def test_couplings_energy_matches_nonfast(self, awsem_model):
        m = self._fresh_fast_model()
        np.testing.assert_allclose(
            m.couplings_energy(), awsem_model.couplings_energy(), rtol=1e-5, atol=1e-5)

    def test_total_frustration_runs(self):
        m = self._fresh_fast_model()
        assert m.sparse_potts_model is None
        val = m.total_frustration(n_decoys=200)
        assert np.isfinite(val)
        assert m.sparse_potts_model is not None  # materialized on demand

    def test_configurational_custom_aa_freq_raises(self, fast_model):
        with pytest.raises(NotImplementedError):
            fast_model.configurational_frustration(aa_freq=np.ones(21) / 21)

    def test_sliding_window_runs(self):
        m = self._fresh_fast_model()
        res = m.sliding_window(win_size=5, ndecoys=100)
        assert len(res['frustration']) == m.N - 4  # win_size=5 -> dif=2 -> L-4 windows
        assert np.all(np.isfinite(res['frustration']))
        assert m.sparse_potts_model is not None  # materialized on demand
