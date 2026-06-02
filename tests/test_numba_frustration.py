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
    """Extract sparse distances and gammas from an existing AWSEM object,
    then build a FrustrationData instance."""
    dm = model._sparse_distance_matrix
    # Use the wider electrostatic distance matrix if available
    elec_dm = getattr(model, '_sparse_distance_matrix_elec', None)
    elec_kw = {}
    if elec_dm is not None:
        elec_kw = dict(
            elec_dist_row=elec_dm.row,
            elec_dist_col=elec_dm.col,
            elec_dist_data=elec_dm.data,
        )
    return FrustrationData.from_sparse(
        dist_row=dm.row,
        dist_col=dm.col,
        dist_data=dm.data,
        L=model.N,
        sequence=model.sequence,
        burial_gamma=model.burial_gamma,
        direct_gamma=model.direct_gamma,
        water_gamma=model.water_gamma,
        protein_gamma=model.protein_gamma,
        k_contact=model.k_contact,
        eta=model.eta,
        r_min=model.r_min,
        r_max=model.r_max,
        r_minII=model.r_minII,
        r_maxII=model.r_maxII,
        eta_sigma=model.eta_sigma,
        rho_0=model.rho_0,
        burial_kappa=model.burial_kappa,
        burial_ro_min=model.burial_ro_min,
        burial_ro_max=model.burial_ro_max,
        min_seq_sep_rho=model.min_sequence_separation_rho,
        min_seq_sep_contact=model.min_sequence_separation_contact,
        distance_cutoff_contact=model.distance_cutoff_contact,
        k_electrostatics=model.k_electrostatics,
        screening_length=model.electrostatics_screening_length,
        min_seq_sep_elec=model.min_sequence_separation_electrostatics,
        chain_breaks=model.chain_breaks,
        **elec_kw,
    )


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


class TestFastAttributes:
    def test_has_frustration_data(self, fast_model):
        assert fast_model._frustration_data is not None

    def test_sequence_preserved(self, awsem_model, fast_model):
        assert fast_model.sequence == awsem_model.sequence

    def test_N_preserved(self, awsem_model, fast_model):
        assert fast_model.N == awsem_model.N

    def test_no_potts_model(self, fast_model):
        assert fast_model.sparse_potts_model is None
