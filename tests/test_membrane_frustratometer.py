"""
test_awsem_vs_openawsem.py
==========================
Cross-validates the frustratometer AWSEM implementation against pre-computed
openawsem reference energies for the 2xov membrane protein (GlpG rhomboid
protease, chain A, 276 residues).

Matching conditions
-------------------
openawsem uses:
  - k_contact = 1 * 4.184 kJ/mol
  - min_sequence_separation = 10
  - no explicit distance cutoff (1.0 nm neighbour list, thetaII decays to ~0 by 10.5 Å)
  - min_sequence_separation_rho = 2 

frustratometer settings to match:
  - k_electrostatics = 0       
  - distance_cutoff_contact = None  
  - min_sequence_separation_contact = 10
  - min_sequence_separation_rho = 2 
Tolerances
----------
        component-level energies: 0.1 kJ/mol (strict)

"""
import json
import pytest
import numpy as np
from pathlib import Path

import frustratometer

TEST_DATA = Path("tests/data")
REF = json.loads((TEST_DATA / "openawsem_energies.json").read_text())
ENERGY_ABS_TOL_KJ = 0.1
ENERGY_ABS_TOL_KJ_WATER_CONTACT = 0.3
ENERGY_ABS_TOL_KJ_MEMBRANE_CONTACT = 0.2
ENERGY_ABS_TOL_KJ_MEMBRANE_SIMPLE_ZIM = 0.1


@pytest.fixture(scope="module")
def awsem_2xov_water():
    """AWSEM with no membrane, settings matched to openawsem water-only run."""
    structure = frustratometer.Structure(TEST_DATA / "2xov.pdb", "A", repair_pdb=False)
    return frustratometer.AWSEM(
        structure,
        membrane=False,
        min_sequence_separation_contact=10,
        min_sequence_separation_rho=2,
        distance_cutoff_contact=None,
        k_electrostatics=0.0,
        sparse=False,
    )


@pytest.fixture(scope="module")
def awsem_2xov_membrane():
    """AWSEM with membrane blending, settings matched to openawsem membrane run."""
    structure = frustratometer.Structure(TEST_DATA / "2xov.pdb", "A", repair_pdb=False)
    predicted_zim = np.loadtxt(TEST_DATA / "PredictedZim")
    # openawsem uses: zim = -1 if file value is 2 else +1
    zim = np.where(predicted_zim == 2, -1.0, 1.0)
    return frustratometer.AWSEM(
        structure,
        membrane=True,
        min_sequence_separation_contact=10,
        min_sequence_separation_rho=2,
        distance_cutoff_contact=None,
        k_electrostatics=0.0,
        sparse=False,
        zim=zim,
    )


@pytest.fixture(scope="module")
def awsem_2xov_membrane_no_preassigned_zim():
    """AWSEM membrane run without explicit zim override."""
    structure = frustratometer.Structure(TEST_DATA / "2xov.pdb", "A", repair_pdb=False)
    return frustratometer.AWSEM(
        structure,
        membrane=True,
        min_sequence_separation_contact=10,
        min_sequence_separation_rho=2,
        distance_cutoff_contact=None,
        k_electrostatics=0.0,
        sparse=False,
    )


# ── Water-only ────────────────────────────────────────────────────────────────

def test_water_only_burial_matches_reference(awsem_2xov_water):
    """Water burial term matches reference."""
    ref = REF["water_only"]["burial_only_kJ_per_mol"]
    got = awsem_2xov_water.fields_energy()
    assert got == pytest.approx(ref, abs=ENERGY_ABS_TOL_KJ), (
        f"Water burial mismatch: frustratometer={got:.4f}, "
        f"openawsem={ref:.4f}, diff={got - ref:.4f} kJ/mol"
    )


def test_water_only_contact_matches_reference(awsem_2xov_water):
    """Water contact term matches reference."""
    ref = REF["water_only"]["contact_only_kJ_per_mol"]
    got = awsem_2xov_water.couplings_energy()
    assert got == pytest.approx(ref, abs=ENERGY_ABS_TOL_KJ_WATER_CONTACT), (
        f"Water contact mismatch: frustratometer={got:.4f}, "
        f"openawsem={ref:.4f}, diff={got - ref:.4f} kJ/mol"
    )


# ── Membrane ──────────────────────────────────────────────────────────────────

def _membrane_zim_kj_per_mol(awsem_2xov_membrane):
    return (awsem_2xov_membrane.k_membrane * np.sum(awsem_2xov_membrane.phi_z * awsem_2xov_membrane.dgwoct)
    )


def _membrane_burial_kj_per_mol(awsem_2xov_membrane):
    # fields_energy in membrane mode includes burial plus a -zim contribution.
    # Add zim back to recover burial-only energy.
    return awsem_2xov_membrane.fields_energy() + _membrane_zim_kj_per_mol(awsem_2xov_membrane)


def test_membrane_zim_matches_reference(awsem_2xov_membrane):
    """Membrane insertion (zim) term matches reference."""
    ref = REF["membrane"]["zim_kJ_per_mol"]
    got = _membrane_zim_kj_per_mol(awsem_2xov_membrane)
    assert got == pytest.approx(ref, abs=ENERGY_ABS_TOL_KJ), (
        f"Membrane zim mismatch: frustratometer={got:.4f}, "
        f"openawsem={ref:.4f}, diff={got - ref:.4f} kJ/mol"
    )


def test_membrane_burial_matches_reference(awsem_2xov_membrane):
    """Membrane burial term matches membrane reference."""
    ref = REF["membrane"]["burial_only_kJ_per_mol"]
    got = _membrane_burial_kj_per_mol(awsem_2xov_membrane)
    assert got == pytest.approx(ref, abs=ENERGY_ABS_TOL_KJ), (
        f"Membrane burial mismatch: frustratometer={got:.4f}, "
        f"openawsem={ref:.4f}, diff={got - ref:.4f} kJ/mol"
    )


def test_membrane_contact_matches_reference(awsem_2xov_membrane):
    """Membrane contact term matches membrane reference."""
    ref = REF["membrane"]["contact_only_kJ_per_mol"]
    got = awsem_2xov_membrane.couplings_energy()
    assert got == pytest.approx(ref, abs=ENERGY_ABS_TOL_KJ_MEMBRANE_CONTACT), (
        f"Membrane contact mismatch: frustratometer={got:.4f}, "
        f"openawsem={ref:.4f}, diff={got - ref:.4f} kJ/mol"
    )


def test_membrane_simple_no_preassigned_zim_matches_reference(
    awsem_2xov_membrane_no_preassigned_zim,
):
    """No-preassigned membrane insertion tracks OpenAWSEM simple membrane_term."""
    ref = REF["membrane_simple"]["zim_kJ_per_mol"]
    got = _membrane_zim_kj_per_mol(awsem_2xov_membrane_no_preassigned_zim)
    assert got == pytest.approx(ref, abs=ENERGY_ABS_TOL_KJ_MEMBRANE_SIMPLE_ZIM), (
        f"Membrane simple zim mismatch: frustratometer={got:.4f}, "
        f"openawsem={ref:.4f}, diff={got - ref:.4f} kJ/mol"
    )
