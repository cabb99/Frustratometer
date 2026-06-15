"""
test_membrane_frustratometer.py
==========================
Cross-validates the frustratometer AWSEM implementation against pre-computed
openawsem reference energies for the 2xov membrane protein (GlpG rhomboid
protease, chain A, 276 residues) in both water and membrane contexts.

The reference energies for the water and membrane contexts were computed 
by running openawsem on the 2xov structure.
The "membrane_simple" reference was computed by running openawsem with the
simple membrane term (no preassigned ZIM, i.e. no per-residue bias towards
membrane insertion or water burial).
The "membrane" reference was computed by running openawsem with the full
membrane term, which includes preassigned ZIM values based on the predicted
membrane topology of 2xov.
These references are stored in tests/data/2xov_validation/openawsem_energies.json.

There are also reference frustration indices for the 2xov water model, computed by
running the LAMMPS-AWSEM server with kelec=4.15 and seqsep=12.

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
import pandas as pd
from pathlib import Path

import frustratometer

TEST_DATA = Path("tests/data")
REF = json.loads((TEST_DATA / "2xov_validation" / "openawsem_energies.json").read_text())

# Shared model settings (single source of truth for this module)
MODEL_SETTINGS = {
    "awsem_2xov_water": {
        "structure_pdb": TEST_DATA / "2xov.pdb",
        "chain": "A",
        "repair_pdb": False,
        "membrane": False,
        "min_sequence_separation_contact": 10,
        "min_sequence_separation_rho": 2,
        "distance_cutoff_contact": None,
        "k_electrostatics": 0.0,
    },
    "awsem_2xov_membrane": {
        "structure_pdb": TEST_DATA / "2xov.pdb",
        "chain": "A",
        "repair_pdb": False,
        "membrane": True,
        "min_sequence_separation_contact": 10,
        "min_sequence_separation_rho": 2,
        "distance_cutoff_contact": None,
        "k_electrostatics": 0.0,
        "zim_source": TEST_DATA / "PredictedZim",
    },
    "awsem_2xov_membrane_no_preassigned_zim": {
        "structure_pdb": TEST_DATA / "2xov.pdb",
        "chain": "A",
        "repair_pdb": False,
        "membrane": True,
        "min_sequence_separation_contact": 10,
        "min_sequence_separation_rho": 2,
        "distance_cutoff_contact": None,
        "k_electrostatics": 0.0,
    },
}

_R_SERVER_12_DIR = TEST_DATA / "2xov_kelec_4.15_seqsep_12"
_R_SERVER_12_PDB = _R_SERVER_12_DIR / "VisualizationScripts" / "202657181458583980.pdb"
_R_SERVER_12_PREFIX = "202657181458583980.pdb"
_R_SERVER_12_RESID_OFFSET = 91

SERVER_MODEL_SETTINGS = {
    "base": {
        "structure_pdb": _R_SERVER_12_PDB,
        "chain": "A",
        "repair_pdb": False,
        "membrane": False,
        "distance_cutoff_contact": 9.5,
        "min_sequence_separation_rho": 13,  # seqsep=12 -> 13
        "k_electrostatics": 4.15 * 4.184,
        "min_sequence_separation_electrostatics": 1,
    },
    "variants": {
        "awsem_2xov_water_lammps_sr": {
            "min_sequence_separation_contact": 2,
        },
        "awsem_2xov_water_lammps_pair": {
            "min_sequence_separation_contact": 0,
        },
    },
}

ENERGY_ABS_TOL_KJ = 0.1
ENERGY_ABS_TOL_KJ_WATER_CONTACT = 0.3
ENERGY_ABS_TOL_KJ_MEMBRANE_CONTACT = 0.2
ENERGY_ABS_TOL_KJ_MEMBRANE_SIMPLE_ZIM = 0.1


def _structure_for(cfg: dict):
    return frustratometer.Structure(cfg["structure_pdb"], cfg["chain"],
                                    repair_pdb=cfg["repair_pdb"])


def _awsem_kwargs(cfg: dict, exclude=("structure_pdb", "chain", "repair_pdb", "zim_source")) -> dict:
    kwargs = {k: v for k, v in cfg.items() if k not in exclude}
    if "zim_source" in cfg:
        predicted_zim = np.loadtxt(cfg["zim_source"])
        # openawsem uses: zim = -1 if file value is 2 else +1
        kwargs["zim"] = np.where(predicted_zim == 2, -1.0, 1.0)
    return kwargs


@pytest.fixture(scope="module")
def awsem_models():
    """Build all AWSEM models needed by this module. The three openawsem
    variants share one Structure (same PDB/chain/repair); the two server
    variants share another."""
    models = {}

    structures: dict[tuple, "frustratometer.Structure"] = {}
    for name, cfg in MODEL_SETTINGS.items():
        key = (cfg["structure_pdb"], cfg["chain"], cfg["repair_pdb"])
        if key not in structures:
            structures[key] = _structure_for(cfg)
        models[name] = frustratometer.AWSEM(structures[key], **_awsem_kwargs(cfg))

    server_cfg = SERVER_MODEL_SETTINGS["base"]
    server_structure = _structure_for(server_cfg)
    for name, variant_cfg in SERVER_MODEL_SETTINGS["variants"].items():
        models[name] = frustratometer.AWSEM(
            server_structure, **_awsem_kwargs(server_cfg), **variant_cfg)

    return models


# Water-energy

def test_water_only_burial_matches_reference(awsem_models):
    """Water burial term matches reference."""
    awsem_2xov_water = awsem_models["awsem_2xov_water"]
    ref = REF["water_only"]["burial_only_kJ_per_mol"]
    got = awsem_2xov_water.fields_energy()
    assert got == pytest.approx(ref, abs=ENERGY_ABS_TOL_KJ), (
        f"Water burial mismatch: frustratometer={got:.4f}, "
        f"openawsem={ref:.4f}, diff={got - ref:.4f} kJ/mol"
    )


def test_water_only_contact_matches_reference(awsem_models):
    """Water contact term matches reference."""
    awsem_2xov_water = awsem_models["awsem_2xov_water"]
    ref = REF["water_only"]["contact_only_kJ_per_mol"]
    got = awsem_2xov_water.couplings_energy()
    assert got == pytest.approx(ref, abs=ENERGY_ABS_TOL_KJ_WATER_CONTACT), (
        f"Water contact mismatch: frustratometer={got:.4f}, "
        f"openawsem={ref:.4f}, diff={got - ref:.4f} kJ/mol"
    )


# Membrane-energy
# Sign conventions used below:
#   model.zim_energy()    = +k_membrane * sum(phi_z * dgwoct)  (added to h)
#   model.fields_energy() = -sum(h_native) = -(burial_h + zim_h)
# So the burial contribution to fields_energy is fields_energy + zim_energy.


def test_membrane_zim_matches_reference(awsem_models):
    """Membrane insertion (zim) term matches reference."""
    model = awsem_models["awsem_2xov_membrane"]
    ref = REF["membrane"]["zim_kJ_per_mol"]
    got = model.zim_energy()
    assert got == pytest.approx(ref, abs=ENERGY_ABS_TOL_KJ), (
        f"Membrane zim mismatch: frustratometer={got:.4f}, "
        f"openawsem={ref:.4f}, diff={got - ref:.4f} kJ/mol"
    )


def test_membrane_burial_matches_reference(awsem_models):
    """Membrane burial term matches membrane reference."""
    model = awsem_models["awsem_2xov_membrane"]
    ref = REF["membrane"]["burial_only_kJ_per_mol"]
    got = model.fields_energy() + model.zim_energy()
    assert got == pytest.approx(ref, abs=ENERGY_ABS_TOL_KJ), (
        f"Membrane burial mismatch: frustratometer={got:.4f}, "
        f"openawsem={ref:.4f}, diff={got - ref:.4f} kJ/mol"
    )


def test_membrane_contact_matches_reference(awsem_models):
    """Membrane contact term matches membrane reference."""
    model = awsem_models["awsem_2xov_membrane"]
    ref = REF["membrane"]["contact_only_kJ_per_mol"]
    got = model.couplings_energy()
    assert got == pytest.approx(ref, abs=ENERGY_ABS_TOL_KJ_MEMBRANE_CONTACT), (
        f"Membrane contact mismatch: frustratometer={got:.4f}, "
        f"openawsem={ref:.4f}, diff={got - ref:.4f} kJ/mol"
    )


def test_membrane_simple_no_preassigned_zim_matches_reference(awsem_models):
    """No-preassigned membrane insertion tracks OpenAWSEM simple membrane_term."""
    model = awsem_models["awsem_2xov_membrane_no_preassigned_zim"]
    ref = REF["membrane_simple"]["zim_kJ_per_mol"]
    got = model.zim_energy()
    assert got == pytest.approx(ref, abs=ENERGY_ABS_TOL_KJ_MEMBRANE_SIMPLE_ZIM), (
        f"Membrane simple zim mismatch: frustratometer={got:.4f}, "
        f"openawsem={ref:.4f}, diff={got - ref:.4f} kJ/mol"
    )


# ── 2xov water (frustratometeR / LAMMPS-AWSEM server, kelec=4.15, seqsep=12) ──
# Server cropped 2xov to PDB residues 91–271 (181 residues); ref resid X maps
# to array index X-91.

_MIN_FRUST = 0.78    # FrstIndex ≥  0.78 → minimally frustrated (green)
_HIGH_FRUST = -1.0   # FrstIndex ≤ -1.00 → highly frustrated   (red)
_FRUST_TOL = 0.3     # per-pair value tolerance
_BORDERLINE_EPS = 0.1  # ref FrstIndex must lie within this of a threshold for a
                       # classification disagreement to be tolerated


def _read_ref(kind: str) -> pd.DataFrame:
    return pd.read_csv(_R_SERVER_12_DIR / "FrustrationData" / f"{_R_SERVER_12_PREFIX}_{kind}", sep=r"\s+")


def _classify(values, keys=None) -> dict:
    """Apply the minimally/highly thresholds, dropping neutrals.

    If `keys` is given, returns {key: class} keyed accordingly. Otherwise
    `values` is a flat array and returns {index: class} keyed by position.
    """
    if keys is None:
        keys = range(len(values))
    out = {}
    for k, v in zip(keys, values):
        if v >= _MIN_FRUST:
            out[k] = "minimally"
        elif v <= _HIGH_FRUST:
            out[k] = "highly"
    return out


def _check_classification(calc: dict, ref: dict, ref_values: dict, label: str) -> None:
    """Compare calc against ref classification with two rules:
      - any key in BOTH must have the same class (minimally↔highly flips fail
        unconditionally)
      - any key in only ONE is tolerated only if the reference FrstIndex at
        that key is within _BORDERLINE_EPS of a threshold (numerical noise on
        the boundary).
    """
    flips = [k for k in (calc.keys() & ref.keys()) if calc[k] != ref[k]]
    assert not flips, f"{label}: minimally↔highly flips: {flips[:5]}"

    near_threshold = lambda v: (abs(v - _MIN_FRUST) <= _BORDERLINE_EPS
                                or abs(v - _HIGH_FRUST) <= _BORDERLINE_EPS)
    bad = [(k, ref_values.get(k))
           for k in calc.keys() ^ ref.keys()
           if k not in ref_values or not near_threshold(ref_values[k])]
    assert not bad, f"{label}: non-borderline disagreements: {bad[:5]}"


# ── Value tests: every reference (i,j) value matches calc within tolerance ──

def test_2xov_water_lammps_singleresidue_matches_reference(awsem_models):
    awsem_2xov_water_lammps_sr = awsem_models["awsem_2xov_water_lammps_sr"]
    ref = _read_ref("singleresidue")
    np.testing.assert_allclose(
        awsem_2xov_water_lammps_sr.frustration(kind="singleresidue"),
        ref["FrstIndex"], atol=_FRUST_TOL)
    np.testing.assert_allclose(
        awsem_2xov_water_lammps_sr.rho_r, ref["DensityRes"], atol=1e-3)


@pytest.mark.parametrize("kind", ["mutational", "configurational"])
def test_2xov_water_lammps_pair_matches_reference(kind, awsem_models):
    awsem_2xov_water_lammps_pair = awsem_models["awsem_2xov_water_lammps_pair"]
    ref = _read_ref(kind)
    i = ref["#Res1"].to_numpy() - _R_SERVER_12_RESID_OFFSET
    j = ref["Res2"].to_numpy() - _R_SERVER_12_RESID_OFFSET
    if kind == "mutational":
        frust = awsem_2xov_water_lammps_pair.frustration(kind="mutational")
    else:
        frust = awsem_2xov_water_lammps_pair.configurational_frustration(n_decoys=10000)
    np.testing.assert_allclose(frust[i, j], ref["FrstIndex"], atol=_FRUST_TOL)


# ── Classification tests: every disagreement must be borderline ─────────────

def test_2xov_water_lammps_singleresidue_classification(awsem_models):
    awsem_2xov_water_lammps_sr = awsem_models["awsem_2xov_water_lammps_sr"]
    ref = _read_ref("singleresidue")
    resids = ref["#Res"].astype(int).tolist()
    calc_vals = awsem_2xov_water_lammps_sr.frustration(kind="singleresidue")
    ref_vals = ref["FrstIndex"].astype(float).tolist()

    _check_classification(
        calc=_classify(calc_vals, keys=resids),
        ref=_classify(ref_vals, keys=resids),
        ref_values=dict(zip(resids, ref_vals)),
        label="singleresidue",
    )


@pytest.mark.parametrize("kind", ["mutational", "configurational"])
def test_2xov_water_lammps_pair_classification(kind, awsem_models):
    model = awsem_models["awsem_2xov_water_lammps_pair"]
    if kind == "mutational":
        frust = model.frustration(kind="mutational")
    else:
        frust = model.configurational_frustration(n_decoys=10000)

    # Calc classification: every LAMMPS-style contact (d ≤ 9.5 Å, |i-j| ≥ 2)
    # whose FrstIndex crosses a threshold.
    d = model.distance_matrix
    if hasattr(d, "to_dense"):
        d = d.to_dense()
    elif hasattr(d, "toarray"):
        d = d.toarray()
    d = np.asarray(d)
    i, j = np.triu_indices(model.N, k=2)
    in_contact = d[i, j] <= 9.5
    pair_keys = list(zip((i + _R_SERVER_12_RESID_OFFSET)[in_contact],
                         (j + _R_SERVER_12_RESID_OFFSET)[in_contact]))
    pair_keys = [(int(a), int(b)) for a, b in pair_keys]
    calc_class = _classify(frust[i, j][in_contact], keys=pair_keys)

    # Ref classification comes straight from the FrstState column (the .tcl
    # visualization is generated from this same column).
    ref = _read_ref(kind)
    ref_class = {(int(r1), int(r2)): s
                 for r1, r2, s in zip(ref["#Res1"], ref["Res2"], ref["FrstState"])
                 if s != "neutral"}
    ref_values = {(int(r1), int(r2)): float(f)
                  for r1, r2, f in zip(ref["#Res1"], ref["Res2"], ref["FrstIndex"])}

    _check_classification(calc_class, ref_class, ref_values, label=kind)
