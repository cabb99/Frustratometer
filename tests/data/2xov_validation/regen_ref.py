"""Regenerate OpenAWSEM reference energies for membrane tests.

This script generates the JSON schema expected by
tests/test_membrane_frustratometer.py, including:
    - water_only
    - membrane (preassigned Zim path)
    - membrane_preassigned (alias of membrane for clarity)
    - membrane_simple (no-preassigned simple membrane_term path)
"""
import argparse
import json
import os
import openmm
import openmm.unit as u
import openawsem
from openawsem import openAWSEM
from openawsem.functionTerms.contactTerms import contact_term
from openawsem.functionTerms.membraneTerms import membrane_preassigned_term, membrane_term

os.chdir(os.path.dirname(os.path.abspath(__file__)))

PDB = "crystal_structure-openmmawsem.pdb"
PARAMS = openawsem.data_path.parameters
print(f"Using PARAMS: {PARAMS}")


def _parse_args():
    parser = argparse.ArgumentParser(description="Regenerate OpenAWSEM reference energies JSON.")
    parser.add_argument(
        "--output",
        default=os.path.join(".", "openawsem_energies.json"),
        help="Output JSON path (default: ./openawsem_energies.json)",
    )
    return parser.parse_args()


def build_sim(forces):
    oa = openAWSEM.OpenMMAWSEMSystem(PDB, k_awsem=1.0)
    for f in forces(oa):
        oa.system.addForce(f)
    integrator = openmm.LangevinIntegrator(
        300 * u.kelvin, 1.0 / u.picosecond, 2 * u.femtosecond
    )
    sim = openmm.app.Simulation(
        oa.pdb.topology, oa.system, integrator,
        openmm.Platform.getPlatformByName("CPU"),
    )
    sim.context.setPositions(oa.pdb.positions)
    return sim


def get_energy(sim, group):
    state = sim.context.getState(getEnergy=True, groups={group})
    kJ = state.getPotentialEnergy().value_in_unit(u.kilojoule_per_mole)
    kCal = state.getPotentialEnergy().value_in_unit(u.kilocalorie_per_mole)
    return float(kJ), float(kCal)


def run_pair(forces_burial, forces_contact_only, label):
    sim_b = build_sim(forces_burial)
    kJ_total, kCal_total = get_energy(sim_b, 22)
    kJ_zim, kCal_zim = get_energy(sim_b, 24)

    sim_c = build_sim(forces_contact_only)
    kJ_contact, kCal_contact = get_energy(sim_c, 22)

    kJ_burial = kJ_total - kJ_contact
    kCal_burial = kCal_total - kCal_contact

    print(f"{label} burial+contact: {kJ_total:.6f} kJ/mol  ({kCal_total:.6f} kcal/mol)")
    print(f"{label} contact-only:   {kJ_contact:.6f} kJ/mol  ({kCal_contact:.6f} kcal/mol)")
    print(f"{label} burial-only:    {kJ_burial:.6f} kJ/mol  ({kCal_burial:.6f} kcal/mol)")
    print(f"{label} zim (group 24): {kJ_zim:.6f} kJ/mol  ({kCal_zim:.6f} kcal/mol)")

    return {
        "contact_burial_kJ_per_mol": kJ_total,
        "contact_burial_kcal_per_mol": kCal_total,
        "contact_only_kJ_per_mol": kJ_contact,
        "contact_only_kcal_per_mol": kCal_contact,
        "burial_only_kJ_per_mol": kJ_burial,
        "burial_only_kcal_per_mol": kCal_burial,
        "zim_kJ_per_mol": kJ_zim,
        "zim_kcal_per_mol": kCal_zim,
    }


# Water-only
def water_forces_burial(oa):
    return [contact_term(oa, k_contact=1 * 4.184, z_dependent=False, inMembrane=False,
                         parametersLocation=PARAMS, burialPartOn=True, forceGroup=22)]

def water_forces_contact_only(oa):
    return [contact_term(oa, k_contact=1 * 4.184, z_dependent=False, inMembrane=False,
                         parametersLocation=PARAMS, burialPartOn=False, forceGroup=22)]

sim_water = build_sim(water_forces_burial)
kJ_w, kCal_w = get_energy(sim_water, 22)
sim_water_c = build_sim(water_forces_contact_only)
kJ_wc, kCal_wc = get_energy(sim_water_c, 22)
kJ_wb = kJ_w - kJ_wc
kCal_wb = kCal_w - kCal_wc

print(f"water burial+contact: {kJ_w:.6f} kJ/mol  ({kCal_w:.6f} kcal/mol)")
print(f"water contact-only:   {kJ_wc:.6f} kJ/mol  ({kCal_wc:.6f} kcal/mol)")
print(f"water burial-only:    {kJ_wb:.6f} kJ/mol  ({kCal_wb:.6f} kcal/mol)")


# Membrane (z_dependent=True, PredictedZim = all 1s)
def mem_forces_burial(oa):
    return [
        contact_term(oa, k_contact=1 * 4.184, z_dependent=True, inMembrane=True,
                     membrane_center=0 * u.angstrom, k_relative_mem=1.0,
                     parametersLocation=PARAMS, burialPartOn=True, forceGroup=22),
        membrane_preassigned_term(oa, k=1 * u.kilocalorie_per_mole,
                                  membrane_center=0 * u.angstrom,
                                  zimFile="PredictedZim", forceGroup=24)
    ]

def mem_forces_contact_only(oa):
    return [
        contact_term(oa, k_contact=1 * 4.184, z_dependent=True, inMembrane=True,
                     membrane_center=0 * u.angstrom, k_relative_mem=1.0,
                     parametersLocation=PARAMS, burialPartOn=False, forceGroup=22),
        membrane_preassigned_term(oa, k=1 * u.kilocalorie_per_mole,
                                  membrane_center=0 * u.angstrom,
                                  zimFile="PredictedZim", forceGroup=24)
    ]

membrane_preassigned = run_pair(mem_forces_burial, mem_forces_contact_only, "mem preassigned")


# Membrane simple term (no preassigned zim mapping)
def mem_simple_forces_burial(oa):
    return [
        contact_term(
            oa,
            k_contact=1 * 4.184,
            z_dependent=True,
            inMembrane=True,
            membrane_center=0 * u.angstrom,
            k_relative_mem=1.0,
            parametersLocation=PARAMS,
            burialPartOn=True,
            forceGroup=22,
        ),
        membrane_term(
            oa,
            k=1 * u.kilocalorie_per_mole,
            membrane_center=0 * u.angstrom,
            forceGroup=24,
        ),
    ]


def mem_simple_forces_contact_only(oa):
    return [
        contact_term(
            oa,
            k_contact=1 * 4.184,
            z_dependent=True,
            inMembrane=True,
            membrane_center=0 * u.angstrom,
            k_relative_mem=1.0,
            parametersLocation=PARAMS,
            burialPartOn=False,
            forceGroup=22,
        ),
        membrane_term(
            oa,
            k=1 * u.kilocalorie_per_mole,
            membrane_center=0 * u.angstrom,
            forceGroup=24,
        ),
    ]


membrane_simple = run_pair(mem_simple_forces_burial, mem_simple_forces_contact_only, "mem simple")

result = {
    "protein": "2xov chain A (GlpG rhomboid protease, 276 residues)",
    "openawsem_version_note": "Generated with installed OpenAWSEM parameters and explicit membrane preassigned + simple term coverage.",
    "water_only": {
        "contact_burial_kJ_per_mol": kJ_w,
        "contact_burial_kcal_per_mol": kCal_w,
        "contact_only_kJ_per_mol": kJ_wc,
        "contact_only_kcal_per_mol": kCal_wc,
        "burial_only_kJ_per_mol": kJ_wb,
        "burial_only_kcal_per_mol": kCal_wb,
    },
    "membrane": {
        **membrane_preassigned,
    },
    "membrane_preassigned": {
        **membrane_preassigned,
    },
    "membrane_simple": {
        **membrane_simple,
        "note": "OpenAWSEM membrane_term with local zim file (no preassigned PredictedZim mapping).",
    },
    "parameters": {
        "k_contact": "1 * 4.184 kJ/mol",
        "k_relative_mem": 1.0,
        "membrane_center": "0 Angstrom",
        "min_sequence_separation": 10,
        "z_dependent": True,
        "burialPartOn": True,
        "parameter_source": str(PARAMS),
        "preassigned_zim_file": "PredictedZim",
        "simple_zim_file": "zim",
    },
}

args = _parse_args()
out_path = args.output
with open(out_path, "w") as f:
    json.dump(result, f, indent=2)
print(f"\nSaved to {out_path}")
print("JSON:", json.dumps(result, indent=2))
