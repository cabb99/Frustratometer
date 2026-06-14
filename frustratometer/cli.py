"""Command-line interface for the frustratometer package.

Compute single-residue / mutational / configurational frustration on a PDB
structure with AWSEM, write frustratometeR-style .tsv files, and render a
VMD movie per requested frustration kind (movies on by default).

Examples
--------
# Default: r_server preset, all kinds, movies on, output to ./frustration_<stem>/
$ frustratometer mystructure.pdb

# Disable electrostatics
$ frustratometer mystructure.pdb --k-electrostatics 0

# Use the lenient short-range density (rho_min_sep=2 instead of default 13)
$ frustratometer mystructure.pdb --min-sequence-separation-rho 2

# Just configurational, no movies
$ frustratometer mystructure.pdb --kind configurational --no-movie

# OpenAWSEM convention with membrane
$ frustratometer mystructure.pdb --profile openawsem --membrane
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import frustratometer


# ─── Presets ────────────────────────────────────────────────────────────────

# frustratometeR / LAMMPS-AWSEM server default. Singleresidue uses
# min_sequence_separation_contact=2 while pair frustration uses 0; we patch
# this when building the per-kind models.
_LAMMPS_PRESET = dict(
    distance_cutoff_contact=9.5,
    min_sequence_separation_rho=2,
    min_sequence_separation_contact=0,
    k_electrostatics=4.15 * 4.184,
    min_sequence_separation_electrostatics=1,
)

_LAMMPS_12_PRESET = dict(
    distance_cutoff_contact=9.5,
    min_sequence_separation_rho=13,                  # frustratometeR seqsep=12
    min_sequence_separation_contact=0,
    k_electrostatics=4.15 * 4.184,
    min_sequence_separation_electrostatics=1,
)

# OpenMM / Python AWSEM convention, with electrostatics on by default.
_OPENAWSEM_PRESET = dict(
    distance_cutoff_contact=None,
    min_sequence_separation_rho=2,
    min_sequence_separation_contact=10,
    k_electrostatics=4.15 * 4.184,
    min_sequence_separation_electrostatics=1,
)

PRESETS = {
    "r_server": _LAMMPS_PRESET,
    "r_server_12": _LAMMPS_12_PRESET,
    "openawsem": _OPENAWSEM_PRESET,
    "default": {},
}

KINDS = ("singleresidue", "mutational", "configurational")
MIN_FRUST = 0.78
HIGH_FRUST = -1.0


_EPILOG = """\
Common adjustments
  Disable electrostatics:
      --k-electrostatics 0
  Change density threshold (rho_min_sep=13 instead of 2):
      --min-sequence-separation-rho 13
  Skip movie rendering (faster, no VMD needed):
      --no-movie
  Render still images alongside the movies:
      --still
  OpenAWSEM convention with membrane:
      --profile openawsem --membrane
"""


# ─── Argument parser ────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="frustratometer",
        description="Compute frustration for a PDB structure with AWSEM.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_EPILOG,
    )
    p.add_argument("pdb", type=Path, help="Path to PDB file.")
    p.add_argument("--chain", default=None,
                   help="Chain identifier; omit to load every chain in the PDB.")
    p.add_argument("--profile", choices=list(PRESETS), default="r_server",
                   help="Preset configuration (default: r_server).")
    p.add_argument("--kind", nargs="+",
                   choices=list(KINDS) + ["all"], default=["all"],
                   help="Frustration kind(s) to compute (default: all).")
    p.add_argument("--membrane", action="store_true",
                   help="Enable membrane mode.")
    p.add_argument("--no-repair-pdb", dest="repair_pdb", action="store_false",
                   help="Disable automatic PDB repair.")

    g = p.add_argument_group("parameter overrides (override the preset)")
    g.add_argument("--k-electrostatics", type=float,
                   help="Override k_electrostatics (kJ/mol). 0 to disable.")
    g.add_argument("--distance-cutoff-contact", type=str,
                   help="Override distance cutoff (Å). 'none' to disable.")
    g.add_argument("--min-sequence-separation-rho", type=int,
                   help="Min sequence separation for density (default: 13).")
    g.add_argument("--min-sequence-separation-contact", type=int,
                   help="Min sequence separation for contact (preset-dependent).")
    g.add_argument("--min-sequence-separation-electrostatics", type=int)
    g.add_argument("--n-decoys", type=int, default=10000,
                   help="Decoys for configurational frustration (default: 10000).")

    o = p.add_argument_group("output")
    o.add_argument("--output-dir", type=Path, default=None,
                   help="Output directory (default: ./frustration_<pdb_stem>).")
    o.add_argument("--no-movie", dest="movie", action="store_false",
                   help="Skip VMD movie rendering (movies are on by default).")
    o.add_argument("--still", action="store_true",
                   help="Also render a still image (.png) per kind.")
    return p


# ─── Helpers ────────────────────────────────────────────────────────────────

def resolve_params(args: argparse.Namespace) -> dict:
    """Merge the chosen preset with any explicit per-parameter overrides."""
    params = dict(PRESETS[args.profile])
    overrides = {
        "k_electrostatics": args.k_electrostatics,
        "min_sequence_separation_rho": args.min_sequence_separation_rho,
        "min_sequence_separation_contact": args.min_sequence_separation_contact,
        "min_sequence_separation_electrostatics": args.min_sequence_separation_electrostatics,
    }
    if args.distance_cutoff_contact is not None:
        overrides["distance_cutoff_contact"] = (
            None if args.distance_cutoff_contact.lower() == "none"
            else float(args.distance_cutoff_contact)
        )
    for k, v in overrides.items():
        if v is not None:
            params[k] = v
    params["membrane"] = args.membrane
    return params


def _build_model(args: argparse.Namespace, params: dict):
    structure = frustratometer.Structure(args.pdb, args.chain,
                                         repair_pdb=args.repair_pdb)
    return frustratometer.AWSEM(structure, **params)


def _expand_kinds(kinds: list[str]) -> list[str]:
    return list(KINDS) if "all" in kinds else list(dict.fromkeys(kinds))


def _frust_state(f: float) -> str:
    if f >= MIN_FRUST:
        return "minimally"
    if f <= HIGH_FRUST:
        return "highly"
    return "neutral"


def _pdb_resids_chains(model) -> tuple[np.ndarray, np.ndarray]:
    """Return per-residue (resnum, chain_id) arrays in model order."""
    sel = model.structure.select("name CB or (resname GLY IGL and name CA)")
    return np.asarray(sel['resid'].to_numpy()), np.asarray(sel['chain'].to_numpy())


def _output_path(out_dir: Path, kind: str, ext: str = "") -> Path:
    return out_dir / f"{kind}{ext}"


def _classify_welltypes(model, i: np.ndarray, j: np.ndarray,
                        d_ij: np.ndarray) -> np.ndarray:
    """Per-pair AWSEM well classification using the model's own switching.

    Picks the largest of three indicator weights:
      direct           = theta(d)                              (4.5–6.5 Å well)
      water-mediated   = thetaII(d) * sigma_water(rho_i, rho_j)
      protein-mediated = thetaII(d) * sigma_protein(rho_i, rho_j)
    """
    rho = np.asarray(model.rho_r)
    theta = (0.25
             * (1 + np.tanh(model.eta * (d_ij - model.r_min)))
             * (1 + np.tanh(model.eta * (model.r_max - d_ij))))
    thetaII = (0.25
               * (1 + np.tanh(model.eta * (d_ij - model.r_minII)))
               * (1 + np.tanh(model.eta * (model.r_maxII - d_ij))))
    sigma_water = (0.25
                   * (1 - np.tanh(model.eta_sigma * (rho[i] - model.rho_0)))
                   * (1 - np.tanh(model.eta_sigma * (rho[j] - model.rho_0))))
    weights = np.stack([theta, thetaII * sigma_water, thetaII * (1 - sigma_water)])
    labels = np.array(["direct", "water-mediated", "protein-mediated"])
    return labels[np.argmax(weights, axis=0)]


# ─── Output writers ─────────────────────────────────────────────────────────

def _write_singleresidue(model, frust: np.ndarray, out_dir: Path) -> Path:
    rho = np.asarray(model.rho_r)
    resids, chains = _pdb_resids_chains(model)
    df = pd.DataFrame({
        "#Res": resids,
        "ChainRes": chains,
        "DensityRes": np.round(rho, 3),
        "AA": list(model.sequence),
        "FrstIndex": np.round(frust, 3),
        "FrstState": [_frust_state(f) for f in frust],
    })
    out = _output_path(out_dir, "singleresidue", ".tsv")
    df.to_csv(out, sep="\t", index=False)
    return out


def _write_pair(model, frust: np.ndarray, out_dir: Path, kind: str
                ) -> tuple[Path, pd.DataFrame]:
    rho = np.asarray(model.rho_r)
    d = model.distance_matrix
    d = d.toarray() if hasattr(d, "toarray") else np.asarray(d)
    resids, chains = _pdb_resids_chains(model)

    i, j = np.triu_indices(model.N, k=2)
    d_ij = d[i, j]
    keep = d_ij <= 9.5
    i, j, d_ij = i[keep], j[keep], d_ij[keep]
    f = frust[i, j]

    df = pd.DataFrame({
        "#Res1": resids[i],
        "Res2": resids[j],
        "ChainRes1": chains[i],
        "ChainRes2": chains[j],
        "DensityRes1": np.round(rho[i], 3),
        "DensityRes2": np.round(rho[j], 3),
        "AA1": [model.sequence[k] for k in i],
        "AA2": [model.sequence[k] for k in j],
        "FrstIndex": np.round(f, 3),
        "Welltype": _classify_welltypes(model, i, j, d_ij),
        "FrstState": [_frust_state(v) for v in f],
    })
    out = _output_path(out_dir, kind, ".tsv")
    df.to_csv(out, sep="\t", index=False)
    return out, df


# ─── Stdout summary ─────────────────────────────────────────────────────────

def _print_summary(label: str, count_label: str, values: np.ndarray, out: Path) -> None:
    n_min = int(np.sum(values >= MIN_FRUST))
    n_high = int(np.sum(values <= HIGH_FRUST))
    n_neutral = len(values) - n_min - n_high
    print(f"  {label:15s} {len(values):5d} {count_label:9s}  "
          f"min={n_min:4d}  high={n_high:4d}  neutral={n_neutral:4d}  "
          f"range [{values.min():+.2f}, {values.max():+.2f}]")
    print(f"                  → {out}")


# ─── Visualization ─────────────────────────────────────────────────────────

def _render_visualizations(viz_models: dict, viz_arrays: dict, kinds: list[str],
                           out_dir: Path, movie: bool, still: bool) -> None:
    """Render one VMD movie/still per kind. Singleresidue colors residues by
    SR frustration; pair kinds draw pair lines for that kind."""
    import multiprocessing
    
    if not movie and not still:
        return
    for kind in kinds:
        model = viz_models[kind]
        single = -viz_arrays["singleresidue"] if kind == "singleresidue" else None
        pair = -viz_arrays[kind] if kind != "singleresidue" else None
        out_base = str(_output_path(out_dir, kind))
        if movie:
            print(f"  rendering movie:  {out_base}.mp4 ...", flush=True)
        try:
            vmd_process = model.vmd(single=single,
                                    pair=pair,
                                    movie_name=out_base if movie else None,
                                    still_image_name=out_base if still else None,
                                    minimum_sequence_separation=2)
            vmd_process.communicate()  # Wait for VMD to finish rendering before proceeding
        except Exception as exc:
            print(f"    failed to render {out_base}: {exc}", file=sys.stderr)
            continue
        if still:
            print(f"                  → {out_base}.png (still)")


# ─── Main ───────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    params = resolve_params(args)
    kinds = _expand_kinds(args.kind)

    out_dir = args.output_dir or Path(f"frustration_{args.pdb.stem}")
    out_dir.mkdir(parents=True, exist_ok=True)

    frustratometer.utils.autolog_run(
        results_dir=out_dir,
        program_version=frustratometer.__version__,
        extra={
            "profile": args.profile,
            "params": params,
            "parsed_args": vars(args),
        },
    )

    pair_kinds = [k for k in kinds if k != "singleresidue"]
    needs_separate_sr = (
        "singleresidue" in kinds
        and args.profile in ("r_server", "r_server_12")
        and args.min_sequence_separation_contact is None
    )

    print(
        f"  profile={args.profile}  membrane={args.membrane}  "
        f"k_elec={params.get('k_electrostatics', 0):.4f} kJ/mol  "
        f"rho_sep={params.get('min_sequence_separation_rho', '?')}"
    )

    model_pair = _build_model(args, params) if pair_kinds else None
    model_sr = None
    if "singleresidue" in kinds:
        if needs_separate_sr:
            model_sr = _build_model(args, dict(params, min_sequence_separation_contact=2))
        else:
            model_sr = model_pair if model_pair is not None else _build_model(args, params)

    viz_models: dict[str, "frustratometer.AWSEM"] = {}
    viz_arrays: dict[str, np.ndarray] = {}

    if "singleresidue" in kinds:
        sr_frust = model_sr.frustration(kind="singleresidue")
        out = _write_singleresidue(model_sr, sr_frust, out_dir)
        _print_summary("singleresidue", "residues", sr_frust, out)
        viz_models["singleresidue"] = model_sr
        viz_arrays["singleresidue"] = sr_frust

    for kind in pair_kinds:
        if kind == "mutational":
            frust = model_pair.frustration(kind="mutational")
        else:
            frust = model_pair.configurational_frustration(n_decoys=args.n_decoys)
        out, df = _write_pair(model_pair, frust, out_dir, kind)
        _print_summary(kind, "contacts", df["FrstIndex"].to_numpy(), out)
        viz_models[kind] = model_pair
        viz_arrays[kind] = frust

    _render_visualizations(viz_models, viz_arrays, kinds, out_dir, args.movie, args.still)
    return 0


if __name__ == "__main__":
    sys.exit(main())
