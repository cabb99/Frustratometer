from pathlib import Path
import logging
import multiprocessing
import tempfile

# Import guard: if pdbfixer is not installed, this ImportError
# propagates to __init__.py which provides a friendly fallback.
import pdbfixer  # noqa: F401

logger = logging.getLogger(__name__)


def _repair_worker(pdb_file_str: str, chain: str, cleaned_path_str: str) -> None:
    """Run PDBFixer in an isolated process to prevent OpenMM memory leaks."""
    import pdbfixer

    PDBFile = pdbfixer.pdbfixer.app.PDBFile
    fixer = pdbfixer.PDBFixer(pdb_file_str)

    # Keep only the requested chain(s)
    if chain is not None:
        chains = list(fixer.topology.chains())
        chains_to_remove = [i for i, c in enumerate(chains) if c.id not in chain]
        fixer.removeChains(chains_to_remove)

    # Fill in missing internal residues (skip terminal gaps)
    fixer.findMissingResidues()
    for key in list(fixer.missingResidues):
        chain_obj = list(fixer.topology.chains())[key[0]]
        if key[1] == 0 or key[1] == len(list(chain_obj.residues())):
            del fixer.missingResidues[key]

    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    fixer.removeHeterogens(keepWater=False)
    fixer.findMissingAtoms()
    try:
        fixer.addMissingAtoms()
    except Exception:
        pass
    fixer.addMissingHydrogens(7.0)

    with open(cleaned_path_str, 'w') as f:
        PDBFile.writeFile(fixer.topology, fixer.positions, f)


def repair_pdb(pdb_file: str, chain: str, pdb_directory: Path = None) -> Path:
    """
    Repair a PDB or CIF file using PDBFixer.

    Runs PDBFixer in a child process so that OpenMM's C-level memory
    is fully reclaimed by the OS when the process exits.

    Fills in missing internal residues, replaces nonstandard residues,
    removes heterogens, adds missing atoms and hydrogens, and writes
    a cleaned PDB file.

    Parameters
    ----------
    pdb_file : str or Path
        PDB / mmCIF file location.
    chain : str or None
        Chain ID to keep.  If None, all chains are kept.
    pdb_directory : Path, optional
        Directory for the cleaned output file.  Defaults to the system temp dir.

    Returns
    -------
    cleaned_path : Path
        Path to the repaired PDB file (``<pdb_directory>/<stem>_cleaned.pdb``).
    """
    if pdb_directory is None:
        pdb_directory = Path(tempfile.gettempdir())
    pdb_directory = Path(pdb_directory)
    pdb_file = Path(pdb_file)

    cleaned_path = pdb_directory / f"{pdb_file.stem}_cleaned.pdb"

    process = multiprocessing.Process(
        target=_repair_worker,
        args=(str(pdb_file), chain, str(cleaned_path)),
    )
    process.start()
    process.join()

    if process.exitcode != 0:
        raise RuntimeError(
            f"PDB repair failed for {pdb_file.name} (exit code {process.exitcode})"
        )

    return cleaned_path