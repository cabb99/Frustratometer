from pathlib import Path
import pdbfixer

PDBFile = pdbfixer.pdbfixer.app.PDBFile
PDBFixer = pdbfixer.PDBFixer

def repair_pdb(pdb_file: str, chain: str, pdb_directory: Path= Path.cwd()) -> PDBFixer:
    """
    Repairs a pdb or cif file using pdbfixer. Note that a pdb file will be produced, regardless of input file format

    Parameters
    ----------
    pdb_file: str,
        PDB file location.
    chain: str,
        Chain ID
    pdb_directory: str,
        PDB file location

    Returns
    -------
    fixer : object
        Repaired PDB Object
    """
    pdb_directory=Path(pdb_directory)
    pdb_file=Path(pdb_file)
    
    pdbID=pdb_file.stem
    fixer = PDBFixer(str(pdb_file))

    chains = list(fixer.topology.chains())
    if chain!=None:
        chains_to_remove = [i for i, x in enumerate(chains) if x.id not in chain]
        fixer.removeChains(chains_to_remove)

    fixer.findMissingResidues()
    #Filling in missing residues inside chain
    chains = list(fixer.topology.chains())
    keys = fixer.missingResidues.keys()
    for key in list(keys):
        chain_tmp = chains[key[0]]
        if key[1] == 0 or key[1] == len(list(chain_tmp.residues())):
            del fixer.missingResidues[key]
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    fixer.removeHeterogens(keepWater=False)
    fixer.findMissingAtoms()
    try:
        fixer.addMissingAtoms()
    except:
        print("Unable to add missing atoms")

    fixer.addMissingHydrogens(7.0)
    PDBFile.writeFile(fixer.topology, fixer.positions, open(f"{pdb_directory}/{pdbID}_cleaned.pdb", 'w'))
    return fixer