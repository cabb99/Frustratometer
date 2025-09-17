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
        Chain ID -- can be formatted as str or list (or None)
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
    
    # renumber residues so that each chain starts at 1
    new_top = type(fixer.topology)() # an openmm.app.Topology accesed without needing to import it separately
    for old_chain in fixer.topology.chains():
        new_chain = new_top.addChain(id=old_chain.id)
        for old_residue in old_chain.residues():
            new_residue = new_top.addResidue(old_residue.name,new_chain,id=None) # allow the class to choose residue id
            for old_atom in old_residue.atoms():
                new_atom = new_top.addAtom(old_atom.name,old_atom.element,new_residue,id=old_atom.id)

    # use keepIds=True when writing to preserve chain IDs
    # changing it to False causes test_multichain_density to fail for structure_file1-density_file1
    # because chains a, b, c, ... get renamed to A, B, C, ... and end up getting confused
    # with the real chains A, B, C, ...
    PDBFile.writeFile(new_top, fixer.positions, open(f"{pdb_directory}/{pdbID}_cleaned.pdb", 'w'),keepIds=True)
    return fixer