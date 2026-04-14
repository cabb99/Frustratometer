from Bio.PDB import PDBParser,MMCIFParser
import numpy as np
import os
from pathlib import Path
from typing import Union

three_to_one = {'ALA':'A', 'ARG':'R', 'ASN':'N', 'ASP':'D', 'CYS':'C',
                'GLU':'E', 'GLN':'Q', 'GLY':'G', 'HIS':'H', 'ILE':'I',
                'LEU':'L', 'LYS':'K', 'MET':'M', 'PHE':'F', 'PRO':'P',
                'SER':'S', 'THR':'T', 'TRP':'W', 'TYR':'Y', 'VAL':'V'}


def download(pdbID: str,directory: Union[Path,str]=Path.cwd()) -> Path:
    """
    Downloads a single pdb file

    Parameters
    ----------
    pdbID: str,
        PDB ID
    directory: Path or str,
        Directory where PDB file will be downloaded.

    Returns
    -------
    pdb_file : Path
        PDB file location.
    """

    import urllib.request
    pdb_file=Path(directory) / f'{pdbID}.pdb'
    urllib.request.urlretrieve('http://www.rcsb.org/pdb/files/%s.pdb' % pdbID, pdb_file)
    return pdb_file

def get_sequence(pdb_file: str, 
                 chain: str
                 ) -> str:
    """
    Get a protein sequence from a pdb file

    Parameters
    ----------
    pdb_file : str,
        PDB file location.
    chain: str,
        Chain ID of the selected protein.

    Returns
    -------
    sequence : str
        Protein sequence.
    """
    """
    Get a protein sequence from a PDB file
    
    :param pdb: PDB file location
    :param chain: chain name of PDB file to get sequence
    :return: protein sequence
    """

    if ".cif" in str(pdb_file):
        parser = MMCIFParser()
    else:
        parser = PDBParser()
    structure = parser.get_structure('name', pdb_file)
    if chain==None:
        all_chains=[i.get_id() for i in structure.get_chains()]
    else:
        all_chains=[chain]
    sequence = ""
    for chain in all_chains:
        c = structure[0][chain]
        chain_seq = ""
        for residue in c:
            is_regular_res = residue.has_id('CA') and residue.has_id('O')
            res_id = residue.get_id()[0]
            if (res_id==' ' or res_id=='H_MSE' or res_id=='H_M3L' or res_id=='H_CAS') and is_regular_res:
                residue_name = residue.get_resname()
                chain_seq += three_to_one[residue_name]
        sequence += chain_seq
    return sequence


def full_to_filtered_aligned_mapping(aligned_sequence: str,
                                    filtered_aligned_sequence: str)->dict:

    """
    Get a dictionary mapping residue positions in the full pdb sequence to the aligned pdb sequence

    Parameters
    ----------
    aligned_sequence : str,
        Raw aligned PDB sequence.
    filtered_aligned_sequence: str,
        Filtered aligned PDB sequence (columns with insertions and deletions, i.e. dashes, that are
        typically filtered in MSA file processing are removed)

    Returns
    -------
    full_to_aligned_index_dict : dict
        Dictionary
    """
    full_to_aligned_index_dict={}; counter=0
    for i,x in enumerate(aligned_sequence):
        if x != "-" and x==x.upper():
            full_to_aligned_index_dict[counter]=i
        if x!="-":
            counter+=1

    dash_indices=[i for i,x in enumerate(filtered_aligned_sequence) if x!="-"]
    counter=0
    for entry in full_to_aligned_index_dict:
        full_to_aligned_index_dict[entry]=dash_indices[counter]
        counter+=1

    return full_to_aligned_index_dict