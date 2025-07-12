from Bio.PDB import PDBParser,MMCIFParser
import prody
import scipy.spatial.distance as sdist
import pandas as pd
import numpy as np
import itertools
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
                 chain: str,
                 return_start_mask: bool=False
                 ) -> str:
    """
    Get a protein sequence from a pdb file

    Parameters
    ----------
    pdb_file : str,
        PDB file location.
    chain: str or list,
        Chain ID(s) of the selected protein.
    return_start_mask: bool,
        Return binary mask list indicating whether each position is the start of a chain

    Returns
    -------
    sequence : str
        Protein sequence.
    """
    """
    Get a protein sequence from a PDB file
    
    :param pdb: PDB file location
    :param chain: chain name(s) of PDB file to get sequence
    :return: protein sequence
    """

    if ".cif" in str(pdb_file): # BIOPYTHON
        parser = MMCIFParser() # BIOPYTHON
    else:                     # BIOPYTHON
        parser = PDBParser() # BIOPYTHON
    structure = parser.get_structure('name', pdb_file) #BIOPYTHON
    #structure = prody.parsePDB(str(pdb_file)) # PRODY
    #hv = structure.getHierView() # PRODY
    if chain==None:
        all_chains=[i.get_id() for i in structure.get_chains()] # BIOPYTHON
        #all_chains = [structure_chain.getChid() for structure_chain in hv] # PRODY
    else:
        if type(chain) == list:
            all_chains = chain
        elif type(chain) == str:
            all_chains = [id for id in chain if id != " "] # remove spaces if present in string
        else:
            raise TypeError(f"chain must be list or str but was {type(chain)}")
    sequence = ""
    start_mask = []
    for single_chain in all_chains:
        c = structure[0][single_chain] # BIOPYTHON
        #c = hv[single_chain] # PRODY
        chain_seq = ""
        for residue in c:
            is_regular_res = residue.has_id('CA') and residue.has_id('O') # BIOPYTHON
            #atom_names = [atom.getName() for atom in residue] # PRODY
            #is_regular_res = ("CA" in atom_names and "O" in atom_names) # PRODY
            res_id = residue.get_id()[0] #BIOPYTHON
            if (res_id==' ' or res_id=='H_MSE' or res_id=='H_M3L' or res_id=='H_CAS') and is_regular_res: # BIOPYTHON
            # i don't know what H_HSE, H_M3L, and H_CAS are doing 
            # because they aren't in three_to_one, so those should throw an error
            # long story short, I don't think we have to worry about them when switching from biopython to prody
            #if is_regular_res: # PRODY
                residue_name = residue.get_resname() # BIOPYTHON
                #residue_name = residue.getResname() # PRODY
                chain_seq += three_to_one[residue_name]
        if chain_seq == "": # empty chain, like a nucleic acid chain (see 8ZWK)
            continue        # FYI, currently, a non-empty chain with certain invalid residues will throw an error at the three_to_one[residue_name] above
        sequence += chain_seq
        start_mask.append(1)
        for _ in range(1,len(chain_seq)):
            start_mask.append(0)
    if return_start_mask:
        return (sequence,start_mask)
    else:
        return sequence



def get_distance_matrix(pdb_file: Union[Path,str],
                        chain: str,
                        method: str = 'CB',
                        return_distance_midpoints: bool = False,
                        ) -> np.array:
    """
    Calculate the distance matrix of the specified atoms in a PDB file.

    Parameters
    ----------
    pdb_file: Path or str 
        The path to the PDB file.
    chain: str
        The chainID or chainIDs (space separated) of the protein.
    method: str
        The method to use for calculating the distance matrix. 
        Defaults to 'CB', which uses the CB atom for all residues except GLY, which uses the CA atom. 
        Options:
            'CA' for using only the CA atom,
            'minimum' for using the minimum distance between all atoms in each residue,
            'CB_force' computes a new coordinate for the CB atom based on the CA, C, and N atoms and uses CB distance even for glycine.
    return_distance_midpoints: bool
        Whether to return a matrix of the same shape as distance_matrix representing the same contacts as distance_matrix
        that indicates the absolute coordinates of the midpoint between the pair of atoms. This helps us compute the pair distribution
        functions of the different classes of contacts. So this matrix isn't really a matrix because each "element" has 3 channels: x, y, and z

    Returns:
        np.array: The distance matrix of the selected atoms.

    Raises:
        IndexError: If the selection of atoms is empty.
        ValueError: If the method is not recognized.
    """

    structure = prody.parsePDB(str(pdb_file))
    chain_selection = '' if chain is None else f' and chain {chain}'

    if method == 'CA':
        coords = structure.select('protein and name CA' + chain_selection).getCoords()
    elif method == 'CB':
        coords = structure.select('(protein and (name CB) or (resname GLY IGL and name CA))' + chain_selection).getCoords()
    elif method == 'minimum':
        selection = structure.select('protein' + chain_selection)
        coords = selection.getCoords()
        distance_matrix = sdist.squareform(sdist.pdist(coords))
        resids = selection.getResindices()
        residues = pd.Series(resids).unique()
        selections = np.array([resids == a for a in residues])
        dm = np.zeros((len(residues), len(residues)))
        for i, j in itertools.combinations(range(len(residues)), 2):
            d = distance_matrix[selections[i]][:, selections[j]].min()
            dm[i, j] = d
            dm[j, i] = d
        return dm
    elif method == 'CB_force':
        sel_CA = structure.select('name CA')
        sel_N = structure.select('name N')
        sel_C = structure.select('name C')
        
        # Base vectors
        vector_CA_C=sel_C.getCoords() - sel_CA.getCoords()
        vector_CA_N=sel_N.getCoords() - sel_CA.getCoords()

        vector_CA_C/=np.linalg.norm(vector_CA_C,axis=1,keepdims=True)
        vector_CA_N/=np.linalg.norm(vector_CA_N,axis=1,keepdims=True)

        #First Ortogonal vector
        cross_CA_C_CA_N=np.cross(vector_CA_C,vector_CA_N)
        cross_CA_C_CA_N/=np.linalg.norm(cross_CA_C_CA_N,axis=1,keepdims=True)

        #Second Ortogonal vector
        cross_cross_CA_N=np.cross(cross_CA_C_CA_N,vector_CA_N)
        cross_cross_CA_N/= np.linalg.norm(cross_cross_CA_N,axis=1,keepdims=True)
        
        #Precomputed CB coordinates
        coords = -0.531020*vector_CA_N-1.206181*cross_CA_C_CA_N+0.789162*cross_cross_CA_N+sel_CA.getCoords()
    else:
        raise ValueError(f"Invalid method '{method}'. Accepted methods are 'CA', 'CB', 'minimum', and 'CB_force'.")

    if len(coords) == 0:
        raise IndexError('Empty selection for distance map')

    # coords should be a numpy array of shape (N,3)
    distance_matrix = sdist.squareform(sdist.pdist(coords))
    assert distance_matrix.shape[0] == distance_matrix.shape[1]
    if return_distance_midpoints:
        midpoint_matrix = np.zeros((distance_matrix.shape[0],distance_matrix.shape[1],3))
        for i in range(distance_matrix.shape[0]):
            for j in range(distance_matrix.shape[1]):
                midpoint_matrix[i,j,:] = (coords[None,i,:] + coords[None,j,:])/2
                # check that indexing is consistent with distance_matrix
                assert np.allclose(np.linalg.norm(coords[i,:]-coords[j,:]),distance_matrix[i,j])
        assert np.allclose(midpoint_matrix,midpoint_matrix.transpose(1,0,2)) # check symmetry
        return distance_matrix, midpoint_matrix
    else:    
        return distance_matrix


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