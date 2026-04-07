"""
Utilities for sequence energies, decoys, and frustration metrics from Potts models.

Conventions
-----------
- Alphabet: `_AA = '-ACDEFGHIKLMNPQRSTVWY'` (gap '-' at index 0)
- Sequence length: L
- Fields and couplings:
    * potts_model['h'] has shape (L, Q)
    * potts_model['J'] has shape (L, L, Q, Q)
- Energies: A positive value in the potts model or a negative value in the energy is a favorable contribution.
- Masks: boolean array of shape (L, L). True values indicate included residue pairs.

Notes
-----
All functions assume `len(seq) == potts_model['h'].shape[0] == potts_model['J'].shape[0]`.
"""

import prody
import scipy.spatial.distance as sdist
import numpy as np
from typing import Union
from pathlib import Path

_AA = '-ACDEFGHIKLMNPQRSTVWY'

def compute_mask(distance_matrix: np.array,
                 maximum_contact_distance: Union[float, None] = None,
                 minimum_sequence_separation: Union[int, None] = None,
                 chain_breaks: Union[list, None] = None) -> np.array:
    """
    Computes a 2D Boolean mask (L, L) for pairwise interactions from a given distance matrix using a distance cutoff and/or a minimum sequence-separation cutoff.
    Both cutoffs are inclusive. True indicates that the residue pair meets the criteria.

    Parameters
    ----------
    distance_matrix : np.array (L,L)
        A 2D array where the element at index [i, j] represents the spatial distance
        between residues i and j (e.g., Ca-Ca, Cb-Cb). This matrix is assumed to be symmetric.
    maximum_contact_distance : float, optional
        The maximum distance of a contact. Include i,j if distance_matrix[i,j] <= maximum_contact_distance. 
        If None, no distance filtering is applied. Default is None.
    minimum_sequence_separation : int, optional
        A minimum sequence separation threshold. Include i,j if |i-j| >= minimum_sequence_separation.
        If None, no sequence separation is applied . Default is None.
    chain_breaks : list of int, optional
        Indices where new chains begin (excluding the implicit 0). For example, [50, 80]
        means three chains: residues 0-49, 50-79, 80-end. Cross-chain pairs always satisfy
        the minimum sequence separation, since they are not bonded in sequence.
        If None, all residues are treated as a single chain. Default is None.

    Returns
    -------
    mask : np.array (L,L)
        A 2D Boolean array of the same dimensions as `distance_matrix`. Elements of the mask
        are True where the residue pairs meet the specified `distance_cutoff` and
        `sequence_distance_cutoff` criteria.

    Examples
    --------
    >>> import numpy as np
    >>> dm = np.array([[0, 1, 2], [1, 0, 1], [2, 1, 0]])
    >>> print(compute_mask(dm, distance_cutoff=1.5, sequence_distance_cutoff=1))
    [[False  True False]
     [ True False  True]
     [False  True False]]
    """
    seq_len = len(distance_matrix)
    mask = np.ones([seq_len, seq_len])
    if minimum_sequence_separation is not None:
        positions = np.arange(seq_len, dtype=np.float64)
        if chain_breaks is not None:
            for brk in chain_breaks:
                positions[brk:] += minimum_sequence_separation
        sequence_distance = sdist.squareform(sdist.pdist(positions[:, np.newaxis]))
        mask *= sequence_distance >= minimum_sequence_separation
    if maximum_contact_distance is not None:
        mask *= distance_matrix <= maximum_contact_distance

    return mask.astype(np.bool_)


def compute_native_energy(seq: str,
                          potts_model: dict,
                          mask: np.array,
                          ignore_gap_couplings: bool = False,
                          ignore_gap_fields: bool = False) -> float:
    
    """
    Computes the native energy of a protein sequence based on a given Potts model and an interaction mask.
    
    .. math::
        E = \\sum_i h_i + \\frac{1}{2} \\sum_{i,j} J_{ij} \\Theta_{ij}
        
    Parameters
    ----------
    seq : str
        The amino acid sequence of the protein. The sequence is assumed to be in one-letter code. Gaps are represented as '-'. The length of the sequence (L) should match the dimensions of the Potts model.
    potts_model : dict
        A dictionary containing the Potts model parameters 'h' (fields) and 'J' (couplings). The fields are a 2D array of shape (L, 20), where L is the length of the sequence and 20 is the number of amino acids. The couplings are a 4D array of shape (L, L, 20, 20). The fields and couplings are assumed to be in units of energy.
    mask : np.array
        A 2D Boolean array that determines which residue pairs should be considered in the energy computation. The mask should have dimensions (L, L), where L is the length of the sequence.
    ignore_couplings_of_gaps : bool, optional
        If True, couplings involving gaps ('-') in the sequence are set to 0 in the energy calculation.
        Default is False.
    ignore_fields_of_gaps : bool, optional
        If True, fields corresponding to gaps ('-') in the sequence are set to 0 in the energy calculation.
        Default is False.

    Returns
    -------
    energy : float
        The computed energy of the protein sequence based on the Potts model and the interaction mask.

    Examples
    --------
    >>> seq = "ACDEFGHIKLMNPQRSTVWY"
    >>> potts_model = {
        'h': np.random.rand(20, 20),  # Random fields
        'J': np.random.rand(20, 20, 20, 20)  # Random couplings
    }
    >>> mask = np.ones((len(seq), len(seq)), dtype=bool) # Include all pairs
    >>> energy = compute_native_energy(seq, potts_model, mask)
    >>> print(f"Computed energy: {energy:.2f}")

    Notes
    -----
    The energy is computed as the sum of the fields and the half-sum of the couplings for all pairs of residues
    where the mask is True. The division by 2 for the couplings accounts for double-counting in symmetric
    matrices.

    .. todo:: Optimize the computation.
    """
        
    seq_index = np.array([_AA.find(aa) for aa in seq])
    seq_len = len(seq_index)

    pos1, pos2 = np.meshgrid(np.arange(seq_len), np.arange(seq_len), indexing='ij', sparse=True)
    aa1, aa2 = np.meshgrid(seq_index, seq_index, indexing='ij', sparse=True)

    h = -potts_model['h'][range(seq_len), seq_index]
    j = -potts_model['J'][pos1, pos2, aa1, aa2]
    j_prime = j * mask 

    gap_indices=[int(i) for i,j in enumerate(seq) if j=="-"]

    if ignore_gap_couplings==True:
        if len(gap_indices)>0:
            j_prime[gap_indices,:]=False
            j_prime[:,gap_indices]=False

    if ignore_gap_fields==True:
        if len(gap_indices)>0:
            h[gap_indices]=False

    energy = h.sum() + j_prime.sum() / 2
    return energy

def compute_fields_energy(seq: str,
                          potts_model: dict,
                          ignore_fields_of_gaps: bool = False) -> float:
    """
    Computes the fields energy of a protein sequence based on a given Potts model.
    
    .. math::
        E = \\sum_i h_i
        
    Parameters
    ----------
    seq : str
        The amino acid sequence of the protein. The sequence is assumed to be in one-letter code. Gaps are represented as '-'. The length of the sequence (L) should match the dimensions of the Potts model.
    potts_model : dict
        A dictionary containing the Potts model parameters 'h' (fields) and 'J' (couplings). The fields are a 2D array of shape (L, 20), where L is the length of the sequence and 20 is the number of amino acids. The couplings are a 4D array of shape (L, L, 20, 20). The fields and couplings are assumed to be in units of energy.
    ignore_fields_of_gaps : bool, optional
        If True, fields corresponding to gaps ('-') in the sequence are set to 0 in the energy calculation.
        Default is False.

    Returns
    -------
    fields_energy : float
        The computed fields energy of the protein sequence based on the Potts model

    Examples
    --------
    >>> seq = "ACDEFGHIKLMNPQRSTVWY"
    >>> potts_model = {
        'h': np.random.rand(20, 20),  # Random fields
        'J': np.random.rand(20, 20, 20, 20)  # Random couplings
    }
    >>> fields_energy = compute_fields_energy(seq, potts_model)
    >>> print(f"Computed fields energy: {fields_energy:.2f}")
    """
    seq_index = np.array([_AA.find(aa) for aa in seq])
    seq_len = len(seq_index)

    h = -potts_model['h'][range(seq_len), seq_index]
    
    if ignore_fields_of_gaps==True:
        gap_indices=[int(i) for i,j in enumerate(seq) if j=="-"]
        if len(gap_indices)>0:
            h[gap_indices]=False
    fields_energy=h.sum()
    return fields_energy

def compute_couplings_energy(seq: str,
                      potts_model: dict,
                      mask: np.array,
                      ignore_couplings_of_gaps: bool = False) -> float:
    """
    Computes the couplings energy of a protein sequence based on a given Potts model and an interaction mask.
    
    .. math::
        E = \\frac{1}{2} \\sum_{i,j} J_{ij} \\Theta_{ij}
        
    Parameters
    ----------
    seq : str
        The amino acid sequence of the protein. The sequence is assumed to be in one-letter code. Gaps are represented as '-'. The length of the sequence (L) should match the dimensions of the Potts model.
    potts_model : dict
        A dictionary containing the Potts model parameters 'h' (fields) and 'J' (couplings). The fields are a 2D array of shape (L, 20), where L is the length of the sequence and 20 is the number of amino acids. The couplings are a 4D array of shape (L, L, 20, 20). The fields and couplings are assumed to be in units of energy.
    mask : np.array
        A 2D Boolean array that determines which residue pairs should be considered in the energy computation. The mask should have dimensions (L, L), where L is the length of the sequence.
    ignore_couplings_of_gaps : bool, optional
        If True, couplings involving gaps ('-') in the sequence are set to 0 in the energy calculation.
        Default is False.

    Returns
    -------
    couplings_energy : float
        The computed couplings energy of the protein sequence based on the Potts model and the interaction mask.

    Examples
    --------
    >>> seq = "ACDEFGHIKLMNPQRSTVWY"
    >>> potts_model = {
        'h': np.random.rand(20, 20),  # Random fields
        'J': np.random.rand(20, 20, 20, 20)  # Random couplings
    }
    >>> mask = np.ones((len(seq), len(seq)), dtype=bool) # Include all pairs
    >>> couplings_energy = compute_couplings_energy(seq, potts_model, mask)
    >>> print(f"Computed couplings energy: {couplings_energy:.2f}")

    Notes
    -----
    The couplings energy is computed as the half-sum of the couplings for all pairs of residues
    where the mask is True. The division by 2 for the couplings accounts for double-counting in symmetric
    matrices.

    .. todo:: Optimize the computation.
    """
    seq_index = np.array([_AA.find(aa) for aa in seq])
    seq_len = len(seq_index)
    pos1, pos2 = np.meshgrid(np.arange(seq_len), np.arange(seq_len), indexing='ij', sparse=True)
    aa1, aa2 = np.meshgrid(seq_index, seq_index, indexing='ij', sparse=True)

    j = -potts_model['J'][pos1, pos2, aa1, aa2]
    j_prime = j * mask
    if ignore_couplings_of_gaps==True:
        gap_indices=[i for i,j in enumerate(seq) if j=="-"]
        if len(gap_indices)>0:
            j_prime[:,gap_indices]=False
            j_prime[gap_indices,:]=False
    couplings_energy = j_prime.sum() / 2
    return couplings_energy

def compute_sequences_energy(seqs: list,
                             potts_model: dict,
                             mask: np.array,
                             split_couplings_and_fields = False) -> np.array:
    """
    Computes the energy of multiple protein sequences based on a given Potts model and an interaction mask.
    
    .. math::
        E = \\sum_i h_i + \\frac{1}{2} \\sum_{i,j} J_{ij} \\Theta_{ij}
        
    Parameters
    ----------
    seqs : list
        List of amino acid sequences in string format, separated by commas. The sequences are assumed to be in one-letter code. Gaps are represented as '-'. The length of each sequence (L) should all match the dimensions of the Potts model.
    potts_model : dict
        A dictionary containing the Potts model parameters 'h' (fields) and 'J' (couplings). The fields are a 2D array of shape (L, 20), where L is the length of the sequences and 20 is the number of amino acids. The couplings are a 4D array of shape (L, L, 20, 20). The fields and couplings are assumed to be in units of energy.
    mask : np.array
        A 2D Boolean array that determines which residue pairs should be considered in the energy computation. The mask should have dimensions (L, L), where L is the length of the sequences.
    split_couplings_and_fields : bool, optional
        If True, two lists of the sequences' couplings and fields energies are returned.
        Default is False.

    Returns
    -------
    energy (if split_couplings_and_fields==False): float
        The computed energies of the protein sequences based on the Potts model and the interaction mask.
    fields_couplings_energy (if split_couplings_and_fields==True): np.array
        Array containing computed fields and couplings energies of the protein sequences based on the Potts model and the interaction mask. 

    Examples
    --------
    >>> seq_list = ["ACDEFGHIKLMNPQRSTVWY","AKLWYMNPQRSTCDEFGHIV"]
    >>> potts_model = {
        'h': np.random.rand(20, 20),  # Random fields
        'J': np.random.rand(20, 20, 20, 20)  # Random couplings
    }
    >>> mask = np.ones((len(seq_list[0]), len(seq_list[0])), dtype=bool) # Include all pairs
    >>> energies = compute_sequences_energy(seq_list, potts_model, mask)
    >>> print(f"Sequence 1 energy: {energies[0]:.2f}")
    >>> print(f"Sequence 2 energy: {energies[1]:.2f}")

    Notes
    -----
    The couplings energy is computed as the half-sum of the couplings for all pairs of residues
    where the mask is True. The division by 2 for the couplings accounts for double-counting in symmetric
    matrices.

    .. todo:: Optimize the computation.
    """

    seq_index = np.array([[_AA.find(aa) for aa in seq] for seq in seqs])
    N_seqs, seq_len = seq_index.shape
    pos_index=np.repeat([np.arange(seq_len)], N_seqs,axis=0)


    pos1=np.array([np.meshgrid(p, p, indexing='ij', sparse=True)[0] for p in pos_index])
    pos2=np.array([np.meshgrid(p, p, indexing='ij', sparse=True)[1] for p in pos_index])
    aa1=np.array([np.meshgrid(s, s, indexing='ij', sparse=True)[0] for s in seq_index])
    aa2=np.array([np.meshgrid(s, s, indexing='ij', sparse=True)[1] for s in seq_index])
    
    h = -potts_model['h'][pos_index,seq_index]
    j = -potts_model['J'][pos1, pos2, aa1, aa2]
    j_prime = j * mask

    if split_couplings_and_fields:
        fields_couplings_energy=np.array([h.sum(axis=-1),j_prime.sum(axis=-1).sum(axis=-1) / 2])
        return fields_couplings_energy
    else:
        energy = h.sum(axis=-1) + j_prime.sum(axis=-1).sum(axis=-1) / 2
        return energy


def compute_singleresidue_decoy_energy_fluctuation(seq: str,
                                                   potts_model: dict,
                                                   mask: np.array) -> np.array:

    """
    Computes a (Lx21) matrix for a sequence of length L. Row i contains all possible changes in energy upon mutating residue i.
    
    .. math::
        \\Delta H_i = \\Delta h_i + \\sum_k \\Delta j_{ik}
        
    Parameters
    ----------
    seq : str
        The amino acid sequence of the protein. The sequence is assumed to be in one-letter code. Gaps are represented as '-'. The length of the sequence (L) should match the dimensions of the Potts model.
    potts_model : dict
        A dictionary containing the Potts model parameters 'h' (fields) and 'J' (couplings). The fields are a 2D array of shape (L, 20), where L is the length of the sequence and 20 is the number of amino acids. The couplings are a 4D array of shape (L, L, 20, 20). The fields and couplings are assumed to be in units of energy.
    mask : np.array
        A 2D Boolean array that determines which residue pairs should be considered in the energy computation. The mask should have dimensions (L, L), where L is the length of the sequence.

    Returns
    -------
    decoy_energy: np.array
        (Lx21) matrix describing the energetic changes upon mutating a single residue.

    Examples
    --------
    >>> seq = "ACDEFGHIKLMNPQRSTVWY"
    >>> potts_model = {
        'h': np.random.rand(20, 20),  # Random fields
        'J': np.random.rand(20, 20, 20, 20)  # Random couplings
    }
    >>> mask = np.ones((len(seq), len(seq)), dtype=bool) # Include all pairs
    >>> decoy_energy = compute_singleresidue_decoy_energy_fluctuation(seq, potts_model, mask)
    >>> print(f"Matrix of Residue Decoy Energy Fluctuations: "); print(decoy_energy)
    >>> print(f"Matrix Size: "); print(shape(decoy_energy))

    Notes
    -----
    The couplings energy is computed as the half-sum of the couplings for all pairs of residues
    where the mask is True. The division by 2 for the couplings accounts for double-counting in symmetric
    matrices.

    .. todo:: Optimize the computation.
    """
    seq_index = np.array([_AA.find(aa) for aa in seq])
    seq_len = len(seq_index)

    # Create decoys
    pos1, aa1 = np.meshgrid(np.arange(seq_len), np.arange(21), indexing='ij', sparse=True)

    decoy_energy = np.zeros([seq_len, 21])
    decoy_energy -= (potts_model['h'][pos1, aa1] - potts_model['h'][pos1, seq_index[pos1]])  # h correction aa1

    j_correction = np.zeros([seq_len, seq_len, 21])
    # J correction interactions with other aminoacids
    reduced_j = potts_model['J'][range(seq_len), :, seq_index, :].astype(np.float32)
    j_correction += reduced_j[:, pos1, seq_index[pos1]] * mask[:, pos1]
    j_correction -= reduced_j[:, pos1, aa1] * mask[:, pos1]

    # J correction, interaction with self aminoacids
    decoy_energy += j_correction.sum(axis=0)

    return decoy_energy


def compute_mutational_decoy_energy_fluctuation(seq: str,
                                                potts_model: dict,
                                                mask: np.array, ) -> np.array:
    r"""
    Computes a (LxLx21x21) matrix for a sequence of length L. Matrix[i,j] describes all possible changes in energy upon mutating residue i and j simultaneously.
    
    .. math::
        \Delta H_{ij} = H_i - H_{i'} + H_{j}-H_{j'} + J_{ij} -J_{ij'} + J_{i'j'} - J_{i'j} + \\sum_k {J_{ik} - J_{i'k} + J_{jk} -J_{j'k}}
        
    Parameters
    ----------
    seq : str
        The amino acid sequence of the protein. The sequence is assumed to be in one-letter code. Gaps are represented as '-'. The length of the sequence (L) should match the dimensions of the Potts model.
    potts_model : dict
        A dictionary containing the Potts model parameters 'h' (fields) and 'J' (couplings). The fields are a 2D array of shape (L, 20), where L is the length of the sequence and 20 is the number of amino acids. The couplings are a 4D array of shape (L, L, 20, 20). The fields and couplings are assumed to be in units of energy.
    mask : np.array
        A 2D Boolean array that determines which residue pairs should be considered in the energy computation. The mask should have dimensions (L, L), where L is the length of the sequence.

    Returns
    -------
    decoy_energy2: np.array
        (LxLx21x21) matrix describing the energetic changes upon mutating two residues simultaneously.

    Examples
    --------
    >>> seq = "ACDEFGHIKLMNPQRSTVWY"
    >>> potts_model = {
        'h': np.random.rand(20, 20),  # Random fields
        'J': np.random.rand(20, 20, 20, 20)  # Random couplings
    }
    >>> mask = np.ones((len(seq), len(seq)), dtype=bool) # Include all pairs
    >>> decoy_energy2 = compute_mutational_decoy_energy_fluctuation(seq, potts_model, mask)
    >>> print(f"Matrix of Contact Mutational Decoy Energy Fluctuations: "); print(decoy_energy2)
    >>> print(f"Matrix Size: "); print(shape(decoy_energy2))

    Notes
    -----
    The couplings energy is computed as the half-sum of the couplings for all pairs of residues
    where the mask is True. The division by 2 for the couplings accounts for double-counting in symmetric
    matrices.

    .. todo:: Optimize the computation.
    """
    seq_index = np.array([_AA.find(aa) for aa in seq])
    seq_len = len(seq_index)

    # Create masked decoys
    pos1_flat,pos2_flat=np.where(mask>0)
    contacts_len=len(pos1_flat)

    pos1,aa1,aa2=np.meshgrid(pos1_flat, np.arange(21), np.arange(21), indexing='ij', sparse=True)
    pos2,aa1,aa2=np.meshgrid(pos2_flat, np.arange(21), np.arange(21), indexing='ij', sparse=True)

    #Compute fields
    decoy_energy = np.zeros([contacts_len, 21, 21])
    decoy_energy -= (potts_model['h'][pos1, aa1] - potts_model['h'][pos1, seq_index[pos1]])  # h correction aa1
    decoy_energy -= (potts_model['h'][pos2, aa2] - potts_model['h'][pos2, seq_index[pos2]])  # h correction aa2

    #Compute couplings
    # j_correction = np.zeros([contacts_len, 21, 21])
    # for pos, aa in enumerate(seq_index):
    #     # J correction interactions with other aminoacids
    #     reduced_j = potts_model['J'][pos, :, aa, :].astype(np.float32)
    #     j_correction += reduced_j[pos1, seq_index[pos1]] * mask[pos, pos1]
    #     j_correction -= reduced_j[pos1, aa1] * mask[pos, pos1]
    #     j_correction += reduced_j[pos2, seq_index[pos2]] * mask[pos, pos2]
    #     j_correction -= reduced_j[pos2, aa2] * mask[pos, pos2]

    # Vectorized: Compute reduced_j[pos1, aa1] * mask[pos, pos1] in einsum
    # V[j, a] = sum_i J[i, j, seq_index[i], a] * mask[i, j] 
    j_reduced = potts_model['J'][np.arange(seq_len), :, seq_index, :].astype(np.float32)  # (L, L, 21)
    mask_float = mask.astype(np.float32)
    V = np.einsum('ijk,ij->jk', j_reduced, mask_float)  # (L, 21)
    j_correction = (  V[pos1_flat, seq_index[pos1_flat]][:, None, None]
                    - V[pos1_flat, :][:, :, None]
                    + V[pos2_flat, seq_index[pos2_flat]][:, None, None]
                    - V[pos2_flat, :][:, None, :])
    
    # J correction, interaction with self aminoacids
    j_correction -= potts_model['J'][pos1, pos2, seq_index[pos1], seq_index[pos2]] * mask[pos1, pos2]  # Taken two times
    j_correction += potts_model['J'][pos1, pos2, aa1, seq_index[pos2]] * mask[pos1, pos2]  # Added mistakenly
    j_correction += potts_model['J'][pos1, pos2, seq_index[pos1], aa2] * mask[pos1, pos2]  # Added mistakenly
    j_correction -= potts_model['J'][pos1, pos2, aa1, aa2] * mask[pos1, pos2]  # Correct combination
    decoy_energy += j_correction
    
    decoy_energy2=np.zeros([seq_len,seq_len,21,21])
    decoy_energy2[mask]=decoy_energy
    return decoy_energy2


def compute_configurational_decoy_energy_fluctuation(seq: str,
                                                     potts_model: dict,
                                                     mask: np.array, ) -> np.array:
    r"""
    Computes a (LxLx21x21) matrix for a sequence of length L. Matrix[i,j] describes all possible changes in energy upon mutating and altering the 
    local densities of residue i and j simultaneously.
    
    .. math::
        \Delta H_{ij} = H_i - H_{i'} + H_{j}-H_{j'} + J_{ij} -J_{ij'} + J_{i'j'} - J_{i'j} + \\sum_k {J_{ik} - J_{i'k} + J_{jk} -J_{j'k}}
        
    Parameters
    ----------
    seq : str
        The amino acid sequence of the protein. The sequence is assumed to be in one-letter code. Gaps are represented as '-'. The length of the sequence (L) should match the dimensions of the Potts model.
    potts_model : dict
        A dictionary containing the Potts model parameters 'h' (fields) and 'J' (couplings). The fields are a 2D array of shape (L, 20), where L is the length of the sequence and 20 is the number of amino acids. The couplings are a 4D array of shape (L, L, 20, 20). The fields and couplings are assumed to be in units of energy.
    mask : np.array
        A 2D Boolean array that determines which residue pairs should be considered in the energy computation. The mask should have dimensions (L, L), where L is the length of the sequence.

    Returns
    -------
    decoy_energy2: np.array
        (LxLx21x21) matrix describing the energetic changes upon mutating and altering the local densities of two residues simultaneously.

    Examples
    --------
    >>> seq = "ACDEFGHIKLMNPQRSTVWY"
    >>> potts_model = {
        'h': np.random.rand(20, 20),  # Random fields
        'J': np.random.rand(20, 20, 20, 20)  # Random couplings
    }
    >>> mask = np.ones((len(seq), len(seq)), dtype=bool) # Include all pairs
    >>> decoy_energy2 = compute_configurational_decoy_energy_fluctuation(seq, potts_model, mask)
    >>> print(f"Matrix of Contact Configurational Decoy Energy Fluctuations: "); print(decoy_energy2)
    >>> print(f"Matrix Size: "); print(shape(decoy_energy2))

    Notes
    -----
    The couplings energy is computed as the half-sum of the couplings for all pairs of residues
    where the mask is True. The division by 2 for the couplings accounts for double-counting in symmetric
    matrices.

    .. todo:: Optimize the computation.
    """
    seq_index = np.array([_AA.find(aa) for aa in seq])
    seq_len = len(seq_index)

    # Create masked decoys
    pos1,pos2=np.where(mask>0)
    contacts_len=len(pos1)

    pos1,aa1,aa2=np.meshgrid(pos1, np.arange(21), np.arange(21), indexing='ij', sparse=True)
    pos2,aa1,aa2=np.meshgrid(pos2, np.arange(21), np.arange(21), indexing='ij', sparse=True)

    #Compute fields
    decoy_energy = np.zeros([contacts_len, 21, 21])
    decoy_energy -= (potts_model['h'][pos1, aa1] - potts_model['h'][pos1, seq_index[pos1]])  # h correction aa1
    decoy_energy -= (potts_model['h'][pos2, aa2] - potts_model['h'][pos2, seq_index[pos2]])  # h correction aa2

    #Compute couplings
    j_correction = np.zeros([contacts_len, 21, 21])
    for pos, aa in enumerate(seq_index):
        # J correction interactions with other aminoacids
        reduced_j = potts_model['J'][pos, :, aa, :].astype(np.float32)
        j_correction += reduced_j[pos1, seq_index[pos1]] * mask[pos, pos1]
        j_correction -= reduced_j[pos1, aa1] * mask.mean()
        j_correction += reduced_j[pos2, seq_index[pos2]] * mask[pos, pos2]
        j_correction -= reduced_j[pos2, aa2] * mask.mean()
    # J correction, interaction with self aminoacids
    j_correction -= potts_model['J'][pos1, pos2, seq_index[pos1], seq_index[pos2]] * mask[pos1, pos2]  # Taken two times
    j_correction += potts_model['J'][pos1, pos2, aa1, seq_index[pos2]] * mask.mean()  # Added mistakenly
    j_correction += potts_model['J'][pos1, pos2, seq_index[pos1], aa2] * mask.mean()  # Added mistakenly
    j_correction -= potts_model['J'][pos1, pos2, aa1, aa2] * mask.mean()  # Correct combination
    decoy_energy += j_correction
    
    decoy_energy2=np.zeros([seq_len,seq_len,21,21])
    decoy_energy2[mask]=decoy_energy
    return decoy_energy2


def compute_contact_decoy_energy_fluctuation(seq: str,
                                             potts_model: dict,
                                             mask: np.array) -> np.array:
    r"""
    $$ \Delta DCA_{ij} = \Delta j_{ij} $$
    :param seq:
    :param potts_model:
    :param mask:
    :return:
    """

    seq_index = np.array([_AA.find(aa) for aa in seq])
    seq_len = len(seq_index)

    # Create decoys
    pos1, pos2, aa1, aa2 = np.meshgrid(np.arange(seq_len), np.arange(seq_len), np.arange(21), np.arange(21),
                                       indexing='ij', sparse=True)

    decoy_energy = np.zeros([seq_len, seq_len, 21, 21])
    decoy_energy += potts_model['J'][pos1, pos2, seq_index[pos1], seq_index[pos2]] * mask[pos1, pos2]  # Old coupling
    decoy_energy -= potts_model['J'][pos1, pos2, aa1, aa2] * mask[pos1, pos2]  # New Coupling

    return decoy_energy


def compute_decoy_energy(seq: str, potts_model: dict, mask: np.array, kind='singleresidue') -> np.array:
    """
    Computes all possible decoy energies.
    
    Parameters
    ----------
    seq : str
        The amino acid sequence of the protein. The sequence is assumed to be in one-letter code. Gaps are represented as '-'. The length of the sequence (L) should match the dimensions of the Potts model.
    potts_model : dict
        A dictionary containing the Potts model parameters 'h' (fields) and 'J' (couplings). The fields are a 2D array of shape (L, 20), where L is the length of the sequence and 20 is the number of amino acids. The couplings are a 4D array of shape (L, L, 20, 20). The fields and couplings are assumed to be in units of energy.
    mask : np.array
        A 2D Boolean array that determines which residue pairs should be considered in the energy computation. The mask should have dimensions (L, L), where L is the length of the sequence.
    kind : str
        Kind of decoys generated. Options: "singleresidue," "mutational," "configurational," and "contact." 
    Returns
    -------
    decoy_energy: np.array
        Matrix describing all possible decoy energies.

    Examples
    --------
    >>> seq = "ACDEFGHIKLMNPQRSTVWY"
    >>> potts_model = {
        'h': np.random.rand(20, 20),  # Random fields
        'J': np.random.rand(20, 20, 20, 20)  # Random couplings
    }
    >>> mask = np.ones((len(seq), len(seq)), dtype=bool) # Include all pairs
    >>> kind = "singleresidue"
    >>> decoy_energy = compute_decoy_energy(seq, potts_model, mask, kind)
    >>> print(f"Matrix of Single Residue Decoy Energo: "); print(decoy_energy2)
    >>> print(f"Matrix Size: "); print(shape(decoy_energy2))

    Notes
    -----
    The couplings energy is computed as the half-sum of the couplings for all pairs of residues
    where the mask is True. The division by 2 for the couplings accounts for double-counting in symmetric
    matrices.

    .. todo:: Optimize the computation.
    """

    native_energy = compute_native_energy(seq, potts_model, mask)
    if kind == 'singleresidue':
        decoy_energy=native_energy + compute_singleresidue_decoy_energy_fluctuation(seq, potts_model, mask)
    elif kind == 'mutational':
        decoy_energy=native_energy + compute_mutational_decoy_energy_fluctuation(seq, potts_model, mask)
    elif kind == 'configurational':
        decoy_energy=native_energy + compute_configurational_decoy_energy_fluctuation(seq, potts_model, mask)
    elif kind == 'contact':
        decoy_energy=native_energy + compute_contact_decoy_energy_fluctuation(seq, potts_model, mask)
    return decoy_energy

def compute_aa_freq(seq, include_gaps=True):
    """
    Calculates amino acid frequencies in given sequence

    Parameters
    ----------
    seq :  str
        The amino acid sequence of the protein. The sequence is assumed to be in one-letter code. Gaps are represented as '-'.
    include_gaps: bool
        If True, frequencies of gaps ('-') in the sequence are set to 0.
        Default is True.
        

    Returns
    -------
    aa_freq: np.array
        Array of frequencies of all 21 possible amino acids within sequence
    """
    seq_index = np.array([_AA.find(aa) for aa in seq])
    aa_freq = np.array([(seq_index == i).sum() for i in range(21)])
    if not include_gaps:
        aa_freq[0] = 0
    return aa_freq


def compute_contact_freq(seq):
    """
    Calculates contact frequencies in given sequence

    Parameters
    ----------
    seq :  str
        The amino acid sequence of the protein. The sequence is assumed to be in one-letter code. Gaps are represented as '-'.
        
    Returns
    -------
    contact_freq: np.array
        21x21 array of frequencies of all possible contacts within sequence.
    """
    seq_index = np.array([_AA.find(aa) for aa in seq])
    aa_freq = np.array([(seq_index == i).sum() for i in range(21)], dtype=np.float64)
    aa_freq /= aa_freq.sum()
    contact_freq = (aa_freq[:, np.newaxis] * aa_freq[np.newaxis, :])
    return contact_freq


def compute_single_frustration(decoy_fluctuation,
                               aa_freq=None,
                               correction=0):
    """
    Calculates single residue frustration indices

    Parameters
    ----------
    decoy_fluctuation: np.array
        (Lx21) matrix for a sequence of length L, describing the energetic changes upon mutating a single residue. 
    aa_freq: np.array
        Array of frequencies of all 21 possible amino acids within sequence
        
    Returns
    -------
    frustration: np.array
        Array of length L featuring single residue frustration indices.
    """
    if aa_freq is None:
        aa_freq = np.ones(21)
    mean_energy = (aa_freq * decoy_fluctuation).sum(axis=1) / aa_freq.sum()
    std_energy = np.sqrt(
        ((aa_freq * (decoy_fluctuation - mean_energy[:, np.newaxis]) ** 2) / aa_freq.sum()).sum(axis=1))
    frustration = -mean_energy / (std_energy + correction)
    frustration *= -1
    return frustration


def compute_pair_frustration(decoy_fluctuation,
                             contact_freq: Union[None, np.array],
                             correction=0) -> np.array:
    """
    Calculates pair residue frustration indices

    Parameters
    ----------
    decoy_fluctuation: np.array
        (LxLx21x21) matrix for a sequence of length L, describing the energetic changes upon mutating two residues simultaneously. 
    contact_freq: np.array
        21x21 array of frequencies of all possible contacts within sequence.
        
    Returns
    -------
    contact_frustration: np.array
        LxL array featuring pair frustration indices (mutational or configurational frustration, depending on 
        decoy_fluctuation matrix provided)
    """
    if contact_freq is None:
        contact_freq = np.ones([21, 21])
    decoy_energy = decoy_fluctuation
    seq_len = decoy_fluctuation.shape[0]
    average = np.average(decoy_energy.reshape(seq_len * seq_len, 21 * 21), weights=contact_freq.flatten(), axis=-1)
    variance = np.average((decoy_energy.reshape(seq_len * seq_len, 21 * 21) - average[:, np.newaxis]) ** 2,
                          weights=contact_freq.flatten(), axis=-1)
    mean_energy = average.reshape(seq_len, seq_len)
    std_energy = np.sqrt(variance).reshape(seq_len, seq_len)
    contact_frustration = -mean_energy / (std_energy + correction)
    contact_frustration *= -1
    return contact_frustration


def compute_scores(potts_model: dict) -> np.array:
    """
    Computes contact scores based on the Frobenius norm
    
    .. math::
        CN[i,j] = \\frac{F[i,j] - F[i,:] * F[:,j}{F[:,:]}

    Parameters
    ----------
    potts_model :  dict
        Potts model containing the couplings in the "J" key

    Returns
    -------
    corr_norm : np.array
        Contact score matrix (N x N)
    """
    j = potts_model['J']
    n, _, __, q = j.shape
    norm = np.linalg.norm(j.reshape(n * n, q * q), axis=1).reshape(n, n)  # Frobenius norm
    norm_mean = np.mean(norm, axis=0) / (n - 1) * n
    norm_mean_all = np.mean(norm) / (n - 1) * n
    corr_norm = norm - norm_mean[:, np.newaxis] * norm_mean[np.newaxis, :] / norm_mean_all
    corr_norm[np.diag_indices(n)] = 0
    corr_norm = np.mean([corr_norm, corr_norm.T], axis=0)  # Symmetrize matrix
    return corr_norm


def compute_roc(scores, distance_matrix, cutoff):

    """
    Computes Receiver Operating Characteristic (ROC) curve of 
    predicted and true contacts (identified from the distance matrix).

    Parameters
    ----------
    scores :  np.array
        Contact score matrix (N x N)
    distance_matrix : np.array
        LxL array for sequence of length L, describing distances between contacts
    cutoff : float
        Distance cutoff for contacts

    Returns
    -------
    roc_score : np.array
        Array containing lists of false and true positive rates 
    """

    scores = sdist.squareform(scores)
    distance = sdist.squareform(distance_matrix)
    results = np.array([np.array(scores), np.array(distance)])
    results = results[:, results[0, :].argsort()[::-1]]  # Sort results by score
    if cutoff!= None:
        contacts = results[1] <= cutoff
    else:
        contacts = results[1]>0
    not_contacts = ~contacts
    tpr = np.concatenate([[0], contacts.cumsum() / contacts.sum()])
    fpr = np.concatenate([[0], not_contacts.cumsum() / not_contacts.sum()])
    roc_score=np.array([fpr, tpr])
    return roc_score


def compute_auc(roc_score):
    """
    Computes Area Under Curve (AUC) of calculated ROC distribution

    Parameters
    ----------
    roc_score : np.array
        Array containing lists of false and true positive rates 

    Returns
    -------
    auc : float
        AUC value
    """
    fpr, tpr = roc
    auc = np.sum(tpr[:-1] * (fpr[1:] - fpr[:-1]))
    return auc


def plot_roc(roc_score):
    """
    Plot ROC distribution

    Parameters
    ----------
    roc_score : np.array
        Array containing lists of false and true positive rates 
    """
    import matplotlib.pyplot as plt
    plt.plot(roc[0], roc[1])
    plt.xlabel('False positive rate (1-specificity)')
    plt.ylabel('True positive rate (sensiticity)')
    plt.suptitle('Receiver operating characteristic')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.plot([0, 1], [0, 1], '--')


def plot_singleresidue_decoy_energy(decoy_energy, native_energy, method='clustermap'):
    """
    Plot comparison of single residue decoy energies, relative to the native energy

    Parameters
    ----------
    decoy_energy : np.array
        Lx21 array of decoy energies
    native_energy : float
        Native energy value
    method : str
        Options: "clustermap", "heatmap"
    """
    import seaborn as sns
    if method=='clustermap':
        f=sns.clustermap
    elif method == 'heatmap':
        f = sns.heatmap
    g = f(decoy_energy, cmap='RdBu_r',
          vmin=native_energy - decoy_energy.std() * 3,
          vmax=native_energy + decoy_energy.std() * 3)
    AA_dict = {str(i): _AA[i] for i in range(len(_AA))}
    new_ticklabels = []
    if method == 'clustermap':
        ax_heatmap = g.ax_heatmap
    else:
        ax_heatmap = g.axes
    for t in ax_heatmap.get_xticklabels():
        t.set_text(AA_dict[t.get_text()])
        new_ticklabels += [t]
    ax_heatmap.set_xticklabels(new_ticklabels)
    return g


def write_tcl_script(pdb_file: Union[Path,str], chain: str, mask: np.array, distance_matrix: np.array, distance_cutoff: float, single_frustration: np.array,
                    pair_frustration: np.array, tcl_script: Union[Path, str] ='frustration.tcl',max_connections: int =None, movie_name: Union[Path, str] =None, still_image_name: Union[Path, str] =None) -> Union[Path, str]:
    """
    Writes a tcl script that can be run with VMD to superimpose the frustration patterns onto the corresponding PDB structure. 

    Parameters
    ----------
    pdb_file :  Path or str
        pdb file name
    chain : str
        Select chain from pdb
    mask : np.array
        A 2D Boolean array that determines which residue pairs should be considered in the energy computation. The mask should have dimensions (L, L), where L is the length of the sequence.
    distance_matrix : np.array
        LxL array for sequence of length L, describing distances between contacts
    distance_cutoff : float
        Maximum distance at which a contact occurs
    single_frustration : np.array
        Array containing single residue frustration index values
    pair_frustration : np.array
        Array containing pair (ex. configurational, mutational, contact) frustration index values
    tcl_script : Path or str
        Output tcl script file with static structure
    max_connections : int
        Maximum number of pair frustration values visualized in tcl file
    movie_name : Path or str
        Output movie file with rotating structure
    still_image_name : Path or str
        Output image file with still image
    

    Returns
    -------
    tcl_script : Path or str
        tcl script file
    """
    fo = open(tcl_script, 'w+')
    single_frustration = np.nan_to_num(single_frustration,nan=0,posinf=0,neginf=0)
    pair_frustration = np.nan_to_num(pair_frustration,nan=0,posinf=0,neginf=0)
    
    
    structure = prody.parsePDB(str(pdb_file))
    selection = structure.select('protein', chain=chain)
    residues = np.unique(selection.getResnums())

    fo.write(f'[atomselect top all] set beta 0\n')
    # Single residue frustration
    for r, f in zip(residues, single_frustration):
        # print(f)
        fo.write(f'[atomselect top "chain {chain} and residue {int(r)}"] set beta {f}\n')

    # Mutational frustration:
    r1, r2 = np.meshgrid(residues, residues, indexing='ij')
    sel_frustration = np.array([r1.ravel(), r2.ravel(), pair_frustration.ravel(),distance_matrix.ravel(), mask.ravel()]).T
    #Filter with mask and distance
    if distance_cutoff:
        mask_dist=(sel_frustration[:, -2] <= distance_cutoff)
    else:
        mask_dist=np.ones(len(sel_frustration),dtype=bool)
    sel_frustration = sel_frustration[mask_dist & (sel_frustration[:, -1] > 0)]
    
    minimally_frustrated = sel_frustration[sel_frustration[:, 2] < -0.78]
    #minimally_frustrated = sel_frustration[sel_frustration[:, 2] < -1.78]
    sort_index = np.argsort(minimally_frustrated[:, 2])
    minimally_frustrated = minimally_frustrated[sort_index]
    if max_connections:
        minimally_frustrated = minimally_frustrated[:max_connections]
    fo.write('draw color green\n')
    

    for (r1, r2, f, d ,m) in minimally_frustrated:
        r1=int(r1)
        r2=int(r2)
        if abs(r1-r2) == 1: # don't draw interactions between residues adjacent in sequence
            continue
        pos1 = selection.select(f'resid {r1} and chain {chain} and (name CB or (resname GLY and name CA))').getCoords()[0]
        pos2 = selection.select(f'resid {r2} and chain {chain} and (name CB or (resname GLY and name CA))').getCoords()[0]
        distance = np.linalg.norm(pos1 - pos2)
        if d > 9.5 or d < 3.5:
            continue
        fo.write(f'lassign [[atomselect top "resid {r1} and name CA and chain {chain}"] get {{x y z}}] pos1\n')
        fo.write(f'lassign [[atomselect top "resid {r2} and name CA and chain {chain}"] get {{x y z}}] pos2\n')
        if 3.5 <= distance <= 6.5:
            fo.write(f'draw line $pos1 $pos2 style solid width 2\n')
        else:
            fo.write(f'draw line $pos1 $pos2 style dashed width 2\n')

    frustrated = sel_frustration[sel_frustration[:, 2] > 1]
    #frustrated = sel_frustration[sel_frustration[:, 2] > 0]
    sort_index = np.argsort(frustrated[:, 2])[::-1]
    frustrated = frustrated[sort_index]
    if max_connections:
        frustrated = frustrated[:max_connections]
    fo.write('draw color red\n')
    for (r1, r2, f ,d, m) in frustrated:
        r1=int(r1)
        r2=int(r2)
        if d > 9.5 or d < 3.5:
            continue
        fo.write(f'lassign [[atomselect top "resid {r1} and name CA and chain {chain}"] get {{x y z}}] pos1\n')
        fo.write(f'lassign [[atomselect top "resid {r2} and name CA and chain {chain}"] get {{x y z}}] pos2\n')
        if 3.5 <= d <= 6.5:
            fo.write(f'draw line $pos1 $pos2 style solid width 2\n')
        else:
            fo.write(f'draw line $pos1 $pos2 style dashed width 2\n')
    
    fo.write('''mol delrep top 0
            mol color Beta
            mol representation NewCartoon 0.300000 10.000000 4.100000 0
            mol selection all
            mol material Opaque
            mol addrep top
            color scale method GWR
            ''')
    
    if movie_name:
        fo.write('''axes location Off
            color Display Background white
            display resize 800 800
            display projection Orthographic
            display depthcue off
            display resetview
            display resize [expr [lindex [display get size] 0]/2*2] [expr [lindex [display get size] 1]/2*2] ;#Resize display to even height and width
            display update ui

            # Set up the movie directory and base file name
            mkdir movie_tmp
            set workdir "movie_tmp"
            ''' + f'set basename "{movie_name}"' + '''
            set numframes 360
            set framerate 25

            # Function to rotate the molecule and capture frames
            proc captureFrames {} {
                global workdir basename numframes
                for {set i 0} {$i < $numframes} {incr i} {
                    # Rotate the molecule around the Y-axis
                    rotate y by 1
                    
                    # Capture the frame
                    set output [format "%s/$basename.%05d.tga" $workdir $i]
                    render snapshot $output
                }
            }

            # Function to convert frames to MP4
            proc convertToMP4 {} {
                global workdir basename numframes framerate

                set mybasefilename [format "%s/%s" $workdir $basename]
                set outputFile [format "%s.mp4" $basename]
                
                # Construct and execute the ffmpeg command
                
                set command "ffmpeg -y -framerate $framerate -i $mybasefilename.%05d.tga -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p $outputFile"
                puts "Executing: $command"
                exec ffmpeg -y -framerate $framerate -i $mybasefilename.%05d.tga -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p $outputFile >&@ stdout
            }

            # Main script execution
            captureFrames
            convertToMP4

            # Cleanup the TGA files if desired
            for {set i 0} {$i < $numframes} {incr i} {
                set output [format "%s/$basename.%05d.tga" $workdir $i]
                exec rm $output
            }
            exit
        ''')
    elif still_image_name:
        fo.write(f'set output "{still_image_name}"' + '''
            render snapshot $output
            exit
        ''')
    fo.close()
    return tcl_script


def call_vmd(pdb_file: Union[Path,str], tcl_script: Union[Path,str]):
    """
    Calls VMD program with given pdb file and tcl script to visualize frustration patterns

    Parameters
    ----------
    pdb_file :  Path or str
        pdb file name
    tcl_script : Path or str
        Output tcl script file with static structure
    """
    import subprocess
    return subprocess.Popen(['vmd', '-e', tcl_script, pdb_file], stdin=subprocess.PIPE)


def canvas(with_attribution=True):
    """
    Placeholder function to show example docstring (NumPy format).

    Replace this function and doc string for your own project.

    Parameters
    ----------
    with_attribution : bool, Optional, default: True
        Set whether or not to display who the quote is from.

    Returns
    -------
    quote : str
        Compiled string including quote and optional attribution.
    """

    quote = "The code is but a canvas to our imagination."
    if with_attribution:
        quote += "\n\t- Adapted from Henry David Thoreau"
    return quote


###

def make_decoy_seqs(seq, ndecoys=1000):
    """
    Creates permutated, decoy sequences using a given sequence residue composition and length.
    
    Parameters
    ----------
    seq : str
        The amino acid sequence of the protein. The sequence is assumed to be in one-letter code. Gaps are represented as '-'. The length of the sequence (L) should match the dimensions of the Potts model.
    n_decoys: int
        Number of sequence decoys to create
    
    Return
    -------
    decoy_seqs : list
        List of decoy sequences. The sequences are assumed to be in one-letter code. Gaps are represented as '-'. The length of the sequences (L) should match the dimensions of the Potts model.
    """
    
    seq_array = np.array(list(seq))
    decoy_seqs = [''.join(np.random.permutation(seq_array)) for _ in range(ndecoys)]
    return decoy_seqs


def compute_fragment_mask(mask: np.array,
                  fragment_pos: np.array)-> np.array:
    """
    Creates a mask for a sequence fragment such that:
    - position i belongs to the fragment, all j
    - position j belongs to the fragment, all i
    
    The new mask consider all the interactions within the fragment and also the interactions between the fragment and other sequence positions.
    
    Parameters
    ----------
    
    mask : np.array
       A 2D Boolean array that determines which residue pairs should be considered in the energy computation. The mask should have dimensions (L, L), where L is the length of the sequence.
   fragment_pos: np:array
        Array of sequence positions selected. 
        
    Return
    -------
    fragment_mask: np.array
        New 2D Boolean array that determines which residue pairs should be considered in the energy computation. The mask should have dimensions (L, L), where L is the length of the sequence.
    """

    custom_mask=np.zeros((mask.shape[0],mask.shape[0]),dtype=bool)
    custom_mask[fragment_pos]=True
    custom_mask[:,fragment_pos]=True
    fragment_mask = custom_mask*mask
    return fragment_mask



def compute_fragment_total_native_energy(seq: str,
                                         potts_model: dict,
                                         mask: np.array,
                                         fragment_pos : Union[None, np.array] = None,
                                         fragment_in_context = False ) -> float:
    
    """
    Calculates the energy for the complete protein or for a fragment in context

    Parameters
    ----------
    seq : str
        The amino acid sequence of the protein. The sequence is assumed to be in one-letter code. Gaps are represented as '-'. The length of the sequence (L) should match the dimensions of the Potts model.
    potts_model : dict
        A dictionary containing the Potts model parameters 'h' (fields) and 'J' (couplings). The fields are a 2D array of shape (L, 20), where L is the length of the sequence and 20 is the number of amino acids. The couplings are a 4D array of shape (L, L, 20, 20). The fields and couplings are assumed to be in units of energy.
    mask : np.array
        A 2D Boolean array that determines which residue pairs should be considered in the energy computation. The mask should have dimensions (L, L), where L is the length of the sequence.
   fragment_pos: np:array
        Array of sequence positions selected.
   fragment_in_context: bool
        If True, the energetics calculations take into account the interactions between the fragment and other sequence positions
    
    Return
    -------
    energy: float
        Native energy of the protein 
    """   
    
    
    seq_index = np.array([_AA.find(aa) for aa in seq])
    seq_len = len(seq_index)

    pos1, pos2 = np.meshgrid(np.arange(seq_len), np.arange(seq_len), indexing='ij', sparse=True)
    aa1, aa2 = np.meshgrid(seq_index, seq_index, indexing='ij', sparse=True)
    if len(potts_model['J'].shape)==4:
      #  print('potts')
        h = -potts_model['h'][range(seq_len), seq_index]
        j = -potts_model['J'][pos1, pos2, aa1, aa2]
    else:
        #MJ 
       # print('MJ')
        h = 0
        j = -potts_model['J'][aa1, aa2]
    
    if fragment_in_context:
        h_mask=np.zeros(seq_len,dtype=int)    
        h_mask[fragment_pos]=1
        j_mask=compute_fragment_mask(mask,fragment_pos)
    else:
        h_mask = 1
        j_mask = 1
    
    h_prime= h*h_mask
    j_prime = j * j_mask

    energy = h_prime.sum() + j_prime.sum() / 2
    return energy

def compute_fragment_total_decoy_energy(decoy_seqs: list,
                                        potts_model: dict,
                                        mask: np.array,
                                        fragment_pos : Union[None, np.array] = None,
                                        fragment_in_context = False, 
                                        split_couplings_and_fields = False,
                                        config_decoys = False,
                                        msa_mask = 1) -> np.array:
    """
    Calculates decoy energies for the complete protein or for a fragment in context

    Parameters
    ----------
    decoy_seqs : list
        List of decoy sequences. The sequences are assumed to be in one-letter code. Gaps are represented as '-'. The length of the sequences (L) should match the dimensions of the Potts model.
    potts_model : dict
        A dictionary containing the Potts model parameters 'h' (fields) and 'J' (couplings). The fields are a 2D array of shape (L, 20), where L is the length of the sequence and 20 is the number of amino acids. The couplings are a 4D array of shape (L, L, 20, 20). The fields and couplings are assumed to be in units of energy.
    mask : np.array
        A 2D Boolean array that determines which residue pairs should be considered in the energy computation. The mask should have dimensions (L, L), where L is the length of the sequence.
   fragment_pos: np:array
        Array of sequence positions selected.
   fragment_in_context: bool
        If True, the energetics calculations take into account the interactions between the fragment and other sequence positions
    split_couplings_and_fields: bool
        Separate output into coupling and local fields contribution to energy.
    config_decoys: bool
        If True, use the configurational decoys approximation, shuffling index positions for configurational decoys energy calculation. If False, mutational decoys.
    msa_mask: np.array
        Extra mask to use a Multiple Sequence Alignment that do not cover completely the reference PDB
    
    Return
    -------
    energy: np.array
        Decoy energies
    """   
    
    seq_index = np.array([[_AA.find(aa) for aa in seq] for seq in decoy_seqs])
    N_seqs, seq_len = seq_index.shape
    pos_index=np.repeat([np.arange(seq_len)], N_seqs,axis=0)
    
    if config_decoys:

        pos_index=np.array([np.random.choice(pos_index[0],
                                             size=len(pos_index[0]),
                                             replace=False) for x in range(pos_index.shape[0])])
        mask=np.ones(mask.shape)*mask.mean()

    mask=mask*msa_mask    
       
    pos1=np.array([np.meshgrid(p, p, indexing='ij', sparse=True)[0] for p in pos_index])
    pos2=np.array([np.meshgrid(p, p, indexing='ij', sparse=True)[1] for p in pos_index])
    aa1=np.array([np.meshgrid(s, s, indexing='ij', sparse=True)[0] for s in seq_index])
    aa2=np.array([np.meshgrid(s, s, indexing='ij', sparse=True)[1] for s in seq_index])
    if len(potts_model['J'].shape)==4:
        h = -potts_model['h'][pos_index,seq_index]
        j = -potts_model['J'][pos1, pos2, aa1, aa2]
    else:
        #MJ 
        h = 0
        j = -potts_model['J'][aa1, aa2]
    
    if fragment_in_context:
        h_mask=np.zeros(seq_len,dtype=int)    
        h_mask[fragment_pos]=1
        j_mask=compute_fragment_mask(mask,fragment_pos)
    else:
        h_mask = 1
        j_mask = 1
        
    h_prime= h*h_mask
    j_prime = j * j_mask  
    
    if split_couplings_and_fields:
        return np.array([h_prime.sum(axis=-1),j_prime.sum(axis=-1).sum(axis=-1) / 2])
    else:
        energy = h_prime.sum(axis=-1) + j_prime.sum(axis=-1).sum(axis=-1) / 2
        return energy
    
    
def compute_total_frustration(seq,
                              potts_model,
                              mask, 
                              ndecoys = 1000,
                              config_decoys = False,
                              msa_mask = 1,
                              fragment_pos = None,
                              fragment_in_context = False,
                              output_kind = 'frustration'):
    """
    Calculates the total frustration of a protein fragment.

    Parameters
    ----------
    seq : str
        The amino acid sequence of the protein. The sequence is assumed to be in one-letter code. Gaps are represented as '-'. The length of the sequence (L) should match the dimensions of the Potts model.
    potts_model : dict
        A dictionary containing the Potts model parameters 'h' (fields) and 'J' (couplings). The fields are a 2D array of shape (L, 20), where L is the length of the sequence and 20 is the number of amino acids. The couplings are a 4D array of shape (L, L, 20, 20). The fields and couplings are assumed to be in units of energy.
    mask : np.array
        A 2D Boolean array that determines which residue pairs should be considered in the energy computation. The mask should have dimensions (L, L), where L is the length of the sequence.
    n_decoys: int
        Number of sequence decoys to create
    config_decoys: bool
        If True, use the configurational decoys approximation, shuffling index positions for configurational decoys energy calculation. If False, mutational decoys.
    msa_mask: np.array
        Extra mask to use a Multiple Sequence Alignment that do not cover completely the reference PDB
    fragment_pos: np.array
        Fragment positions. If None, use the complete model
    fragment_in_context: bool
        If True, the energetics calculations take into account the interactions between the fragment and other sequence positions
    output_kind: str
        If 'frustration', returns frustration. If not, returns native energy, decoy energy average and decoy energy standard deviation.
    Return
    -------
    total_frustration : float
        Total frustration of the fragment or complete protein
    native_energy: float
        Native energy of the given sequence
    decoy_energy_average: float
        Average of the decoy energy distribution
    decoy_energy_std: float
        Standard deviation of the decoy energy distribution
    """
   
    native_energy = compute_fragment_total_native_energy(seq,
                                                         potts_model,
                                                         mask,
                                                         fragment_pos,
                                                         fragment_in_context)
    decoy_seqs = make_decoy_seqs(seq,ndecoys=ndecoys)

    decoy_energies = compute_fragment_total_decoy_energy(decoy_seqs,
                                                        potts_model,
                                                        mask,
                                                        fragment_pos,
                                                        fragment_in_context,
                                                        config_decoys = config_decoys,
                                                        msa_mask = msa_mask)
    decoy_energy_average = decoy_energies.mean()
    decoy_energy_std = decoy_energies.std()

    total_frustration = (native_energy - decoy_energy_average)/ decoy_energy_std
    if output_kind == 'frustration':
    
        return total_frustration
    else:
        return native_energy, decoy_energy_average, decoy_energy_std



def compute_native_h_J(seq: str,
                       potts_model: dict,
                       mask: np.array) -> tuple:

    """
    Computes the applied fields h_i(a_i) and J_{ij}(a_i,a_j) for the residues a_i of the sequence  
    
          
    Parameters
    ----------
    seq : str
        The amino acid sequence of the protein. The sequence is assumed to be in one-letter code. Gaps are represented as '-'. The length of the sequence (L) should match the dimensions of the Potts model.
    potts_model : dict
        A dictionary containing the Potts model parameters 'h' (fields) and 'J' (couplings). The fields are a 2D array of shape (L, 20), where L is the length of the sequence and 20 is the number of amino acids. The couplings are a 4D array of shape (L, L, 20, 20). The fields and couplings are assumed to be in units of energy.
    mask : np.array
        A 2D Boolean array that determines which residue pairs should be considered in the energy computation. The mask should have dimensions (L, L), where L is the length of the sequence.
    
    Returns
    -------
    h: np.array
        Values of the local field for the sequence.
    j: np.array
        Values of the coupling field for the sequence. 
    """
    
    seq_index = np.array([_AA.find(aa) for aa in seq])
    seq_len = len(seq_index)

    pos1, pos2 = np.meshgrid(np.arange(seq_len), np.arange(seq_len), indexing='ij', sparse=True)
    aa1, aa2 = np.meshgrid(seq_index, seq_index, indexing='ij', sparse=True)
    if len(potts_model['J'].shape)==4:
      #  print('potts')
        h = -potts_model['h'][range(seq_len), seq_index]
        j = -potts_model['J'][pos1, pos2, aa1, aa2]
    else:
        #MJ 
       # print('MJ')
        h = 0
        j = -potts_model['J'][aa1, aa2]
    return h, j


def compute_decoy_h_J(decoy_seqs: list,
                      potts_model: dict,
                      mask: np.array,
                      config_decoys: bool = False) -> tuple:
    """
    Computes the applied fields h_i(a_i) and J_{ij}(a_i,a_j) for the residues a_i of a set of decoy sequences 
    
          
    Parameters
    ----------
    decoy_seqs : list
        List of decoy sequences. The sequences are assumed to be in one-letter code. Gaps are represented as '-'. The length of the sequences (L) should match the dimensions of the Potts model.
    potts_model : dict
        A dictionary containing the Potts model parameters 'h' (fields) and 'J' (couplings). The fields are a 2D array of shape (L, 20), where L is the length of the sequence and 20 is the number of amino acids. The couplings are a 4D array of shape (L, L, 20, 20). The fields and couplings are assumed to be in units of energy.
    mask : np.array
        A 2D Boolean array that determines which residue pairs should be considered in the energy computation. The mask should have dimensions (L, L), where L is the length of the sequence.
    config_decoys: bool
        If True, use the configurational decoys approximation, shuffling index positions for configurational decoys energy calculation. If False, mutational decoys.
    
    Returns
    -------
    h: np.array
        Values of the local field for each decoy sequence.
    j: np.array
        Values of the coupling field for each decoy sequence. 
    """
    
    seq_index = np.array([[_AA.find(aa) for aa in seq] for seq in decoy_seqs])
    N_seqs, seq_len = seq_index.shape
    pos_index=np.repeat([np.arange(seq_len)], N_seqs,axis=0)
    
    if config_decoys:
        
        pos_index=np.array([np.random.choice(pos_index[0],
                                             size=len(pos_index[0]),
                                             replace=False) for x in range(pos_index.shape[0])])
        mask=np.ones(mask.shape)*mask.mean()

       
    pos1=np.array([np.meshgrid(p, p, indexing='ij', sparse=True)[0] for p in pos_index])
    pos2=np.array([np.meshgrid(p, p, indexing='ij', sparse=True)[1] for p in pos_index])
    aa1=np.array([np.meshgrid(s, s, indexing='ij', sparse=True)[0] for s in seq_index])
    aa2=np.array([np.meshgrid(s, s, indexing='ij', sparse=True)[1] for s in seq_index])
    if len(potts_model['J'].shape)==4:
        h = -potts_model['h'][pos_index,seq_index]
        j = -potts_model['J'][pos1, pos2, aa1, aa2]
    else:
        #MJ 
        h = 0
        j = -potts_model['J'][aa1, aa2]
    return h,j


def compute_native_fragment_energy_from_h_j(fragment_pos: np.array,
                                            h: np.array,
                                            j: np.array,
                                            mask: np.array)-> float:
    """
    Computes the energy from the applied fields h_i(a_i) and J_{ij}(a_i,a_j) 
          
    Parameters
    ----------
    fragment_pos: np:array
        Array of sequence positions selected.
    h: np.array
        Values of the local field for each decoy sequence.
    j: np.array
        Values of the coupling field for each decoy sequence. 
    mask : np.array
        A 2D Boolean array that determines which residue pairs should be considered in the energy computation. The mask should have dimensions (L, L), where L is the length of the sequence.
     
    Returns
    -------
    energy: float
        Native energy of the protein 
        
    """
    h_mask=np.zeros(len(h),dtype=int)    
    h_mask[fragment_pos]=1
    j_mask=compute_fragment_mask(mask,fragment_pos)
    h_prime= h*h_mask
    j_prime = j * j_mask

    energy = h_prime.sum() + j_prime.sum() / 2
    return energy

def compute_decoy_fragment_energy_from_h_j(fragment_pos: np.array,
                                            h: np.array,
                                            j: np.array,
                                            mask: np.array)-> tuple:
    """
    Computes the energy from the applied fields h_i(a_i) and J_{ij}(a_i,a_j) for each decoy sequence.
          
    Parameters
    ----------
    fragment_pos: np:array
        Array of sequence positions selected.
    h: np.array
        Values of the local field for each decoy sequence.
    j: np.array
        Values of the coupling field for each decoy sequence. 
    mask : np.array
        A 2D Boolean array that determines which residue pairs should be considered in the energy computation. The mask should have dimensions (L, L), where L is the length of the sequence.
     
    Returns
    -------
    energy_average: float
        Average of the decoy energies
    energy_std: float
        Standard deviation of the decoy energies
        
    """
    h_mask=np.zeros(h.shape[1],dtype=int)    
    h_mask[fragment_pos]=1
    j_mask=compute_fragment_mask(mask,fragment_pos)
    h_prime= h*h_mask
    j_prime = j * j_mask  
    energy = h_prime.sum(axis=-1) + j_prime.sum(axis=-1).sum(axis=-1) / 2
    
    energy_average = energy.mean()
    energy_std = energy.std() 
    
    return energy_average, energy_std 


def compute_energy_sliding_window(seq: str,
                                  potts_model: dict,
                                  mask: np.array,
                                  win_size: int,
                                  ndecoys: int,
                                  config_decoys: bool) -> dict:
    
    """
    Computes the total frustration, the native energy, the decoy average energy and the decoy standard deviation for fragments on a sliding window
    
    Parameters
    ----------
    seq : str
        The amino acid sequence of the protein. The sequence is assumed to be in one-letter code. Gaps are represented as '-'. The length of the sequence (L) should match the dimensions of the Potts model.
    potts_model : dict
        A dictionary containing the Potts model parameters 'h' (fields) and 'J' (couplings). The fields are a 2D array of shape (L, 20), where L is the length of the sequence and 20 is the number of amino acids. The couplings are a 4D array of shape (L, L, 20, 20). The fields and couplings are assumed to be in units of energy.
    mask : np.array
        A 2D Boolean array that determines which residue pairs should be considered in the energy computation. The mask should have dimensions (L, L), where L is the length of the sequence.
    win_size: int
        Size of the sliding window
    ndecoys: int
        Number of decoy sequences to use
    config_decoys: bool
        If True, use the configurational decoys approximation, shuffling index positions for configurational decoys energy calculation. If False, mutational decoys.

    Returns
    -------
    results: dict
        Dictionary with the results, containing
        'fragment_center': center position of each window 
        'win_size': size of the sliding windows
        'native_energy': native energy for each window
        'decoy_energy_av': decoy energy average for each window
        'decoy_energy_std': decoy energy standard deviation for each window
        'frustration': total frustration index for each window
        
    """
    h, j = compute_native_h_J(seq, potts_model, mask)
    
    decoy_seqs = make_decoy_seqs(seq, ndecoys=ndecoys)    
    h_decoys,j_decoys = compute_decoy_h_J(decoy_seqs, potts_model, mask, config_decoys)

    dif = (win_size - 1) // 2
    positions = np.arange(dif, len(seq) - dif)

    e_native, e_decoy_av, e_decoy_std, frustration_sw = [], [], [], []

    for i in positions:
        fragment_pos = np.arange(i - dif, i + dif + 1)  

        native_energy = compute_native_fragment_energy_from_h_j(fragment_pos,h,j,mask)
        
        decoy_avg, decoy_std = compute_decoy_fragment_energy_from_h_j(fragment_pos,
                                                                      h_decoys,
                                                                      j_decoys,
                                                                      mask) 
        
        frustration_score = (native_energy - decoy_avg) / decoy_std if decoy_std != 0 else 0

        e_native.append(native_energy)
        e_decoy_av.append(decoy_avg)
        e_decoy_std.append(decoy_std)
        frustration_sw.append(frustration_score)
        
    results = {
        'fragment_center': positions,
        'win_size': [win_size] * len(positions),
        'native_energy': e_native,
        'decoy_energy_av': e_decoy_av,
        'decoy_energy_std': e_decoy_std,
        'frustration': frustration_sw
    }

    return results


def compute_mask_sparse(contact_i: np.ndarray,
                        contact_j: np.ndarray,
                        contact_distances: np.ndarray,
                        L: int,
                        maximum_contact_distance: Union[float, None] = None,
                        minimum_sequence_separation: Union[int, None] = None,
                        chain_breaks: Union[list, None] = None):
    """
    Compute a sparse mask from sparse distance data.

    Parameters
    ----------
    contact_i : np.ndarray (N,)
        Row indices of distance pairs (from sparse distance matrix).
    contact_j : np.ndarray (N,)
        Column indices of distance pairs.
    contact_distances : np.ndarray (N,)
        Distances for each (i, j) pair.
    L : int
        Sequence length (number of residues).
    maximum_contact_distance : float, optional
        Include pair if distance <= this value. If None, no distance filtering.
    minimum_sequence_separation : int, optional
        Include pair if |i - j| >= this value. If None, no sequence separation filtering.
    chain_breaks : list of int, optional
        Indices where new chains begin (excluding the implicit 0). Cross-chain pairs
        always satisfy the minimum sequence separation. Default is None.

    Returns
    -------
    mask_i : np.ndarray
        Row indices of pairs that pass the mask criteria.
    mask_j : np.ndarray
        Column indices of pairs that pass the mask criteria.
    """
    keep = np.ones(len(contact_i), dtype=bool)

    if maximum_contact_distance is not None:
        keep &= contact_distances <= maximum_contact_distance

    if minimum_sequence_separation is not None:
        pos_i = contact_i.astype(np.float64)
        pos_j = contact_j.astype(np.float64)
        if chain_breaks is not None:
            for brk in chain_breaks:
                pos_i = np.where(contact_i >= brk, pos_i + minimum_sequence_separation, pos_i)
                pos_j = np.where(contact_j >= brk, pos_j + minimum_sequence_separation, pos_j)
        keep &= np.abs(pos_i - pos_j) >= minimum_sequence_separation

    return contact_i[keep], contact_j[keep]


def potts_model_dense_to_sparse(potts_model: dict, mask: np.ndarray) -> dict:
    """
    Convert a dense Potts model to sparse format, keeping only couplings at masked positions.

    Both (i,j) and (j,i) entries are stored to preserve the existing ``sum / 2``
    energy summation pattern.

    Parameters
    ----------
    potts_model : dict
        Dense Potts model with 'h' (L, Q) and 'J' (L, L, Q, Q).
    mask : np.ndarray
        Boolean mask (L, L). True means the pair is a contact.

    Returns
    -------
    sparse_potts_model : dict
        Dictionary with keys:
        - 'h': np.ndarray (L, Q) — unchanged
        - 'J': np.ndarray (N_contacts, Q, Q) — couplings at contact positions
        - 'contact_i': np.ndarray (N_contacts,) — row indices
        - 'contact_j': np.ndarray (N_contacts,) — column indices
        - 'L': int — sequence length
    """
    contact_i, contact_j = np.where(mask)
    J_sparse = potts_model['J'][contact_i, contact_j, :, :]
    return {
        'h': potts_model['h'],
        'J': J_sparse,
        'contact_i': contact_i.astype(np.intp),
        'contact_j': contact_j.astype(np.intp),
        'L': potts_model['h'].shape[0],
    }


def potts_model_sparse_to_dense(sparse_potts_model: dict) -> dict:
    """
    Convert a sparse Potts model back to dense format.

    Couplings at positions not stored in the sparse model are set to zero.

    Parameters
    ----------
    sparse_potts_model : dict
        Sparse Potts model (see ``potts_model_dense_to_sparse``).

    Returns
    -------
    potts_model : dict
        Dense Potts model with 'h' (L, Q) and 'J' (L, L, Q, Q).
    """
    L = sparse_potts_model['L']
    Q = sparse_potts_model['h'].shape[1]
    J = np.zeros((L, L, Q, Q), dtype=sparse_potts_model['J'].dtype)
    J[sparse_potts_model['contact_i'], sparse_potts_model['contact_j'], :, :] = sparse_potts_model['J']
    return {
        'h': sparse_potts_model['h'],
        'J': J,
    }


def build_contact_lookup(contact_i: np.ndarray, contact_j: np.ndarray, L: int) -> tuple:
    """
    Build CSR-like flat arrays mapping each position to its contacts.

    For position ``p``, contacts are at indices
    ``lookup_data[lookup_offsets[p]:lookup_offsets[p+1]]`` in the sparse J array,
    and the partner positions are
    ``lookup_partners[lookup_offsets[p]:lookup_offsets[p+1]]``.

    Parameters
    ----------
    contact_i : np.ndarray (N_contacts,)
        Row indices of contacts.
    contact_j : np.ndarray (N_contacts,)
        Column indices of contacts.
    L : int
        Sequence length.

    Returns
    -------
    lookup_offsets : np.ndarray (L+1,)
        CSR-style offset array.
    lookup_partners : np.ndarray (N_contacts,)
        Partner position for each contact entry.
    lookup_indices : np.ndarray (N_contacts,)
        Index into sparse J array for each contact entry.
    """
    N = len(contact_i)
    # Count contacts per position (using contact_i as the "row")
    counts = np.zeros(L, dtype=np.intp)
    for k in range(N):
        counts[contact_i[k]] += 1

    lookup_offsets = np.zeros(L + 1, dtype=np.intp)
    np.cumsum(counts, out=lookup_offsets[1:])

    # Fill partner and index arrays
    lookup_partners = np.empty(N, dtype=np.intp)
    lookup_indices = np.empty(N, dtype=np.intp)
    current = lookup_offsets[:-1].copy()
    for k in range(N):
        p = contact_i[k]
        pos = current[p]
        lookup_partners[pos] = contact_j[k]
        lookup_indices[pos] = k
        current[p] += 1

    return lookup_offsets, lookup_partners, lookup_indices


def compute_native_energy_sparse(seq: str,
                                 sparse_potts_model: dict,
                                 ignore_gap_couplings: bool = False,
                                 ignore_gap_fields: bool = False) -> float:
    """
    Compute the native energy using a sparse Potts model.

    Equivalent to ``compute_native_energy`` but avoids materializing the full
    (L, L) coupling matrix.

    Parameters
    ----------
    seq : str
        Amino acid sequence (length L).
    sparse_potts_model : dict
        Sparse Potts model with keys 'h', 'J', 'contact_i', 'contact_j', 'L'.
    ignore_gap_couplings : bool
        Zero out couplings involving gap positions. Default False.
    ignore_gap_fields : bool
        Zero out fields at gap positions. Default False.

    Returns
    -------
    energy : float
    """
    seq_index = np.array([_AA.find(aa) for aa in seq])
    L = len(seq_index)

    h = -sparse_potts_model['h'][np.arange(L), seq_index].copy()
    contact_i = sparse_potts_model['contact_i']
    contact_j = sparse_potts_model['contact_j']
    J_sparse = sparse_potts_model['J']

    j_vals = -J_sparse[np.arange(len(contact_i)), seq_index[contact_i], seq_index[contact_j]]

    gap_indices = np.array([i for i, aa in enumerate(seq) if aa == '-'])

    if ignore_gap_fields and len(gap_indices) > 0:
        h[gap_indices] = 0.0

    if ignore_gap_couplings and len(gap_indices) > 0:
        gap_set = set(gap_indices)
        gap_mask = np.array([contact_i[k] in gap_set or contact_j[k] in gap_set for k in range(len(contact_i))])
        j_vals[gap_mask] = 0.0

    energy = h.sum() + j_vals.sum() / 2
    return energy


def compute_couplings_energy_sparse(seq: str,
                                    sparse_potts_model: dict,
                                    ignore_couplings_of_gaps: bool = False) -> float:
    """
    Compute the couplings energy using a sparse Potts model.

    Parameters
    ----------
    seq : str
        Amino acid sequence (length L).
    sparse_potts_model : dict
        Sparse Potts model.
    ignore_couplings_of_gaps : bool
        Zero out couplings involving gap positions. Default False.

    Returns
    -------
    couplings_energy : float
    """
    seq_index = np.array([_AA.find(aa) for aa in seq])
    contact_i = sparse_potts_model['contact_i']
    contact_j = sparse_potts_model['contact_j']
    J_sparse = sparse_potts_model['J']

    j_vals = -J_sparse[np.arange(len(contact_i)), seq_index[contact_i], seq_index[contact_j]]

    if ignore_couplings_of_gaps:
        gap_indices = set(i for i, aa in enumerate(seq) if aa == '-')
        if gap_indices:
            gap_mask = np.array([contact_i[k] in gap_indices or contact_j[k] in gap_indices for k in range(len(contact_i))])
            j_vals[gap_mask] = 0.0

    return j_vals.sum() / 2


def compute_sequences_energy_sparse(seqs: list,
                                    sparse_potts_model: dict,
                                    split_couplings_and_fields: bool = False) -> np.ndarray:
    """
    Compute energies for multiple sequences using a sparse Potts model.

    Parameters
    ----------
    seqs : list of str
        Amino acid sequences (all length L).
    sparse_potts_model : dict
        Sparse Potts model.
    split_couplings_and_fields : bool
        If True, return (2, N_seqs) array of [fields, couplings]. Default False.

    Returns
    -------
    energy : np.ndarray (N_seqs,) or (2, N_seqs)
    """
    seq_index = np.array([[_AA.find(aa) for aa in seq] for seq in seqs])
    N_seqs, L = seq_index.shape
    contact_i = sparse_potts_model['contact_i']
    contact_j = sparse_potts_model['contact_j']
    J_sparse = sparse_potts_model['J']

    h = -sparse_potts_model['h'][np.arange(L)[np.newaxis, :], seq_index]  # (N_seqs, L)
    j_vals = -J_sparse[np.arange(len(contact_i)),
                        seq_index[:, contact_i],
                        seq_index[:, contact_j]]  # (N_seqs, N_contacts)

    if split_couplings_and_fields:
        return np.array([h.sum(axis=1), j_vals.sum(axis=1) / 2])
    else:
        return h.sum(axis=1) + j_vals.sum(axis=1) / 2

    return lookup_offsets, lookup_partners, lookup_indices