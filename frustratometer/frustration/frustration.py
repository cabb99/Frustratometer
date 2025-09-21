import prody
import scipy.spatial.distance as sdist
import numpy as np
from typing import Union
from pathlib import Path

_AA = '-ACDEFGHIKLMNPQRSTVWY'

def compute_mask(distance_matrix: np.array,
                 maximum_contact_distance: Union[float, None] = None,
                 minimum_sequence_separation: Union[int, None] = None,
                 maximum_sequence_separation: Union[int, None] = None,
                 start_mask: list = None) -> np.array:
    """
    Computes a 2D Boolean mask from a given distance matrix based on a distance cutoff and a sequence separation cutoff.

    Parameters
    ----------
    distance_matrix : np.array
        A 2D array where the element at index [i, j] represents the spatial distance
        between residues i and j. This matrix is assumed to be symmetric.
    maximum_contact_distance : float, optional
        The maximum distance of a contact. Pairs of residues with distances less than this
        threshold are marked as True in the mask. If None, the spatial distance criterion
        is ignored and all distances are included. Default is None.
    minimum_sequence_separation : int, optional
        A minimum sequence distance threshold. Pairs of residues with sequence indices
        differing by at least this value are marked as True in the mask. If None,
        the sequence distance criterion is ignored. Default is None.
    maximum_sequence_separation : int, optional
        Maximum sequence distance threshold. Pairs of residues with sequence indices 
        differing by no more than this value are marked as True in the mask. If None,
        the sequence distance criterion is ignore. Default is None.
    start_mask: list, optional
        A list where the value at each index indicates whether the same index in the 
        protein structure is the first amino acid in a chain (1 for True, 0 for False).
        This is needed to accurately evaluate the minimum sequence separation (it shouldn't 
        apply for residues in different chains)

    Returns
    -------
    mask : np.array
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

    .. todo:: Add chain information for sequence separation--hope this fixed it :)
    """
    seq_len = len(distance_matrix)
    mask = np.ones([seq_len, seq_len])
    if minimum_sequence_separation is not None:
        sequence_distance = sdist.squareform(sdist.pdist(np.arange(seq_len)[:, np.newaxis]))
        mask *= sequence_distance >= minimum_sequence_separation
        if start_mask: # adjust mask to allow interactions between close in sequence residues that are part of different chains
            for i in range(0,mask.shape[0]):
                for j in range(0,mask.shape[1]):
                    if sum(start_mask[:i+1])!=sum(start_mask[:j+1]) and abs(i-j)<minimum_sequence_separation:
                        assert mask[i,j]==0
                        mask[i,j]=1
    #if maximum_sequence_separation is not None:
    #    sequence_distance = sdist.squareform(sdist.pdist(np.arange(seq_len)[:, np.newaxis]))
    #    mask *= sequence_distance <= maximum_sequence_separation
    #    # not going to bother with interchain stuff cause the concept of a maximum sequence separation is unphysical anyway
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
    """
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
        j_correction -= reduced_j[pos1, aa1] * mask[pos, pos1]
        j_correction += reduced_j[pos2, seq_index[pos2]] * mask[pos, pos2]
        j_correction -= reduced_j[pos2, aa2] * mask[pos, pos2]
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
    """
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

def compute_aa_freq(seq, include_gaps=True, segment_aa_freq=False, start_mask=None, new_AA=None):
    """
    Calculates amino acid frequencies in given sequence

    Parameters
    ----------
    seq :  str
        The amino acid sequence of the protein. The sequence is assumed to be in one-letter code. Gaps are represented as '-'.
    include_gaps: bool
        If True, frequencies of gaps ('-') in the sequence are set to 0.
        Default is True.
    segment_aa_freq: bool
        If True, compute aa_freq separately for each contiguous segment indicated in start_mask
        (the idea is that you might do this for each chain, but start_mask could be changed
        to include subsegments of chains). We could modify this function to accept
        a differently formatted mask where we use different numbers to indicate the residues that 
        belong to the same class for separate aa_freq calculations.
    start_mask: list, optional
        A list where the value at each index indicates whether the same index in the 
        protein structure is the first amino acid in a chain (1 for True, 0 for False).
        This is needed to confine each aa_freq calculation to each subsegment of the entire system
        (for example, each chain)
    new_AA: str 
        alternative order for amino acid identities used in _AA. Gap must be place in first position

    Returns
    -------
    aa_freq: np.array
        Array of frequencies of all 21 possible amino acids within sequence
    """

    if new_AA and new_AA[0] != "-":
        raise ValueError("First position in new_AA must be a gap, '-' !")

    # compute aa_freq without regard to chain or segment
    seq_index = np.array([_AA.find(aa) for aa in seq])
    aa_freq = np.array([(seq_index == i).sum() for i in range(21)])
    if not include_gaps:
        aa_freq[0] = 0 # set frequency of gap to 0
    if new_AA: # adjust to new amino acid identity order
        aa_freq = [aa_freq[_AA.index(aa)] for aa in new_AA]

    if segment_aa_freq:
        if not start_mask:
            raise ValueError("Segment-wise aa_freq calculation requested but no start_mask was given")
        aa_freq_by_chain = []
        seq_index = [] 
        subseq = []
        for bit,aa in zip(start_mask,seq):
            """
            if bit==1:
                seq_index.append(subseq)
                subseq = []
            else:
                if new_AA:
                    subseq.append(new_AA.find(aa))
                else:
                    subseq.append(_AA.find(aa)) 
            """
            if bit==1:
                seq_index.append(subseq)
                aa_freq_by_chain.append([(np.array(subseq)==aa_number).sum() for aa_number in range(21)]) 
                subseq = []
            if new_AA:
                subseq.append(new_AA.find(aa))
            else:
                subseq.append(_AA.find(aa))
        seq_index.append(subseq) # add the last one
        aa_freq_by_chain.append([(np.array(subseq)==aa_number).sum() for aa_number in range(21)])     
        seq_index = seq_index[1:] # get rid of empty list at the beginning
        aa_freq_by_chain = aa_freq_by_chain[1:] # get rid of empty list at the beginning
        ##########################################################################################################
        #seq_index = np.array(seq_index)
        #aa_freq_by_chain = np.array([[(seq_index == i).sum() for i in range(21)] for _ in range(sum(start_mask))])
        #if not include_gaps:
        #    assert aa_freq_by_chain.shape[1]==1, aa_freq_by_chain.shape
        #    for counter in range(aa_freq_by_chain.shape[0]): # axis 0 corresponds to subsegments
        ###########################################################################################################
        if not include_gaps:
            for counter,chain_aa_freq in enumerate(aa_freq_by_chain):
                #aa_freq_by_chain[counter,0] = 0 # set frequency of gap to 0
                aa_freq_by_chain[counter][0] = 0 # set frequency of gap to 0
        return (aa_freq, aa_freq_by_chain)
    else:
        return aa_freq


def compute_contact_freq(seq, segment_aa_freq=False, start_mask=None, new_AA=None):
    """
    Calculates contact frequencies in given sequence

    Parameters
    ----------
    seq :  str
        The amino acid sequence of the protein. The sequence is assumed to be in one-letter code. Gaps are represented as '-'.
    start_mask: list, optional
        A list where the value at each index indicates whether the same index in the 
        protein structure is the first amino acid in a chain (1 for True, 0 for False).
        This is needed to confine each aa_freq calculation to each chain (important for
        heterogeneous systems)
    new_AA: str
        See definition of compute_aa_freq
        
    Returns
    -------
    contact_freq: np.array
        21x21 array of frequencies of all possible contacts within sequence.
    """
    if segment_aa_freq:
        aa_freq, aa_freq_by_chain = compute_aa_freq(seq, segment_aa_freq=segment_aa_freq, start_mask=start_mask, new_AA=new_AA)
        aa_freq = np.array(aa_freq, dtype=np.float64) # processing the overall (non-segmented) aa_freq/contact_freq the same as below (else block)
        aa_freq /= aa_freq.sum() # processing the overall (non-segmented) aa_freq/contact_freq the same as below (else block)
        contact_freq = (aa_freq[:, np.newaxis] * aa_freq[np.newaxis, :]) # processing the overall (non-segmented) aa_freq/contact_freq the same as below (else block)
        aa_freq_by_chain = np.array(aa_freq_by_chain, dtype=np.float64)
        aa_freq_by_chain /= aa_freq_by_chain.sum(axis=1)[:,np.newaxis] # divide shape (NUM_CHAINS, 21) by (NUM_CHAINS, 1) (the 1 is broadcasted along each row)
        #import pdb; pdb.set_trace()                                    # so the resulting shape is (NUM_CHAINS, 21)
        contact_freq_by_chain = (aa_freq_by_chain[:, np.newaxis] * aa_freq_by_chain[np.newaxis, :])
        return (contact_freq, contact_freq_by_chain)
    else:
        aa_freq = compute_aa_freq(seq, segment_aa_freq=segment_aa_freq, start_mask=start_mask, new_AA=new_AA)
        aa_freq = np.array(aa_freq, dtype=np.float64)
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
                    pair_frustration: np.array, tcl_script: Union[Path, str] ='frustration.tcl',max_connections: int =None, movie_name: Union[Path, str] =None, still_image_name: Union[Path, str] =None,
                    min_contact_distance=3.5) -> Union[Path, str]:
    """
    Writes a tcl script that can be run with VMD to superimpose the frustration patterns onto the corresponding PDB structure. 

    Parameters
    ----------
    pdb_file :  Path or str
        pdb file name
    chain : str
        Select chain from pdb
    mask : np.array
        A 2D Boolean array that determines which residue pairs should be drawn 
        The mask should have dimensions (L, L), where L is the length of the sequence.
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
    if type(chain) == list: # chain could also be None
        chain = "".join(chain)

    fo = open(tcl_script, 'w+')
    single_frustration = np.nan_to_num(single_frustration,nan=0,posinf=0,neginf=0)
    pair_frustration = np.nan_to_num(pair_frustration,nan=0,posinf=0,neginf=0)
    
    
    structure = prody.parsePDB(str(pdb_file))
    if chain:
        selection = structure.select(f'protein and chain {chain}')#, chain=chain)
    else:
        selection = structure.select('protein')
    try:
        residues = np.unique(selection.getResindices())
    except AttributeError:
        print(chain)
        raise

    fo.write(f'[atomselect top all] set beta 0\n')
    # Single residue frustration
    for r, f in zip(residues, single_frustration):
        # print(f)
        fo.write(f'[atomselect top "chain {chain} and residue {int(r)}"] set beta {f}\n')

    # Mutational frustration:
    r1, r2 = np.meshgrid(residues, residues, indexing='ij')
    try:
        sel_frustration = np.array([r1.ravel(), r2.ravel(), pair_frustration.ravel(),distance_matrix.ravel(), mask.ravel()]).T
    except ValueError:
        print(chain)
        print(r1.shape)
        print(r2.shape)
        print(pair_frustration.shape)
        print(distance_matrix.shape)
        print(mask.shape)
        raise

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
        pos1 = selection.select(f'resindex {r1} and (name CB or (resname GLY and name CA))').getCoords()[0] # chain is unnecessary because resindex is unique
        pos2 = selection.select(f'resindex {r2} and (name CB or (resname GLY and name CA))').getCoords()[0] # chain is unnecessary because resindex is unique
        distance = np.linalg.norm(pos1 - pos2)
        if d > 9.5 or d < min_contact_distance:
            continue
        fo.write(f'lassign [[atomselect top "residue {r1} and name CA"] get {{x y z}}] pos1\n') # chain is unnecessary because resindex is unique
        fo.write(f'lassign [[atomselect top "residue {r2} and name CA"] get {{x y z}}] pos2\n') # chain is unnecessary because resindex is unique
        if min_contact_distance <= distance <= 6.5:
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
        if abs(r1-r2) == 1: # don't draw interactions between residues adjacent in sequence
            continue
        if d > 9.5 or d < min_contact_distance:
            continue
        fo.write(f'lassign [[atomselect top "residue {r1} and name CA"] get {{x y z}}] pos1\n')
        fo.write(f'lassign [[atomselect top "residue {r2} and name CA"] get {{x y z}}] pos2\n')
        if min_contact_distance <= d <= 6.5:
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

def write_tcl_script_v2(pdb_file: Union[Path,str], chain: str, solid_mask: np.array, dashed_mask: np.array,
                   single_frustration: np.array, pair_frustration: np.array, 
                   tcl_script: Union[Path, str] ='frustration.tcl', movie_name: Union[Path, str] =None, still_image_name: Union[Path, str] =None,
                   ) -> Union[Path, str]:
    """
    Writes a tcl script that can be run with VMD to superimpose the frustration patterns onto the corresponding PDB structure. 

    Maybe we can eventually integrate this method into the AWSEM class and add an attribute to the AWSEM class
    to keep track of the residue index in our original pdb file corresponding to residue index 0 in our subselection.
    Then we could use this attribute to write the correct numbers to the tcl script (which is intended to be used with the original pdb file)
    without having to read the pdb file again in this method. We would also need another way to get atomic coordinates to write 
    the correct coordinates for the ends of the line segments in vmd (currently we get them from the prody structure)

    Parameters
    ----------
    pdb_file :  Path or str
        pdb file name
    chain : str
        Select chain from pdb
    solid_mask : np.array
        A 2D Boolean array that determines which residue pairs should be drawn with a solid line
        The mask should have dimensions (L, L), where L is the length of the sequence.
    dashed_mask : np.array
        A 2D Boolean array that determines which residue pairs should be drawn with a dashed line
        The mask should have dimensions (L, L), where L is the length of the sequence.
    single_frustration : np.array
        Array containing single residue frustration index values
    pair_frustration : np.array
        Array containing pair (ex. configurational, mutational, contact) frustration index values
    tcl_script : Path or str
        Output tcl script file with static structure
    movie_name : Path or str
        Output movie file with rotating structure
    still_image_name : Path or str
        Output image file with still image
    

    Returns
    -------
    tcl_script : Path or str
        tcl script file
    """

    # check input
    solid_unique = np.unique(solid_mask.astype(int)) 
    dashed_unique = np.unique(dashed_mask.astype(int)) 
    if not np.all(solid_unique==np.array([0,1])):
        raise ValueError("found values not equal to 0 or 1 or bools in solid_mask")
    if not np.all(dashed_unique==np.array([0,1])):
        raise ValueError("found values not equal to 0 or 1 or bool in dashed_mask")
    if solid_mask.shape != dashed_mask.shape:
        raise ValueError(f"solid_mask.shape was {solid_mask.shape} and dashed_mask.shape was {dashed_mask.shape},\
                          but they should be the same!")
    if not (len(solid_mask.shape)==2 or solid_mask.shape[0]!=solid_mask.shape[1]):
        raise ValueError("solid_mask must be a 2D, square np.ndarray")
    if not (len(dashed_mask.shape)==2 or dashed_mask.shape[0]!=dashed_mask.shape[1]):
        raise ValueError("dashed_mask must be a 2D, square np.ndarray")
    if not dashed_mask.shape == solid_mask.shape:
        raise ValueError(f"dashed_mask (shape {dashed_mask.shape}) and solid_mask (shape {solid_mask.shape}) must have the same shape")
    if not np.all(solid_mask==solid_mask.T):
        solid_mask += solid_mask.T
        if np.max(solid_mask) > 1:
            raise ValueError("solid_mask was not symmetric")
        else: # we probably just filled in upper or lower triangle but not both
            pass
    if not np.all(dashed_mask==dashed_mask.T):
        dashed_mask += dashed_mask.T
        if np.max(dashed_mask) > 1:
            raise ValueError("dashed_mask was not symmetric")
        else: # we probably just filled in upper or lower triangle but not both
            pass
    if np.max(solid_mask+dashed_mask) > 1:
        raise ValueError(f"found one or more index where both solid_mask and dashed_mask were nonzero!\
            We can't draw both a solid and a dashed line between the same two residues.")

    if type(chain) == list: # chain could also be None
        chain = "".join(chain)

    fo = open(tcl_script, 'w+')
    single_frustration = np.nan_to_num(single_frustration,nan=0,posinf=0,neginf=0)
    pair_frustration = np.nan_to_num(pair_frustration,nan=0,posinf=0,neginf=0)
    
    structure = prody.parsePDB(str(pdb_file))
    if chain:
        selection = structure.select(f'protein and chain {chain}')#, chain=chain)
    else:
        selection = structure.select('protein')
    try:
        residues = np.unique(selection.getResindices())
    except AttributeError:
        print(chain)
        raise

    fo.write(f'[atomselect top all] set beta 0\n')
    # Single residue frustration
    for r, f in zip(residues, single_frustration):
        fo.write(f'[atomselect top "chain {chain} and residue {int(r)}"] set beta {f}\n')

    # pair frustration (mutational or configuration, depending on the matrix that was passed in)
    r1, r2 = np.meshgrid(residues, residues, indexing='ij')
    try:
        sel_frustration = np.array([r1.ravel(), r2.ravel(), pair_frustration.ravel(), solid_mask.ravel(), dashed_mask.ravel()]).T
        # so each row of the matrix is an instance (pair) and each column is a feature (r1, r2, etc.)
    except ValueError:
        print(chain)
        print(r1.shape)
        print(r2.shape)
        print(pair_frustration.shape)
        print(solid_mask.shape)
        print(dashed_mask.shape)
        raise
    
    # sort indices to be drawn by line style and frustration class,
    # and reorder according to the same criterion (frustration index) used in the older version of this function
    minimally_frustrated = sel_frustration[((sel_frustration[:,-2] == 1) | (sel_frustration[:,-1] == 1)) & (sel_frustration[:,2] < -0.78)]
    minimally_frustrated = minimally_frustrated[np.argsort(minimally_frustrated[:,2])]
    highly_frustrated = sel_frustration[((sel_frustration[:,-2] == 1) | (sel_frustration[:,-1] == 1)) & (sel_frustration[:,2] > 1)]
    highly_frustrated = highly_frustrated[np.argsort(highly_frustrated[:,2])]

    # draw lines
    fo.write('draw color green\n')
    draw_write_loop(minimally_frustrated, fo)
    fo.write("draw color red\n")
    draw_write_loop(highly_frustrated, fo)
    
    # boilerplate
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


def draw_write_loop(frustration_info, fo):
    """
    Just moving some code that gets called twice outside of write_tcl_script_v2
    """
    for (r1, r2, f, solid_bool , dashed_bool) in frustration_info:
        # check bool mask values
        if solid_bool and dashed_bool:
            raise AssertionError(f"solid_bool: {solid_bool}, dashed_bool: {dashed_bool}\
                                    I expected this to be caught by the input checks at the beginning of this function!")
        elif (not solid_bool) and (not dashed_bool):
            raise AssertionError("solid_bool and dashed_bool False! This should have been filtered out already")
        elif solid_bool and not dashed_bool:
            line_style = "solid"
        elif dashed_bool and not solid_bool:
            line_style = "dashed"
        else:
            raise AssertionError(f"logical issue with if-elif-else block! solid_bool: {solid_bool}, dashed_bool: {dashed_bool}")
        r1=int(r1)
        r2=int(r2)
        fo.write(f'lassign [[atomselect top "residue {r1} and name CA"] get {{x y z}}] pos1\n') # chain is unnecessary because residue is unique
        fo.write(f'lassign [[atomselect top "residue {r2} and name CA"] get {{x y z}}] pos2\n') # chain is unnecessary because residue is unique
        fo.write(f'draw line $pos1 $pos2 style {line_style} width 2\n')


def write_line_draw_commands(fo,info):
    """ 
    Small function to avoid repeating code

    fo: opened file object
    info: zip(residue indices, residue indices, frustration_indices, draw solid line bools, draw dashed line bools)

    Returns:
    None
    """
    for (r1, r2, f, solid_line, dashed_line) in info:
        r1=int(r1)
        r2=int(r2)
        if not (solid_line or dashed_line): # the masks are telling us to skip these
            continue
        fo.write(f'lassign [[atomselect top "residue {r1} and name CA"] get {{x y z}}] pos1\n') # chain is unnecessary because residue is a unique index
        fo.write(f'lassign [[atomselect top "residue {r2} and name CA"] get {{x y z}}] pos2\n') # chain is unnecessary because residue is a unique index
        if solid_line: # conventionally used for shorter distances (0-6.5 or 3.5-6.5 or 4.5-6.5)
            fo.write(f'draw line $pos1 $pos2 style solid width 2\n')
        elif dashed_line: # conventionally used for longer distances (6.5-9.5)
            fo.write(f'draw line $pos1 $pos2 style dashed width 2\n')
        else:
            raise AssertionError(f"unexpected else block. solid_line was {solid_line} and dashed_line was {dashed_line}\
                                         (one of them should be True if we've reached this point in the code)")

def mask_based_write_tcl_script(solid_mask: np.array, dashed_mask: np.array, start_residue_index: int,
                                single_frustration: np.array,pair_frustration: np.array,
                                tcl_script: Union[Path, str] ='frustration.tcl',
                                movie_name: Union[Path, str] =None, still_image_name: Union[Path, str] =None,
                                show_minimally: bool = True, show_highly: bool = True, show_neutral: bool = False,
                                ) -> Union[Path, str]:
    """
    Writes a tcl script that can be run with VMD to superimpose the frustration patterns onto the corresponding PDB structure. 

    This function removes the distance and sequence separation constraints in write_tcl_script and instead relies on the mask
    for all information about which pairs to include and which pairs to exclude. This allows us to pass in the same unified mask
    used to determine which pairs (i,j) the frustration statistics were computed for, ensuring that our frustratograms accurately
    represent our calculated proportions of minimally and highly frustrated contacts.

    Parameters
    ----------
    solid_mask : np.array
        A 2D Boolean array indicating each pair (i,j) for which a solid line is to be drawn in the frustratogram,
        conditioned on the frustration index (i,j) falling into a bin (category) that is to be shown.
        Typically, neutral contacts are not shown, so a line is typically not drawn between i and j if their frustration
        index falls in the neutral range, regardless of the values of solid_mask[i,j] and dashed_mask[i,j] 
        The mask should have dimensions (L, L), where L is the length of the sequence in the entire protein
        or the subselection (usually, a chain) that was extracted from a larger system when initializing the Structure.
    dashed_mask: np.array
        A 2D Boolean array indicating each pair (i,j) for which a dashed line is to be drawn in the frustratogram,
        subject to the same qualification described above for solid_mask.
    start_residue_index: int
        Residue index of the first residue in our subselection, mask[0], in the complete structure file that
        was used to initialize the Structure class
    single_frustration : np.array
        Array containing single residue frustration index values
    pair_frustration : np.array
        Array containing pair (ex. configurational, mutational, contact) frustration index values
    tcl_script : Path or str
        Output tcl script file with static structure
    movie_name : Path or str
        Output movie file with rotating structure
    still_image_name : Path or str
        Output image file with still image
    show_minimally: bool = True
        Draw lines for minimally frustrated contacts
    show_highly: bool = True
        Draw lines for highly frustrated contacts
    show_neutral: bool = False
        Draw lines for neutral contacts

    Returns
    -------
    tcl_script : Path or str
        tcl script file
    """

    # check input
    if np.max(solid_mask+dashed_mask) > 1:
        raise ValueError("Found at least one pair (i,j) where both solid and dashed lines are set to be drawn!")

    # open output file the old fashoined way
    fo = open(tcl_script, 'w+')\

    # clean data
    single_frustration = np.nan_to_num(single_frustration,nan=0,posinf=0,neginf=0)
    pair_frustration = np.nan_to_num(pair_frustration,nan=0,posinf=0,neginf=0)
    
    # adjust residue indices if necessary
    assert single_frustration.shape[0] == pair_frustration.shape[0] == pair_frustration.shape[1]
    residues = np.arange(start_residue_index, start_residue_index+single_frustration.shape[0])

    # initialize all residue colors at neutral (0) beta value
    fo.write(f'[atomselect top all] set beta 0\n')

    # modify beta values to indicate single residue frustration (will all be 0 if we chose not to calculate single residue frustration)
    for r, f in zip(residues, single_frustration):
        fo.write(f'[atomselect top "chain {chain} and residue {int(r)}"] set beta {f}\n')

    # pair frustration (either mutational or configurational)
    r1, r2 = np.meshgrid(residues, residues, indexing='ij')
    try:
        sel_frustration = np.array([r1.ravel(), r2.ravel(), pair_frustration.ravel(), solid_mask.ravel(), dashed_mask.ravel()]).T
    except ValueError:
        print(r1.shape)
        print(r2.shape)
        print(pair_frustration.shape)
        print(mask.shape)
        raise
    # minimally frustrated
    minimally_frustrated = sel_frustration[sel_frustration[:, 2] < -0.78]
    sort_index = np.argsort(minimally_frustrated[:, 2])
    minimally_frustrated = minimally_frustrated[sort_index]
    fo.write('draw color green\n')
    if show_minimally:
        write_line_draw_commands(fo,minimally_frustrated)
    # neutral
    neutral = sel_frustration[(sel_frustration[:, 2] > -0.78) & (sel_frustration[:, 2] < 1)]
    sort_index = np.argsort(minimally_frustrated[:, 2])
    minimally_frustrated = minimally_frustrated[sort_index]
    fo.write('draw color yellow\n')
    if show_neutral:
        write_line_draw_commands(fo,neutral)
    # highly frustrated
    frustrated = sel_frustration[sel_frustration[:, 2] > 1]
    sort_index = np.argsort(frustrated[:, 2])[::-1] # i don't know why we reverse the order of elements in this case
    frustrated = frustrated[sort_index]
    fo.write('draw color red\n')
    if show_highly:
        write_line_draw_commands(fo,frustrated)
    
    # boilerplate vmd visualization commands 
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
    
    # close tcl script file the old fashioned way
    fo.close()

    # return tcl_script as a string, but we don't usually capture or use the return value
    return tcl_script



#def write_tcl_script(pdb_file: Union[Path,str], chain: str, mask: np.array, distance_matrix: np.array, distance_cutoff: float, single_frustration: np.array,
#                    pair_frustration: np.array, tcl_script: Union[Path, str] ='frustration.tcl',max_connections: int =None, movie_name: Union[Path, str] =None, still_image_name: Union[Path, str] =None) -> Union[Path, str]:
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
    """
    fo = open(tcl_script, 'w+')
    single_frustration = np.nan_to_num(single_frustration,nan=0,posinf=0,neginf=0)
    pair_frustration = np.nan_to_num(pair_frustration,nan=0,posinf=0,neginf=0)
    
    if chain == None:
        chain_selection = '".*"'
    else:
        chain_selection = chain
    
    structure = prody.parsePDB(str(pdb_file))

    # do single residue frustration for each chain and pair frustration within each chain
    for chain in structure.iterChains():
        selection = structure.select('protein', chain=chain)
        residues = np.unique(selection.getResindices())

        fo.write(f'[atomselect top all] set beta 0\n')
        # Single residue frustration
        for r, f in zip(residues, single_frustration):
            # print(f)
            fo.write(f'[atomselect top "chain {chain} and residue {int(r)}"] set beta {f}\n')

        # Mutational frustration:
        r1, r2 = np.meshgrid(residues, residues, indexing='ij')
        try:
            sel_frustration = np.array([r1.ravel(), r2.ravel(), pair_frustration.ravel(),distance_matrix.ravel(), mask.ravel()]).T
        except ValueError:
            print(r1.shape)
            print(r2.shape)
            print(pair_frustration.shape)
            print(distance_matrix.shape)
            print(mask.shape)
            raise
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
            #minimally_frustrated = minimally_frustrated[:max_connections]
            raise NotImplementedError("need to fix this because we changed to support multichain files")

        fo.write('draw color green\n')
        for (r1, r2, f, d ,m) in minimally_frustrated:
            r1=int(r1)
            r2=int(r2)
            if abs(r1-r2) == 1: # don't draw interactions between residues adjacent in sequence
                continue
            pos1 = selection.select(f'resindex {r1} and chain {chain} and (name CB or (resname GLY and name CA))').getCoords()[0]
            pos2 = selection.select(f'resindex {r2} and chain {chain} and (name CB or (resname GLY and name CA))').getCoords()[0]
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
"""

#def write_tcl_script(pdb_file: Union[Path,str], chain: str, mask: np.array, distance_matrix: np.array, distance_cutoff: float, single_frustration: np.array,
#                    pair_frustration: np.array, tcl_script: Union[Path, str] ='frustration.tcl',max_connections: int =None, movie_name: Union[Path, str] =None, still_image_name: Union[Path, str] =None) -> Union[Path, str]:
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
    """
    fo = open(tcl_script, 'w+')
    single_frustration = np.nan_to_num(single_frustration,nan=0,posinf=0,neginf=0)
    pair_frustration = np.nan_to_num(pair_frustration,nan=0,posinf=0,neginf=0)
    
    if chain == None:
        chain_selection = '".*"'
    else:
        chain_selection = chain
    
    structure = prody.parsePDB(str(pdb_file))

    # do single residue frustration for each chain and pair frustration within each chain
    for chain in structure.iterChains():
        selection = structure.select('protein', chain=chain)
        residues = np.unique(selection.getResindices())

        fo.write(f'[atomselect top all] set beta 0\n')
        # Single residue frustration
        for r, f in zip(residues, single_frustration):
            # print(f)
            fo.write(f'[atomselect top "chain {chain} and residue {int(r)}"] set beta {f}\n')

        # Mutational frustration:
        r1, r2 = np.meshgrid(residues, residues, indexing='ij')
        try:
            sel_frustration = np.array([r1.ravel(), r2.ravel(), pair_frustration.ravel(),distance_matrix.ravel(), mask.ravel()]).T
        except ValueError:
            print(r1.shape)
            print(r2.shape)
            print(pair_frustration.shape)
            print(distance_matrix.shape)
            print(mask.shape)
            raise
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
            #minimally_frustrated = minimally_frustrated[:max_connections]
            raise NotImplementedError("need to fix this because we changed to support multichain files")

        fo.write('draw color green\n')
        for (r1, r2, f, d ,m) in minimally_frustrated:
            r1=int(r1)
            r2=int(r2)
            if abs(r1-r2) == 1: # don't draw interactions between residues adjacent in sequence
                continue
            pos1 = selection.select(f'resindex {r1} and chain {chain} and (name CB or (resname GLY and name CA))').getCoords()[0]
            pos2 = selection.select(f'resindex {r2} and chain {chain} and (name CB or (resname GLY and name CA))').getCoords()[0]
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
"""

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