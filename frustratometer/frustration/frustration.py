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

import molscene
import scipy.spatial.distance as sdist
import numpy as np
import warnings
from typing import Union
from pathlib import Path
import subprocess

_AA = '-ACDEFGHIKLMNPQRSTVWY'


def compute_seq_index(seq, alphabet: str = _AA, dtype=np.int64) -> np.ndarray:
    """Map a one-letter sequence to integer indices in ``alphabet`` (gap '-' at 0).

    Letters absent from the alphabet (e.g. 'X', 'B', 'Z', 'U') are mapped to the
    gap index (0) and a warning is emitted. This avoids ``str.find`` returning -1,
    which would otherwise silently alias an unknown residue to the last alphabet
    entry ('Y') under numpy/numba indexing.
    """
    idx = np.array([alphabet.find(aa) for aa in seq], dtype=dtype)
    unknown = idx < 0
    if unknown.any():
        bad = sorted({seq[i] for i in np.nonzero(unknown)[0]})
        warnings.warn(
            f"Unknown residue(s) {bad} not in alphabet {alphabet!r}; "
            f"mapping to gap ('-', index 0).",
            stacklevel=2,
        )
        idx[unknown] = 0
    return idx

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
        A minimum sequence separation threshold. Include i,j if ``abs(i-j) >= minimum_sequence_separation``.
        If None, no sequence separation is applied. Default is None.
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
        
    seq_index = compute_seq_index(seq)
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
    seq_index = compute_seq_index(seq)
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
    seq_index = compute_seq_index(seq)
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

    seq_index = np.array([compute_seq_index(s) for s in seqs])
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
    seq_index = compute_seq_index(seq)
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
    seq_index = compute_seq_index(seq)
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


def compute_pseudoconfigurational_decoy_energy_fluctuation(seq: str,
                                                     potts_model: dict,
                                                     mask: np.array, ) -> np.array:
    r"""
    Computes a (LxLx21x21) matrix for a sequence of length L. Matrix[i,j] describes all possible changes in energy upon mutating and altering the 
    local densities of residue i and j simultaneously (pseudo-configurational approximation).
    
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
    >>> decoy_energy2 = compute_pseudoconfigurational_decoy_energy_fluctuation(seq, potts_model, mask)
    >>> print(f"Matrix of Pseudo-configurational Decoy Energy Fluctuations: "); print(decoy_energy2)
    >>> print(f"Matrix Size: "); print(shape(decoy_energy2))

    Notes
    -----
    The couplings energy is computed as the half-sum of the couplings for all pairs of residues
    where the mask is True. The division by 2 for the couplings accounts for double-counting in symmetric
    matrices.

    .. todo:: Optimize the computation.
    """
    seq_index = compute_seq_index(seq)
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

    seq_index = compute_seq_index(seq)
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
        Kind of decoys generated. Options: "singleresidue," "mutational," "pseudoconfigurational," and "contact." 
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
    elif kind == 'pseudoconfigurational':
        decoy_energy=native_energy + compute_pseudoconfigurational_decoy_energy_fluctuation(seq, potts_model, mask)
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
    seq_index = compute_seq_index(seq)
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
    seq_index = compute_seq_index(seq)
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
        LxL array featuring pair frustration indices (mutational or pseudoconfigurational frustration, depending on 
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


def write_tcl_script(pdb_file: Union[Path,str], chain: str, mask: np.array, distance_matrix: np.array,
                     single_frustration: Union[np.ndarray, None] = None,
                     pair_frustration: Union[np.ndarray, None] = None,
                     minimum_distance_cutoff: float = None, maximum_distance_cutoff: float = None, 
                     minimum_sequence_separation: int = None, max_connections: int = None,
                     tcl_script: Union[Path, str] ='frustration.tcl',
                     movie_name: Union[Path, str] =None, still_image_name: Union[Path, str] =None, 
                     exit_afterwards=True, initial_rotation: tuple = (-90, -90, 0),
                     debug_directory: Union[Path, str] = None) -> Union[Path, str]:
    """
    Write a TCL script for VMD to visualize single and pair frustration.

    Parameters
    ----------
    pdb_file : Path or str
        Path to the PDB file used for visualization.
    chain : str
        Chain identifier used to select residues from the PDB.
    mask : np.array
        Boolean mask of shape (L, L) indicating eligible residue pairs.
    distance_matrix : np.array
        Pairwise distance matrix of shape (L, L).
    single_frustration : np.array, optional
        Per-residue frustration values (length L). If provided, written to the VMD beta field.
    pair_frustration : np.array, optional
        Pair frustration matrix of shape (L, L). If provided, pair connections are drawn.
    minimum_distance_cutoff : float, optional
        Lower inclusive cutoff applied to pair distances when filtering pair contacts.
    maximum_distance_cutoff : float, optional
        Upper inclusive cutoff applied to pair distances when filtering pair contacts.
    minimum_sequence_separation : int, optional
        Minimum sequence separation applied only to intra-chain pairs.
        Inter-chain pairs are not filtered by sequence separation.
    max_connections : int, optional
        Maximum number of minimally frustrated and highly frustrated connections to draw
        (applied independently to each class after sorting).
    tcl_script : Path or str
        Output TCL script path.
    movie_name : Path or str, optional
        Base filename for a rotating movie rendering (MP4).
    still_image_name : Path or str, optional
        Output path for a still image snapshot.
    exit_afterwards : bool, optional
        If True, append ``exit`` to the VMD script when rendering output.
    

    Returns
    -------
    tcl_script : Path or str
        Path to the generated TCL script.

    """
    fo = open(tcl_script, 'w+')
    from ..classes.Structure import SparseMatrix
    if isinstance(mask, SparseMatrix):
        mask = mask.to_dense(fill=0.0)
    if isinstance(distance_matrix, SparseMatrix):
        distance_matrix = distance_matrix.to_dense(fill=np.inf)
    if isinstance(pair_frustration, SparseMatrix):
        pair_frustration = pair_frustration.to_dense(fill=0.0)
    if single_frustration is not None:
        single_frustration = np.nan_to_num(single_frustration, nan=0, posinf=0, neginf=0)
    
    if pair_frustration is not None:
        pair_frustration = np.nan_to_num(pair_frustration, nan=0, posinf=0, neginf=0)
    
    
    scene = molscene.Scene.from_pdb(str(pdb_file))
    sel = 'protein' if chain is None else f'protein and chain {chain}'
    ca_selection = scene.select(sel).select(altloc=['A']).select('name CA')
    residues = ca_selection['residue'].to_numpy()
    chids = ca_selection['chain'].to_numpy()
    unique_chains = list(dict.fromkeys(chids))
    chain_to_index = {ch: i for i, ch in enumerate(unique_chains)}
    chain_indices = np.array([chain_to_index[ch] for ch in chids])


    fo.write(f'[atomselect top all] set beta 0\n')
    
    if single_frustration is not None:

        for r, f in zip(residues, single_frustration):
            fo.write(f'[atomselect top "residue {int(r)}"] set beta {f}\n')

    
    if pair_frustration is not None:
        # Mutational frustration:
        r1, r2 = np.meshgrid(residues, residues, indexing='ij')
        c1, c2 = np.meshgrid(chain_indices, chain_indices, indexing='ij')
        
        sel_frustration = np.array([r1.ravel(), r2.ravel(), c1.ravel(), c2.ravel(), pair_frustration.ravel(), distance_matrix.ravel(), mask.ravel()]).T
        #Filter with mask and distance
        mask_dist=np.ones(len(sel_frustration),dtype=bool)
        if maximum_distance_cutoff is not None:
            mask_dist= mask_dist & (sel_frustration[:, -2] <= maximum_distance_cutoff)
            
        if minimum_distance_cutoff is not None:
            mask_dist = mask_dist & (sel_frustration[:, -2] >= minimum_distance_cutoff)
            
        if minimum_sequence_separation is not None:
            #Filter with sequence separation for residues in the same chain
            mask_seqsep = np.ones(len(sel_frustration), dtype=bool)
            same_chain = sel_frustration[:, 2] == sel_frustration[:, 3]
            seq_sep = np.abs(sel_frustration[:, 0] - sel_frustration[:, 1])
            mask_seqsep[same_chain] = seq_sep[same_chain] >= minimum_sequence_separation
        else:
            mask_seqsep = np.ones(len(sel_frustration), dtype=bool) 

        sel_frustration = sel_frustration[mask_dist & (sel_frustration[:, -1] > 0) & mask_seqsep]
        # print(f"Number of contacts after filtering: {len(sel_frustration)}")
        draw_contacts = sel_frustration[(sel_frustration[:, 4] < -0.78) | (sel_frustration[:, 4] > 1)]
        # print(f"Number of contacts to draw: {len(draw_contacts)}")
        
        minimally_frustrated = sel_frustration[sel_frustration[:, 4] < -0.78]
        #minimally_frustrated = sel_frustration[sel_frustration[:, 2] < -1.78]
        sort_index = np.argsort(minimally_frustrated[:, 4])
        minimally_frustrated = minimally_frustrated[sort_index]
        if max_connections:
            minimally_frustrated = minimally_frustrated[:max_connections]
        fo.write('draw color green\n')
        
        counter = 0
        for (r1, r2, c1, c2, f, d ,m) in minimally_frustrated:
            r1=int(r1)
            r2=int(r2)
            if r1>r2: #Remove symmetric pairs
                continue
            if d > 9.5 or d < 3.5:
                continue
            # pos1_CA = selection.select(f'resindex {r1} and (name CA or (resname GLY and name CA))').getCoords()[0]
            # pos2_CA = selection.select(f'resindex {r2} and (name CA or (resname GLY and name CA))').getCoords()[0]
            # distance_CA = np.linalg.norm(pos1_CA - pos2_CA)
            # if distance_CA > 9.5 or distance_CA < 3.5:
            #     continue

            fo.write(f'lassign [[atomselect top "residue {r1} and name CA"] get {{x y z}}] pos1\n')
            fo.write(f'lassign [[atomselect top "residue {r2} and name CA"] get {{x y z}}] pos2\n')
            if 3.5 <= d <= 6.5:
                fo.write(f'draw line $pos1 $pos2 style solid width 2\n')
            else:
                fo.write(f'draw line $pos1 $pos2 style dashed width 2\n')
            counter += 1
        
        # print("Number of minimally frustrated contacts drawn: ", counter)

        frustrated = sel_frustration[sel_frustration[:, 4] > 1]
        #frustrated = sel_frustration[sel_frustration[:, 2] > 0]
        sort_index = np.argsort(frustrated[:, 4])[::-1]
        frustrated = frustrated[sort_index]
        if max_connections:
            frustrated = frustrated[:max_connections]
        fo.write('draw color red\n')
        for (r1, r2, c1, c2, f ,d, m) in frustrated:
            r1=int(r1)
            r2=int(r2)
            if r1>r2: #Remove symmetric pairs
                continue
            # pos1_CA = selection.select(f'resindex {r1} and (name CA or (resname GLY and name CA))').getCoords()[0]
            # pos2_CA = selection.select(f'resindex {r2} and (name CA or (resname GLY and name CA))').getCoords()[0]
            # distance_CA = np.linalg.norm(pos1_CA - pos2_CA)
            # if distance_CA > 9.5 or distance_CA < 3.5:
            #     continue
            if d > 9.5 or d < 3.5:
                continue
            fo.write(f'lassign [[atomselect top "residue {r1} and name CA"] get {{x y z}}] pos1\n')
            fo.write(f'lassign [[atomselect top "residue {r2} and name CA"] get {{x y z}}] pos2\n')
            if 3.5 <= d <= 6.5:
                fo.write(f'draw line $pos1 $pos2 style solid width 2\n')
            else:
                fo.write(f'draw line $pos1 $pos2 style dashed width 2\n')
            counter += 1
        # print("Number of connections drawn: ", counter)
    
    fo.write('''mol delrep top 0
            mol color Beta
            mol representation NewCartoon 0.300000 10.000000 4.100000 0
            mol selection all
            mol material Opaque
            mol addrep top
            color scale method GWR
            ''')

    fo.write(f'rotate x by {initial_rotation[0]}\n')
    fo.write(f'rotate y by {initial_rotation[1]}\n')
    fo.write(f'rotate z by {initial_rotation[2]}\n')

    if movie_name:
        fo.write('''axes location Off
            color Display Background white
            display resize 800 800
            display projection Orthographic
            display depthcue off
            display resetview
            display resize [expr [lindex [display get size] 0]/2*2] [expr [lindex [display get size] 1]/2*2] ;#Resize display to even height and width
            display update ui
            ''')

        fo.write(f'rotate x by {initial_rotation[0]}\n')
        fo.write(f'rotate y by {initial_rotation[1]}\n')
        fo.write(f'rotate z by {initial_rotation[2]}\n')

        fo.write('''
            # Set up the movie directory and base file name
            set workdir "''' +f'{debug_directory}"' + '''
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

            # # Cleanup the TGA files if desired
            # for {set i 0} {$i < $numframes} {incr i} {
            #     set output [format "%s/$basename.%05d.tga" $workdir $i]
            #     exec rm $output
            }

        ''' + ("exit\n" if exit_afterwards else '')
        
        )
    elif still_image_name:
        fo.write(f'set output "{still_image_name}"' + '''
            render snapshot $output
            ''' + ("exit\n" if exit_afterwards else ''))
    fo.close()
    return tcl_script




def call_vmd(pdb_file: Union[Path,str], tcl_script: Union[Path,str],pipe=False):
    """
    Calls VMD program with given pdb file and tcl script to visualize frustration patterns

    Parameters
    ----------
    pdb_file :  Path or str
        pdb file name
    tcl_script : Path or str
        Output tcl script file with static structure
    """
    if pipe:
        return subprocess.Popen(['vmd', '-e', tcl_script, pdb_file], stdin=subprocess.PIPE)
    else:    
        return subprocess.Popen(['vmd', '-e', tcl_script, pdb_file])


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
    fragment_pos : np.array
        Array of sequence positions selected.

    Returns
    -------
    fragment_mask : np.array
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
    Calculates the energy for the complete protein or for a fragment in context.

    Parameters
    ----------
    seq : str
        The amino acid sequence of the protein. The sequence is assumed to be in one-letter code. Gaps are represented as '-'. The length of the sequence (L) should match the dimensions of the Potts model.
    potts_model : dict
        A dictionary containing the Potts model parameters 'h' (fields) and 'J' (couplings). The fields are a 2D array of shape (L, 20), where L is the length of the sequence and 20 is the number of amino acids. The couplings are a 4D array of shape (L, L, 20, 20). The fields and couplings are assumed to be in units of energy.
    mask : np.array
        A 2D Boolean array that determines which residue pairs should be considered in the energy computation. The mask should have dimensions (L, L), where L is the length of the sequence.
    fragment_pos : np.array
        Array of sequence positions selected.
    fragment_in_context : bool
        If True, the energetics calculations take into account the interactions between the fragment and other sequence positions.

    Returns
    -------
    energy : float
        Native energy of the protein.
    """   
    
    
    seq_index = compute_seq_index(seq)
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
    Calculates decoy energies for the complete protein or for a fragment in context.

    Parameters
    ----------
    decoy_seqs : list
        List of decoy sequences. The sequences are assumed to be in one-letter code. Gaps are represented as '-'. The length of the sequences (L) should match the dimensions of the Potts model.
    potts_model : dict
        A dictionary containing the Potts model parameters 'h' (fields) and 'J' (couplings). The fields are a 2D array of shape (L, 20), where L is the length of the sequence and 20 is the number of amino acids. The couplings are a 4D array of shape (L, L, 20, 20). The fields and couplings are assumed to be in units of energy.
    mask : np.array
        A 2D Boolean array that determines which residue pairs should be considered in the energy computation. The mask should have dimensions (L, L), where L is the length of the sequence.
    fragment_pos : np.array
        Array of sequence positions selected.
    fragment_in_context : bool
        If True, the energetics calculations take into account the interactions between the fragment and other sequence positions.
    split_couplings_and_fields : bool
        Separate output into coupling and local fields contribution to energy.
    config_decoys : bool
        If True, use the pseudoconfigurational decoys approximation, shuffling index positions for pseudoconfigurational decoys energy calculation. If False, mutational decoys.
    msa_mask : np.array
        Extra mask to use a Multiple Sequence Alignment that do not cover completely the reference PDB.

    Returns
    -------
    energy : np.array
        Decoy energies.
    """   
    
    seq_index = np.array([compute_seq_index(s) for s in decoy_seqs])
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
        If True, use the pseudoconfigurational decoys approximation, shuffling index positions for pseudoconfigurational decoys energy calculation. If False, mutational decoys.
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


def compute_total_frustration_sparse(seq,
                                     sparse_potts_model,
                                     ndecoys=1000,
                                     output_kind='frustration'):
    """
    Total frustration of a full protein from a sparse Potts model (mutational decoys).

    Sparse-native counterpart of ``compute_total_frustration`` for the full-protein,
    mutational-decoy case: reuses ``compute_native_energy_sparse`` and
    ``compute_sequences_energy_sparse`` so neither the (L, L, Q, Q) coupling tensor nor
    the (N_decoys, L, L) decoy-coupling array is ever materialized. Fragment masking,
    pseudoconfigurational (shuffled) decoys, and an MSA mask are not handled here; the
    caller routes those to the dense path.

    Parameters
    ----------
    seq : str
        Amino acid sequence (length L).
    sparse_potts_model : dict
        Sparse Potts model with keys 'h', 'J', 'contact_i', 'contact_j', 'L'.
    ndecoys : int
        Number of permuted decoy sequences.
    output_kind : str
        If 'frustration', return the z-scored total frustration; otherwise return
        (native_energy, decoy_energy_average, decoy_energy_std).
    """
    native_energy = compute_native_energy_sparse(seq, sparse_potts_model)
    decoy_seqs = make_decoy_seqs(seq, ndecoys=ndecoys)
    decoy_energies = compute_sequences_energy_sparse(decoy_seqs, sparse_potts_model)
    decoy_energy_average = decoy_energies.mean()
    decoy_energy_std = decoy_energies.std()

    if output_kind == 'frustration':
        return (native_energy - decoy_energy_average) / decoy_energy_std
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
    
    seq_index = compute_seq_index(seq)
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
        If True, use the pseudoconfigurational decoys approximation, shuffling index positions for pseudoconfigurational decoys energy calculation. If False, mutational decoys.
    
    Returns
    -------
    h: np.array
        Values of the local field for each decoy sequence.
    j: np.array
        Values of the coupling field for each decoy sequence. 
    """
    
    seq_index = np.array([compute_seq_index(s) for s in decoy_seqs])
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
        If True, use the pseudoconfigurational decoys approximation, shuffling index positions for pseudoconfigurational decoys energy calculation. If False, mutational decoys.

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


def compute_energy_sliding_window_sparse(seq,
                                         sparse_potts_model,
                                         win_size,
                                         ndecoys):
    """
    Sliding-window total frustration from a sparse Potts model (mutational decoys).

    Sparse-native counterpart of ``compute_energy_sliding_window`` for the
    ``config_decoys=False`` case. Keeps per-residue field (L,) and per-contact coupling
    (N_contacts,) terms for the native and decoy sequences instead of the dense (L, L) and
    (N_decoys, L, L) coupling tensors; each window then sums the fields inside it and the
    couplings touching it. Pseudoconfigurational (shuffled-position) decoys are routed to
    the dense path by the caller. Returns the same dict shape as the dense version.
    """
    h = sparse_potts_model['h']
    contact_i = sparse_potts_model['contact_i']
    contact_j = sparse_potts_model['contact_j']
    J_sparse = sparse_potts_model['J']
    L = sparse_potts_model['L']
    nc_index = np.arange(len(contact_i))

    seq_index = compute_seq_index(seq)
    field_native = -h[np.arange(L), seq_index]                                          # (L,)
    coupling_native = -J_sparse[nc_index, seq_index[contact_i], seq_index[contact_j]]   # (Nc,)

    decoy_seqs = make_decoy_seqs(seq, ndecoys=ndecoys)
    decoy_index = np.array([compute_seq_index(s) for s in decoy_seqs])                  # (N, L)
    field_decoy = -h[np.arange(L)[np.newaxis, :], decoy_index]                          # (N, L)
    coupling_decoy = -J_sparse[nc_index,
                               decoy_index[:, contact_i],
                               decoy_index[:, contact_j]]                               # (N, Nc)

    dif = (win_size - 1) // 2
    positions = np.arange(dif, len(seq) - dif)

    e_native, e_decoy_av, e_decoy_std, frustration_sw = [], [], [], []
    for i in positions:
        lo, hi = i - dif, i + dif + 1
        contact_in_window = (((contact_i >= lo) & (contact_i < hi))
                             | ((contact_j >= lo) & (contact_j < hi)))

        native_energy = field_native[lo:hi].sum() + coupling_native[contact_in_window].sum() / 2

        decoy_energy = (field_decoy[:, lo:hi].sum(axis=1)
                        + coupling_decoy[:, contact_in_window].sum(axis=1) / 2)
        decoy_avg = decoy_energy.mean()
        decoy_std = decoy_energy.std()
        frustration_score = (native_energy - decoy_avg) / decoy_std if decoy_std != 0 else 0

        e_native.append(native_energy)
        e_decoy_av.append(decoy_avg)
        e_decoy_std.append(decoy_std)
        frustration_sw.append(frustration_score)

    return {
        'fragment_center': positions,
        'win_size': [win_size] * len(positions),
        'native_energy': e_native,
        'decoy_energy_av': e_decoy_av,
        'decoy_energy_std': e_decoy_std,
        'frustration': frustration_sw,
    }


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
        Include pair if ``abs(i - j) >= this value``. If None, no sequence separation filtering.
    chain_breaks : list of int, optional
        Indices where new chains begin (excluding the implicit 0). Cross-chain pairs
        always satisfy the minimum sequence separation. Default is None.

    Returns
    -------
    SparseMatrix
        Mask-only (data=None) SparseMatrix of pairs that pass the criteria.
    """
    from frustratometer.classes.Structure import SparseMatrix as _SM

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

    return _SM(contact_i[keep], contact_j[keep], data=None, shape=L)

def mask_mean(L: int, minimum_sequence_separation: Union[int, None] = None, chain_breaks: Union[list, None] = None):
    """
    Compute the fraction of valid residue pairs analytically for a sequence-separation-only mask.

    Instead of constructing the full (L, L) boolean mask and calling `.mean()`, this
    function counts the number of excluded pairs directly and returns the valid fraction.
    This is used as a normalization factor for pseudoconfigurational frustration calculations.

    Parameters
    ----------
    L : int
        The number of residues (sequence length). The full mask would have shape (L, L).
    minimum_sequence_separation : int, optional
        Minimum sequence separation ``abs(i - j)`` required for a pair to be included.
        Pairs with ``abs(i - j) < minimum_sequence_separation`` are excluded.
        If None, no sequence-separation filtering is applied and 1.0 is returned.
        Default is None.
    chain_breaks : list of int, optional
        Indices where new chains begin (excluding the implicit 0). For example, [50, 80]
        means three chains: residues 0–49, 50–79, 80–end. Cross-chain pairs always satisfy
        the minimum sequence separation criterion since they are not bonded in sequence.
        If None, all residues are treated as a single chain. Default is None.

    Returns
    -------
    fraction : float
        The fraction of the (L, L) pairs that pass the sequence-separation filter,
        equivalent to ``compute_mask(distance_matrix, minimum_sequence_separation=minimum_sequence_separation,
        chain_breaks=chain_breaks).mean()`` but computed without building the full matrix.

    Examples
    --------
    >>> mask_mean(10, minimum_sequence_separation=2)
    0.8  # (100 - 20) / 100

    Notes
    -----
    Without chain breaks, the number of excluded pairs is:
    L (diagonal) + 2*(L-1) + 2*(L-2) + ... + 2*(L - minimum_sequence_separation + 1).

    With chain breaks, positions are offset so that inter-chain distances are
    artificially inflated, ensuring cross-chain pairs always pass the filter.
    """
    if chain_breaks is None or len(chain_breaks) == 0:
        n_excluded = L  # diagonal (|i-j| = 0)
        for k in range(1, minimum_sequence_separation):
            n_excluded += 2 * (L - k)
        return float(L * L - n_excluded) / (L * L)
    else:
        # With chain breaks, use position-offsetting approach
        positions = np.arange(L, dtype=np.float64)
        for brk in chain_breaks:
            positions[brk:] += minimum_sequence_separation
        # Count valid pairs efficiently via sorted positions + binary search
        sorted_pos = np.sort(positions)
        n_excluded = 0
        for i in range(L):
            lo = np.searchsorted(sorted_pos, positions[i] - minimum_sequence_separation + 1e-9, side='left')
            hi = np.searchsorted(sorted_pos, positions[i] + minimum_sequence_separation - 1e-9, side='right')
            n_excluded += hi - lo
        return float(L * L - n_excluded) / (L * L)


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
    counts = np.bincount(contact_i, minlength=L).astype(np.intp)

    lookup_offsets = np.zeros(L + 1, dtype=np.intp)
    np.cumsum(counts, out=lookup_offsets[1:])

    # Sort by contact_i to group contacts by position, then fill partner/index arrays
    order = np.argsort(contact_i, kind='stable')
    lookup_partners = contact_j[order].astype(np.intp)
    lookup_indices = order.astype(np.intp)

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
    seq_index = compute_seq_index(seq)
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
        gap_mask = np.isin(contact_i, gap_indices) | np.isin(contact_j, gap_indices)
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
    seq_index = compute_seq_index(seq)
    contact_i = sparse_potts_model['contact_i']
    contact_j = sparse_potts_model['contact_j']
    J_sparse = sparse_potts_model['J']

    j_vals = -J_sparse[np.arange(len(contact_i)), seq_index[contact_i], seq_index[contact_j]]

    if ignore_couplings_of_gaps:
        gap_indices = np.array([i for i, aa in enumerate(seq) if aa == '-'])
        if len(gap_indices) > 0:
            gap_mask = np.isin(contact_i, gap_indices) | np.isin(contact_j, gap_indices)
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
    seq_index = np.array([compute_seq_index(s) for s in seqs])
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


def compute_singleresidue_decoy_energy_fluctuation_sparse(seq: str,
                                                          sparse_potts_model: dict) -> np.ndarray:
    """
    Compute single-residue decoy energy fluctuation using a sparse Potts model.

    Returns the same (L, 21) shape as the dense version.

    Parameters
    ----------
    seq : str
        Amino acid sequence (length L).
    sparse_potts_model : dict
        Sparse Potts model.

    Returns
    -------
    decoy_energy : np.ndarray (L, 21)
    """
    seq_index = compute_seq_index(seq)
    L = len(seq_index)
    h = sparse_potts_model['h']
    J_sparse = sparse_potts_model['J']
    contact_i = sparse_potts_model['contact_i']
    contact_j = sparse_potts_model['contact_j']
    N_contacts = len(contact_i)

    # Fields correction: -(h[i, aa_new] - h[i, aa_native])
    decoy_energy = np.zeros((L, 21))
    decoy_energy -= (h - h[np.arange(L), seq_index][:, np.newaxis])

    # Couplings correction via V
    # V[j, a] = sum_i J[i, j, seq[i], a] * mask[i, j]
    # Dense formula: correction_i(a) = V[i, seq[i]] - V[i, a]
    V = np.zeros((L, 21))
    values = J_sparse[np.arange(N_contacts), seq_index[contact_i], :]  # (N_contacts, 21)
    np.add.at(V, contact_j, values)

    decoy_energy += V[np.arange(L), seq_index][:, np.newaxis] - V

    return decoy_energy


def compute_mutational_decoy_energy_fluctuation_sparse(seq: str,
                                                       sparse_potts_model: dict) -> np.ndarray:
    """
    Compute mutational decoy energy fluctuation using a sparse Potts model.

    Returns (N_contacts, 21, 21) instead of (L, L, 21, 21).

    Parameters
    ----------
    seq : str
        Amino acid sequence (length L).
    sparse_potts_model : dict
        Sparse Potts model.

    Returns
    -------
    decoy_energy : np.ndarray (N_contacts, 21, 21)
        Energy changes for each contact upon mutating both residues.
    """
    seq_index = compute_seq_index(seq)
    L = len(seq_index)
    h = sparse_potts_model['h']
    J_sparse = sparse_potts_model['J']
    contact_i = sparse_potts_model['contact_i']
    contact_j = sparse_potts_model['contact_j']
    N_contacts = len(contact_i)

    aa_range = np.arange(21)
    idx = np.arange(N_contacts)
    pos1 = contact_i
    pos2 = contact_j

    # Fields correction
    decoy_energy = np.zeros((N_contacts, 21, 21))
    dh1 = h[pos1[:, np.newaxis], aa_range[np.newaxis, :]] - h[pos1, seq_index[pos1]][:, np.newaxis]  # (N, 21)
    dh2 = h[pos2[:, np.newaxis], aa_range[np.newaxis, :]] - h[pos2, seq_index[pos2]][:, np.newaxis]  # (N, 21)
    decoy_energy -= dh1[:, :, np.newaxis]
    decoy_energy -= dh2[:, np.newaxis, :]

    # Environment coupling correction via V
    V = np.zeros((L, 21))
    values = J_sparse[np.arange(N_contacts), seq_index[contact_i], :]  # (N_contacts, 21)
    np.add.at(V, contact_j, values)

    decoy_energy += (V[pos1, seq_index[pos1]][:, np.newaxis, np.newaxis]
                     - V[pos1][:, :, np.newaxis]
                     + V[pos2, seq_index[pos2]][:, np.newaxis, np.newaxis]
                     - V[pos2][:, np.newaxis, :])

    # Self-interaction corrections (pair i,j was counted in V for both pos1 and pos2)
    decoy_energy -= J_sparse[idx, seq_index[pos1], seq_index[pos2]][:, np.newaxis, np.newaxis]
    term2 = J_sparse[idx[:, np.newaxis], aa_range[np.newaxis, :], seq_index[pos2][:, np.newaxis]]  # (N, 21)
    decoy_energy += term2[:, :, np.newaxis]
    term3 = J_sparse[idx[:, np.newaxis], seq_index[pos1][:, np.newaxis], aa_range[np.newaxis, :]]  # (N, 21)
    decoy_energy += term3[:, np.newaxis, :]
    decoy_energy -= J_sparse

    return decoy_energy


def compute_pseudoconfigurational_decoy_energy_fluctuation_sparse(seq: str,
                                                            sparse_potts_model: dict,
                                                            mask_mean: float) -> np.ndarray:
    """
    Compute pseudo-configurational decoy energy fluctuation using a sparse Potts model.

    Uses ``mask_mean`` as a scalar weight for decoy interactions (shuffled densities).

    Parameters
    ----------
    seq : str
        Amino acid sequence (length L).
    sparse_potts_model : dict
        Sparse Potts model.
    mask_mean : float
        Mean of the original dense mask, used as weight for pseudoconfigurational decoys.

    Returns
    -------
    decoy_energy : np.ndarray (N_contacts, 21, 21)
    """
    seq_index = compute_seq_index(seq)
    L = len(seq_index)
    h = sparse_potts_model['h']
    J_sparse = sparse_potts_model['J']
    contact_i = sparse_potts_model['contact_i']
    contact_j = sparse_potts_model['contact_j']
    N_contacts = len(contact_i)

    aa_range = np.arange(21)
    idx = np.arange(N_contacts)
    pos1 = contact_i
    pos2 = contact_j

    # Fields correction
    decoy_energy = np.zeros((N_contacts, 21, 21))
    dh1 = h[pos1[:, np.newaxis], aa_range[np.newaxis, :]] - h[pos1, seq_index[pos1]][:, np.newaxis]
    dh2 = h[pos2[:, np.newaxis], aa_range[np.newaxis, :]] - h[pos2, seq_index[pos2]][:, np.newaxis]
    decoy_energy -= dh1[:, :, np.newaxis]
    decoy_energy -= dh2[:, np.newaxis, :]

    # Environment coupling correction
    V = np.zeros((L, 21))
    values = J_sparse[np.arange(N_contacts), seq_index[contact_i], :]  # (N_contacts, 21)
    np.add.at(V, contact_j, values)

    # Native environment uses actual mask (contacts), decoy uses mask_mean
    decoy_energy += (V[pos1, seq_index[pos1]][:, np.newaxis, np.newaxis]
                     - V[pos1][:, :, np.newaxis] * mask_mean
                     + V[pos2, seq_index[pos2]][:, np.newaxis, np.newaxis]
                     - V[pos2][:, np.newaxis, :] * mask_mean)

    # Self-interaction corrections
    decoy_energy -= J_sparse[idx, seq_index[pos1], seq_index[pos2]][:, np.newaxis, np.newaxis]
    term2 = J_sparse[idx[:, np.newaxis], aa_range[np.newaxis, :], seq_index[pos2][:, np.newaxis]] * mask_mean
    decoy_energy += term2[:, :, np.newaxis]
    term3 = J_sparse[idx[:, np.newaxis], seq_index[pos1][:, np.newaxis], aa_range[np.newaxis, :]] * mask_mean
    decoy_energy += term3[:, np.newaxis, :]
    decoy_energy -= J_sparse * mask_mean

    return decoy_energy


def compute_contact_decoy_energy_fluctuation_sparse(seq: str,
                                                    sparse_potts_model: dict) -> np.ndarray:
    """
    Compute contact decoy energy fluctuation using a sparse Potts model.

    Only the coupling at the contact itself changes.

    Parameters
    ----------
    seq : str
        Amino acid sequence (length L).
    sparse_potts_model : dict
        Sparse Potts model.

    Returns
    -------
    decoy_energy : np.ndarray (N_contacts, 21, 21)
    """
    seq_index = compute_seq_index(seq)
    J_sparse = sparse_potts_model['J']
    contact_i = sparse_potts_model['contact_i']
    contact_j = sparse_potts_model['contact_j']
    N_contacts = len(contact_i)
    idx = np.arange(N_contacts)

    # Old coupling - new coupling (mask is implicit = 1 for all stored contacts)
    decoy_energy = (J_sparse[idx, seq_index[contact_i], seq_index[contact_j]][:, np.newaxis, np.newaxis]
                    - J_sparse)
    return decoy_energy


def compute_pair_frustration_sparse(decoy_fluctuation: np.ndarray,
                                    contact_freq: Union[None, np.ndarray] = None,
                                    correction: float = 0) -> np.ndarray:
    """
    Compute pair frustration indices from sparse decoy fluctuation.

    Parameters
    ----------
    decoy_fluctuation : np.ndarray (N_contacts, 21, 21)
        Decoy energy fluctuation at each contact.
    contact_freq : np.ndarray (21, 21) or None
        Amino acid pair frequencies as weights. If None, uniform.
    correction : float
        Added to denominator to avoid division by zero. Default 0.

    Returns
    -------
    frustration : np.ndarray (N_contacts,)
        Frustration index for each contact.
    """
    if contact_freq is None:
        contact_freq = np.ones((21, 21))
    N_contacts = decoy_fluctuation.shape[0]
    flat_fluct = decoy_fluctuation.reshape(N_contacts, 21 * 21)
    flat_freq = contact_freq.flatten()
    average = np.average(flat_fluct, weights=flat_freq, axis=1)
    variance = np.average((flat_fluct - average[:, np.newaxis]) ** 2, weights=flat_freq, axis=1)
    std_energy = np.sqrt(variance)
    frustration = -(-average / (std_energy + correction))
    return frustration


def sparse_frustration_to_dense(frustration_values: np.ndarray,
                                contact_i: np.ndarray,
                                contact_j: np.ndarray,
                                L: int) -> np.ndarray:
    """
    Convert sparse frustration values to a dense (L, L) matrix.

    Parameters
    ----------
    frustration_values : np.ndarray (N_contacts,)
        Frustration index for each contact.
    contact_i : np.ndarray (N_contacts,)
        Row indices.
    contact_j : np.ndarray (N_contacts,)
        Column indices.
    L : int
        Sequence length.

    Returns
    -------
    dense : np.ndarray (L, L)
        Dense frustration matrix (zero at non-contact positions).
    """
    dense = np.zeros((L, L))
    dense[contact_i, contact_j] = frustration_values
    return dense


def _build_charges_array():
    """
    Build the (21,) charge vector in the DCA alphabet order.

    Returns
    -------
    charges : np.ndarray (21,)
        Charge of each amino acid in ``_AA`` order. D, E = -1; K, R = +1.
    """
    charges = np.zeros(21)
    for idx, aa in enumerate(_AA):
        if aa in 'DE':
            charges[idx] = -1.0
        elif aa in 'KR':
            charges[idx] = 1.0
    return charges


# Module-level constant
_CHARGES = _build_charges_array()
_Q_VAR = (_CHARGES ** 2).sum() / 21  # Var(q) under uniform freq = 4/21


def compute_elec_indicator(distance_matrix: np.ndarray,
                           k_electrostatics: float = 17.3636,
                           screening_length: float = 10.0) -> np.ndarray:
    """
    Compute the electrostatic indicator matrix (Debye-Hückel screening).

    .. math::
        \\text{indicator}(i,j) = -k_{\\text{elec}} \\frac{\\exp(-d_{ij}/\\lambda)}{d_{ij}}

    Parameters
    ----------
    distance_matrix : np.ndarray (L, L)
        Distance matrix between residues.
    k_electrostatics : float
        Electrostatic coupling coefficient (kJ/mol). Default 17.3636.
    screening_length : float
        Debye-Hückel screening length (Angstrom). Default 10.0.

    Returns
    -------
    indicator : np.ndarray (L, L)
        Electrostatic indicator. Negative for all pair distances > 0.
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        indicator = -k_electrostatics * np.exp(-distance_matrix / screening_length) / distance_matrix
        indicator[np.isnan(indicator)] = 0.0
        indicator[np.isinf(indicator)] = 0.0
    return indicator


def build_elec_data(distance_matrix: np.ndarray,
                    mask: np.ndarray,
                    sequence: str,
                    sparse_potts_model: dict,
                    k_electrostatics: float = 17.3636,
                    screening_length: float = 10.0,
                    min_sequence_separation_electrostatics: int = 1,
                    chain_breaks: list = None) -> dict:
    """
    Precompute all electrostatic data needed for sparse corrections.

    Parameters
    ----------
    distance_matrix : np.ndarray (L, L)
        Distance matrix between residues.
    mask : np.ndarray (L, L)
        Contact mask (the mask used for frustration, may differ from
        the electrostatic mask).
    sequence : str
        Amino acid sequence (length L).
    sparse_potts_model : dict
        Sparse Potts model.
    k_electrostatics : float
        Electrostatic coupling coefficient.
    screening_length : float
        Screening length (Angstrom).
    min_sequence_separation_electrostatics : int
        Minimum sequence separation for electrostatic interactions.

    Returns
    -------
    elec_data : dict
        Dictionary with keys:
        - 'charges': (21,) charge array
        - 'q_var': scalar, Var(q) under uniform frequency
        - 'q_native': (L,) native charges
        - 'indicator': (L, L) full indicator matrix
        - 'indicator_at_contacts': (N_contacts,) indicator values
        - 'phi': (L,) indicator * mask @ q_native
        - 'phi_raw': (L,) indicator @ q_native (unmasked, for pseudoconfigurational)
        - 'mask_at_contacts': (N_contacts,) mask values at contacts
        - 'mask_mean': scalar, mean of mask
    """
    seq_index = compute_seq_index(sequence)
    L = len(seq_index)
    charges = _CHARGES
    q_native = charges[seq_index]

    # Full indicator matrix
    indicator = compute_elec_indicator(distance_matrix, k_electrostatics, screening_length)

    # Apply electrostatic sequence separation
    elec_mask = compute_mask(distance_matrix,
                             maximum_contact_distance=None,
                             minimum_sequence_separation=min_sequence_separation_electrostatics,
                             chain_breaks=chain_breaks)
    indicator = indicator * elec_mask

    ci = sparse_potts_model['contact_i']
    cj = sparse_potts_model['contact_j']

    # phi vectors
    indicator_masked = indicator * mask
    phi = indicator_masked @ q_native          # (L,)
    phi_raw = indicator @ q_native             # (L,) — unmasked

    return {
        'charges': charges,
        'q_var': _Q_VAR,
        'q_native': q_native,
        'indicator': indicator,
        'indicator_at_contacts': indicator[ci, cj],
        'phi': phi,
        'phi_raw': phi_raw,
        'mask_at_contacts': mask[ci, cj].astype(float),
        'mask_mean': float(mask.mean()),
    }


def compute_native_energy_elec(sequence: str,
                               elec_data: dict,
                               mask: np.ndarray) -> float:
    """
    Compute the electrostatic contribution to the native energy.

    Parameters
    ----------
    sequence : str
        Amino acid sequence.
    elec_data : dict
        Electrostatic data from ``build_elec_data``.
    mask : np.ndarray (L, L)
        Contact mask.

    Returns
    -------
    energy : float
    """
    q_native = elec_data['q_native']
    indicator = elec_data['indicator']
    energy = -0.5 * (indicator * mask * np.outer(q_native, q_native)).sum()
    return energy


def build_elec_data_sparse(elec_dm,
                           mask_dm,
                           sequence: str,
                           sparse_potts_model: dict,
                           k_electrostatics: float = 17.3636,
                           screening_length: float = 10.0,
                           min_sequence_separation_electrostatics: int = 1,
                           chain_breaks: list = None,
                           mask_sequence_cutoff: int = None,
                           mask_chain_breaks: list = None) -> dict:
    """
    Build electrostatic data from sparse COO distance arrays.

    Parameters
    ----------
    elec_dm : SparseMatrix
        40 Å sparse distance matrix for electrostatic calculations.
    mask_dm : SparseMatrix
        Frustration mask (data=None).
    sequence : str
        Amino acid sequence.
    sparse_potts_model : dict
        Sparse Potts model with 'contact_i', 'contact_j'.
    k_electrostatics : float
        Electrostatic coupling coefficient.
    screening_length : float
        Debye-Hückel screening length.
    min_sequence_separation_electrostatics : int
        Minimum sequence separation for electrostatic interactions.
    chain_breaks : list, optional
        Chain break positions.
    mask_sequence_cutoff : int, optional
        When provided, check sequence separation directly for mask membership
        instead of using mask_i/mask_j.
    mask_chain_breaks : list, optional
        Chain breaks for the mask sequence separation check.

    Returns
    -------
    elec_data : dict
        Same keys as ``build_elec_data`` output.
    """
    elec_ci = elec_dm.row
    elec_cj = elec_dm.col
    elec_dists = elec_dm.data
    elec_L = elec_dm.shape
    mask_i = mask_dm.row
    mask_j = mask_dm.col

    seq_index = compute_seq_index(sequence)
    L = len(seq_index)
    charges = _CHARGES
    q_native = charges[seq_index]

    # Apply electrostatic sequence separation to full elec sparse data
    filt_mask = compute_mask_sparse(
        elec_ci, elec_cj, elec_dists, elec_L,
        maximum_contact_distance=None,
        minimum_sequence_separation=min_sequence_separation_electrostatics,
        chain_breaks=chain_breaks)
    filt_ci = filt_mask.row
    filt_cj = filt_mask.col

    # Build a lookup for filtered elec distances
    # We need the distances at (filt_ci, filt_cj) positions
    # Create index mapping from original elec arrays
    elec_key = elec_ci.astype(np.int64) * elec_L + elec_cj.astype(np.int64)
    filt_key = filt_ci.astype(np.int64) * elec_L + filt_cj.astype(np.int64)
    # Find matching indices
    sort_idx = np.argsort(elec_key)
    filt_pos = np.searchsorted(elec_key, filt_key, sorter=sort_idx)
    filt_dists = elec_dists[sort_idx[filt_pos]]

    # Indicator values at filtered positions
    with np.errstate(divide='ignore', invalid='ignore'):
        ind_vals = -k_electrostatics * np.exp(-filt_dists / screening_length) / filt_dists
        ind_vals[np.isnan(ind_vals)] = 0.0
        ind_vals[np.isinf(ind_vals)] = 0.0

    # Build set of mask entries for fast lookup
    ind_masked = ind_vals.copy()
    if mask_sequence_cutoff is not None:
        # Check sequence separation directly (mask not limited by sparse distance cutoff)
        def _check_seqsep(arr_i, arr_j, seq_cutoff, brks):
            pos_i = arr_i.astype(np.float64)
            pos_j = arr_j.astype(np.float64)
            if brks is not None:
                for brk in brks:
                    pos_i = np.where(arr_i >= brk, pos_i + seq_cutoff, pos_i)
                    pos_j = np.where(arr_j >= brk, pos_j + seq_cutoff, pos_j)
            return np.abs(pos_i - pos_j) >= seq_cutoff

        filt_in_mask = _check_seqsep(filt_ci, filt_cj, mask_sequence_cutoff, mask_chain_breaks)
    else:
        mask_key = mask_i.astype(np.int64) * L + mask_j.astype(np.int64)
        sort_idx_mk = np.argsort(mask_key)
        sorted_mask_key = mask_key[sort_idx_mk]
        filt_lookup_key = filt_ci.astype(np.int64) * L + filt_cj.astype(np.int64)
        if len(sorted_mask_key) == 0:
            filt_in_mask = np.zeros(len(filt_lookup_key), dtype=bool)
        else:
            filt_pos = np.searchsorted(sorted_mask_key, filt_lookup_key)
            filt_pos = np.clip(filt_pos, 0, len(sorted_mask_key) - 1)
            filt_in_mask = sorted_mask_key[filt_pos] == filt_lookup_key
    ind_masked[~filt_in_mask] = 0.0

    # phi = indicator * mask @ q_native  per row i
    phi = np.bincount(filt_ci, weights=ind_masked * q_native[filt_cj], minlength=L).astype(np.float64)
    # phi_raw = indicator @ q_native per row i (unmasked)
    phi_raw = np.bincount(filt_ci, weights=ind_vals * q_native[filt_cj], minlength=L).astype(np.float64)

    # Indicator at Potts contact positions
    potts_ci = sparse_potts_model['contact_i']
    potts_cj = sparse_potts_model['contact_j']
    potts_key = potts_ci.astype(np.int64) * L + potts_cj.astype(np.int64)

    sort_idx_filt = np.argsort(filt_key)
    sorted_filt_key = filt_key[sort_idx_filt]
    if len(sorted_filt_key) == 0:
        indicator_at_contacts = np.zeros(len(potts_key))
    else:
        potts_pos = np.searchsorted(sorted_filt_key, potts_key)
        potts_pos = np.clip(potts_pos, 0, len(sorted_filt_key) - 1)
        found = sorted_filt_key[potts_pos] == potts_key
        indicator_at_contacts = np.where(found, ind_vals[sort_idx_filt[potts_pos]], 0.0)

    if mask_sequence_cutoff is not None:
        mask_at_contacts = _check_seqsep(potts_ci, potts_cj, mask_sequence_cutoff, mask_chain_breaks).astype(float)
    else:
        # mask_key and sorted_mask_key already computed in the filt_in_mask branch above
        if len(sorted_mask_key) == 0:
            mask_at_contacts = np.zeros(len(potts_key))
        else:
            mask_pos = np.searchsorted(sorted_mask_key, potts_key)
            mask_pos = np.clip(mask_pos, 0, len(sorted_mask_key) - 1)
            mask_at_contacts = (sorted_mask_key[mask_pos] == potts_key).astype(float)

    if mask_sequence_cutoff is not None:
        _mask_mean = mask_mean(L, mask_sequence_cutoff, mask_chain_breaks)
    else:
        _mask_mean = float(len(mask_i)) / (L * L)

    return {
        'charges': charges,
        'q_var': _Q_VAR,
        'q_native': q_native,
        'indicator': None,  # No dense indicator in sparse mode
        'indicator_at_contacts': indicator_at_contacts,
        'phi': phi,
        'phi_raw': phi_raw,
        'mask_at_contacts': mask_at_contacts,
        'mask_mean': _mask_mean,
        # Sparse representation for reconstructing dense indicator_masked if needed
        'indicator_masked_i': filt_ci[filt_in_mask],
        'indicator_masked_j': filt_cj[filt_in_mask],
        'indicator_masked_vals': ind_masked[filt_in_mask],
        'L': L,
    }


def compute_native_energy_elec_sparse(sequence: str,
                                      elec_data: dict) -> float:
    """
    Compute the electrostatic contribution to native energy from sparse elec_data.

    Uses phi vectors instead of full indicator matrix.

    Parameters
    ----------
    sequence : str
        Amino acid sequence.
    elec_data : dict
        Electrostatic data from ``build_elec_data_sparse``.

    Returns
    -------
    energy : float
    """
    q_native = elec_data['q_native']
    phi = elec_data['phi']
    # E = -0.5 * sum_i q_i * phi_i where phi_i = sum_j indicator_ij * mask_ij * q_j
    energy = -0.5 * (q_native * phi).sum()
    return energy


def apply_elec_correction_singleresidue(decoy_fluctuation: np.ndarray,
                                        elec_data: dict) -> np.ndarray:
    """
    Apply electrostatic correction to single-residue decoy fluctuation.

    Parameters
    ----------
    decoy_fluctuation : np.ndarray (L, 21)
        Decoy energy fluctuation from Potts-only sparse computation.
    elec_data : dict
        Electrostatic data from ``build_elec_data``.

    Returns
    -------
    corrected : np.ndarray (L, 21)
    """
    charges = elec_data['charges']
    q_native = elec_data['q_native']
    phi = elec_data['phi']

    correction = -(charges[np.newaxis, :] - q_native[:, np.newaxis]) * phi[:, np.newaxis]
    return decoy_fluctuation + correction


def apply_elec_correction_mutational(decoy_fluctuation: np.ndarray,
                                     sparse_potts_model: dict,
                                     elec_data: dict) -> np.ndarray:
    """
    Apply electrostatic correction to mutational decoy fluctuation.

    Parameters
    ----------
    decoy_fluctuation : np.ndarray (N_contacts, 21, 21)
        Decoy energy fluctuation from Potts-only sparse computation.
    sparse_potts_model : dict
        Sparse Potts model.
    elec_data : dict
        Electrostatic data from ``build_elec_data``.

    Returns
    -------
    corrected : np.ndarray (N_contacts, 21, 21)
    """
    charges = elec_data['charges']
    q_native = elec_data['q_native']
    phi = elec_data['phi']
    ind_vals = elec_data['indicator_at_contacts']
    ci = sparse_potts_model['contact_i']
    cj = sparse_potts_model['contact_j']

    qn1 = q_native[ci]
    qn2 = q_native[cj]

    dq_a = charges[np.newaxis, :] - qn1[:, np.newaxis]   # (N, 21)
    dq_b = charges[np.newaxis, :] - qn2[:, np.newaxis]   # (N, 21)

    correction = -(dq_a * phi[ci][:, np.newaxis])[:, :, np.newaxis]
    correction = correction - (dq_b * phi[cj][:, np.newaxis])[:, np.newaxis, :]
    correction = correction - ind_vals[:, np.newaxis, np.newaxis] * dq_a[:, :, np.newaxis] * dq_b[:, np.newaxis, :]

    return decoy_fluctuation + correction


def apply_elec_correction_contact(decoy_fluctuation: np.ndarray,
                                  sparse_potts_model: dict,
                                  elec_data: dict) -> np.ndarray:
    """
    Apply electrostatic correction to contact decoy fluctuation.

    Parameters
    ----------
    decoy_fluctuation : np.ndarray (N_contacts, 21, 21)
        Decoy energy fluctuation from Potts-only sparse computation.
    sparse_potts_model : dict
        Sparse Potts model.
    elec_data : dict
        Electrostatic data from ``build_elec_data``.

    Returns
    -------
    corrected : np.ndarray (N_contacts, 21, 21)
    """
    charges = elec_data['charges']
    q_native = elec_data['q_native']
    ind_vals = elec_data['indicator_at_contacts']
    ci = sparse_potts_model['contact_i']
    cj = sparse_potts_model['contact_j']

    qn1 = q_native[ci]
    qn2 = q_native[cj]

    correction = ind_vals[:, np.newaxis, np.newaxis] * (
        (qn1 * qn2)[:, np.newaxis, np.newaxis]
        - charges[np.newaxis, :, np.newaxis] * charges[np.newaxis, np.newaxis, :]
    )

    return decoy_fluctuation + correction


def apply_elec_correction_pseudoconfigurational(decoy_fluctuation: np.ndarray,
                                          sparse_potts_model: dict,
                                          elec_data: dict) -> np.ndarray:
    """
    Apply electrostatic correction to pseudo-configurational decoy fluctuation.

    Parameters
    ----------
    decoy_fluctuation : np.ndarray (N_contacts, 21, 21)
        Decoy energy fluctuation from Potts-only sparse computation.
    sparse_potts_model : dict
        Sparse Potts model.
    elec_data : dict
        Electrostatic data from ``build_elec_data``.

    Returns
    -------
    corrected : np.ndarray (N_contacts, 21, 21)
    """
    charges = elec_data['charges']
    q_native = elec_data['q_native']
    phi = elec_data['phi']
    phi_raw = elec_data['phi_raw']
    ind_vals = elec_data['indicator_at_contacts']
    mask_vals = elec_data['mask_at_contacts']
    mask_mean = elec_data['mask_mean']
    ci = sparse_potts_model['contact_i']
    cj = sparse_potts_model['contact_j']

    qn1 = q_native[ci]
    qn2 = q_native[cj]

    c00 = qn1 * phi[ci] + qn2 * phi[cj] - ind_vals * qn1 * qn2 * mask_vals
    c10 = mask_mean * (ind_vals * qn2 - phi_raw[ci])
    c01 = mask_mean * (ind_vals * qn1 - phi_raw[cj])
    c11 = -ind_vals * mask_mean

    qa = charges[np.newaxis, :, np.newaxis]   # (1, 21, 1)
    qb = charges[np.newaxis, np.newaxis, :]   # (1, 1, 21)
    correction = (c00[:, np.newaxis, np.newaxis]
                  + c10[:, np.newaxis, np.newaxis] * qa
                  + c01[:, np.newaxis, np.newaxis] * qb
                  + c11[:, np.newaxis, np.newaxis] * qa * qb)

    return decoy_fluctuation + correction

    