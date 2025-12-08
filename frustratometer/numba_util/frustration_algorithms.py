"""
Functions for frustration calculations with numba.
Relies upon numba_hamiltonian module to evaluate the potential.

Sometimes, the Potts model for a system requires more RAM than we have available.
One solution to this challenge is to calculate energies on the fly instead of 
storing them in a massive array. To speed up evaluation of the many loops needed 
to calculate quantities on the fly, we would like to use numba. Unfortunately,
numba struggles to jit-compile most python objects, like Structure and
AWSEM. Our solution is to define functions that take attributes from our
python objects as parameters, which we can then jit without issue.

The object-oriented interfaces found elsewhere in this repository should
offer an option called something like "use_numba" or "ram_limited" that,
when set to True, results in these numba utilities being called.
"""

import numpy as np
import numba
from numba import njit, prange, int64, float64, boolean

from . import hamiltonian as ham

signature = numba.types.UniTuple(float64,2)(
    float64[:,:], 
    float64[:], float64[:],
    float64, float64[:,:],
    float64, float64[:,:],
    float64, float64[:,:],
    float64, float64[:,:],
    float64, float64[:,:],
    int64, int64[:], int64[:])
def pair_decoy_stats(
    allowed_thetaIthetaIIelectrostatic,
    allowed_rho_i, allowed_rho_j,
    lambda_direct, direct_gamma,
    lambda_protein, protein_gamma, 
    lambda_water, water_gamma, 
    lambda_burial, burial_gamma,
    lambda_electrostatic, electrostatic_gamma,
    n_decoys, seq_index_i, seq_index_j):
    """
    Generate distribution of pair energies by randomly sampling
    indicators and gammas, then compute the mean and
    and standard deviation of the distribution.

    The sampling is performed in the following way:
    -   Randomly select a row from thetaIthetaIIelectrostatic_array,
        representing the pairwise distance-based indicator functions
        for a particular pair of residues (i,j)
    -   Randomly select a rho value for residue i from allowed_rho_i
    -   Randomly select a rho value for residue j from allowed_rho_j
    -   Randomly select an amino acid type for residue i
        from seq_index_i
    -   Randomly select an amino acid type for residue j
        from seq_index_j
    -   Randomly select an amino acid type for residue j
    -   Get the appropriate gammas for the pair (i,j)
    -   Compute the pair energy given the indicators and gammas
    
    When writing this function, I'm thinking of seq_index_i and
    seq_index_j as the seq_index of the protein (list equal to
    the length of the protein where each element represents the
    amino acid type at its position). But you can get aa_freq
    behavior by replacing seq_index with a different array having
    different amino acid types in your desired proportions.

    Similarly, I'm thinking of allowed_thetaIthetaIIelectrostatic as
    including one set of {thetaI, thetaII, electrostatic_indicator}
    for each pair of residues in the protein meeting some mask
    condition (applied by the user before calling this function).
    
    Note that this function uses the deprecated np.random.choice()
    function to take a uniform random sample of our arrays. Getting
    random number generators to work with numba is tricky, so it's
    probably best to stick with this way of doing things.

    Parameters
    ----------
    - allowed_thetaIthetaIIelectrostatic : np.array(C_1, 3)
         thetaI, thetaII, and electrostatic indicator values for
         all C_1 allowed contacts. 
         Each set {thetaI_i, thetaII_i, electrostatic_indicator_i}
         should be repeated multiple times in proportion to the 
         desired probability.
    - allowed_rho_i : np.array(C_2,)
         All C_2 choices of rho allowed for residue "i".
         Each unique value should be repeated multiple times
         in proportion to the desired probability.
    - allowed_rho_j : np.array(C_3,)
         All C_3 choices of rho allowed for residue "j". 
         Each unique value should be repeated multiple times
         in proportion to the desired probability.
    - lambda_direct : float
         Scale factor for direct interaction energies.
         Should probably be 1 kcal/mol (4.184 kJ/mol),
         but has sometimes been set to 0.75 kcal/mol along with lambda_protein and lambda_water.
    - direct_gamma : np.array(20,20)
         Array formatted in the same way as self.direct_gamma from the AWSEM class.
         Order may vary (ACDE vs. ARND). 
    - lambda_protein : float
         Scale factor for protein-mediated interaction energies.
         Should probably be 1 kcal/mol (4.184 kJ/mol),
         but has sometimes been set to 0.75 kcal/mol along with lambda_direct and lambda_water.
    - protein_gamma : np.array(20,20)
         Array formatted in the same way as self.protein_gamma from the AWSEM class.
         Order may vary (ACDE vs. ARND). 
    - lambda_water : float
         Scale factor for water-mediated interaction energies.
         Should probably be 1 kcal/mol (4.184 kJ/mol),
         but has sometimes been set to 0.75 kcal/mol along with lambda_direct and lambda_protein.
    - water_gamma : np.array(20,20)
         Array formatted in the same way as self.water_gamma from the AWSEM class.
         Order may vary (ACDE vs. ARND). 
    - lambda_burial : float
         Scale factor for burial interaction energies.
         Should be 1 kcal/mol (4.184 kJ/mol).
    - burial_gamma : np.array(20,3)
         Array formatted in the same way as self.burial_gamma from the AWSEM class.
         Order along axis 0 may vary (ACDE vs. ARND), but should always be ordered as 
         [low density, medium density, high density] along axis 1.
    - lambda_electrostatic : float
         Our electrostatic "lambda" and "gamma" are different from those for our other
         terms in that they seek to represent fundamental from the bottom up,
         rather than the top-down optimization followed for the other gammas.
         Specifically, the "lambda" is the conversion factor from fundamental
         charge units to kJ/mol, adjusted for the (uniform component of the)
         solvent dielectric screening. (Heterogeneities in the solvation structures of
         ions are accounted for in the electrostatics indicator function).
    - electrostatic_gamma : np.array(20,20)
         Our electrostatic "lambda" and "gamma" are different from those for our other
         terms in that they seek to represent fundamental from the bottom up,
         rather than the top-down optimization followed for the other gammas.
         Specifically, the "gamma" is the product of the expected fundamental charges
         of the side chain -- usually +/-1, but we could do -2 for phosphorylation
    - n_decoys : int
         Number of samples draw to construct the distribution of pair energies.
         Ideally, n_decoys = infinity.
    - seq_index_i : np.array(C_4,)
         All C_4 choices of amino acid type allowed for residue "i". 
         Necessarily repeats amino acid types for C_4 > 20. 
         Each unique value should be repeated multiple times
         in proportion to the desired probability.
    - seq_index_j : np.array(C_5,)
         All C_5 choices of amino acid type allowed for residue "j". 
         Necessarily repeats amino acid types for C_5 > 20. 
         Each unique value should be repeated multiple times
         in proportion to the desired probability.

    Returns
    -------
    mean : float
        Average energy of the decoys
    stdev : float
        Standard deviation of the energies of the decoys
    """
    # randomly choose (with replacement) indices to sample, 
    # then generate arrays containing the randomly sampled values
    thetaIthetaIIelectrostatic_array = allowed_thetaIthetaIIelectrostatic\
        [np.random.choice(allowed_thetaIthetaIIelectrostatic.shape[0],size=n_decoys),:]
    rho_i_array = allowed_rho_i[np.random.choice(allowed_rho_i.shape[0],size=n_decoys)]
    rho_j_array = allowed_rho_j[np.random.choice(allowed_rho_j.shape[0],size=n_decoys)]
    aa_i_array = seq_index_i[np.random.choice(seq_index_i.shape[0],size=n_decoys)]
    aa_j_array = seq_index_j[np.random.choice(seq_index_j.shape[0],size=n_decoys)]
    # calculate pair energies and fill array
    pair_energies = np.zeros(n_decoys)
    for counter in prange(n_decoys):
        thetaI = thetaIthetaIIelectrostatic_array[counter,0]
        thetaII = thetaIthetaIIelectrostatic_array[counter,1]
        electrostatic_indicator = thetaIthetaIIelectrostatic_array[counter,2]
        rho_i = rho_i_array[counter]
        rho_j = rho_j_array[counter]
        aa_i = seq_index_i[aa_i_array[counter]]
        aa_j = seq_index_j[aa_j_array[counter]]
        gamma_bi = burial_gamma[aa_i,:]
        gamma_bj = burial_gamma[aa_j,:]
        gamma_d = direct_gamma[aa_i, aa_j]
        gamma_p = protein_gamma[aa_i, aa_j]
        gamma_w = water_gamma[aa_i, aa_j]
        gamma_e = electrostatic_gamma[aa_i, aa_j]
        pair_energy = ham.compute_pair_energy_ij_useful(
            rho_i, rho_j, thetaI, thetaII, electrostatic_indicator,
            lambda_direct, gamma_d, lambda_protein, gamma_p, lambda_water, gamma_w, 
            lambda_burial, gamma_bi, gamma_bj, lambda_electrostatic, gamma_e)
        #burial_energy_i =              ham.compute_burial_potential_i_from_rho_gamma(rho_i, lambda_burial, gamma_bi)
        #burial_energy_j =              ham.compute_burial_potential_i_from_rho_gamma(rho_j, lambda_burial, gamma_bj)
        #direct_energy =                ham.compute_direct_potential_ij_from_thetaI_gamma(thetaI, lambda_direct, gamma_d)
        #protein_energy, water_energy = ham.compute_long_potentials_ij_from_rho_thetaII_gamma(
        #    rho_i, rho_j, thetaII, lambda_protein, gamma_p, lambda_water, gamma_w)
        #electrostatic_energy =         ham.compute_electrostatic_potential_ij_from_indicator_gamma(
        #                                          lambda_electrostatic, gamma_e, electrostatic_indicator)
        #pair_energy = burial_energy_i + burial_energy_j + direct_energy +\
        #    protein_energy + water_energy + electrostatic_energy
        pair_energies[counter] = pair_energy 
    mean = np.average(pair_energies)
    stdev = np.std(pair_energies)
    return mean, stdev
pair_decoy_stats_parallel = njit(signature_or_function=signature, parallel=True)(pair_decoy_stats)
pair_decoy_stats = njit(signature_or_function=signature)(pair_decoy_stats)
#pair_decoy_stats_parallel = njit(parallel=True)(pair_decoy_stats).compile(signature)
#pair_decoy_stats = njit()(pair_decoy_stats).compile(signature)
#pair_decoy_stats_parallel = njit(pair_decoy_stats, 
#    signature_or_function=signature, parallel=True)
#pair_decoy_stats = njit(pair_decoy_stats, signature_or_function=signature)
#
@njit(signature_or_function=numba.types.UniTuple(float64,2)(
      float64, int64, int64[:], int64[:], float64[:,:],
      float64, float64,
      float64, float64[:,:],
      float64, float64[:,:],
      float64, float64[:,:],
      float64, float64[:,:],
      float64, float64[:,:],
      int64, int64[:]),
      parallel=True) # we definitely want to parallelize this function
def standard_config_decoy_stats(
    l_D, min_seq_sep_rho, chain_starts, chain_ends, dist_mat, 
    min_dist_decoy_gen, max_dist_decoy_gen, 
    lambda_direct, direct_gamma,
    lambda_protein, protein_gamma, 
    lambda_water, water_gamma, 
    lambda_burial, burial_gamma,
    lambda_electrostatic, electrostatic_gamma,
    n_decoys, seq_index):
    """
    Get mean and standard deviation of decoy energies
    following the standard configurational frustration algorithm.

    Parameters
    ----------
    l_D : float
        Screening length for Debye-Huckel electrostatics, in units of Angstroms
    min_seq_sep_rho : int
        The minimum distance in sequence for two residues to contribute to each others'
        rho. Include i,j (i.e., set mask bit bool to True) if |i-j| >= min_seq_sep_rho.
    chain_starts : np.array(N_c)
        List of 0-indexed residue indices marking the start of each chain,
        for example, array([0]) for the case of a single chain (N_c==1).
    chain_ends : np.array(N_c)
        List of 0-indexed residue indices marking the end of each chain,
        for example, array([L-1]) for the case of a single chain (N_c==1).
    dist_mat : np.array(L,L)
        Pairwise distance matrix for the entire protein system
    min_dist_decoy_gen : float
        Discard distances lower than this value from the distribution
    max_dist_decoy_gen : float
        Discard distances greater than this value from the distribution
    lambda_direct : float
        Scale factor for direct interaction energies.
        Should probably be 1 kcal/mol (4.184 kJ/mol),
        but has sometimes been set to 0.75 kcal/mol along with lambda_protein and lambda_water.
    direct_gamma : np.array(20,20)
        Array formatted in the same way as self.direct_gamma from the AWSEM class.
        Order may vary (ACDE vs. ARND). 
    lambda_protein : float
        Scale factor for protein-mediated interaction energies.
        Should probably be 1 kcal/mol (4.184 kJ/mol),
        but has sometimes been set to 0.75 kcal/mol along with lambda_direct and lambda_water.
    protein_gamma : np.array(20,20)
        Array formatted in the same way as self.protein_gamma from the AWSEM class.
        Order may vary (ACDE vs. ARND). 
    lambda_water : float
        Scale factor for water-mediated interaction energies.
        Should probably be 1 kcal/mol (4.184 kJ/mol),
        but has sometimes been set to 0.75 kcal/mol along with lambda_direct and lambda_protein.
    water_gamma : np.array(20,20)
        Array formatted in the same way as self.water_gamma from the AWSEM class.
        Order may vary (ACDE vs. ARND). 
    lambda_burial : float
        Scale factor for burial interaction energies.
        Should be 1 kcal/mol (4.184 kJ/mol).
    burial_gamma : np.array(20,3)
        Array formatted in the same way as self.burial_gamma from the AWSEM class.
        Order along axis 0 may vary (ACDE vs. ARND), but should always be ordered as 
        [low density, medium density, high density] along axis 1.    
    lambda_electrostatic : float
        Our electrostatic "lambda" and "gamma" are different from those for our other
        terms in that they seek to represent fundamental from the bottom up,
        rather than the top-down optimization followed for the other gammas.
        Specifically, the "lambda" is the conversion factor from fundamental
        charge units to kJ/mol, adjusted for the (uniform component of the)
        solvent dielectric screening. (Heterogeneities in the solvation structures of
        ions are accounted for in the electrostatics indicator function).
    electrostatic_gamma : np.array(20,20)
        Our electrostatic "lambda" and "gamma" are different from those for our other
        terms in that they seek to represent fundamental from the bottom up,
        rather than the top-down optimization followed for the other gammas.
        Specifically, the "gamma" is the product of the expected fundamental charges
        of the side chain -- usually +/-1, but we could do -2 for phosphorylation
    n_decoys : int
        Number of samples draw to construct the distribution of pair energies.
        Ideally, n_decoys = infinity.
    seq_index : np.array(L,)
        Array equal in length to the number of amino acids in the protein,
        where each element is the numerical code for the amino acid
        at that position. Numerical codes are determined by the position
        of the one-letter code of the amino acid in the string of all 
        one-letter amino acid codes, and so should range from 0 to 19. 
        The string of all one-letter amino acid codes is probably
        "ARND..." or "ACDE...", alphabetical by 3-letter code or 1-letter code.

    Returns
    -------
    mean : float
        Average energy of the decoys
    stdev : float
        Standard deviation of the energies of the decoys
    """
    # calculate rho
    C_2 = dist_mat.shape[0]
    allowed_rho_i = np.zeros(C_2)
    for counter in prange(C_2):
        allowed_rho_i[counter] = ham.compute_rho_i(counter, 
            min_seq_sep_rho, chain_starts, chain_ends, dist_mat)
    allowed_rho_j = allowed_rho_i
    # calculate distance-based indicators
    #triu_indices = np.triu_indices(C_2,k=1)
    #distances = dist_mat[triu_indices[0], triu_indices[1]]
    #distances = distances[(distances<=max_dist_decoy_gen)&(distances>=min_dist_decoy_gen)]
    distances = np.zeros(((C_2**2)-C_2)//2) # maximum possible number of distances
    num_distances = 0
    for i in range(C_2):
        for j in range(i+1, C_2):
            dist_ij = dist_mat[i,j]
            if min_dist_decoy_gen <= dist_ij <= max_dist_decoy_gen:
                distances[num_distances] = dist_ij
                num_distances += 1
    distances = distances[:num_distances+1]
    C_1 = distances.shape[0]
    allowed_thetaIthetaIIelectrostatic = np.zeros((C_1, 3))
    for counter in prange(C_1):
        dist_ij = distances[counter]
        allowed_thetaIthetaIIelectrostatic[counter,0] = ham.compute_thetaI(dist_ij)
        allowed_thetaIthetaIIelectrostatic[counter,1] = ham.compute_thetaII(dist_ij)
        allowed_thetaIthetaIIelectrostatic[counter,2] = ham.compute_electrostatic_indicator(l_D, dist_ij) 
    # assign pools of aa types to draw from
    seq_index_i = seq_index
    seq_index_j = seq_index
    # send our formatted data to numba function for rapid sampling
    mean, stdev = pair_decoy_stats(allowed_thetaIthetaIIelectrostatic,
                                   allowed_rho_i, allowed_rho_j,
                                   lambda_direct, direct_gamma,
                                   lambda_protein, protein_gamma, 
                                   lambda_water, water_gamma, 
                                   lambda_burial, burial_gamma,
                                   lambda_electrostatic, electrostatic_gamma,
                                   n_decoys, seq_index_i, seq_index_j)
    return mean, stdev
#


## no numba for this function
#def compute_frustration_matrix(dist_mat, 
#                       min_seq_sep_rho, min_seq_sep_frust_index,
#                       chain_starts, chain_ends,
#                       seq_index,
#                       lambda_direct, direct_gamma,
#                       lambda_protein, protein_gamma, 
#                       lambda_water, water_gamma, 
#                       lambda_burial, burial_gamma,
#                       lambda_electrostatic, electrostatic_gamma, l_D,
#                       decoy_stats_method):
#    """
#    Calculate matrix of frustration indices 
#
#    Parameters
#    ----------
#    decoy_stats_method : callable
#        function that returns decoy mean and standard deviation
#        (recommend numba_util.pair_decoy_stats_config)
#    others :
#        See module-level docstring
#
#    Returns
#    -------
#    frustration_matrix:
#        Matrix of the same shape as dist_mat, where each element (i,j)
#        is, if unmasked, the frustration index of the pair (i,j), or,
#        if masked, np.nan.
#    """
#    pair_energy_matrix = compute_pair_energy_matrix(
#        dist_mat, 
#        min_seq_sep_rho, min_seq_sep_frust_index,
#        chain_starts, chain_ends,
#        seq_index,
#        lambda_direct, direct_gamma,
#        lambda_protein, protein_gamma, 
#        lambda_water, water_gamma, 
#        lambda_burial, burial_gamma,
#        lambda_electrostatic, electrostatic_gamma, l_D)
#    mean, stdev = decoy_stats_method(dist_mat, 
#        min_dist_decoy_gen, max_dist_decoy_gen, 
#        min_seq_sep_rho,
#        lambda_direct, direct_gamma,
#        lambda_protein, protein_gamma, 
#        lambda_water, water_gamma, 
#        lambda_burial, burial_gamma,
#        lambda_electrostatic, electrostatic_gamma, l_D)
#    # will generate warnings about np.nan
#    frustration_matrix = (pair_energy_matrix - mean) / stdev
#    return frustration_matrix