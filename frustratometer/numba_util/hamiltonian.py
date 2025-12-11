"""
Hierarchy of functions for AMW/tertiary/frustratometer/potts
Hamiltonian calculations with numba.

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

Conventions
-----------

This is the complete list of parameters that may be used by any function:
i, j, l_D, min_seq_sep_rho, min_seq_sep_contact, min_seq_sep_electrostatic,
min_seq_sep_frust_index, min_seq_sep, seq_sep, chain_starts, chain_ends, 
same_chain, min_dist, max_dist, dist_mat, dist_ij, rho_i, rho_j, 
thetaI, thetaII, sigma_water, lambda_direct, direct_gamma, lambda_protein,
protein_gamma, gamma_p, lambda_water, water_gamma, gamma_w, lambda_burial, burial_gamma, 
lambda_electrostatic, electrostatic_gamma, gamma, seq_index, parallel

No function uses all these parameters, but all functions use a subset of
these parameters. The subset of parameters is always ordered the same 
as it is in the above list. The meanings of the parameters are given 
below, in order.

Parameters to select the residue(s) for a computation
- i : int
     0-indexed position of residue "i" in the complete system
- j : int
     0-indexed position of residue "j" in the complete system

Mathematical parameters of the indicator functions
- l_D: float
     Screening length for Debye-Huckel electrostatics, in units of Angstroms

Parameters for evaluating mask conditions
- min_seq_sep_rho : int
     The minimum distance in sequence for two residues to contribute to each others'
     rho. Include i,j (i.e., set mask bit bool to True) if |i-j| >= min_seq_sep_rho.
- min_seq_sep_contact : int
     The minimum distance in sequence for a contact to be considered "real" and unmasked. 
     Include i,j (i.e., set mask bit bool to True) if |i-j| >= min_seq_sep_contact.
- min_seq_sep_electrostatic : int
     The minimum distance in sequence for a charged pair to be considered "real" and unmasked. 
     Include i,j (i.e., set mask bit bool to True) if |i-j| >= min_seq_sep_electrostatic.
- min_seq_sep_frust_index : int
     The minimum distance in sequence for a pair's frustration index to be
     calculated (frustration index is set to np.nan if not satisfied)
- min_seq_sep : int 
     Sequence separation used to determine whether two residues "see" each other;
     what it means to two residues to "see" each other depends on the context
     (see min_seq_sep_contact and min_seq_sep_rho)
- seq_sep : int
     Actual distance in sequence between two residues, |i-j|
- chain_starts : np.array(N_c)
     List of 0-indexed residue indices marking the start of each chain,
     for example, array([0]) for the case of a single chain (N_c==1).
- chain_ends : np.array(N_c)
     List of 0-indexed residue indices marking the end of each chain,
     for example, array([L-1]) for the case of a single chain (N_c==1).
- same_chain : bool
     Whether the two residues i and j are part of the same chain
- min_dist : float
     Residues closer in space than this distance are masked
- max_dist : float
     Residues further in space than this distance are masked
- max_dist_contact : float
     Like the plain max_dist argument (see above)
- max_dist_electrostatic : float
     Like the plain max_dist argument (see above)

Parameters holding the values of indicator functions or
quantities needed to compute indicator functions
- dist_mat : np.array (L,L)
     Distance matrix for all residue pairs
- dist_ij : float
     Distance between two residues, in angstroms
- rho_i : float
     Rho value of residue "i"
- rho_j : float
     Rho value of residue "j"
- burial_indicator : np.array(3,)
     Low, medium, and high burial components for a burial indicator
     function for a particular residue
- thetaI : float
     Value of the short-range indicator function for a pair of residues.
     This indicator function is used to compute the direct interaction
     and as an input to the rho computation
- thetaII : float
     Value of the long-range indicator function for a pair of residues.
     This indicator function is used to compute the protein-mediated
     and water-mediated interactions
- sigma_water : float
     Used to determine whether a pair of residues is in a solvent-
     exposed or buried environment.
- electrostatic_indicator : float
     Effective interaction strength of two charged residues,
     based on their distance and debye-huckel screening length

Parameters needed to compute energies but not indicator functions
- lambda_direct : float
     Scale factor for direct interaction energies.
     Should probably be 1 kcal/mol (4.184 kJ/mol),
     but has sometimes been set to 0.75 kcal/mol along with lambda_protein and lambda_water.
- direct_gamma : np.array(20,20)
     Array formatted in the same way as self.direct_gamma from the AWSEM class.
     Order may vary (ACDE vs. ARND). 
- gamma_d : float
     Like the plain gamma argument (see below)
- lambda_protein : float
     Scale factor for protein-mediated interaction energies.
     Should probably be 1 kcal/mol (4.184 kJ/mol),
     but has sometimes been set to 0.75 kcal/mol along with lambda_direct and lambda_water.
- protein_gamma : np.array(20,20)
     Array formatted in the same way as self.protein_gamma from the AWSEM class.
     Order may vary (ACDE vs. ARND). 
- gamma_p : float
    Like the plain gamma argument (see below), but differentiates the protein gamma from
    the water gamma in the long-range potential calculation function
- lambda_water : float
     Scale factor for water-mediated interaction energies.
     Should probably be 1 kcal/mol (4.184 kJ/mol),
     but has sometimes been set to 0.75 kcal/mol along with lambda_direct and lambda_protein.
- water_gamma : np.array(20,20)
     Array formatted in the same way as self.water_gamma from the AWSEM class.
     Order may vary (ACDE vs. ARND). 
- gamma_w : float
    Like the plain gamma argument (see below), but differentiates the water gamma from
    the protein gamma in the long-range potential calculation function
- lambda_burial : float
     Scale factor for burial interaction energies.
     Should be 1 kcal/mol (4.184 kJ/mol).
- burial_gamma : np.array(20,3)
     Array formatted in the same way as self.burial_gamma from the AWSEM class.
     Order along axis 0 may vary (ACDE vs. ARND), but should always be ordered as 
     [low density, medium density, high density] along axis 1.
- gamma_bi : float
     Like the plain gamma argument (see below), but is np.array(3,) instead of float
- gamma_bj : float
     Like the plain gamma argument (see below), but is np.array(3,) instead of float
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
- gamma_e : float
    Like the plain gamma argument (see below)
- gamma : float
    Scalar gamma that has been selected from a gamma array based on the
    amino acid types of residues i and j
- seq_index : np.array(L,)
     Array equal in length to the number of amino acids in the protein,
     where each element is the numerical code for the amino acid
     at that position. Numerical codes are determined by the position
     of the one-letter code of the amino acid in the string of all 
     one-letter amino acid codes, and so should range from 0 to 19. 
     The string of all one-letter amino acid codes is probably
     "ARND..." or "ACDE...", alphabetical by 3-letter code or 1-letter code.

Parameters to optimize computation efficiency
- parallel : bool
     Whether to call numba parallelized or not

Notes
-----
What we call the "(AWSEM) tertiary Hamiltonian" 
or the "frustratometer Hamiltonian" or the "AMW Hamiltonian"
without electrostatics was defined in its modern form in

Papoian, Ulander, Eastwood, Luthey-Schulten, and Wolynes,
PNAS 2004 (https://www.pnas.org/doi/10.1073/pnas.0307851100)

This paper also gave us the gammas for the contact and burial interactions.

Electrostatics were introduced in

Tsai, Zheng, Balamurugan, Schafer, Kim, Cheung, and Wolynes,
Prot. Sci. 2016 (https://doi.org/10.1002%2Fpro.2751)
"""

import numpy as np
import numba
from numba import njit, prange, int64, float64, boolean

################################################################################
# FUNCTIONS TO CALCULATE masks, thetaI, thetaII, rho, sigma_wat, sigma_prot,
# burial_indicator, AND THE ELECTROSTATICS INDICATOR,
# GIVEN A SINGLE RESIDUE i OR A PAIR OF RESIDUES (i,j), AS APPROPRIATE
# THESE FUNCTIONS **DON'T** CHECK MASK CONDITIONS!
#
@njit(signature_or_function=boolean(int64, int64, int64[:], int64[:]))
def check_same_chain(i, j, chain_starts, chain_ends):
    """
    Checks whether two zero-indexed residue indices, i and j, belong to the same chain

    Parameters
    ----------
    See module-level docstring.

    Returns
    -------
    same_chain : bool
        Whether i and j are in the same chain
    """
    same_chain = False
    for counter in range(len(chain_starts)): # should be same length as chain_ends
        if (chain_starts[counter] <= i <= chain_ends[counter]) and (chain_starts[counter] <= j <= chain_ends[counter]):
            same_chain = True
            break # this could save us a couple iterations, probably doesn't matter
    return same_chain
#
@njit(signature_or_function=boolean(int64, int64, float64, float64, boolean, float64))
def mask_of_pair(min_seq_sep, seq_sep, min_dist, max_dist, same_chain, dist_ij):
    """
    Get a bool representing whether a pair of residues having
    sequence separation seq_sep and distance dist_ij should be 
    considered (True) or ignored (False), given the supplied parameters.

    Parameters
    ----------
    See module-level docstring.
    
    Returns
    -------
    mask_bit : bool
        Whether pair should be considered unmasked (True) or masked (False).
    """
    if (min_dist<=dist_ij) and (dist_ij<=max_dist) and ((min_seq_sep<=seq_sep) or (not same_chain)):
        mask_bit = True
    else:
        mask_bit = False
    return mask_bit
#
@njit(signature_or_function=float64(float64, float64, float64))
def _compute_theta(dist_ij, r_min, r_max):
    # This function may be called to evaluate either thetaI or thetaII.
    # Since thetaI is used to compute both contact indicators and rho,
    # we have to worry about min_seq_sep_contact vs min_seq_sep_rho.
    # So we do not check the mask here, but instead check it before thetaI 
    # or thetaII is called.
    # 5 (Angstrom^-1) is "eta"
    theta = 0.25 * (1 + np.tanh(5*(dist_ij-r_min))) * (1 + np.tanh(5*(r_max-dist_ij)))
    return theta
@njit(signature_or_function=float64(float64))
def compute_thetaI(dist_ij):
    """
    Computes thetaI, the short-range indicator function
    that tells us whether two residues are close but not overlapping.
    This function does not check whether the ij interaction should
    be blocked by a mask; this should be done in the calling scope.

    Parameters
    ----------
    See module-level docstring.

    Returns
    -------
    thetaI : float
        The short-range indicator function.
    """
    return _compute_theta(dist_ij, 4.5, 6.5)
@njit(signature_or_function=float64(float64))
def compute_thetaII(dist_ij):
    """
    Computes thetaII, the long-range switching function
    that tells us whether two residues are close but not in direct contact.
    This function does not check whether the ij interaction should
    be blocked by a mask; this should be done in the calling scope.

    Parameters
    ----------
    See module-level docstring.

    Returns
    -------
    thetaII : float
        The long-range indicator function
    """
    return _compute_theta(dist_ij, 6.5, 9.5)
#
signature = float64(int64, int64, int64[:], int64[:], float64[:,:])
def compute_rho_i(i, min_seq_sep_rho, chain_starts, chain_ends, dist_mat):
    """
    Compute the "local density," rho, of a given 0-indexed
    residue index, i. The quantity rho_i may be loosely thought of
    as the number of neighbors (coordination number) of residue i.

    Parameters
    ----------
    See module-level docstring.   

    Returns 
    -------
    rho_i : float
        The local density of residue i
    """
    rho_i = 0.0
    for j in prange(dist_mat.shape[1]):
        # check mask
        same_chain = check_same_chain(i, j, chain_starts, chain_ends)
        # 2.5 and 8.5 cutoffs: effectively, 
        #    we're truncating the potential where the indicators are almost zero
        #    (thetaI(2.5)==thetaI(8.5)==2.0611536367E-9)
        if mask_of_pair(min_seq_sep_rho, abs(i-j), 2.5, 8.5, 
                        same_chain, dist_mat[i,j]):
            # only let the residue contribute if it isn't caught by the mask
            rho_i += compute_thetaI(dist_mat[i,j])   
    return rho_i
compute_rho_i_parallel = njit(signature_or_function=signature, parallel=True)(compute_rho_i)
compute_rho_i = njit(signature_or_function=signature)(compute_rho_i)
#
@njit(signature_or_function=float64[:](float64))
def compute_burial_indicator_i(rho_i):
    """
    Compute the vector-valued burial indicator function (one element 
    each for low, medium, and high density).

    Parameters
    ----------
    See module-level docstring

    Returns
    -------
    burial_indicator : np.array(3,)
        The burial indicator for residue i in each well
        Remember that the burial indicator is defined as ranging from 0 to 2
    """

    burial_indicator = np.zeros(3)
    # 4.0 is "burial_kappa"
    burial_indicator[0] = (np.tanh(4.0*(rho_i-0.0)) + np.tanh(4.0*(3.0-rho_i)))
    burial_indicator[1] = (np.tanh(4.0*(rho_i-3.0)) + np.tanh(4.0*(6.0-rho_i)))
    burial_indicator[2] = (np.tanh(4.0*(rho_i-6.0)) + np.tanh(4.0*(9.0-rho_i)))
    return burial_indicator
#
@njit(signature_or_function=float64(float64,float64))
def compute_sigma_water(rho_i, rho_j):
    """
    Compute sigma_water based on local densities of the two residues in the pair.

    If both residues are exposed ((rho_i < rho_0) && (rho_j < rho_0)),
    then water-mediated interactions dominate (sigma_water ~ 1).
    If either is buried ((rho_i > rho_0) || (rho_j > rho_0)),
    then water-mediated interactions are small (sigma_water ~ 0).

    Parameters
    ----------
    See module-level docstring.

    Returns
    -------
    sigma_water : float
        Fraction of water-mediated interactions (0 to 1)
    """
    #sigma_water = 0.25 * (1 - np.tanh(eta_sigma * (rho_i - rho_0))) * (1 - np.tanh(eta_sigma * (rho_j - rho_0)))
    sigma_water = 0.25*(1-np.tanh(7*(rho_i-2.6)))*(1-np.tanh(7*(rho_j-2.6)))
    return sigma_water
#
@njit(signature_or_function=float64(float64, float64))
def compute_electrostatic_indicator(l_D, dist_ij):
    """
    Computes electrostatics indicator function, which gives an
    effective proximity (higher <==> closer) of two residues,
    capturing not only the 1/r decay of the coulomb energy,
    but also the screening effects of counterions; l_D
    should be negatively correlated with the ionic strength.

    Parameters
    ----------
    See module-level docstring

    Returns
    -------
    electrostatics_indicator(i,j,dist_mat[i,j]), parameterized by l_D
    """
    if dist_ij >= 1:
        safe_dist = dist_ij
    else:
        raise ValueError("Distance between two residue was less than 1 angstrom!")
    electrostatics_indicator = np.exp(-safe_dist / l_D) / safe_dist
    return electrostatics_indicator
#
###########################################################################
# FUNCTIONS TO CALCULATE ENERGIES, GIVEN A SINGLE RESIDUE i OR A PAIR (i,j)
# THESE FUNCTIONS **DON'T** CHECK MASK CONDITIONS!
#
# BURIAL POTENTIAL
@njit(signature_or_function=float64(float64[:], float64, float64[:]))
def compute_burial_potential_i_from_indicator_gamma(burial_indicator, lambda_burial, gamma):
    """
    Compute the burial energy for residue i based on its local density.
    Note that this function computes and sums the 3 types of burial energies:
    low-density, medium-density, and high-density.

    Parameters
    ----------
    See module-level docstring.

    Returns
    -------
    burial_energy : float
        Total burial energy for residue i, sum across all three burial wells.
    """
    # Caution: the burial indicator functions range from 0 to 2,
    # not 0 to 1, like the other indicator functions.
    # This is why we have a coefficient of 0.5 in the energy expression.
    low_indicator = burial_indicator[0]
    low_gamma = gamma[0]
    medium_indicator = burial_indicator[1]
    medium_gamma = gamma[1]
    high_indicator = burial_indicator[2]
    high_gamma = gamma[2]
    burial_energy = -0.5*lambda_burial *\
        (low_indicator*low_gamma+medium_indicator*medium_gamma+high_indicator*high_gamma)
    return burial_energy
@njit(signature_or_function=float64(float64, float64, float64[:]))
def compute_burial_potential_i_from_rho_gamma(rho_i, lambda_burial, gamma):
    """
    Compute the burial energy for residue i based on its local density.
    Note that this function computes and sums the 3 types of burial energies:
    low-density, medium-density, and high-density.

    Parameters
    ----------
    See module-level docstring.

    Returns
    -------
    burial_energy : float
        Total burial energy for residue i, sum across all three burial wells.
    """
    burial_indicator = compute_burial_indicator_i(rho_i)
    burial_energy = compute_burial_potential_i_from_indicator_gamma(burial_indicator, lambda_burial, gamma)
    return burial_energy
@njit(signature_or_function=float64(int64, float64, float64, float64[:,:], int64[:]))
def compute_burial_potential_i_from_rho(i, rho_i, lambda_burial, burial_gamma, seq_index):
    """
    Compute the burial energy for residue i based on its local density.
    Note that this function computes and sums the 3 types of burial energies:
    low-density, medium-density, and high-density.

    Parameters
    ----------
    See module-level docstring.

    Returns
    -------
    burial_energy : float
        Total burial energy for residue i, sum across all three burial wells.
    """
    gamma = burial_gamma[seq_index[i]]
    burial_energy = compute_burial_potential_i_from_rho_gamma(rho_i, lambda_burial, gamma)
    return burial_energy
@njit(signature_or_function=float64(int64, int64, int64[:], int64[:], float64[:,:], float64, float64[:]))
def compute_burial_potential_i_from_gamma(i, min_seq_sep_rho, chain_starts, chain_ends, dist_mat, lambda_burial, gamma):
    """
    Compute the burial energy for residue i based on its local density.
    Note that this function computes and sums the 3 types of burial energies:
    low-density, medium-density, and high-density.

    Parameters
    ----------
    See module-level docstring.

    Returns
    -------
    burial_energy : float
        Total burial energy for residue i, sum across all three burial wells.
    """    
    rho_i = compute_rho_i(i, min_seq_sep_rho, chain_starts, chain_ends, dist_mat)
    burial_energy = compute_burial_potential_i_from_rho_gamma(rho_i, lambda_burial, gamma)
    return burial_energy
@njit(signature_or_function=float64(int64, int64, int64[:], int64[:], float64[:,:], float64, float64[:,:], int64[:]))
def compute_burial_potential_i(i, min_seq_sep_rho, chain_starts, chain_ends, dist_mat, lambda_burial, burial_gamma, seq_index):
    """
    Compute the burial energy for residue i based on its local density.
    Note that this function computes and sums the 3 types of burial energies:
    low-density, medium-density, and high-density.

    Parameters
    ----------
    See module-level docstring.

    Returns
    -------
    burial_energy : float
        Total burial energy for residue i, sum across all three burial wells.
    """
    gamma = burial_gamma[seq_index[i], :]
    burial_energy = compute_burial_potential_i_from_gamma(i, min_seq_sep_rho, chain_starts, chain_ends, dist_mat, lambda_burial, gamma)
    return burial_energy
# feel free to add more functions with different signatures for greater flexibility of use
#
# DIRECT POTENTIAL
@njit(signature_or_function=float64(float64, float64, float64))
def compute_direct_potential_ij_from_thetaI_gamma(thetaI, lambda_direct, gamma):
    """
    Compute the direct interaction potential for a pair of residues.

    Parameters
    ----------
    See module-level docstring. 

    Returns
    -------
    direct_energy : float
        Energy of the direct contact term for the pair (i,j),
        set to 0 if the pair is masked.
    """    
    return -lambda_direct * thetaI * gamma
@njit(signature_or_function=float64(float64, float64, float64))
def compute_direct_potential_ij_from_distij_gamma(dist_ij, lambda_direct, gamma):
    """
    Compute the direct interaction potential for a pair of residues.

    Parameters
    ----------
    See module-level docstring. 

    Returns
    -------
    direct_energy : float
        Energy of the direct contact term for the pair (i,j),
        set to 0 if the pair is masked.
    """
    # get indicator
    thetaI = compute_thetaI(dist_ij)
    # put it all together
    direct_energy = compute_direct_potential_ij_from_thetaI_gamma(thetaI, lambda_direct, gamma)
    return direct_energy
@njit(signature_or_function=float64(int64, int64, float64[:,:], float64, float64))
def compute_direct_potential_ij_from_gamma(i, j, dist_mat, lambda_direct, gamma):
    """
    Compute the direct interaction potential for a pair of residues.

    Parameters
    ----------
    See module-level docstring. 

    Returns
    -------
    direct_energy : float
        Energy of the direct contact term for the pair (i,j),
        set to 0 if the pair is masked.
    """    
    dist_ij = dist_mat[i,j]
    return compute_direct_potential_ij_from_distij_gamma(dist_ij, lambda_direct, gamma)
@njit(signature_or_function=float64(int64, int64, float64[:,:], float64, float64[:,:], int64[:]))
def compute_direct_potential_ij(i, j, dist_mat, lambda_direct, direct_gamma, seq_index):
    """
    Compute the direct interaction potential for a pair of residues.

    Parameters
    ----------
    See module-level docstring. 

    Returns
    -------
    direct_energy : float
        Energy of the direct contact term for the pair (i,j),
        set to 0 if the pair is masked.
    """    
    gamma = direct_gamma[seq_index[i], seq_index[j]]
    return compute_direct_potential_ij_from_gamma(i, j, dist_mat, lambda_direct, gamma)
#
# LONG RANGE (protein-mediated and water-mediated) CONTACT POTENTIALS
@njit(signature_or_function=numba.types.UniTuple(float64,2)(
    float64, float64, float64, float64, float64, float64))
def compute_long_potentials_ij_from_sigmawater_thetaII_gamma(thetaII, sigma_water, 
    lambda_protein, gamma_p, lambda_water, gamma_w):    
    """
    Compute the protein-mediated and water-mediated (long-range) potentials 
    for a pair of residues. 

    Parameters
    ----------
    See module-level docstring.

    Returns
    -------
    protein_energy : float
        Energy of the protein-mediated contact term for the pair (i,j),
        set to 0 if the pair is masked.
    water_energy : float
        Energy of the water-mediated contact term for the pair (i,j),
        set to 0 if the pair is masked.
    """    
    # this function is defined so that we have the details of the 
    # calculation in one place and don't have to type the equation
    # in several different places. probably not a big deal,
    # but just trying to follow best practices
    sigma_protein = 1.0 - sigma_water
    protein_energy = -lambda_protein * thetaII * sigma_protein * gamma_p
    water_energy = -lambda_water * thetaII * sigma_water * gamma_w
    return protein_energy, water_energy
@njit(signature_or_function=numba.types.UniTuple(float64,2)(float64, float64, float64, float64, float64, float64))
def compute_long_potentials_ij_from_sigmawater_distij_gamma(dist_ij, sigma_water, 
        lambda_protein, gamma_p, lambda_water, gamma_w):
    """
    Compute the protein-mediated and water-mediated (long-range) potentials 
    for a pair of residues. 

    Parameters
    ----------
    See module-level docstring.

    Returns
    -------
    protein_energy : float
        Energy of the protein-mediated contact term for the pair (i,j),
        set to 0 if the pair is masked.
    water_energy : float
        Energy of the water-mediated contact term for the pair (i,j),
        set to 0 if the pair is masked.
    """
    # get indicators and sigma values
    thetaII = compute_thetaII(dist_ij)
    # compute energies
    protein_energy, water_energy = compute_long_potentials_ij_from_sigmawater_thetaII_gamma(
            thetaII, sigma_water, lambda_protein, gamma_p, lambda_water, gamma_w)
    return protein_energy, water_energy
@njit(signature_or_function=numba.types.UniTuple(float64,2)(int64, int64, float64, float64,
        float64, float64[:,:], float64, float64[:,:], int64[:]))
def compute_long_potentials_ij_from_sigmawater_distij(i, j, dist_ij, sigma_water, 
        lambda_protein, protein_gamma, lambda_water, water_gamma, seq_index):
    """
    Compute the protein-mediated and water-mediated (long-range) potentials 
    for a pair of residues. 

    Parameters
    ----------
    See module-level docstring.

    Returns
    -------
    protein_energy : float
        Energy of the protein-mediated contact term for the pair (i,j),
        set to 0 if the pair is masked.
    water_energy : float
        Energy of the water-mediated contact term for the pair (i,j),
        set to 0 if the pair is masked.
    """
    gamma_p = protein_gamma[seq_index[i], seq_index[j]]
    gamma_w = water_gamma[seq_index[i], seq_index[j]]
    # compute energies
    protein_energy, water_energy = compute_long_potentials_ij_from_sigmawater_distij_gamma(
            dist_ij, sigma_water, lambda_protein, gamma_p, lambda_water, gamma_w)
    return protein_energy, water_energy
@njit(signature_or_function=numba.types.UniTuple(float64,2)(int64, int64, float64[:,:], float64,
        float64, float64[:,:], float64, float64[:,:], int64[:]))
def compute_long_potentials_ij_from_sigmawater(i, j, dist_mat, sigma_water, 
        lambda_protein, protein_gamma, lambda_water, water_gamma, seq_index):
    """
    Compute the protein-mediated and water-mediated (long-range) potentials 
    for a pair of residues. 

    Parameters
    ----------
    See module-level docstring.

    Returns
    -------
    protein_energy : float
        Energy of the protein-mediated contact term for the pair (i,j),
        set to 0 if the pair is masked.
    water_energy : float
        Energy of the water-mediated contact term for the pair (i,j),
        set to 0 if the pair is masked.
    """
    dist_ij = dist_mat[i,j]
    # compute energies
    protein_energy, water_energy = compute_long_potentials_ij_from_sigmawater_distij(i, j,
            dist_ij, sigma_water, lambda_protein, protein_gamma, lambda_water, water_gamma, seq_index)
    return protein_energy, water_energy
@njit(signature_or_function=numba.types.UniTuple(float64,2)(
    float64, float64, float64, float64, float64, float64, float64))
def compute_long_potentials_ij_from_rho_thetaII_gamma(rho_i, rho_j, thetaII,
    lambda_protein, gamma_p, lambda_water, gamma_w):
    """
    Compute the protein-mediated and water-mediated (long-range) potentials 
    for a pair of residues. 

    Parameters
    ----------
    See module-level docstring.

    Returns
    -------
    protein_energy : float
        Energy of the protein-mediated contact term for the pair (i,j),
        set to 0 if the pair is masked.
    water_energy : float
        Energy of the water-mediated contact term for the pair (i,j),
        set to 0 if the pair is masked.
    """    
    sigma_water = compute_sigma_water(rho_i, rho_j)
    protein_energy, water_energy = compute_long_potentials_ij_from_sigmawater_thetaII_gamma(
        thetaII, sigma_water, lambda_protein, gamma_p, lambda_water, gamma_w)
    return protein_energy, water_energy    
@njit(signature_or_function=numba.types.UniTuple(float64,2)(
    float64, float64, float64, float64, float64, float64, float64))
def compute_long_potentials_ij_from_rho_distij_gamma(dist_ij, rho_i, rho_j, 
    lambda_protein, gamma_p, lambda_water, gamma_w):
    """
    Compute the protein-mediated and water-mediated (long-range) potentials 
    for a pair of residues. 

    Parameters
    ----------
    See module-level docstring.

    Returns
    -------
    protein_energy : float
        Energy of the protein-mediated contact term for the pair (i,j),
        set to 0 if the pair is masked.
    water_energy : float
        Energy of the water-mediated contact term for the pair (i,j),
        set to 0 if the pair is masked.
    """    
    sigma_water = compute_sigma_water(rho_i, rho_j)
    #assert 0 < sigma_water < 1, f'rho_i: {repr(rho_i)}, rho_j: {repr(rho_j)}, sigma_water: {repr(sigma_water)}'
    protein_energy, water_energy = compute_long_potentials_ij_from_sigmawater_distij_gamma(
        dist_ij, sigma_water, lambda_protein, gamma_p, lambda_water, gamma_w)
    return protein_energy, water_energy
@njit(signature_or_function=numba.types.UniTuple(float64,2)(
    int64, int64, int64, int64[:], int64[:], float64[:,:], float64, float64, float64, float64))
def compute_long_potentials_ij_from_gamma(i, j, min_seq_sep_rho, chain_starts, chain_ends, dist_mat,
    lambda_protein, gamma_p, lambda_water, gamma_w):
    """
    Compute the protein-mediated and water-mediated (long-range) potentials 
    for a pair of residues. 

    Parameters
    ----------
    See module-level docstring.

    Returns
    -------
    protein_energy : float
        Energy of the protein-mediated contact term for the pair (i,j),
        set to 0 if the pair is masked.
    water_energy : float
        Energy of the water-mediated contact term for the pair (i,j),
        set to 0 if the pair is masked.
    """    
    rho_i = compute_rho_i(i, min_seq_sep_rho, chain_starts, chain_ends, dist_mat)
    rho_j = compute_rho_i(j, min_seq_sep_rho, chain_starts, chain_ends, dist_mat)
    protein_energy, water_energy = compute_long_potentials_ij_from_rho_distij_gamma(
        dist_mat[i,j], rho_i, rho_j, lambda_protein, gamma_p, lambda_water, gamma_w)
    return protein_energy, water_energy
@njit(numba.types.UniTuple(float64,2)(int64, int64, int64, int64[:], int64[:], float64[:,:],
                                      float64, float64[:,:], float64, float64[:,:], int64[:]))
def compute_long_potentials_ij(i, j, min_seq_sep_rho, chain_starts, chain_ends, dist_mat,
                               lambda_protein, protein_gamma, lambda_water, water_gamma, seq_index):
    """
    Compute the protein-mediated and water-mediated (long-range) potentials 
    for a pair of residues. 

    Parameters
    ----------
    See module-level docstring.

    Returns
    -------
    protein_energy : float
        Energy of the protein-mediated contact term for the pair (i,j),
        set to 0 if the pair is masked.
    water_energy : float
        Energy of the water-mediated contact term for the pair (i,j),
        set to 0 if the pair is masked.
    """    
    gamma_p = protein_gamma[seq_index[i], seq_index[j]]
    gamma_w = water_gamma[seq_index[i], seq_index[j]]
    return compute_long_potentials_ij_from_gamma(i, j, min_seq_sep_rho, chain_starts, chain_ends, dist_mat,
          lambda_protein, lambda_water, gamma_p, gamma_w)
# feel free to add more functions with different signatures for greater flexibility of use
#
@njit(signature_or_function=float64(float64, float64, float64))
def compute_electrostatic_potential_ij_from_indicator_gamma(electrostatic_indicator, lambda_electrostatic, gamma):
    """
    Compute the solvation-averaged electrostatic potential
    for a pair of residues.   

    Parameters
    ----------
    See module-level docstring.

    Returns
    -------
    electrostatic_energy : float
        Energy of the electrostatic interaction between residues i and j
    """
    # gamma is negative if interaction is favorable and positive if
    # unfavorable, and our lambdas and indicators are all positive by convention,
    # so we don't precede this equation with a negative sign
    #return -lambda_electrostatic * electrostatic_indicator * gamma
    return lambda_electrostatic * electrostatic_indicator * gamma
@njit(signature_or_function=float64(float64, float64, float64, float64))
def compute_electrostatic_potential_ij_from_distij_gamma(l_D, dist_ij, lambda_electrostatic, gamma):
    """
    Compute the solvation-averaged electrostatic potential
    for a pair of residues.   

    Parameters
    ----------
    See module-level docstring.

    Returns
    -------
    electrostatic_energy : float
        Energy of the electrostatic interaction between residues i and j
    """
    indicator = compute_electrostatic_indicator(l_D, dist_ij)
    electrostatic_energy = compute_electrostatic_potential_ij_from_indicator_gamma(
        indicator, lambda_electrostatic, gamma)
    return electrostatic_energy
@njit(signature_or_function=float64(int64, int64, float64, float64[:,:], float64, float64))
def compute_electrostatic_potential_ij_from_gamma(i, j, l_D, dist_mat, lambda_electrostatic, gamma):
    """
    Compute the solvation-averaged electrostatic potential
    for a pair of residues.   

    Parameters
    ----------
    See module-level docstring.

    Returns
    -------
    electrostatic_energy : float
        Energy of the electrostatic interaction between residues i and j
    """
    dist_ij = dist_mat[i,j]
    electrostatic_energy = compute_electrostatic_potential_ij_from_distij_gamma(l_D, dist_ij, lambda_electrostatic, gamma)
    return electrostatic_energy
@njit(signature_or_function=float64(int64, int64, float64, float64[:,:], float64, float64[:,:], int64[:]))
def compute_electrostatic_potential_ij(i, j, l_D, dist_mat, lambda_electrostatic, electrostatic_gamma, seq_index):
    """
    Compute the solvation-averaged electrostatic potential
    for a pair of residues.   

    Parameters
    ----------
    See module-level docstring.

    Returns
    -------
    electrostatic_energy : float
        Energy of the electrostatic interaction between residues i and j
    """
    gamma = electrostatic_gamma[seq_index[i], seq_index[j]]
    electrostatic_energy = compute_electrostatic_potential_ij_from_distij_gamma(
        l_D, dist_mat[i,j], lambda_electrostatic, gamma)
    return electrostatic_energy
# feel free to add more functions with different signatures for greater flexibility of use
#
##########################################################################
# FUNCTIONS TO SUM DIFFERENT ENERGY TYPES OVER AN ENTIRE PROTEIN SYSTEM.
# THESE FUNCTIONS **DO** CHECK MASK CONDITIONS!
#
signature = float64(int64,
                    int64[:], int64[:],
                    float64[:,:],
                    float64, float64[:,:], int64[:])
def compute_burial_potential_total(min_seq_sep_rho, 
                                   chain_starts, chain_ends,
                                   dist_mat, 
                                   lambda_burial, burial_gamma, seq_index):
    """
    Compute the total burial potential for all residues in the protein system.
    Iterates over all residues and sums burial energies.

    Parameters
    ----------
    See module-level docstring.

    Returns
    -------
    total_burial_energy : float
        Sum of burial energies for all residues
    """
    num_res = dist_mat.shape[0]
    total_burial_energy = 0.0
    rho_array = np.zeros(num_res)
    burial_indicators = np.zeros((num_res,3)) # axis 1 ordered (low, medium, high)
    for i in prange(num_res):
        rho_array[i] = compute_rho_i(i, min_seq_sep_rho, chain_starts, chain_ends, dist_mat)
        burial_indicators[i] = compute_burial_indicator_i(rho_array[i])
        energy = compute_burial_potential_i(i, min_seq_sep_rho, chain_starts, chain_ends, dist_mat, lambda_burial, burial_gamma, seq_index)
        total_burial_energy += energy 
    return total_burial_energy
compute_burial_potential_total_parallel = njit(signature_or_function=signature, parallel=True)(compute_burial_potential_total)
compute_burial_potential_total = njit(signature_or_function=signature)(compute_burial_potential_total)
#
signature = float64(int64,
                    int64[:], int64[:],
                    float64[:,:],  
                    float64, float64[:,:], int64[:],)
def compute_direct_potential_total(min_seq_sep_contact,
                                   chain_starts, chain_ends, 
                                   dist_mat,
                                   lambda_direct, direct_gamma, seq_index,):
    """
    Compute the total direct contact potential for the entire protein system.
    Iterates over all residue pairs and sums direct interaction energies.

    Parameters
    ----------
    See module-level docstring

    Returns
    -------
    total_direct_energy : float
        Sum of all direct contact energies
    """
    num_res = dist_mat.shape[0]
    total_direct_energy = 0.0
    # loop over all pairs of residues
    for i in prange(num_res):
        # parallelizing inner loop doesn't make much of a difference
        #for j in prange(i+1, num_res):
        for j in range(i+1, num_res):
            # check mask
            same_chain = check_same_chain(i, j, chain_starts, chain_ends)
            # 2.5 and 8.5 cutoffs: effectively, 
            #    we're truncating the potential where the indicators are almost zero
            #    (thetaI(2.5)==thetaI(8.5)==2.0611536367E-9)
            if not mask_of_pair(min_seq_sep_contact, abs(i-j), 2.5, 8.5,
                                same_chain, dist_mat[i,j]):
                continue # just call it 0 energy if the pair is masked
            energy = compute_direct_potential_ij(i, j, dist_mat, lambda_direct, direct_gamma, seq_index)
            total_direct_energy += energy
    return total_direct_energy
compute_direct_potential_total_parallel = njit(signature_or_function=signature, parallel=True)(compute_direct_potential_total)
compute_direct_potential_total = njit(signature_or_function=signature)(compute_direct_potential_total)
#
signature = numba.types.UniTuple(float64,2)(int64, int64,
                    int64[:], int64[:],
                    float64[:,:],
                    float64, float64[:,:],
                    float64, float64[:,:],
                    int64[:])
def compute_long_potentials_total(min_seq_sep_rho, min_seq_sep_contact,
                                  chain_starts, chain_ends,
                                  dist_mat, 
                                  lambda_protein, protein_gamma, 
                                  lambda_water, water_gamma,
                                  seq_index):
    """
    Compute the total protein-mediated and water-mediated contact potentials 
    for the entire protein structure. Iterates over all residue pairs and sums 
    long-range interaction energies, considering local densities
    for the sigma (protein vs. water mediated) weighting.
    This function also applies the mask as appropriate.

    Parameters
    ----------
    See module-level docstring.

    Returns
    -------
    total_protein_energy : float
        Sum of all protein-mediated contact energies
    total_water_energy : float
        Sum of all water-mediated contact energies
    """
    num_res = dist_mat.shape[0]
    total_protein_energy = 0.0
    total_water_energy = 0.0
    # Pre-compute rho for all residues
    rho_array = np.zeros(num_res)
    for i in prange(num_res):
        rho_array[i] = compute_rho_i(i, min_seq_sep_rho, chain_starts, chain_ends, dist_mat)
    # compute pairwise energies and add to the total of each type
    for i in prange(num_res):
        # parallelizing inner loop doesn't make much of a difference
        #for j in prange(i+1, num_res):
        for j in range(i+1, num_res):
            # check contact mask 
            same_chain = check_same_chain(i, j, chain_starts, chain_ends)
            # 4.5 and 11.5 cutoffs: effectively, 
            #    we're truncating the potential where the indicators are almost zero
            #    (thetaII(4.5)==thetaII(11.5)==2.0611536367E-9
            if not mask_of_pair(min_seq_sep_contact, abs(i-j),
                                4.5, 11.5, same_chain, dist_mat[i,j]):
                continue # just call it 0 energy if the pair is masked
            # compute sigma for this pair from precomputed rhos, then call long potentials
            sigma_water = compute_sigma_water(rho_array[i], rho_array[j])
            protein_energy, water_energy = compute_long_potentials_ij_from_rho_distij_gamma(
                dist_mat[i,j], rho_array[i], rho_array[j], 
                lambda_protein, protein_gamma[seq_index[i], seq_index[j]], 
                lambda_water, water_gamma[seq_index[i], seq_index[j]])
            total_protein_energy += protein_energy
            total_water_energy += water_energy
    return total_protein_energy, total_water_energy
compute_long_potentials_total_parallel = njit(signature_or_function=signature, parallel=True)(compute_long_potentials_total)
compute_long_potential_total = njit(signature_or_function=signature)(compute_long_potentials_total)
#
signature = float64(float64, int64,
                    int64[:], int64[:],
                    float64[:,:], 
                    float64, float64[:,:],
                    int64[:],)
def compute_electrostatic_potential_total(l_D, min_seq_sep_electrostatic,
                                          chain_starts, chain_ends,
                                          dist_mat,
                                          lambda_electrostatic, electrostatic_gamma,
                                          seq_index):
    """
    Compute the total Debye-Huckel electrostatic potential 
    for the entire protein structure. Iterates over all residue pairs and sums 
    electrostatic interaction energies.
    This function also applies the mask as appropriate.

    Parameters
    ----------
    See module-level docstring.

    Returns
    -------
    total_electrostatic_energy : float
        Sum of all electrostatic energies, masked as appropriate
    """
    # will move this check and/or get rid of it
    #if lambda_electrostatic == 0:
    #    return 0.0 # save some time if we're going to set everything to 0 anyway
    num_res = dist_mat.shape[0]
    total_electrostatic_energy = 0.0
    # loop over all pairs of residues
    for i in prange(num_res):
        # parallelizing inner loop doesn't make much of a difference
        #for j in prange(i+1, num_res):
        for j in range(i+1, num_res):
            # check mask
            same_chain = check_same_chain(i, j, chain_starts, chain_ends)
            # unlike the other potentials, the electrostatic potential doesn't
            # decay below some minimum distance, so the lower bound is 0;
            # the upper bound varies with the debye length
            if not mask_of_pair(min_seq_sep_electrostatic, abs(i-j), 0, 10*l_D,
                                same_chain, dist_mat[i,j]):
                continue # just call it 0 energy if the pair is masked
            energy = compute_electrostatic_potential_ij(i, j, l_D, dist_mat, lambda_electrostatic, electrostatic_gamma, seq_index)
            total_electrostatic_energy += energy    
    return total_electrostatic_energy
compute_electrostatic_potential_total_parallel = njit(signature_or_function=signature, parallel=True)(compute_electrostatic_potential_total)
compute_electrostatic_potential_total = njit(signature_or_function=signature)(compute_electrostatic_potential_total)
#
# no numba for this function, since it doesn't have any loops or do any intensive computation
def compute_potential_total(l_D, min_seq_sep_rho, min_seq_sep_contact, min_seq_sep_electrostatic,
                            chain_starts, chain_ends,
                            dist_mat,
                            lambda_direct, direct_gamma,
                            lambda_protein, protein_gamma, 
                            lambda_water, water_gamma, 
                            lambda_burial, burial_gamma,
                            lambda_electrostatic, electrostatic_gamma,
                            seq_index, parallel):
    """
    Compute the total AWSEM energy for the entire protein system.

    CAUTION: this is NOT the sum over all i and j of compute_pair_energy_ij
    (compute_pair_energy_ij is found below with the frustration utilities).
    Taking the sum of compute_pair_energy_ij over all i and j would overcount
    each residue's burial energy, since it is included in the pair energy
    of all contacts in which that residue participates, but the burial energy
    should only be counted once for each residue.
    
    Aggregates direct , protein-mediated, water-mediated, 
    burial, and electrostatic terms.

    Parameters
    ----------
    See module-level docstring.

    Returns
    -------
    total_energy : float
        Total AWSEM energy for the protein system
    """
    direct_args        = (min_seq_sep_contact, 
                          chain_starts, chain_ends,
                          dist_mat,
                          lambda_direct, direct_gamma,
                          seq_index)
    long_args          = (min_seq_sep_rho, min_seq_sep_contact,
                          chain_starts, chain_ends,
                          dist_mat,
                          lambda_protein, protein_gamma,
                          lambda_water, water_gamma,
                          seq_index)
    burial_args        = (min_seq_sep_rho,
                          chain_starts, chain_ends,
                          dist_mat, 
                          lambda_burial, burial_gamma,
                          seq_index)
    electrostatic_args = (l_D, min_seq_sep_electrostatic,
                          chain_starts, chain_ends,
                          dist_mat, 
                          lambda_electrostatic, electrostatic_gamma,
                          seq_index)
    if parallel:
        direct_e = compute_direct_potential_total_parallel(*direct_args)
        protein_e, water_e = compute_long_potentials_total_parallel(*long_args)
        burial_e = compute_burial_potential_total_parallel(*burial_args)
        electrostatic_e = compute_electrostatic_potential_total_parallel(*electrostatic_args)
    else:
        direct_e = compute_direct_potential_total(*direct_args)
        protein_e, water_e = compute_long_potentials_total(*long_args)
        burial_e = compute_burial_potential_total(*burial_args)
        electrostatic_e = compute_electrostatic_potential_total(*electrostatic_args)
    total_energy = direct_e + protein_e + water_e + burial_e + electrostatic_e
    return total_energy
#
#########################################################################################
# PAIR ENERGY: burial(i)+burial(j)+direct(i,j)+protein(i,j)+water(i,j)+electrostatic(i,j)
# important: total energy is NOT sum over all pairs ij of pair_energy(i,j)
# these functions DO NOT check mask conditions
@njit(signature_or_function=float64(
    float64, float64, float64, float64, float64,
    float64, float64, float64, float64, float64, float64,
    float64, float64[:], float64[:], float64, float64))
def compute_pair_energy_ij_useful(
    rho_i, rho_j, thetaI, thetaII, electrostatic_indicator,
    lambda_direct, gamma_d, lambda_protein, gamma_p, lambda_water, gamma_w, 
    lambda_burial, gamma_bi, gamma_bj, lambda_electrostatic, gamma_e):
    # useful parameter set for frustration calculations
    burial_energy_i      = compute_burial_potential_i_from_rho_gamma(rho_i, lambda_burial, gamma_bi)
    burial_energy_j      = compute_burial_potential_i_from_rho_gamma(rho_j, lambda_burial, gamma_bj)
    direct_energy        = compute_direct_potential_ij_from_thetaI_gamma(thetaI, lambda_direct, gamma_d)
    protein_energy, water_energy = compute_long_potentials_ij_from_rho_thetaII_gamma(
                           rho_i, rho_j, thetaII, lambda_protein, gamma_p, lambda_water, gamma_w)
    electrostatic_energy = compute_electrostatic_potential_ij_from_indicator_gamma(
                           electrostatic_indicator, lambda_electrostatic, gamma_e)
    pair_energy = burial_energy_i + burial_energy_j + direct_energy +\
                      protein_energy + water_energy + electrostatic_energy
    return pair_energy
@njit(signature_or_function=float64(
                            int64, int64, float64, int64, int64[:], int64[:],
                            float64[:,:],
                            float64, float64,
                            float64, float64,
                            float64, float64,
                            float64, float64[:], float64[:],
                            float64, float64))
def compute_pair_energy_ij_from_gamma(
                           i, j, l_D, min_seq_sep_rho, chain_starts, chain_ends,
                           dist_mat, 
                           lambda_direct, gamma_d,
                           lambda_protein, gamma_p, 
                           lambda_water, gamma_w, 
                           lambda_burial, gamma_bi, gamma_bj,
                           lambda_electrostatic, gamma_e):
    direct_energy = compute_direct_potential_ij_from_gamma(i, j, 
        dist_mat,
        lambda_direct, gamma_d)
    protein_energy, water_energy = compute_long_potentials_ij_from_gamma(
        i, j, min_seq_sep_rho, chain_starts, chain_ends,
        dist_mat, 
        lambda_protein, gamma_p, lambda_water, gamma_w)
    burial_energy_i = compute_burial_potential_i_from_gamma(i, min_seq_sep_rho, chain_starts, chain_ends,
        dist_mat,
        lambda_burial, gamma_bi)
    burial_energy_j = compute_burial_potential_i_from_gamma(j, min_seq_sep_rho, chain_starts, chain_ends,
        dist_mat,
        lambda_burial, gamma_bj)
    electrostatic_energy = compute_electrostatic_potential_ij_from_gamma(i, j, l_D,
        dist_mat,
        lambda_electrostatic, gamma_e)
    pair_energy = burial_energy_i + burial_energy_j + direct_energy +\
        protein_energy + water_energy + electrostatic_energy
    return pair_energy
#
@njit(signature_or_function=float64(
        int64, int64, float64,
        float64[:,:],
        float64, float64, float64,
        float64, float64[:,:],
        float64, float64[:,:],
        float64, float64[:,:],
        float64, float64[:,:],
        float64, float64[:,:],
        int64[:]))
def compute_pair_energy_ij_from_rho_sigmawater(
        i, j, l_D,
        dist_mat, 
        rho_i, rho_j, sigma_water,
        lambda_direct, direct_gamma,
        lambda_protein, protein_gamma, 
        lambda_water, water_gamma, 
        lambda_burial, burial_gamma,
        lambda_electrostatic, electrostatic_gamma,
        seq_index):
    direct_energy = compute_direct_potential_ij(i, j, dist_mat,
        lambda_direct, direct_gamma,
        seq_index)
    protein_energy, water_energy = compute_long_potentials_ij_from_sigmawater(
        i, j, 
        dist_mat, sigma_water,
        lambda_protein, protein_gamma, lambda_water, water_gamma,
        seq_index)
    burial_energy_i = compute_burial_potential_i_from_rho(i, rho_i,
        lambda_burial, burial_gamma,
        seq_index)
    burial_energy_j = compute_burial_potential_i_from_rho(j, rho_j,
        lambda_burial, burial_gamma,
        seq_index)
    electrostatic_energy = compute_electrostatic_potential_ij(i, j, l_D,
        dist_mat,
        lambda_electrostatic, electrostatic_gamma,
        seq_index)
    pair_energy = burial_energy_i + burial_energy_j + direct_energy +\
        protein_energy + water_energy + electrostatic_energy
    return pair_energy
#
@njit(signature_or_function=float64(int64, int64, float64,
                                    float64[:,:],
                                    float64, float64,
                                    float64, float64[:,:],
                                    float64, float64[:,:],
                                    float64, float64[:,:],
                                    float64, float64[:,:],
                                    float64, float64[:,:],
                                    int64[:]))
def compute_pair_energy_ij_from_rho(i, j, l_D,
                           dist_mat, 
                           rho_i, rho_j,
                           lambda_direct, direct_gamma,
                           lambda_protein, protein_gamma, 
                           lambda_water, water_gamma, 
                           lambda_burial, burial_gamma,
                           lambda_electrostatic, electrostatic_gamma,
                           seq_index):
    """
    Compute the "pair energy" for residues i and j, defined as the sum of:
    - Direct contact energy
    - Protein-mediated contact energy
    - Water-mediated contact energy
    - Burial energies for both residues
    - Electrostatic interaction energy, if requested

    This quantity is used in the calculation of the frustration index:
    Frustration Index = -1 * (pair energy - DECOY_AVERAGE) / DECOY_STDEV

    Parameters
    ----------
    See module-level docstring.

    Returns
    -------
    pair_energy : float
        Total "pair energy" of residues i and j
    """
    sigma_water = compute_sigma_water(rho_i, rho_j)
    pair_energy = compute_pair_energy_ij_from_rho_sigmawater(
        i, j, l_D,
        dist_mat, 
        rho_i, rho_j, sigma_water,
        lambda_direct, direct_gamma,
        lambda_protein, protein_gamma, 
        lambda_water, water_gamma, 
        lambda_burial, burial_gamma,
        lambda_electrostatic, electrostatic_gamma,
        seq_index)
    return pair_energy
#
@njit(signature_or_function=float64(int64, int64, float64, int64, int64[:], int64[:],
                                    float64[:,:],
                                    float64, float64[:,:],
                                    float64, float64[:,:],
                                    float64, float64[:,:],
                                    float64, float64[:,:],
                                    float64, float64[:,:],
                                    int64[:]))
def compute_pair_energy_ij(i, j, l_D, min_seq_sep_rho, chain_starts, chain_ends,
                           dist_mat, 
                           lambda_direct, direct_gamma,
                           lambda_protein, protein_gamma, 
                           lambda_water, water_gamma, 
                           lambda_burial, burial_gamma,
                           lambda_electrostatic, electrostatic_gamma,
                           seq_index):
    """
    Compute the "pair energy" for residues i and j, defined as the sum of:
    - Direct contact energy
    - Protein-mediated contact energy
    - Water-mediated contact energy
    - Burial energies for both residues
    - Electrostatic interaction energy, if requested

    This quantity is used in the calculation of the frustration index:
    Frustration Index = -1 * (pair energy - DECOY_AVERAGE) / DECOY_STDEV

    Parameters
    ----------
    See module-level docstring.

    Returns
    -------
    pair_energy : float
        Total "pair energy" of residues i and j
    """
    direct_energy = compute_direct_potential_ij(i, j, dist_mat,
        lambda_direct, direct_gamma,
        seq_index)
    protein_energy, water_energy = compute_long_potentials_ij(
        i, j, min_seq_sep_rho, chain_starts, chain_ends,
        dist_mat, 
        lambda_protein, protein_gamma, lambda_water, water_gamma,
        seq_index)
    burial_energy_i = compute_burial_potential_i(i, min_seq_sep_rho, chain_starts, chain_ends,
        dist_mat,
        lambda_burial, burial_gamma,
        seq_index)
    burial_energy_j = compute_burial_potential_i(j, min_seq_sep_rho, chain_starts, chain_ends,
        dist_mat,
        lambda_burial, burial_gamma,
        seq_index)
    electrostatic_energy = compute_electrostatic_potential_ij(i, j, l_D,
        dist_mat,
        lambda_electrostatic, electrostatic_gamma,
        seq_index)
    pair_energy = burial_energy_i + burial_energy_j + direct_energy +\
        protein_energy + water_energy + electrostatic_energy
    """
    alternatively, the body of this function could look like this:
    aa_i = seq_index[i]
    aa_j = seq_index[j]
    gamma_d = direct_gamma[aa_i, aa_j]
    gamma_p = protein_gamma[aa_i, aa_j]
    gamma_w = water_gamma[aa_i, aa_j]
    gamma_bi = burial_gamma[aa_i,:]
    gamma_bj = burial_gamma[aa_j,:]
    gamma_e = electrostatic_gamma[aa_i, aa_j]
    pair_energy = compute_pair_energy_ij_from_gamma(
                           i, j, l_D, min_seq_sep_rho, chain_starts, chain_ends,
                           dist_mat, 
                           lambda_direct, gamma_d,
                           lambda_protein, gamma_p, 
                           lambda_water, gamma_w, 
                           lambda_burial, gamma_bi, gamma_bj,
                           lambda_electrostatic, gamma_e)
    """
    return pair_energy
#
#########################################################################################
# POTTS MODEL:  (N,N,q,q) for (N,N) dist_mat and (q,q) gammas
signature = float64[:,:](
    int64,
    int64[:], int64[:],
    float64[:,:],
    float64, float64[:,:])
def compute_potts_model_h(
    min_seq_sep_rho,
    chain_starts, chain_ends,
    dist_mat, 
    lambda_burial, burial_gamma):
    assert dist_mat.shape[0] == dist_mat.shape[1]
    num_aa = dist_mat.shape[0]
    num_aa_types = burial_gamma.shape[0]
    assert burial_gamma.shape[1] == 3
    h = np.zeros((num_aa, num_aa_types))
    for i in prange(num_aa):
        for q in range(num_aa_types):
            gamma = burial_gamma[q]
            h[i,q] = compute_burial_potential_i_from_gamma(
                i, min_seq_sep_rho, chain_starts, chain_ends, dist_mat, lambda_burial, gamma)
    h = -h # i guess we define it as the negative of the actual potential?
    return h
compute_potts_model_h_parallel = njit(signature_or_function=signature, parallel=True)(compute_potts_model_h)
compute_potts_model_h = njit(signature_or_function=signature)(compute_potts_model_h)
#
signature = float64[:,:,:,:](
    float64, int64, int64, int64,
    int64[:], int64[:], float64, float64,
    float64[:,:],
    float64, float64[:,:],
    float64, float64[:,:],
    float64, float64[:,:],
    float64, float64[:,:])
def compute_potts_model_J(
    l_D, min_seq_sep_rho, min_seq_sep_contact, min_seq_sep_electrostatic,
    chain_starts, chain_ends, max_dist_contact, max_dist_electrostatic,
    dist_mat,  
    lambda_direct, direct_gamma, 
    lambda_protein, protein_gamma,
    lambda_water, water_gamma, 
    lambda_electrostatic, electrostatic_gamma):
    # check input
    assert dist_mat.shape[0] == dist_mat.shape[1]
    assert direct_gamma.shape[0] == direct_gamma.shape[1]
    assert direct_gamma.shape == protein_gamma.shape == water_gamma.shape == electrostatic_gamma.shape
    num_aa = dist_mat.shape[0]
    num_aa_types = direct_gamma.shape[0]
    # precompute rho
    rho_array = np.zeros(num_aa)
    for i in prange(num_aa):
        rho_array[i] = compute_rho_i(i, min_seq_sep_rho, chain_starts, chain_ends, dist_mat)
    J = np.zeros((num_aa, num_aa, num_aa_types, num_aa_types))
    for i in prange(num_aa):
        for j in range(num_aa):
            if i==j:
                J[i,j,:,:] = 0.0
                continue
            dist_ij = dist_mat[i,j]
            rho_i = rho_array[i]
            rho_j = rho_array[j]
            same_chain = check_same_chain(i, j, chain_starts, chain_ends)
            contact_mask_ij = mask_of_pair(min_seq_sep=min_seq_sep_contact, seq_sep=abs(j-i), 
                                           min_dist=0.0, max_dist=max_dist_contact, 
                                           same_chain=same_chain, dist_ij=dist_ij)
            electrostatic_mask_ij = mask_of_pair(min_seq_sep_electrostatic, abs(j-i), 0, max_dist_electrostatic, same_chain, dist_ij)
            for qi in range(num_aa_types):
                for qj in range(qi, num_aa_types):
                    gamma_dij = direct_gamma[qi,qj]
                    gamma_pij = protein_gamma[qi,qj]
                    gamma_wij = water_gamma[qi,qj]
                    gamma_eij = electrostatic_gamma[qi,qj]
                    direct_energy = compute_direct_potential_ij_from_distij_gamma(dist_ij, lambda_direct, gamma_dij)
                    protein_energy, water_energy = compute_long_potentials_ij_from_rho_distij_gamma(
                        dist_ij, rho_i, rho_j, lambda_protein, gamma_pij, lambda_water, gamma_wij)
                    contact_energy = contact_mask_ij * (direct_energy + protein_energy + water_energy)
                    electrostatic_energy = electrostatic_mask_ij * compute_electrostatic_potential_ij_from_distij_gamma(
                                                                        l_D, dist_ij, lambda_electrostatic, gamma_eij)
                    #                         
                    energy = contact_energy + electrostatic_energy
                    J[i,j,qi,qj] = energy
                    J[i,j,qj,qi] = energy
    J = -J # i guess we define it as the negative of the actual potential?
    return J
compute_potts_model_J_parallel = njit(signature_or_function=signature, parallel=True)(compute_potts_model_J)
compute_potts_model_J = njit(signature_or_function=signature)(compute_potts_model_J)
#
########################################################################################
# PAIR ENERGY MATRIX FOR FRUSTRATION CALCULATIONS -- NOT SURE WHAT TO DO WITH THIS. MIGHT DELETE
"""
signature = float64[:,:](float64,
                         int64, int64,
                         int64[:], int64[:], float64,
                         float64[:,:],
                         float64, float64[:,:],
                         float64, float64[:,:],
                         float64, float64[:,:],
                         float64, float64[:,:],
                         float64, float64[:,:],
                         int64[:])
def compute_pair_energy_matrix(l_D,
                       min_seq_sep_rho, min_seq_sep_frust_index,
                       chain_starts, chain_ends, max_dist,
                       dist_mat,
                       lambda_direct, direct_gamma,
                       lambda_protein, protein_gamma, 
                       lambda_water, water_gamma, 
                       lambda_burial, burial_gamma,
                       lambda_electrostatic, electrostatic_gamma,
                       seq_index):
    """"""
    Make matrix of the same shape as the distance matrix, 
    where each element is the pair energy, or np.nan if masked.

    Parameters
    ----------
    See module-level docstring

    Returns
    -------
    pair_energy_matrix : np.array(dist_mat.shape)
        matrix where the element (i,j) is the pair energy of (i,j)
        (if unmasked) or np.nan (if masked)
    """"""
    # Pre-compute rho for all residues
    num_res = dist_mat.shape[0]
    rho_array = np.zeros(num_res)
    for i in prange(num_res):
        rho_array[i] = compute_rho_i(i, min_seq_sep_rho, chain_starts, chain_ends, dist_mat)
    # fill in the matrix
    num_res = dist_mat.shape[0]
    pair_energy_matrix = np.empty((num_res, num_res))
    for i in prange(num_res):
        for j in range(i,num_res):
            # check mask
            same_chain = check_same_chain(i, j, chain_starts, chain_ends)
            # the idea is that this is the matrix we'll use to calculate frustration indices,
            # so we set the minimum distance to 0 (as it always is for frustration calculations)
            # and let the maximum distance be a variable
            unmasked = mask_of_pair(min_seq_sep_frust_index, abs(i-j), 0.0, max_dist,
                                 same_chain, dist_mat[i,j],)
            if unmasked:
                pair_energy_matrix[i,j] = compute_pair_energy_ij_from_rho(
                    i, j, l_D, 
                    dist_mat, 
                    rho_array[i], rho_array[j],
                    lambda_direct, direct_gamma,
                    lambda_protein, protein_gamma, 
                    lambda_water, water_gamma, 
                    lambda_burial, burial_gamma,
                    lambda_electrostatic, electrostatic_gamma,
                    seq_index)
            else:
                pair_energy_matrix[i,j] = np.nan
            pair_energy_matrix[j,i] = pair_energy_matrix[i,j]            
    return pair_energy_matrix
compute_pair_energy_matrix_parallel = njit(signature_or_function=signature, parallel=True)(compute_pair_energy_matrix)
compute_pair_energy_matrix = njit(signature_or_function=signature)(compute_pair_energy_matrix)
"""