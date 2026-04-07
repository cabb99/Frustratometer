from pathlib import Path
from typing import Union, Optional, Tuple

import numpy as np
import scipy.spatial.distance as sdist
from scipy.spatial import cKDTree
import prody
import itertools


# ---------------------------------------------------------------------------
# Coordinate selection helpers (one per method)
# ---------------------------------------------------------------------------

def _parse_structure(pdb_file, chain):
    """Parse the structure and construct a chain selection string."""
    structure = prody.parsePDB(str(pdb_file))
    chain_selection = '' if chain is None else f' and chain {chain}'
    return structure, chain_selection


def _select_ca(structure, chain_selection):
    """Select CA atoms."""
    sel = structure.select('protein and name CA' + chain_selection)
    if sel is None or sel.numAtoms() == 0:
        raise IndexError("Empty selection for distance map (method='CA').")
    return sel.getCoords()


def _select_cb(structure, chain_selection):
    """Select CB atoms, using CA for glycine residues."""
    sel = structure.select(
        '(protein and name CB or (resname GLY and name CA))' + chain_selection
    )
    if sel is None or sel.numAtoms() == 0:
        raise IndexError("Empty selection for distance map (method='CB').")
    return sel.getCoords()


def _select_cb_force(structure, chain_selection):
    """Calculate CB coordinates for all residues, even if CB is missing (e.g. glycine)."""
    sel_CA = structure.select('protein and name CA' + chain_selection)
    sel_N  = structure.select('protein and name N'  + chain_selection)
    sel_C  = structure.select('protein and name C'  + chain_selection)

    if sel_CA is None or sel_N is None or sel_C is None:
        raise IndexError("Empty selection for CB_force (missing CA/N/C atoms).")
    if not (sel_CA.numAtoms() == sel_N.numAtoms() == sel_C.numAtoms()):
        raise ValueError("CA, N, and C selections must have the same length for CB_force.")

    CA = sel_CA.getCoords()
    N  = sel_N.getCoords()
    C  = sel_C.getCoords()

    v_CA_C = C - CA
    v_CA_N = N - CA
    v_CA_C /= np.linalg.norm(v_CA_C, axis=1, keepdims=True)
    v_CA_N /= np.linalg.norm(v_CA_N, axis=1, keepdims=True)

    cross_CA_C_CA_N = np.cross(v_CA_C, v_CA_N)
    cross_CA_C_CA_N /= np.linalg.norm(cross_CA_C_CA_N, axis=1, keepdims=True)

    cross_cross_CA_N = np.cross(cross_CA_C_CA_N, v_CA_N)
    cross_cross_CA_N /= np.linalg.norm(cross_cross_CA_N, axis=1, keepdims=True)

    return -0.531020 * v_CA_N - 1.206181 * cross_CA_C_CA_N + 0.789162 * cross_cross_CA_N + CA


def _get_residue_coords(structure, chain_selection, method):
    """Get representative coordinates for each residue based on the specified method."""
    if method == 'CA':
        return _select_ca(structure, chain_selection)
    elif method == 'CB':
        return _select_cb(structure, chain_selection)
    elif method == 'CB_force':
        return _select_cb_force(structure, chain_selection)
    else:
        raise ValueError(
            f"Invalid method '{method}'. Accepted methods are 'CA', 'CB', 'minimum', and 'CB_force'."
        )


def _full_minimum_distance(structure, chain_selection):
    """Calculate the full minimum distance matrix by computing all atom-atom distances and taking minima per residue pair."""
    sel = structure.select('protein' + chain_selection)
    if sel is None or sel.numAtoms() == 0:
        raise IndexError("Empty selection for distance map (method='minimum').")
    coords = sel.getCoords()
    distance_matrix = sdist.squareform(sdist.pdist(coords))
    resids = sel.getResindices()
    unique_res = np.unique(resids)
    selections = np.array([resids == a for a in unique_res])
    n_res = len(unique_res)
    dm = np.zeros((n_res, n_res))
    for i, j in itertools.combinations(range(n_res), 2):
        d = distance_matrix[selections[i]][:, selections[j]].min()
        dm[i, j] = d
        dm[j, i] = d
    return dm


def _sparse_minimum_distance(structure, chain_selection, max_distance):
    """Calculate a sparse minimum distance matrix by computing all atom-atom distances and including only pairs within max_distance."""
    sel = structure.select('protein' + chain_selection)
    if sel is None or sel.numAtoms() == 0:
        raise IndexError("Empty selection for distance map (method='minimum').")

    coords = sel.getCoords()
    resindices = sel.getResindices()

    unique_res = np.unique(resindices)
    n_res = unique_res.size
    res_to_idx = {res: i for i, res in enumerate(unique_res)}
    atom_res_idx = np.vectorize(res_to_idx.get)(resindices)

    tree = cKDTree(coords)
    pairs = tree.query_pairs(r=max_distance, output_type='ndarray')
    if pairs.size == 0:
        empty = np.array([], dtype=np.intp)
        return empty, empty, np.array([], dtype=float), n_res

    i_atoms = pairs[:, 0]
    j_atoms = pairs[:, 1]
    dists = np.linalg.norm(coords[i_atoms] - coords[j_atoms], axis=1)

    min_dist = {}
    for ia, ja, d in zip(i_atoms, j_atoms, dists):
        ri = atom_res_idx[ia]
        rj = atom_res_idx[ja]
        if ri == rj:
            continue
        if ri > rj:
            ri, rj = rj, ri
        key = (ri, rj)
        prev = min_dist.get(key)
        if prev is None or d < prev:
            min_dist[key] = d

    if not min_dist:
        empty = np.array([], dtype=np.intp)
        return empty, empty, np.array([], dtype=float), n_res

    row_list, col_list, data_list = [], [], []
    for (ri, rj), d in min_dist.items():
        row_list.extend([ri, rj])
        col_list.extend([rj, ri])
        data_list.extend([d, d])

    return (np.array(row_list, dtype=np.intp),
            np.array(col_list, dtype=np.intp),
            np.array(data_list, dtype=float),
            n_res)

def _coords_to_sparse(coords, max_distance):
    """Calculate a sparse distance matrix from coordinates by including only pairs within max_distance."""
    n = coords.shape[0]
    tree = cKDTree(coords)
    pairs = tree.query_pairs(r=max_distance, output_type='ndarray')
    if pairs.size == 0:
        empty = np.array([], dtype=np.intp)
        return empty, empty, np.array([], dtype=float), n

    i_idx = pairs[:, 0]
    j_idx = pairs[:, 1]
    dists = np.linalg.norm(coords[i_idx] - coords[j_idx], axis=1)

    contact_i = np.concatenate([i_idx, j_idx]).astype(np.intp)
    contact_j = np.concatenate([j_idx, i_idx]).astype(np.intp)
    data = np.concatenate([dists, dists])
    return contact_i, contact_j, data, n

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_full_distance_matrix(
    pdb_file: Union[Path, str],
    chain: Optional[str],
    method: str = 'CB',
) -> np.ndarray:
    """
    Calculate the full (dense) distance matrix of the specified atoms in a PDB file.

    Parameters
    ----------
    pdb_file : Path or str
        Path to the PDB file.
    chain : str or None
        Chain ID or chain IDs (space-separated) of the protein.
    method : {'CA', 'CB', 'minimum', 'CB_force'}
        Method for defining the representative coordinates.

    Returns
    -------
    np.ndarray
        Symmetric distance matrix of shape (L, L).
    """
    structure, chain_selection = _parse_structure(pdb_file, chain)
    if method == 'minimum':
        return _full_minimum_distance(structure, chain_selection)
    coords = _get_residue_coords(structure, chain_selection, method)
    return sdist.squareform(sdist.pdist(coords))


def get_sparse_distance_matrix(
    pdb_file: Union[Path, str],
    chain: Optional[str],
    method: str,
    max_distance: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Calculate a sparse distance matrix of the specified atoms/residues in a PDB file.

    Parameters
    ----------
    pdb_file : Path or str
        Path to the PDB file.
    chain : str or None
        Chain ID or chain IDs (space-separated) of the protein.
    method : {'CA', 'CB', 'minimum', 'CB_force'}
        Method for defining the representative coordinates.
    max_distance : float
        Maximum distance (angstrom) to include. Pairs beyond this are not stored.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, int]
        (contact_i, contact_j, distances, L). Both (i,j) and (j,i) stored.
    """
    if max_distance <= 0:
        raise ValueError("max_distance must be > 0")
    structure, chain_selection = _parse_structure(pdb_file, chain)
    if method == 'minimum':
        return _sparse_minimum_distance(structure, chain_selection, max_distance)
    coords = _get_residue_coords(structure, chain_selection, method)
    return _coords_to_sparse(coords, max_distance)
