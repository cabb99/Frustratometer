"""Tests for distance matrix computation (full and sparse)."""
import numpy as np
import pytest
from pathlib import Path
from frustratometer.pdb.distance import get_dense_distance_matrix, get_sparse_distance_matrix
from frustratometer.pdb.sparse import SparseDistanceMatrix

PDB_FILE = Path(__file__).parent / "data" / "6u5e.pdb"
CHAIN = "A"
MAX_DISTANCE = 15.0


def test_full_matches_known_reference():
    """Full CB matrix must reproduce the saved reference exactly."""
    ref_path = Path(__file__).parent / "data" / "6JXX_A_CB_CB_Distance_Map.txt"
    pdb_path = Path(__file__).parent / "data" / "6JXX_A.pdb"
    reference = np.loadtxt(ref_path)
    computed = get_dense_distance_matrix(pdb_path, "A", method="CB")
    np.testing.assert_array_equal(computed, reference)


@pytest.mark.parametrize("method", ["CA", "CB", "minimum", "CB_force"])
def test_sparse_is_lossless_subset_of_full(method):
    """Reconstructing a dense matrix from sparse entries must equal
    the full matrix everywhere within the cutoff (and zero elsewhere)."""
    full = get_dense_distance_matrix(PDB_FILE, CHAIN, method=method)
    sdm = get_sparse_distance_matrix(PDB_FILE, CHAIN, method, MAX_DISTANCE)
    assert isinstance(sdm, SparseDistanceMatrix)
    assert sdm.shape == full.shape[0]

    # Reconstruct dense from sparse
    reconstructed = np.zeros((sdm.shape, sdm.shape))
    reconstructed[sdm.row, sdm.col] = sdm.data

    # Where full <= cutoff and off-diagonal, reconstructed must match
    mask = (full > 0) & (full <= MAX_DISTANCE)
    np.testing.assert_allclose(reconstructed[mask], full[mask], atol=1e-10)
    # Where full > cutoff, reconstructed must be zero (not stored)
    np.testing.assert_array_equal(reconstructed[~mask], 0)


def test_sparse_cutoff_respected():
    """Tighter cutoff produces fewer entries, all within bound."""
    sdm_wide = get_sparse_distance_matrix(PDB_FILE, CHAIN, "CB", 15.0)
    sdm_narrow = get_sparse_distance_matrix(PDB_FILE, CHAIN, "CB", 8.0)
    assert len(sdm_narrow) < len(sdm_wide)
    assert sdm_narrow.data.max() <= 8.0
    assert sdm_wide.data.max() <= 15.0
    assert sdm_wide.data.max() > 8.0  


def test_structure_sparse_flag():
    """sparse=True populates _sparse_distance_matrix; False leaves it None."""
    import frustratometer
    s_on = frustratometer.Structure(PDB_FILE, CHAIN, repair_pdb=False, sparse=True)
    assert s_on._sparse_distance_matrix is not None
    assert isinstance(s_on._sparse_distance_matrix, SparseDistanceMatrix)
    assert s_on._sparse_distance_matrix.shape == len(s_on.sequence)

    s_off = frustratometer.Structure(PDB_FILE, CHAIN, repair_pdb=False, sparse=False)
    assert s_off._sparse_distance_matrix is None
