"""Tests for sparse Potts model infrastructure (mask, convert, lookup)."""
import numpy as np
import pytest
from frustratometer.frustration.frustration import (
    compute_mask,
    compute_mask_sparse,
    potts_model_dense_to_sparse,
    potts_model_sparse_to_dense,
    build_contact_lookup,
)


@pytest.fixture
def small_system():
    """10-residue system with a random distance matrix and Potts model."""
    rng = np.random.default_rng(42)
    L, Q = 10, 21

    # Symmetric distance matrix with zero diagonal
    dm = rng.uniform(3, 20, size=(L, L))
    dm = (dm + dm.T) / 2
    np.fill_diagonal(dm, 0)

    # Sparse distance representation (both directions, like cKDTree output)
    ii, jj = np.triu_indices(L, k=1)
    contact_i = np.concatenate([ii, jj]).astype(np.intp)
    contact_j = np.concatenate([jj, ii]).astype(np.intp)
    contact_d = np.concatenate([dm[ii, jj], dm[ii, jj]])

    # Symmetric dense Potts model
    potts_model = {
        'h': rng.standard_normal((L, Q)),
        'J': rng.standard_normal((L, L, Q, Q)),
    }
    potts_model['J'] = (potts_model['J'] + potts_model['J'].transpose(1, 0, 3, 2)) / 2

    return L, Q, dm, contact_i, contact_j, contact_d, potts_model



@pytest.mark.parametrize("max_dist,min_sep", [
    (10.0, 3),
    (15.0, None),
])
def test_sparse_mask_matches_dense(small_system, max_dist, min_sep):
    """Sparse mask must select the same off-diagonal (i,j) pairs as the dense mask."""
    L, Q, dm, ci, cj, cd, _ = small_system

    dense_mask = compute_mask(dm, max_dist, min_sep)
    sparse_i, sparse_j = compute_mask_sparse(ci, cj, cd, L, max_dist, min_sep)

    reconstructed = np.zeros((L, L), dtype=bool)
    reconstructed[sparse_i, sparse_j] = True

    off_diag = ~np.eye(L, dtype=bool)
    np.testing.assert_array_equal(reconstructed[off_diag], dense_mask[off_diag])


def test_sparse_potts_roundtrip_and_lookup(small_system):
    """dense→sparse→dense recovers J at contacts; lookup indexes them correctly."""
    L, Q, dm, _, _, _, potts_model = small_system

    mask = compute_mask(dm, maximum_contact_distance=10.0, minimum_sequence_separation=3)
    sparse_pm = potts_model_dense_to_sparse(potts_model, mask)
    recovered = potts_model_sparse_to_dense(sparse_pm)

    # h unchanged
    np.testing.assert_array_equal(recovered['h'], potts_model['h'])

    # J matches at masked positions, zero elsewhere
    ci, cj = np.where(mask)
    np.testing.assert_allclose(recovered['J'][ci, cj, :, :], potts_model['J'][ci, cj, :, :])
    np.testing.assert_array_equal(recovered['J'][~mask, :, :], 0)


def test_contact_lookup(small_system):
    """Lookup must cover every contact and point back correctly."""
    L, _, dm, _, _, _, potts_model = small_system

    mask = compute_mask(dm, maximum_contact_distance=10.0, minimum_sequence_separation=3)
    sparse_pm = potts_model_dense_to_sparse(potts_model, mask)
    offsets, partners, indices = build_contact_lookup(sparse_pm['contact_i'], sparse_pm['contact_j'], L)

    recovered_pairs = set()
    for p in range(L):
        for k in range(offsets[p], offsets[p + 1]):
            recovered_pairs.add((p, partners[k]))
            assert sparse_pm['contact_i'][indices[k]] == p
            assert sparse_pm['contact_j'][indices[k]] == partners[k]

    original_pairs = set(zip(sparse_pm['contact_i'].tolist(), sparse_pm['contact_j'].tolist()))
    assert recovered_pairs == original_pairs
