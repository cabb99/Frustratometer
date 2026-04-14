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



@pytest.mark.parametrize("max_dist,min_sep,chain_breaks", [
    (10.0, 3, None),
    (15.0, None, None),
    (10.0, 3, [5]),
    (None, 3, [3, 6]),
])
def test_sparse_mask_matches_dense(small_system, max_dist, min_sep, chain_breaks):
    """Dense and sparse masks must agree; cross-chain pairs always pass separation."""
    L, Q, dm, ci, cj, cd, _ = small_system

    dense_mask = compute_mask(dm, max_dist, min_sep, chain_breaks=chain_breaks)
    sparse_mask = compute_mask_sparse(ci, cj, cd, L, max_dist, min_sep, chain_breaks=chain_breaks)

    reconstructed = np.zeros((L, L), dtype=bool)
    reconstructed[sparse_mask.row, sparse_mask.col] = True

    off_diag = ~np.eye(L, dtype=bool)
    np.testing.assert_array_equal(reconstructed[off_diag], dense_mask[off_diag])

    # When chain_breaks + min_sep are active, cross-chain pairs that pass the
    # distance filter must always be included (they are not bonded in sequence).
    if chain_breaks is not None and min_sep is not None:
        boundaries = [0] + chain_breaks + [L]
        for c1 in range(len(boundaries) - 1):
            for c2 in range(c1 + 1, len(boundaries) - 1):
                i = boundaries[c1 + 1] - 1  # last residue of chain c1
                j = boundaries[c2]           # first residue of chain c2
                if max_dist is None or dm[i, j] <= max_dist:
                    assert dense_mask[i, j]


def test_sparse_potts_roundtrip_and_lookup(small_system):
    """dense->sparse->dense recovers J at contacts; lookup indexes them correctly."""
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


def test_structure_detects_chain_breaks():
    """Structure must detect chain boundaries from a multi-chain CIF file."""
    from pathlib import Path
    from frustratometer.classes.Structure import Structure

    cif_path = Path(__file__).parent / "data" / "5msm.cif"
    if not cif_path.exists():
        pytest.skip("5msm.cif not available")

    s = Structure(cif_path, chain=None)
    # 5msm has 6 chains (A,B,C,D,E,F) -> 5 break points
    assert s.chain_breaks is not None
    assert len(s.chain_breaks) == 5
