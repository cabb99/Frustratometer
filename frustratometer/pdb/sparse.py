"""Sparse COO utilities — functional API for sparse distance / mask arrays.

All functions operate on plain NumPy arrays (row, col, data, shape) so users
who prefer a purely functional workflow never need to instantiate a class.
The OOP wrapper :class:`~frustratometer.classes.Structure.SparseMatrix` in
``Structure.py`` delegates to these functions.
"""

import numpy as np

__all__ = [
    'lookup',
    'compute_mask',
    'filter',
    'to_dense',
]


# ------------------------------------------------------------------
# Lookup
# ------------------------------------------------------------------
def lookup(row, col, data, shape, query_i, query_j):
    """Binary-search lookup of distances at specific ``(query_i, query_j)`` positions.

    Equivalent to ``dense_matrix[query_i, query_j]`` without materialising
    the full L×L matrix.  Every queried pair **must** already exist in
    ``(row, col)``.

    Parameters
    ----------
    row, col : array-like (N,)
        Stored coordinate indices.
    data : array-like (N,)
        Distance values.  Must not be *None*.
    shape : int
        Sequence length *L*.
    query_i, query_j : array-like
        Positions to look up.

    Returns
    -------
    np.ndarray
        Distances at the queried positions.
    """
    if data is None:
        raise ValueError("Cannot lookup distances on mask data (data is None).")
    row = np.asarray(row, dtype=np.int64)
    col = np.asarray(col, dtype=np.int64)
    data = np.asarray(data, dtype=float)
    src_key = row * shape + col
    sort_idx = np.argsort(src_key)
    qry_key = np.asarray(query_i, dtype=np.int64) * shape + np.asarray(query_j, dtype=np.int64)
    pos = np.searchsorted(src_key, qry_key, sorter=sort_idx)
    return data[sort_idx[pos]]


# ------------------------------------------------------------------
# Mask computation
# ------------------------------------------------------------------
def compute_mask(row, col, data, shape,
                        maximum_contact_distance=None,
                        minimum_sequence_separation=None,
                        chain_breaks=None):
    """Return indices of pairs passing distance / sequence-separation criteria.

    Parameters
    ----------
    row, col : array-like (N,)
        Stored coordinate indices.
    data : array-like (N,) or None
        Distance values.  Required when *maximum_contact_distance* is set.
    shape : int
        Sequence length *L*.
    maximum_contact_distance : float, optional
        Keep pairs with distance ≤ this value.
    minimum_sequence_separation : int, optional
        Keep pairs with ``|i − j| ≥`` this value.
    chain_breaks : list of int, optional
        Cross-chain pairs always satisfy sequence separation.

    Returns
    -------
    mask_row : np.ndarray
        Row indices of accepted pairs.
    mask_col : np.ndarray
        Column indices of accepted pairs.
    shape : int
        Unchanged sequence length (passed through for convenience).
    """
    row = np.asarray(row, dtype=np.intp)
    col = np.asarray(col, dtype=np.intp)
    keep = np.ones(len(row), dtype=bool)

    if maximum_contact_distance is not None:
        if data is None:
            raise ValueError("Cannot filter by distance on mask data (data is None).")
        keep &= np.asarray(data, dtype=float) <= maximum_contact_distance

    if minimum_sequence_separation is not None:
        pos_i = row.astype(np.float64)
        pos_j = col.astype(np.float64)
        if chain_breaks is not None:
            for brk in chain_breaks:
                pos_i = np.where(row >= brk, pos_i + minimum_sequence_separation, pos_i)
                pos_j = np.where(col >= brk, pos_j + minimum_sequence_separation, pos_j)
        keep &= np.abs(pos_i - pos_j) >= minimum_sequence_separation

    return row[keep], col[keep], int(shape)


# ------------------------------------------------------------------
# Filtering / slicing
# ------------------------------------------------------------------
def filter(row, col, data, shape, init, fin):
    """Restrict to entries in ``[init, fin)`` and re-index to zero.

    Parameters
    ----------
    row, col : array-like (N,)
        Stored coordinate indices.
    data : array-like (N,) or None
        Distance values.
    shape : int
        Original sequence length.
    init, fin : int
        Half-open range to keep.

    Returns
    -------
    new_row, new_col : np.ndarray
        Re-indexed indices.
    new_data : np.ndarray or None
        Filtered distances (or *None* if input was *None*).
    new_shape : int
        ``fin − init``.
    """
    row = np.asarray(row, dtype=np.intp)
    col = np.asarray(col, dtype=np.intp)
    keep = (row >= init) & (row < fin) & (col >= init) & (col < fin)
    new_data = np.asarray(data, dtype=float)[keep] if data is not None else None
    return row[keep] - init, col[keep] - init, new_data, fin - init


# ------------------------------------------------------------------
# Dense reconstruction
# ------------------------------------------------------------------
def to_dense(row, col, data, shape, fill=np.inf):
    """Reconstruct an L×L dense array from COO arrays.

    Parameters
    ----------
    row, col : array-like (N,)
        Stored coordinate indices.
    data : array-like (N,) or None
        Distance values.  If *None*, stored pairs are set to 1.0 (mask mode).
    shape : int
        Sequence length *L*.
    fill : float
        Value for entries not stored.  Default ``np.inf``.
        Diagonal is always 0.0.

    Returns
    -------
    np.ndarray (L, L)
    """
    L = int(shape)
    mat = np.full((L, L), fill, dtype=float)
    np.fill_diagonal(mat, 0.0)
    row = np.asarray(row, dtype=np.intp)
    col = np.asarray(col, dtype=np.intp)
    if data is not None:
        mat[row, col] = np.asarray(data, dtype=float)
    else:
        mat[row, col] = 1.0
    return mat
