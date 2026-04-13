"""Sparse distance matrix stored in COO format."""

import numpy as np

__all__ = ['SparseDistanceMatrix']


class SparseDistanceMatrix:
    """Coordinate-format (COO) sparse distance matrix.

    Stores only non-zero (i, j, distance) triples for residue pairs within a
    distance cutoff.  When used as a mask (no distances), *data* is ``None``.

    Parameters
    ----------
    row : np.ndarray (N,)
        Row indices.
    col : np.ndarray (N,)
        Column indices.
    data : np.ndarray (N,) or None
        Distance values.  ``None`` for mask-only matrices.
    shape : int or None
        Sequence length L.  Inferred from ``max(row, col) + 1`` when *None*.
    """

    __slots__ = ('row', 'col', 'data', 'shape', '_sort_idx')

    def __init__(self, row, col, data=None, shape=None):
        self.row = np.asarray(row, dtype=np.intp)
        self.col = np.asarray(col, dtype=np.intp)
        self.data = np.asarray(data, dtype=float) if data is not None else None
        if shape is None:
            if len(self.row) == 0:
                shape = 0
            else:
                shape = int(max(self.row.max(), self.col.max())) + 1
        self.shape = int(shape)
        self._sort_idx = None  # lazy-cached argsort for lookup

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------
    def lookup(self, query_i, query_j):
        """Look up distances at specific (i, j) positions via binary search.

        Equivalent to ``dense_matrix[query_i, query_j]`` without materialising
        the full L×L matrix.

        Every queried pair **must** already exist in (row, col).
        """
        if self.data is None:
            raise ValueError("Cannot lookup distances on a mask (data is None).")
        if self._sort_idx is None:
            src_key = self.row.astype(np.int64) * self.shape + self.col.astype(np.int64)
            self._sort_idx = np.argsort(src_key)
        src_key = self.row.astype(np.int64) * self.shape + self.col.astype(np.int64)
        qry_key = np.asarray(query_i, dtype=np.int64) * self.shape + np.asarray(query_j, dtype=np.int64)
        pos = np.searchsorted(src_key, qry_key, sorter=self._sort_idx)
        return self.data[self._sort_idx[pos]]

    # ------------------------------------------------------------------
    # Mask computation
    # ------------------------------------------------------------------
    def compute_mask(self, maximum_contact_distance=None,
                     minimum_sequence_separation=None, chain_breaks=None):
        """Return a mask-only SparseDistanceMatrix (data=None) of passing pairs.

        Parameters
        ----------
        maximum_contact_distance : float or None
            Keep pairs with distance <= this value.
        minimum_sequence_separation : int or None
            Keep pairs with |i - j| >= this value.
        chain_breaks : list of int or None
            Cross-chain pairs always satisfy sequence separation.
        """
        keep = np.ones(len(self.row), dtype=bool)

        if maximum_contact_distance is not None:
            if self.data is None:
                raise ValueError("Cannot filter by distance on a mask (data is None).")
            keep &= self.data <= maximum_contact_distance

        if minimum_sequence_separation is not None:
            pos_i = self.row.astype(np.float64)
            pos_j = self.col.astype(np.float64)
            if chain_breaks is not None:
                for brk in chain_breaks:
                    pos_i = np.where(self.row >= brk, pos_i + minimum_sequence_separation, pos_i)
                    pos_j = np.where(self.col >= brk, pos_j + minimum_sequence_separation, pos_j)
            keep &= np.abs(pos_i - pos_j) >= minimum_sequence_separation

        return SparseDistanceMatrix(self.row[keep], self.col[keep],
                                    data=None, shape=self.shape)

    # ------------------------------------------------------------------
    # Filtering / slicing
    # ------------------------------------------------------------------
    def filter(self, init, fin):
        """Return a new SparseDistanceMatrix restricted to [init, fin) and re-indexed."""
        keep = (self.row >= init) & (self.row < fin) & (self.col >= init) & (self.col < fin)
        new_data = self.data[keep] if self.data is not None else None
        return SparseDistanceMatrix(self.row[keep] - init, self.col[keep] - init,
                                    data=new_data, shape=fin - init)

    # ------------------------------------------------------------------
    # Dense reconstruction
    # ------------------------------------------------------------------
    def to_dense(self, fill=np.inf):
        """Reconstruct an L×L dense array.

        Parameters
        ----------
        fill : float
            Value for entries not stored.  Default ``np.inf``.
            Diagonal is always 0.0.  For masks (data=None), stored pairs get 1.0.
        """
        L = self.shape
        mat = np.full((L, L), fill, dtype=float)
        np.fill_diagonal(mat, 0.0)
        if self.data is not None:
            mat[self.row, self.col] = self.data
        else:
            mat[self.row, self.col] = 1.0
        return mat

    # ------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.row)

    def __repr__(self):
        kind = 'mask' if self.data is None else 'distance'
        return f"SparseDistanceMatrix({kind}, nnz={len(self)}, L={self.shape})"
