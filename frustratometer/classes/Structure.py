from .. import pdb
import prody
import os
import Bio.PDB.Polypeptide as poly
import numpy as np
from typing import Union
from pathlib import Path
import tempfile
import logging
import warnings

__all__ = ['Structure', 'SparseMatrix']

logger = logging.getLogger(__name__)

residue_names=[]


class SparseMatrix:
    """COO sparse matrix wrapping the functional API in ``pdb.sparse``.

    Stores ``(row, col, data)`` arrays and a scalar *shape* (sequence length L).
    When used as a mask (no distances), *data* is ``None``.

    Parameters
    ----------
    row : array-like (N,)
        Row indices.
    col : array-like (N,)
        Column indices.
    data : array-like (N,) or None
        Distance values.  ``None`` for mask-only matrices.
    shape : int or None
        Sequence length L.  Inferred from ``max(row, col) + 1`` when *None*.
    """

    __slots__ = ('row', 'col', 'data', 'shape')

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

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------
    def lookup(self, query_i, query_j):
        """Look up distances at ``(query_i, query_j)`` via binary search."""
        return pdb.sparse.lookup(self.row, self.col, self.data, self.shape,
                             query_i, query_j)

    # ------------------------------------------------------------------
    # Mask computation
    # ------------------------------------------------------------------
    def compute_mask(self, maximum_contact_distance=None,
                     minimum_sequence_separation=None, chain_breaks=None):
        """Return a mask-only ``SparseMatrix`` (data=None) of passing pairs."""
        mr, mc, ms = pdb.sparse.compute_mask(
            self.row, self.col, self.data, self.shape,
            maximum_contact_distance=maximum_contact_distance,
            minimum_sequence_separation=minimum_sequence_separation,
            chain_breaks=chain_breaks)
        return SparseMatrix(mr, mc, data=None, shape=ms)

    # ------------------------------------------------------------------
    # Filtering / slicing
    # ------------------------------------------------------------------
    def filter(self, init, fin):
        """Return a new ``SparseMatrix`` restricted to ``[init, fin)`` and re-indexed."""
        nr, nc, nd, ns = pdb.sparse.filter(
            self.row, self.col, self.data, self.shape, init, fin)
        return SparseMatrix(nr, nc, nd, ns)

    # ------------------------------------------------------------------
    # Dense reconstruction
    # ------------------------------------------------------------------
    def to_dense(self, fill=np.inf):
        """Reconstruct an L×L dense array.

        Parameters
        ----------
        fill : float
            Value for entries not stored.  Default ``np.inf``.
            Diagonal is always 0.0.  For masks, stored pairs get 1.0.
        """
        return pdb.sparse.to_dense(self.row, self.col, self.data, self.shape, fill)

    # ------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.row)

    def __repr__(self):
        kind = 'mask' if self.data is None else 'distance'
        return f"SparseMatrix({kind}, nnz={len(self)}, L={self.shape})"

    def __iter__(self):
        # check that our attributes all make sense
        assert self.row.shape == self.col.shape == (self.row.shape[0],) # tuple of length 1
        if isinstance(self.data, np.ndarray):
            assert self.data.shape == self.row.shape
        elif not (self.data is None):
            raise AssertionError(f'self.data was {self.data}')
        N = self.row.shape[0]
        n = 0
        while n < N:
            yield (self.row[n], self.col[n], None if self.data is None else self.data[n]) 
            n += 1

    def __and__(self, other):
        # create a new SparseMatrix that acts as a boolean mask
        # for the indices present in self and other
        if not isinstance(other, SparseMatrix):
            return NotImplemented
        else:
            if self.shape != other.shape:
                raise ValueError("intersection of two boolean arrays of different shapes is ill-defined")
            new_row, new_col = self._and_helper(other)
            return SparseMatrix(new_row, new_col)

    def __rand__(self, other):
        if not isinstance(other, SparseMatrix):
            return NotImplemented
        else:
            return self.__and__(other)

    def __iand__(self, other):
        # in-place modification of self
        if not isinstance(other, SparseMatrix):
            return NotImplemented
        if self.shape != other.shape:
            raise ValueError("intersection of two boolean arrays of different shapes is ill-defined")
        self.row, self.col = self._and_helper(other) # modify self.row and self.col in place
        self.data = None
        return self

    def _and_helper(self, other):
        #new_row = []
        #new_col = []
        #other_set = set(zip(other.row, other.col))
        #for i, j in zip(self.row, self.col):
        #    if (i,j) in other_set:
        #        new_row.append(i)
        #        new_col.append(j)
        #return np.array(new_row), np.array(new_col)
        
        # Vectorized membership using 64-bit integer keys; preserves order of self.
        # Similar speed to the above option, may be more memory-efficient
        # if numpy significantly outperforms native python
        L = self.shape 
        keys_self = (self.row.astype(np.int64) * L) + self.col.astype(np.int64)
        keys_other = (other.row.astype(np.int64) * L) + other.col.astype(np.int64)
        mask = np.in1d(keys_self, keys_other, assume_unique=False)
        return self.row[mask].astype(np.intp), self.col[mask].astype(np.intp)

class Structure:

    def __init__(self, pdb_file: Union[Path,str], chain: Union[str,None]=None, seq_selection: str = None, aligned_sequence: str = None, filtered_aligned_sequence: str = None,
                distance_matrix_method:str = 'CB', pdb_directory: Path = None, repair_pdb: bool = None, sparse: bool = False, max_sparse_distance: float = 1000)->object:
        
        """
        Generates structure object. Both PDB and CIF format files are accepted as input.

        Parameters
        ----------
        pdb_file :  str
            PDB file path

        chain : str
            PDB chain name. If "chain=None", all chains will be included.

        seq_selection: str
            Subsequence selection command, using Prody select module. 

            *If wanting to use original PDB indexing, 
            set seq_selection as: "resnum `{initial_index}to{final_index}`"

            *If wanting to use absolute PDB indexing (first residue has index=0),
            set seq_selection as: "resindex `{initial_index}to{final_index}`"

            Note that using "to" will create a substructure including both the initial and final designated residue.
            If you would like to not include the final desginated residue, replace "to" with ":"

            Note that the algorithm will account for any gaps in the PDB and adjust the provided residue
            range accordingly.
            
        distance_matrix_method: str
            The method to use for calculating the distance matrix. 
            Defaults to 'CB', which uses the CB atom for all residues except GLY, which uses the CA atom. 
            Other options are 'CA' for using only the CA atom, 
            and 'minimum' for using the minimum distance between all atoms in each residue.    

        pdb_directory: str
            Directory where repaired pdb will be stored. Defaults to the system temporary directory.

        repair_pdb: bool or None
            If True, the PDB file is always repaired (missing residues filled, heteroatoms removed).
            If False, no repair is attempted.
            If None (default), the structure is built without repair first; if the
            sequence length does not match the distance matrix, repair is attempted
            automatically.
            Note that a pdb file will be produced, regardless of input file format.

        Returns
        -------
        Structure object
        """        
        if pdb_directory is None:
            pdb_directory = Path(tempfile.gettempdir())

        if repair_pdb is None:
            # Auto-detect: try without repair, retry with repair on validation failure
            try:
                self._init_structure(pdb_file, chain, seq_selection, aligned_sequence,
                                     filtered_aligned_sequence, distance_matrix_method,
                                     pdb_directory, repair_pdb=False, sparse=sparse, 
                                     max_sparse_distance=max_sparse_distance)
                self._validate_structure()
            except Exception as e:
                logger.info("Structure validation failed without repair (%s), retrying with repair_pdb=True", e)
                self._init_structure(pdb_file, chain, seq_selection, aligned_sequence,
                                     filtered_aligned_sequence, distance_matrix_method,
                                     pdb_directory, repair_pdb=True, sparse=sparse, 
                                     max_sparse_distance=max_sparse_distance)
                self._validate_structure()
        else:
            self._init_structure(pdb_file, chain, seq_selection, aligned_sequence,
                                 filtered_aligned_sequence, distance_matrix_method,
                                 pdb_directory, repair_pdb=repair_pdb, sparse=sparse, 
                                 max_sparse_distance=max_sparse_distance)
            self._validate_structure()

    def _validate_structure(self):
        """Check that the structure is internally consistent."""
        L_seq = len(self.sequence)

        #Distance matrix length must match sequence length
        if isinstance(self.distance_matrix, SparseMatrix):
            L_dm = self.distance_matrix.shape
        elif isinstance(self.distance_matrix, np.ndarray):
            L_dm = self.distance_matrix.shape[0]
        else:
            raise TypeError(f"Unexpected distance_matrix type: {type(self.distance_matrix)}")
        
        if L_seq != L_dm:
            raise ValueError(
                f"Sequence length ({L_seq}) does not match distance matrix "
                f"size ({L_dm}). The PDB may have missing residues. "
                f"Try setting repair_pdb=True.")

    @property
    def _is_sparse(self):
        """True when sparse distance data is stored and no dense matrix is cached."""
        return self._sparse_distance_matrix is not None and self._dense_distance_matrix is None

    @property
    def distance_matrix(self):
        """Return dense distance matrix if available, otherwise sparse COO tuple."""
        if self._dense_distance_matrix is not None:
            return self._dense_distance_matrix
        return self._sparse_distance_matrix

    @distance_matrix.setter
    def distance_matrix(self, value):
        """Allow direct assignment for backward compatibility."""
        if isinstance(value, SparseMatrix):
            self._sparse_distance_matrix = value
            self._dense_distance_matrix = None
        else:
            self._dense_distance_matrix = value

    def get_dense_distance_matrix(self):
        """Lazily compute and cache the dense distance matrix from PDB."""
        if self._dense_distance_matrix is None:
            self._dense_distance_matrix = pdb.get_dense_distance_matrix(
                pdb_file=self.pdb_file, chain=self.chain,
                method=self.distance_matrix_method)
            # Apply seq_selection slicing if needed
            if self.seq_selection is not None:
                self._dense_distance_matrix = self._dense_distance_matrix[
                    self.init_index_shift:self.fin_index_shift,
                    self.init_index_shift:self.fin_index_shift]
        return self._dense_distance_matrix

    @staticmethod
    def _filter_sparse_tuple(sparse_dm, init, fin):
        """Filter a SparseMatrix to a sub-range [init, fin) and re-index."""
        return sparse_dm.filter(init, fin)

    def _init_structure(self, pdb_file, chain, seq_selection, aligned_sequence,
                        filtered_aligned_sequence, distance_matrix_method,
                        pdb_directory, repair_pdb, sparse, max_sparse_distance):

        try:
            #Check if file exists
            pdb_file=Path(pdb_file)
            assert pdb_file.exists()
        except AssertionError:
            #Attempt to download pdb file
            pdb_file=str(pdb_file)
            if len(pdb_file)==4:
                self.pdbID=pdb_file
                print(f"Downloading {self.pdbID} from the PDB")
                pdb_file=pdb.download(self.pdbID, pdb_directory)
            else:
                raise FileNotFoundError(f"Provided file {pdb_file} does not exist")

        
        self.pdbID=pdb_file.stem
        self.pdb_file=pdb_file
        self.chain=chain
        self.distance_matrix_method=distance_matrix_method
        self.filtered_aligned_sequence=filtered_aligned_sequence
        self.aligned_sequence=aligned_sequence

        self.seq_selection=seq_selection

        if self.seq_selection==None:
            self.init_index_shift=0

            if repair_pdb:
                pdb.repair_pdb(pdb_file, chain, pdb_directory)
                self.pdb_file=str(pdb_directory/f"{self.pdbID}_cleaned.pdb")

            if ".pdb" in str(pdb_file) or repair_pdb==True:
                self.structure = prody.parsePDB(str(self.pdb_file), chain=self.chain).select(f"protein")
            else:
                self.structure=prody.parseMMCIF(str(self.pdb_file),chain=self.chain).select(f"protein")
        else:
            assert len(self.seq_selection.replace("to"," to ").replace(":"," : ").split())>=4, "Please correctly input your residue selection"
            
            if self.chain==None:
                raise ValueError("Please provide a chain name")

            self.init_index=int(self.seq_selection.replace("to"," to ").replace(":"," : ").split()[1].replace("`",""))
            self.fin_index=int(self.seq_selection.replace("to"," to ").replace(":"," : ").split()[3].replace("`",""))
            
            #Account for pdbs that have starting indices greater than one and find any gaps in the pdb.
            gap_indices=[]; atom_line_count=0

            if ".cif" in str(pdb_file):
                extension="cif"
                shift=2;index_shift=3 
            else:
                extension="pdb"
                shift=0;index_shift=0

            with open(pdb_file,"r") as f:
                for line in f:
                    if line.split()[0]=="ATOM" and line.split()[4+shift]==self.chain:
                        try:
                            res_index=''.join(i for i in line.split()[5+index_shift] if i.isdigit())
                            next_res_index=''.join(i for i in next(f).split()[5+index_shift] if i.isdigit())
                            if int(next_res_index)-int(res_index)>1:
                                gap_indices.extend(list(range(int(res_index)+1,int(next_res_index))))
                            if atom_line_count==0 and poly.is_aa(line.split()[3+shift]):
                                self.pdb_init_index=int(line.split()[5+index_shift])
                            atom_line_count+=1
                        except:
                            continue

            if "resnum" in self.seq_selection:
                assert self.init_index>=self.pdb_init_index, "Please pick an initial index within the pdb's original indices"
                self.init_index_shift=self.init_index-self.pdb_init_index
                self.fin_index_shift=self.fin_index-self.pdb_init_index+1
                if repair_pdb:
                    pdb.repair_pdb(pdb_file, chain, pdb_directory)
                    self.pdb_file=f"{pdb_directory}/{self.pdbID}_cleaned.pdb"
                    self.select_gap_indices=[i for i in gap_indices if self.init_index<=i<=self.fin_index]
                    self.fin_index_shift-=len(self.select_gap_indices)
                    self.seq_selection=f"resnum `{self.init_index_shift+1}to{self.fin_index_shift}`"
            elif "resindex" in self.seq_selection:
                self.init_index_shift=self.init_index
                self.fin_index_shift=self.fin_index+1
                if repair_pdb:
                    pdb.repair_pdb(pdb_file, chain, pdb_directory)
                    self.pdb_file=f"{pdb_directory}/{self.pdbID}_cleaned.pdb"
                    self.chain="A"

            if ".pdb" in str(pdb_file) or repair_pdb==True:
                self.structure = prody.parsePDB(str(self.pdb_file), chain=self.chain).select(f"protein and {self.seq_selection}")
            else:
                self.structure=prody.parseMMCIF(str(self.pdb_file),chain=self.chain).select(f"protein and {self.seq_selection}")

        self.sequence=pdb.get_sequence(self.pdb_file,self.chain)

        # Distance matrix storage
        self._sparse = sparse
        if sparse:
            # Compute the less stringent sparse matrix for electrostatic calculations (max_distance=40.0)
            # then filter it down for the main sparse matrix (max_distance=15.0)
            self._sparse_distance_matrix_elec = SparseMatrix(
                *pdb.get_sparse_distance_matrix(
                    pdb_file=self.pdb_file, chain=self.chain,
                    method=self.distance_matrix_method, max_distance=max_sparse_distance))
            keep = self._sparse_distance_matrix_elec.data <= 15.0
            self._sparse_distance_matrix = SparseMatrix(
                self._sparse_distance_matrix_elec.row[keep],
                self._sparse_distance_matrix_elec.col[keep],
                self._sparse_distance_matrix_elec.data[keep],
                self._sparse_distance_matrix_elec.shape)
            self._dense_distance_matrix = None
        else:
            self._dense_distance_matrix = pdb.get_dense_distance_matrix(
                pdb_file=self.pdb_file, chain=self.chain,
                method=self.distance_matrix_method)
            self._sparse_distance_matrix = None
            self._sparse_distance_matrix_elec = None

        self.full_pdb_distance_matrix = self.distance_matrix

        self.z_coordinates=self.structure.select('((name CB) or (resname GLY and name CA))').getCoords()

        # Detect chain breaks from per-residue chain IDs
        ca_sel = self.structure.select('name CA')
        chids = ca_sel.getChids()
        breaks = np.where(chids[:-1] != chids[1:])[0] + 1
        self.chain_breaks = breaks.tolist() if len(breaks) > 0 else None

        if self.seq_selection!=None:
            if self._is_sparse:
                self._sparse_distance_matrix = self._filter_sparse_tuple(
                    self._sparse_distance_matrix, self.init_index_shift, self.fin_index_shift)
                self._sparse_distance_matrix_elec = self._filter_sparse_tuple(
                    self._sparse_distance_matrix_elec, self.init_index_shift, self.fin_index_shift)
            else:
                self._dense_distance_matrix = self._dense_distance_matrix[
                    self.init_index_shift:self.fin_index_shift,
                    self.init_index_shift:self.fin_index_shift]
            self.sequence=self.sequence[self.init_index_shift:self.fin_index_shift]

        if self.aligned_sequence is not None:
            if self._is_sparse:
                # aligned_sequence mapping requires a dense distance matrix;
                # reconstruct it from the pdb
                self.get_dense_distance_matrix()
            self.full_to_aligned_index_dict=pdb.full_to_filtered_aligned_mapping(self.aligned_sequence,self.filtered_aligned_sequence)
            self.mapped_distance_matrix=np.full((len(self.filtered_aligned_sequence), len(self.filtered_aligned_sequence)), np.inf)
            pos1, pos2 = np.meshgrid(list(self.full_to_aligned_index_dict.keys()), list(self.full_to_aligned_index_dict.keys()), 
                                    indexing='ij', sparse=True)
            modpos1, modpos2 = np.meshgrid(list(self.full_to_aligned_index_dict.values()), list(self.full_to_aligned_index_dict.values()), 
                                    indexing='ij', sparse=True)
            self.mapped_distance_matrix[modpos1,modpos2]=self._dense_distance_matrix[pos1,pos2]
            np.fill_diagonal(self.mapped_distance_matrix, 0)

        else:
            if self.seq_selection==None:
                self.full_to_aligned_index_dict=dict(zip(range(len(self.sequence)), range(len(self.sequence))))
                self.mapped_distance_matrix=self.distance_matrix if not self._is_sparse else None
            else:
                self.full_to_aligned_index_dict=dict(zip(range(self.init_index_shift,self.fin_index_shift+1), range(len(self.sequence))))
                self.mapped_distance_matrix=self.distance_matrix if not self._is_sparse else None

    @classmethod
    def from_pdb_list(cls, pdb_files, chain=None, pdb_directory=None, repair_pdb=None, **kwargs):
        """Build multiple Structures, repairing PDBs in parallel when needed.

        Parameters
        ----------
        pdb_files : list of str or Path
            PDB/CIF file paths.
        chain : str or None
            Chain to use for all structures.
        pdb_directory : Path, optional
            Directory for repaired files.
        repair_pdb : bool or None
            True -> repair all in parallel, then build.
            False -> build all without repair.
            None (default) -> build without repair first; batch-repair
            only the ones that fail validation, then rebuild those.
        **kwargs
            Extra keyword arguments passed to each Structure().

        Returns
        -------
        list[Structure]
        """
        if pdb_directory is None:
            pdb_directory = Path(tempfile.gettempdir())

        if repair_pdb is True:
            cleaned = pdb.repair_pdbs([(f, chain) for f in pdb_files], pdb_directory)
            return [cls(p, chain, repair_pdb=False, **kwargs) for p in cleaned]

        if repair_pdb is False:
            return [cls(f, chain, repair_pdb=False, **kwargs) for f in pdb_files]

        # Auto-detect: try without repair, batch-repair failures
        structures = {}
        needs_repair = []
        for f in pdb_files:
            try:
                structures[str(f)] = cls(f, chain, repair_pdb=False, **kwargs)
            except ValueError:
                needs_repair.append(f)

        if needs_repair:
            cleaned = pdb.repair_pdbs([(f, chain) for f in needs_repair], pdb_directory)
            for f, p in zip(needs_repair, cleaned):
                structures[str(f)] = cls(p, chain, repair_pdb=False, **kwargs)

        return [structures[str(f)] for f in pdb_files]

    @classmethod
    def full_pdb(cls,pdb_file: Union[Path,str], chain: Union[str,None]=None, aligned_sequence: str = None, filtered_aligned_sequence: str = None,
                distance_matrix_method:str = 'CB', pdb_directory: Path = None, repair_pdb: bool = None):
        warnings.warn("The class method 'full_pdb' is now depreciated. You can now simply call the Structure class to create a full pdb or spliced pdb object.")
        return cls(pdb_file=pdb_file,
                   chain=chain,
                   aligned_sequence=aligned_sequence,
                   filtered_aligned_sequence=filtered_aligned_sequence,
                   distance_matrix_method=distance_matrix_method,
                   pdb_directory=pdb_directory,
                   repair_pdb=repair_pdb)


    @classmethod
    def spliced_pdb(cls,pdb_file: Union[Path,str], chain: Union[str,None]=None, seq_selection: str = None, aligned_sequence: str = None, filtered_aligned_sequence: str = None,
                distance_matrix_method:str = 'CB', pdb_directory: Path = None, repair_pdb: bool = None):
        warnings.warn("The class method 'spliced_pdb' is now depreciated. You can now simply call the Structure class to create a full pdb or spliced pdb object.")
        return cls(pdb_file=pdb_file,
                    chain=chain,
                    seq_selection=seq_selection,
                    aligned_sequence=aligned_sequence,
                    filtered_aligned_sequence=filtered_aligned_sequence,
                    distance_matrix_method=distance_matrix_method,
                    pdb_directory=pdb_directory,
                    repair_pdb=repair_pdb,)
    
    def plot_distance_map(self, interaction_type: bool = False, config=None,
                          ax=None, **kwargs):
        """Plot the inter-residue distance map.

        Parameters
        ----------
        interaction_type : bool
            If True, colour contacts by interaction band.
        config : PlotConfig, optional
            Visual settings.  Fields can also be overridden via ``**kwargs``.
        ax : matplotlib Axes, optional
            Axes to draw on.
        **kwargs
            Forwarded to PlotConfig (e.g. ``vmax=15``, ``cmap='hot'``).

        Returns
        -------
        matplotlib Axes
        """
        chain_label = f' chain {self.chain}' if self.chain else ''
        mode = 'sparse' if self._is_sparse else 'dense'
        kwargs.setdefault('title', f'Distance Map ({mode}){chain_label}')

        if self._is_sparse:
            sparse_dm = getattr(self, '_sparse_distance_matrix_elec',
                                None) or self._sparse_distance_matrix
            if interaction_type:
                return pdb.plot_sparse_interaction_map(
                    sparse_dm, config=config, ax=ax, **kwargs)
            return pdb.plot_sparse_distance_map(
                sparse_dm, config=config, ax=ax, **kwargs)
        else:
            if interaction_type:
                return pdb.plot_interaction_map(
                    self.distance_matrix, config=config, ax=ax, **kwargs)
            return pdb.plot_distance_map(
                self.distance_matrix, config=config, ax=ax, **kwargs)

    # @property
    # def sequence(self):
    #     return self._sequence
    
    # # Set a new sequence in case someone needs to calculate the energy of a diferent sequence with the same structure
    # @sequence.setter
    # def sequence(self, value :str):
    #     assert len(value) == len(self._sequence)
    #     self._sequence = value

    # @property
    # def pdb_file(self):
    #     return str(self._pdb_file)
    
    # @pdb_file.setter
    # def pdb_file(self,value: str):
    #     self._pdb_file=value

    # @property
    # def chain(self):
    #     return self._chain
    
    # @chain.setter
    # def chain(self,value):
    #     self._chain=value
    
    # @property
    # def distance_matrix_method(self):
    #     return self._distance_matrix_method
    
    # @distance_matrix_method.setter
    # def distance_matrix_method(self,value):
    #     self._distance_matrix_method = value

    # @property
    # def init_index(self):
    #     return self._init_index
    
    # @init_index.setter
    # def init_index(self,value):
    #     self._init_index = value

    # @property
    # def fin_index(self):
    #     return self._fin_index
    
    # @fin_index.setter
    # def fin_index(self,value):
    #     self._fin_index = value
