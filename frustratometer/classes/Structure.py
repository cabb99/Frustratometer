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

__all__ = ['Structure']

logger = logging.getLogger(__name__)

residue_names=[]

class Structure:

    def __init__(self, pdb_file: Union[Path,str], chain: Union[str,None]=None, seq_selection: str = None, aligned_sequence: str = None, filtered_aligned_sequence: str = None,
                distance_matrix_method:str = 'CB', pdb_directory: Path = None, repair_pdb: bool = None)->object:
        
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
                                     pdb_directory, repair_pdb=False)
                self._validate_structure()
            except Exception as e:
                logger.info("Structure validation failed without repair (%s), retrying with repair_pdb=True", e)
                self._init_structure(pdb_file, chain, seq_selection, aligned_sequence,
                                     filtered_aligned_sequence, distance_matrix_method,
                                     pdb_directory, repair_pdb=True)
                self._validate_structure()
        else:
            self._init_structure(pdb_file, chain, seq_selection, aligned_sequence,
                                 filtered_aligned_sequence, distance_matrix_method,
                                 pdb_directory, repair_pdb=repair_pdb)
            self._validate_structure()

    def _validate_structure(self):
        """Check that the structure is internally consistent."""
        L_seq = len(self.sequence)
        L_dm = self.distance_matrix.shape[0]
        if L_seq != L_dm:
            raise ValueError(
                f"Sequence length ({L_seq}) does not match distance matrix "
                f"shape ({L_dm}x{L_dm}). The PDB may have missing residues. "
                f"Try setting repair_pdb=True.")

    def _init_structure(self, pdb_file, chain, seq_selection, aligned_sequence,
                        filtered_aligned_sequence, distance_matrix_method,
                        pdb_directory, repair_pdb):

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
        self.distance_matrix=pdb.get_distance_matrix(pdb_file=self.pdb_file,chain=self.chain,
                                                     method=self.distance_matrix_method)
        self.full_pdb_distance_matrix=self.distance_matrix

        self.z_coordinates=self.structure.select('((name CB) or (resname GLY and name CA))').getCoords()

        if self.seq_selection!=None:
            self.distance_matrix=self.distance_matrix[self.init_index_shift:self.fin_index_shift,
                                                    self.init_index_shift:self.fin_index_shift]
            self.sequence=self.sequence[self.init_index_shift:self.fin_index_shift]

        if self.aligned_sequence is not None:
            self.full_to_aligned_index_dict=pdb.full_to_filtered_aligned_mapping(self.aligned_sequence,self.filtered_aligned_sequence)
            self.mapped_distance_matrix=np.full((len(self.filtered_aligned_sequence), len(self.filtered_aligned_sequence)), np.inf)
            pos1, pos2 = np.meshgrid(list(self.full_to_aligned_index_dict.keys()), list(self.full_to_aligned_index_dict.keys()), 
                                    indexing='ij', sparse=True)
            modpos1, modpos2 = np.meshgrid(list(self.full_to_aligned_index_dict.values()), list(self.full_to_aligned_index_dict.values()), 
                                    indexing='ij', sparse=True)
            self.mapped_distance_matrix[modpos1,modpos2]=self.distance_matrix[pos1,pos2]
            np.fill_diagonal(self.mapped_distance_matrix, 0)

        else:
            if self.seq_selection==None:
                self.full_to_aligned_index_dict=dict(zip(range(len(self.sequence)), range(len(self.sequence))))
                self.mapped_distance_matrix=self.distance_matrix
            else:
                self.full_to_aligned_index_dict=dict(zip(range(self.init_index_shift,self.fin_index_shift+1), range(len(self.sequence))))
                self.mapped_distance_matrix=self.distance_matrix

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
            True → repair all in parallel, then build.
            False → build all without repair.
            None (default) → build without repair first; batch-repair
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
